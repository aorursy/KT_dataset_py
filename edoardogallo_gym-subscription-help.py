import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("../input/gym-subscription/abbonamenti_all.xlsx")
data.head()
# I divide employees from clients,'Operatori' is employees
operatori = data[data["description"] == 'Operatori']

data_no_work = data.drop(operatori.index)
# get rid of the columns that I think are not useful

data_no_work.drop(['Idsubscription', 'Idcategory', 'description'],axis = 1, inplace = True)
data_no_work
# divide monthly subscription from the dayly subs

data_no_work['Period'].unique()
data_no_work_M = data_no_work[data_no_work['Period'] == 'M']

# we won't use the day dataframe since is very small
data_no_work_D = data_no_work[data_no_work['Period'] == 'G']


# Let's have a look how many times we have a particular iduser 
# because if we have only 1 value it means that he/she hasn't renewed

ren = data_no_work_M['Iduser'].value_counts()
ren
# Select the clients that haven't renewed
no_renewal = ren[ren == 1]
no_renewal.shape
# now I get rid of these clients to analyze better the remaining

data_no_work_M = data_no_work_M.set_index('Iduser')

for i in no_renewal.index:
    data_no_work_M.drop(i, inplace=True)
    
# change to datetime object the starting and ending dates of subscription

data_no_work_M['Start_date'] = pd.to_datetime(data_no_work_M['Start_date'])
data_no_work_M['End_date'] = pd.to_datetime(data_no_work_M['End_date'])

# add the user column back in the dataframe
data_no_work_M['Iduser'] = data_no_work_M.index

data_no_work_M = data_no_work_M.reset_index(drop = True)
data_no_work_M
summer = []
l = []

for i in range(0,len(data_no_work_M)-1):
    
    # if id user is the same of the previous one
    if data_no_work_M.iloc[i+1,-1] == data_no_work_M.iloc[i, -1]: 
        
        # make 'start next sub' - 'end last sub'
        diff = data_no_work_M.iloc[i+1, 7] - data_no_work_M.iloc[i, 8]
        
        # keep track of the end of sub to check if is in summer
        dat = data_no_work_M.iloc[i, 8]
        
        # append timedelta to a list
        if diff.days <= 15:
            l.append('<15')
            
            # don't care if is summer or not(only for this case)
            summer.append(0)
        elif diff.days <= 31:
            l.append('<1M')
            
            # check if the date is in summer
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
        elif diff.days <= 61:
            l.append('<2M')
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
        elif diff.days <= 91:
            l.append('<3M')
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
        elif diff.days <= 121:
            l.append('<4M')
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
        elif diff.days <= 151:
            l.append('<5M')
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
        elif diff.days <= 181:
            l.append('<6M')
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
        else:
            l.append('>6M')
            if (((dat.day >= 15) & (dat.month >= 6)) | ((dat.day <= 15) & (dat.month <= 9))):
                summer.append(1)
            else:
                summer.append(0)
    
    # if the next id user is not the same of the previous one put a 0 to mark that he have left
    else:
        l.append(0)
        summer.append(0)
# since i get that 'l' have 29368 values instead of 29369 of 'data_no_work_M' I do a very bad thing but I don't care XD.

l.append(0)
summer.append(0)

# add the new lists created
data_no_work_M['dist_ren'] = l
data_no_work_M['summer'] = summer

data_no_work_M
# substitute string value with integer to apply kmeans
cleanup = {'dist_ren' : {'<15': 15, '<1M': 30, '<2M': 60, '<3M': 90,
                         '<4M': 120,'<5M': 150, '<6M': 180, '>6M': 200}}
data_no_work_M.replace(cleanup, inplace=True)
# now is where things become a little bit confuse since I don't know if they are right
# I want to create a dataset with features like 'dist_ren_0','dist_ren_1', 'dist_ren_2'... for each client

# here I create all the 65 'dist_ren_x', since before we found that we have a client that renewed 66 times
for i in range(66):
    globals()['dist_ren_' + str(i)] = []
    
count = 0  

for i in range(0,len(data_no_work_M)-1):
    if i == 0:
        dist_ren_0.append(data_no_work_M.iloc[0, 10])
        count += 1
    elif data_no_work_M.iloc[i,9] == data_no_work_M.iloc[i-1, 9]:
        globals()['dist_ren_' + str(count)].append(data_no_work_M.iloc[i,10])
        count += 1 
    else:
        for k in range(count, 66):
            globals()['dist_ren_' + str(k)].append(0)
        count = 0
        globals()['dist_ren_' + str(count)].append(data_no_work_M.iloc[i,10])
        count += 1
# Need to make this correction to make everything work(2nd horrible thing to do but I think it will not affect that much)
for i in range(1,66):
    globals()['dist_ren_' + str(i)].append(0)
     
# then we create a dataframe with all the 'dist_ren'
users = data_no_work_M['Iduser'].unique()

distance= []
for i in range(0,66):
    distance.append(globals()['dist_ren_' + str(i)])    

distance = zip(distance)

data_kmeans = pd.DataFrame(index = users)    
for i in range(0,66):
    data_kmeans['dist_ren_' + str(i)] = globals()['dist_ren_'+ str(i)]
data_kmeans
# since the zero is a problem
data_kmeans.replace(0, 0.0001, inplace = True)

from scipy import stats

def analyze_skewness(x):
    print(data_kmeans[x].skew().round(2))
    print(np.log(data_kmeans[x]).skew().round(2))
    print(np.sqrt(data_kmeans[x]).skew().round(2))
    print(pd.Series(stats.boxcox(data_kmeans[x])[0]).skew().round(2))

# everything seems to be better with boxcox transformation
analyze_skewness('dist_ren_51')

# transform them
for i in range(0,65):
    data_kmeans['dist_ren_' + str(i)] = stats.boxcox(data_kmeans['dist_ren_' + str(i)])[0]
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

data_to_scale = data_kmeans

X = ss.fit_transform(data_to_scale)
X
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
data_kmeans['cluster'] = y_kmeans
data_kmeans
from sklearn.metrics import silhouette_score

vals = silhouette_score(X, y_kmeans)
vals
data_kmeans['cluster'].value_counts()

cluster_3 = data_kmeans[data_kmeans['cluster'] == 3]
cluster_3.mean().values
cluster_4 = data_kmeans[data_kmeans['cluster'] == 4]
cluster_4.mean().values
cluster_1 = data_kmeans[data_kmeans['cluster'] == 1]
cluster_1.mean().values
cluster_0 = data_kmeans[data_kmeans['cluster'] == 0]
cluster_0.mean().values