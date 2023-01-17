# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/IL_201508.csv')
df['driver_age'] = df['driver_age'].apply(lambda x: int(x) if x == x else 999) #df.fillna()
a = df.groupby(['driver_race','driver_age'])['id'].count().reset_index(name='count')

a.iloc[a.groupby(['driver_race']).apply(lambda x: x['count'].idxmax())]
df['stop_time'] = pd.to_datetime(df['stop_time'])

df['stop_time'] = df['stop_time'].apply(lambda x: x.strftime('%H')) 
df.head()
df.groupby(['stop_time'])['stop_time'].count().reset_index(name = 'stop_count').plot(kind = 'bar')
bins =  np.arange(0,40,10)

ind = np.digitize(df['stop_duration'],bins)

df.groupby([ind,'violation','stop_outcome'])['stop_duration'].count().plot(kind = 'barh', figsize = (100,100),fontsize = 70)
binss =  np.arange(0,61,20)

ind2 = np.digitize(df['driver_age'],binss)

df.groupby([ind,'driver_race','driver_gender'])['driver_race'].count().plot(kind = 'barh', figsize = (100,100),fontsize = 70)
df.groupby(['vehicle_type'])['vehicle_type'].count().sort_values(ascending=False).head(10)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor as RFR

import matplotlib.pyplot as plt
df_ml = df.loc[:,['stop_time','fine_grained_location','driver_gender','driver_age','driver_race','violation','search_conducted','contraband_found','drugs_related_stop','stop_outcome']]
df_ml.head()
#df_ml['stop_time'] = df_ml['stop_time'].astype(int)
x = df_ml.loc[:, df_ml.columns != 'stop_outcome']

y = df_ml.loc[:, 'stop_outcome']

one_hot_encoded_x = pd.get_dummies(x)

one_hot_encoded_x.head()
x_train,x_test,y_train,y_test = train_test_split(one_hot_encoded_x, y, test_size = 0.2, random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

print('With KNN (k = 3) accuracy is:', knn.score(x_test,y_test))
# Model complexity

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('k value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
race = df.groupby(['driver_race'])['id'].count().reset_index(name='race_count')

race['race_rate'] = round(race['race_count'] / sum(race['race_count'])*100, 2)

race
race_search = df[df['search_conducted'] == True].groupby(['driver_race'])['id'].count().reset_index(name='search_count')

race_search 
new = pd.merge(race, race_search, on ='driver_race', how = 'left')

new['search_rate'] = round(new['search_count']/new['race_count']*100,2)

new
contraband_found = df[(df['search_conducted'] == True) & (df['contraband_found'] == True)].groupby(['driver_race'])['id'].count().reset_index(name='contraband_count')

contraband_found
new2 = pd.merge(new,contraband_found, on ='driver_race', how = 'left')

new2['contraband_rate'] = round(new2['contraband_count']/new['search_count']*100,2)
citation = df[(df['search_conducted'] == True) & (df['contraband_found'] == True) & (df['stop_outcome'] == 'Citation')].groupby(['driver_race'])['id'].count().reset_index(name='citation_count')

new3 = pd.merge(new2,citation, on ='driver_race', how = 'left')

new3['citation_rate'] = round(new3['citation_count']/new3['contraband_count']*100,2)

new3 = new3.fillna(0)

new3
fig,ax = plt.subplots(figsize=(10,10))

N = 5

ind = np.arange(N)  # the x locations for the groups

width = 0.2

a = ax.bar(ind, new3['race_rate'],width=0.2,color='b',align='center')

b = ax.bar(ind+width*2, new3['search_rate'],width=0.2,color='g',align='center')

c = ax.bar(ind+width*3, new3['contraband_rate'],width=0.2,color='r',align='center')

d = ax.bar(ind+width*4, new3['citation_rate'],width=0.2,color='y',align='center')

ax.set_ylabel('Rate')

ax.set_xticks(ind+width)

ax.set_xticklabels( ('Asian', 'Black', 'Hispanic', 'Other', 'White') )

ax.legend( (a[0], b[0], c[0],d[0]), ('race_rate', 'search_rate', 'contraband_rate','citation_rate') )

#ax.annotate(new3[name][x], xy = (x,new3[name][x]), textcoords='data')

ax.set_xlabel("race")

ax.legend(loc='best')

def autolabel(rects):

    for rect in rects:

        h = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2., 1.0*h, '%d'%int(h),

                ha='center', va='bottom')



autolabel(a)

autolabel(b)

autolabel(c)

autolabel(d)

plt.show()