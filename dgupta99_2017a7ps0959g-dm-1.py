import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import matplotlib.pyplot as plt
df = pd.read_csv('../input/dmassign1/data.csv')

df.head()
df_cols = df.columns[1:198]

df_cols   #doesnt include id and class
df_label = df['Class']

df_x = df.drop(columns = [ 'Class'] )
df_x = df_x.replace('?',np.nan)
def check_float(s):

    try:

        float(s)

        return True

    except ValueError:

        return False
#function to change numeric strings to float

for i in df_x.columns:

    #print(i)

    #print(df[i].dtypes)

    if(check_float(df_x[i][0])):

        #print(type(df_x[i][0]))

        df_x[i] = df_x[i].astype(float)

        #print(type(df_x[i][0]))
df_x.fillna(value = df_x.mean(), inplace = True)

df_x.fillna(value = df_x.mode().iloc[0], inplace = True)

df_x['Col197'] = df_x['Col197'].apply(lambda x: x.upper())         #capitalizing values in Col197

df_x['Col197'] = df_x['Col197'].replace('M.E.', 'ME')

df_x.head()
df_enc = df_x.copy()

df_enc['Col197'].unique()

df_enc['Col197'] = df_enc['Col197'].replace(['SM','ME','LA','XL'], [0,1,2,3])

df_enc['Col197'] = df_enc['Col197'].astype(int)

df_id = df['ID']

df_enc = df_enc.drop(columns = ['ID'])

#df_enc= pd.get_dummies(df_enc, prefix_sep='_')

print(

df_enc['Col189'].unique(),

df_enc['Col190'].unique(),

df_enc['Col191'].unique(),

df_enc['Col192'].unique(),

df_enc['Col193'].unique(),

df_enc['Col194'].unique(),

df_enc['Col195'].unique(),

df_enc['Col196'].unique())

# df_enc['Col189'] = df_enc['Col189'].replace(['yes','no'], [0,1,2,3])

df_enc['Col190'] = df_enc['Col190'].replace(['sacc1','sacc2','sacc4','sacc5'], [0,1,2,3])

df_enc['Col191'] = df_enc['Col191'].replace(['time1','time2','time3'], [0,1,2])

df_enc['Col192'] = df_enc['Col192'].replace(['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'], [0,1,2,3,4,5,6,7,8,9])

# df_enc['Col193'] = df_enc['Col193'].replace(['SM','ME','LA','XL'], [0,1,2,3])

# df_enc['Col194'] = df_enc['Col194'].replace(['SM','ME','LA','XL'], [0,1,2,3])

df_enc['Col195'] = df_enc['Col195'].replace(['Jb1','Jb2','Jb3','Jb4'], [0,1,2,3])

df_enc['Col196'] = df_enc['Col196'].replace(['H1','H2','H3'], [0,1,2])



#onehot the remaining

#189, 193, 194

df_enc= pd.get_dummies(df_enc, prefix_sep='_')
df_enc.head()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize



df_enc_sca = df_enc.copy()

stdscalar = StandardScaler()

df_enc_sca = stdscalar.fit_transform(df_enc_sca)

df_enc_sca =  normalize(df_enc_sca, norm='l2')

df_enc_sca

from sklearn.decomposition import PCA

pca = PCA().fit(df_enc_sca)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('No. of Components')

plt.ylabel('Variance (%)') #for each component

plt.show()
#70 when std+normalize

pca = PCA(n_components = 70, random_state = 100)

df_enc_sca_pca = df_enc_sca.copy()

df_enc_sca_pca = pca.fit_transform(df_enc_sca_pca)
pd.DataFrame(df_enc_sca_pca).head()
from sklearn.cluster import KMeans

result = []

def kmeans_func(n):

    kmeans = KMeans(n_clusters = n, random_state = 42 )

    clf = kmeans.fit(df_enc_sca_pca)          # pca after scaling

    pred = clf.labels_

    result.append(pred)
for j in range(5,16):

    kmeans_func(j)
result
for i in range(11):

    unique_elements, counts_elements = np.unique(result[i][:1300], return_counts = True)

    print(np.asarray((unique_elements,counts_elements)))
def labelling(n, sol):

    one , two, three, four , five = 0 , 0, 0 , 0 , 0

    for i in range(1300):

        if(sol.iloc[i]['Class'] == n):

            if(df.iloc[i]['Class'] == 1):

                one = one +1

            elif(df.iloc[i]['Class'] == 2):

                two = two+1

            elif(df.iloc[i]['Class'] == 3):

                three = three+1

            elif(df.iloc[i]['Class'] == 4):

                four = four + 1

            else:

                five = five+1

    print(one, two, three, four, five)

    return np.argmax([one, two, three, four, five]) + 1
sol = pd.DataFrame()

sol['ID'] = df_id

for i in range(5,16):

    print("no. of clusters: "+str(i))

    sol['Class'] = result[i-5]

    label = []

    for j in range(i):

        print("for cluster "+str(j))

        label= labelling(j, sol)

        print(label)

        result[i-5] = np.array(pd.DataFrame(result[i-5]).replace(j, label))

    print(" ")
for i in range(11):

    unique_elements, counts_elements = np.unique(result[i], return_counts = True)

    print(np.asarray((unique_elements,counts_elements)))
from sklearn.metrics import accuracy_score

for i in range(11):

    print(accuracy_score(df_label[0:1300], result[i][0:1300]))
sub = pd.DataFrame()

sub['ID'] = df_id

sub['Class'] = result[9]

sub[1300:].to_csv('sub1.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub[1300:])