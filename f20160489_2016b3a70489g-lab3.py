# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

test_df = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

sample_df = pd.read_csv('/kaggle/input/eval-lab-3-f464/sample_submission.csv')
test=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
train_df.columns
train_df.head()
train_df[['Channel1', 'Channel2', 'Channel3', 'Channel4',

       'Channel5', 'Channel6']]=train_df[['Channel1', 'Channel2', 'Channel3', 'Channel4',

       'Channel5', 'Channel6']].replace({'Yes':1,"No":0})

test_df[['Channel1', 'Channel2', 'Channel3', 'Channel4',

       'Channel5', 'Channel6']]=test_df[['Channel1', 'Channel2', 'Channel3', 'Channel4',

       'Channel5', 'Channel6']].replace({'Yes':1,"No":0})
train_df['Channel'] = train_df['Channel1'] + train_df['Channel2']+train_df['Channel3']+train_df['Channel4']+train_df['Channel5']+train_df['Channel6']

test_df['Channel'] = test_df['Channel1'] + test_df['Channel2']+test_df['Channel3']+test_df['Channel4']+test_df['Channel5']+test_df['Channel6']
cols = ['custId', 'gender', 'SeniorCitizen', 'Married', 'Children',

       'TVConnection', 'Channel1', 'Channel3', 'Channel4',

        'Internet', 'HighSpeed', 'AddedServices',

       'Subscription', 'tenure', 'PaymentMethod', 'MonthlyCharges',

       'TotalCharges', 'Satisfied']

cat_cols = [

       'TVConnection','Internet', 'HighSpeed', 'AddedServices'

        ]#'PaymentMethod'

num_cols = ['MonthlyCharges','TotalCharges','Channel']

lenc=['Subscription']

label=['Satisfied']


y_train=train_df[label]
train_df=train_df[cat_cols+num_cols+lenc]

test_df=test_df[cat_cols+num_cols+lenc]
train_df.head()
train_df[lenc[0]].value_counts()
train_df[lenc[0]]=train_df[lenc[0]].replace({'Monthly':0,'Biannually':6,"Annually":12})

test_df[lenc[0]]=test_df[lenc[0]].replace({'Monthly':0,'Biannually':6,"Annually":12})
train_df[lenc[0]].value_counts()
for i in num_cols:

    try:

        train_df[i] = train_df[i].astype('float')

        test_df[i] = test_df[i].astype('float')

    except:

        new_data=[]

        for v in train_df[i].values:

            try :

                new_data.append(float(v))

            except:

                new_data.append(-1)

        train_df[i] = new_data

        new_data=[]

        for v in test_df[i].values:

            try:

                new_data.append(float(v))

            except:

                new_data.append(-1)

        test_df[i] = new_data
from sklearn.preprocessing import MinMaxScaler

for i in num_cols+lenc:

    scaler = MinMaxScaler()

    train_df[i] = scaler.fit_transform(np.asarray(train_df[i]).reshape(-1,1))

    test_df[i] = scaler.transform(np.asarray(test_df[i]).reshape(-1,1))

train_df[lenc[0]].value_counts()
for i in cat_cols:

    print(i)

    one_hot = pd.get_dummies(train_df[i],prefix=i)

    # Drop column B as it is now encoded

    train_df = train_df.drop(i,axis = 1)

    # Join the encoded df

    train_df = train_df.join(one_hot)

    one_hot = pd.get_dummies(test_df[i],prefix=i)

    # Drop column B as it is now encoded

    test_df = test_df.drop(i,axis = 1)

    # Join the encoded df

    test_df = test_df.join(one_hot)
train_df.head()
y_tr = []

for i in y_train.values:

    y_tr.append(i[0])
y_train=y_tr
from sklearn.cluster import KMeans,DBSCAN
train_df.info()
test_df.info()
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt





#print(X)

data = train_df



sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)

    

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
kmeans = KMeans(n_clusters=110,max_iter=2000 ,random_state=100).fit(train_df)

y_pred = kmeans.labels_
'''db = DBSCAN(eps=3, min_samples=2).fit(train_df)

y_pred = db.labels_'''
mapping_df = pd.DataFrame()

mapping_df['y_true'] = y_train

mapping_df['y_pred'] = y_pred
g = mapping_df.groupby(['y_pred'])
map_dict={}

for grp in g.groups.keys():

    print(grp)

    this_grp = g.groups[grp]

    ones=0

    for i in this_grp:

        if y_train[i]==1:

            ones+=1

    #if ones>(len(this_grp)-ones):

    map_dict[grp] = ones/len(this_grp)

    #else:

     #   map_dict[grp]=0

    
map_dict
train_mean = sum(y_train)/len(y_train)
train_mean
def map_pred(map_dict,train_mean,y_pred):

    final_pred =[]

    for i in y_pred:

        if map_dict[i]>train_mean:

            final_pred.append(1)

        else:

            final_pred.append(0)

    return final_pred



    
set(train_df.columns)-set(test_df.columns)
y_test=kmeans.predict(test_df)
final_pred=map_pred(map_dict,train_mean,y_test)
sum(final_pred)/len(final_pred)
sub=pd.DataFrame()

sub['custId']=test['custId']

sub['Satisfied']=final_pred
sub.head()
sub.shape
sub.to_csv("subLab8.csv",index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data3.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe





# create a link to download the dataframe

create_download_link(sub)