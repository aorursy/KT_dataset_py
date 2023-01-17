import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

import os
os.chdir('../input/dmassign1/')
original_data = pd.read_csv('data.csv',sep=',')

data = original_data
data.head()
data.info()
data = original_data.drop('Class',1)

data
def Par_status(val):

    if val=='?':

        return np.nan

    else:

        return val



data_replaced = data.copy()

for col in data.columns:

    data_replaced[col] = data[col].apply(Par_status) # to apply changes

data_replaced
data_replaced['Col189'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='yes':

        return 1

    if val=='no':

        return 0

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col189'] = data['Col189'].apply(Par_status) # to apply changes

data_replaced
data_replaced['Col190'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='sacc1':

        return 1

    if val=='sacc2':

        return 2

    if val=='sacc4':

        return 4

    if val=='sacc5':

        return 5

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col190'] = data_replaced['Col190'].apply(Par_status) # to apply changes

data_replaced
data_replaced['Col191'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='time1':

        return 1

    if val=='time2':

        return 2

    if val=='time3':

        return 3

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col191'] = data_replaced['Col191'].apply(Par_status) # to apply changes

data_replaced
data_replaced['Col192'].unique() # returns unique values that occur in column
def Par_status(val):

    if str(val) == 'nan':

        return np.nan

    else:

        return val[1]



data_replaced['Col192'] = data_replaced['Col192'].apply(Par_status) # to apply changes

data_replaced
data_replaced['Col192'].unique()
data['Col193'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='F1':

        return 4

    if val=='M1':

        return 2

    if val=='F0':

        return 3

    if val=='M0':

        return 1

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col193'] = data['Col193'].apply(Par_status) # to apply changes

data_replaced['Col193'] = data_replaced['Col193'].fillna(data_replaced['Col193'].median())

data_replaced
data['Col194'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='ab':

        return 1

    if val=='ad':

        return 3

    if val=='ac':

        return 2

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col194'] = data['Col194'].apply(Par_status) # to apply changes

data_replaced['Col194'] = data_replaced['Col194'].fillna(data_replaced['Col194'].median())

data_replaced
data['Col195'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='Jb1':

        return 1

    if val=='Jb2':

        return 2

    if val=='Jb3':

        return 3

    if val=='Jb4':

        return 4

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col195'] = data['Col195'].apply(Par_status) # to apply changes

data_replaced['Col195'] = data_replaced['Col195'].fillna(data_replaced['Col195'].median())

data_replaced
data['Col196'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='H1':

        return 1

    if val=='H2':

        return 2

    if val=='H3':

        return 3

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col196'] = data['Col196'].apply(Par_status) # to apply changes

data_replaced['Col196'] = data_replaced['Col196'].fillna(data_replaced['Col196'].median())

data_replaced
data['Col197'].unique() # returns unique values that occur in column
def Par_status(val):

    if val=='XL':

        return 1

    if val=='SM' or val=='sm':

        return 2

    if val=='me' or val=='ME' or val=='M.E.':

        return 3

    if val=='LA' or val=='la':

        return 4

    else:

        return np.nan



# df1['Pstatus'].apply(Par_status) to show changes

data_replaced['Col197'] = data['Col197'].apply(Par_status) # to apply changes

data_replaced['Col197'] = data_replaced['Col197'].fillna(data_replaced['Col197'].median())

data_replaced
data_replaced = data_replaced.drop('ID',1)
for col in data_replaced.columns:

    if data_replaced[col].dtype==object:

        data_replaced[col] = data_replaced[col].astype(float)
for col in data_replaced.columns:

    data_replaced[col] = data_replaced[col].fillna(data_replaced[col].median())
for col in data_replaced.columns:

    if((data_replaced[col]=='?').sum()>0):

        print(col)
saved_data_replaced = data_replaced.copy()
data_replaced = saved_data_replaced.copy()

data_replaced = data_replaced
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
# StandardScaler



scaler=StandardScaler()

data_replaced=scaler.fit(data_replaced).transform(data_replaced)

data_replaced=pd.DataFrame(data_replaced,columns=saved_data_replaced.columns)

data_replaced.tail()
pcamodel = PCA(n_components = 197)

data_replaced = pcamodel.fit(data_replaced).transform(data_replaced)

data_replaced = pd.DataFrame(data_replaced,columns=saved_data_replaced.columns)
data_replaced
data_train = data_replaced[:1300]

data_test = data_replaced[1300:]

X = data_train

y = original_data[:1300]['Class']
X
y
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier(max_depth=20)

dtree.fit(X, y)

dtree.feature_importances_
selected = (dtree.feature_importances_ > 0)

selected
X_new2 = X[X.columns[selected]]

X_new2.head()
from sklearn.feature_selection import RFECV

from sklearn.model_selection import KFold



rfecv = RFECV(estimator=dtree, step=2, cv=KFold(2), scoring='accuracy')

rfecv.fit(X, y)



selected = rfecv.support_
X_new3 = X[X.columns[selected]]

X_new3.head()
X_new3
data_replaced = data_replaced[data_replaced.columns[selected]]
data_replaced._get_numeric_data()
data_replaced._get_numeric_data().info()
data_train = data_replaced[:1300]

data_test = data_replaced[1300:]
pd.options.mode.use_inf_as_na = True

num_clusters = 100

km = KMeans(n_clusters=num_clusters, random_state=10)

new = data_replaced._get_numeric_data()

newtest = data_replaced._get_numeric_data()

km.fit(new)

predict=km.predict(newtest)

df_kmeans = data.copy(deep=True)

df_kmeans['Cluster KMeans'] = pd.Series(predict, index=df_kmeans.index)

df_kmeans
df_kmeans = df_kmeans[['ID','Cluster KMeans']]

df_kmeans
df_kmeans['Cluster KMeans'].unique()
mapped = [[0 for j in range(num_clusters)] for i in range(5)]



for i in range(1300):

    mapped[int(original_data.iloc[i]['Class'])-1][df_kmeans.iloc[i]['Cluster KMeans']] += 1



mapped
mymap = {}

for i in range(num_clusters):

    mymap[i] = 1

    curr = 0

    for j in range(5):

        if mapped[j][i]>curr:

            curr = mapped[j][i]

            mymap[i] = j+1



mymap
final_df = df_kmeans.copy()
for i in range(len(df_kmeans)):

    final_df['Cluster KMeans'][i] = mymap[final_df['Cluster KMeans'][i]]
correct = 0

for i in range(1300):

    if int(original_data.iloc[i]['Class']) == int(final_df.iloc[i]['Cluster KMeans']):

        correct += 1

print("Accuracy: ", (correct/1300))
final_df[1300:].to_csv('Final Submission.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final_df[1300:])