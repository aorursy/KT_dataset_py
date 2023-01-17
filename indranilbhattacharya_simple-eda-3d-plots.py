import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')



train_identity = pd.read_csv('../input/train_identity.csv')

test_identity = pd.read_csv('../input/test_identity.csv')



sample_submission = pd.read_csv('../input/sample_submission.csv')
train_transaction.head()
test_transaction.head()
train_identity.head()
test_identity.head()
train_transaction.shape , test_transaction.shape
train_identity.shape , test_identity.shape
l1 = train_transaction.columns

l2= train_identity.columns

list(set(l1) & set(l2)) 
train = train_transaction.merge(train_identity , how = 'left' , on = 'TransactionID')

test = test_transaction.merge(test_identity , how = 'left' , on = 'TransactionID')

print(train.shape)

print(test.shape)
import gc



del train_transaction, train_identity, test_transaction, test_identity

gc.collect()
train.isnull().sum()
train.dtypes
cat_cols = [c for c in train.columns if train[c].dtype == object ]

cat_cols
for c in cat_cols:

    print('number of unique entries for column' , c , '=' , train[c].nunique())
train.isFraud.value_counts()
train.isFraud.value_counts().plot('bar')

print('target ratio is', round(20663/len(train)*100,2) , 'percent')
pd.options.display.float_format = '{:.2f}'.format

train.TransactionDT.describe()
test.TransactionDT.describe()
train.TransactionDT.max() < test.TransactionDT.min()
import matplotlib.pyplot as plt



plt.hist(train['TransactionAmt'] , bins = 100)

plt.title('transaction amount for train set')

plt.show()



plt.hist(test['TransactionAmt'] , bins = 100)

plt.title('transaction amount for test set')

plt.show()
train.TransactionAmt.describe()
test.TransactionAmt.describe()
plt.hist(np.log(train['TransactionAmt']) , bins = 100)

plt.title('Log scale transaction amount for train set')

plt.show()



plt.hist(np.log(test['TransactionAmt']), bins = 100)

plt.title('Log scale transaction amount for test set')

plt.show()
## Let's subset the numerical columns in train data ##



num_cols = [c for c in train.columns if train[c].dtype != object ]

train_num = train[num_cols]

#print(train_num.shape)

train_num.head()

missing_cols = [c for c in train_num.columns if train_num[c].isnull().sum()/len(train_num) >0.80 ]

len(missing_cols)
for c in train_num.columns:

    print('number of unique entries for column' , c , '=' , train_num[c].nunique())
num_cols = [c for c in train_num.columns if train_num[c].nunique()>5000 ]

len(num_cols) ## 40 columns

#num_cols

train_num = train_num[num_cols]

train_num = train_num.fillna(train_num.mean())

train_num['target'] = train['isFraud']
import random



data1 = train_num.sample(frac= 0.1 , random_state=10)

data1.head()
data1.target.value_counts()
print('target ratio in the sample data is' , round(2060/len(data1)*100,2) , 'which seems okay')
target = data1['target']

del data1['target'], data1['TransactionDT'], data1['TransactionID']
## Let's try PCA on this dataset ##



from sklearn.preprocessing import StandardScaler

data_pca = StandardScaler().fit_transform(data1)



#data_pca = pd.DataFrame(data_pca)

#data_pca.head()

#data_pca.describe()



from sklearn.decomposition import PCA



pca = PCA(n_components=3)

comps = pca.fit_transform(data_pca)



final_pca_data = pd.DataFrame(data = comps , columns=['pc1' , 'pc2' , 'pc3'])



final_pca_data.head()
import plotly

import plotly.graph_objs as go

from plotly.graph_objs import FigureWidget



traces = go.Scatter3d(

    x=final_pca_data['pc1'],

    y=final_pca_data['pc2'],

    z=final_pca_data['pc3'],

    mode='markers',

    marker=dict(

        size=4,

        opacity=0.2,

        color=target,

        colorscale='Viridis',

     )

)



layout = go.Layout(

    autosize=True,

    showlegend=True,

    width=800,

    height=1000,

)



FigureWidget(data=[traces], layout=layout)
%%time



## same with T-SNE ##



from sklearn.manifold import TSNE



tsne = TSNE(n_components=3 , random_state=0)

data_tsne = tsne.fit_transform(data1)



data_tsne



data_tsne = pd.DataFrame(data_tsne , columns=['tsne1' , 'tsne2' , 'tsne3'])

data_tsne.head()



## 3D plot with TSNE components ##



traces = go.Scatter3d(

    x=data_tsne['tsne1'],

    y=data_tsne['tsne2'],

    z=data_tsne['tsne3'],

    mode='markers',

    marker=dict(

        size=4,

        opacity=0.2,

        color=target,

        colorscale='Viridis',

     )

)



layout = go.Layout(

    autosize=True,

    showlegend=True,

    width=800,

    height=1000,

)



FigureWidget(data=[traces], layout=layout)