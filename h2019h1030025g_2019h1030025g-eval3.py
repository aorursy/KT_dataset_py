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
import numpy as np

import pandas as pd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

df=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

dd=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df.info()
df.describe()
df.isnull().values.sum()
df['gender']=df['gender'].astype('category').cat.codes

df['Married']=df['Married'].astype('category').cat.codes

df['Children']=df['Children'].astype('category').cat.codes

df['TVConnection']=df['TVConnection'].astype('category').cat.codes

df['Channel1']=df['Channel1'].astype('category').cat.codes

df['Channel2']=df['Channel2'].astype('category').cat.codes

df['Channel3']=df['Channel3'].astype('category').cat.codes

df['Channel4']=df['Channel4'].astype('category').cat.codes

df['Channel5']=df['Channel5'].astype('category').cat.codes

df['Channel6']=df['Channel6'].astype('category').cat.codes

df['Internet']=df['Internet'].astype('category').cat.codes

df['HighSpeed']=df['HighSpeed'].astype('category').cat.codes

df['AddedServices']=df['AddedServices'].astype('category').cat.codes

df['Subscription']=df['Subscription'].astype('category').cat.codes

df['PaymentMethod']=df['PaymentMethod'].astype('category').cat.codes

# df.info()

df['TotalCharges']=df['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)
dd['gender']=dd['gender'].astype('category').cat.codes

dd['Married']=dd['Married'].astype('category').cat.codes

dd['Children']=dd['Children'].astype('category').cat.codes

dd['TVConnection']=dd['TVConnection'].astype('category').cat.codes

dd['Channel1']=dd['Channel1'].astype('category').cat.codes

dd['Channel2']=dd['Channel2'].astype('category').cat.codes

dd['Channel3']=dd['Channel3'].astype('category').cat.codes

dd['Channel4']=dd['Channel4'].astype('category').cat.codes

dd['Channel5']=dd['Channel5'].astype('category').cat.codes

dd['Channel6']=dd['Channel6'].astype('category').cat.codes

dd['Internet']=dd['Internet'].astype('category').cat.codes

dd['HighSpeed']=dd['HighSpeed'].astype('category').cat.codes

dd['AddedServices']=dd['AddedServices'].astype('category').cat.codes

dd['Subscription']=dd['Subscription'].astype('category').cat.codes

dd['PaymentMethod']=dd['PaymentMethod'].astype('category').cat.codes

# dd.info()

# dd['TotalCharges']=dd['TotalCharges'].astype(np.float64)

dd['TotalCharges']=dd['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)
df.head()
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
# df['TotalCharges']

# features=['SeniorCitizen','Children','TVConnection','Channel2','Channel4','Channel5','Subscription','tenure','AddedServices','PaymentMethod']

features=['TotalCharges','Subscription','Channel2','Channel4','Channel5','Children','Channel3','TVConnection','Internet']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# features=['gender','SeniorCitizen','Children','TVConnection','Internet','AddedServices','Subscription','PaymentMethod','tenure']

x=scaler.fit_transform(df[features])

y=scaler.fit_transform(df[['Satisfied']])





xx=scaler.fit_transform(dd[features])

xxx=dd['custId']









# x = scaler.fit_transform(xt)

# xx=scaler.fit_transform(xxt)

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.20,random_state=42) 
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(X_train,y_train)

y_pred=kmeans.predict(X_val)







# y_pred.tocsv('temp_output')

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred,y_val)

accuracy

# y_pred.to_csv('temp.csv')

from sklearn.metrics import mean_absolute_error

mae_lr = mean_absolute_error(y_pred,y_val)

print("Mean Absolute Error of K-means: {}".format(mae_lr))
yy=kmeans.predict(xx)
oo=dd

oo['Satisfied'] =yy

# oo['custId']=dd[['custId']].copy()



ans=oo[['custId','Satisfied']].copy()

ans.to_csv('tempo.csv')

ans.head()