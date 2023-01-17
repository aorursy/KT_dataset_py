#importing libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import k_means
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#reading the data.
df=pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()
df.shape
df.describe()
df.dtypes
df.isna().sum()
df.rename(columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)
df.head()
#Gender
sns.countplot(x=df['Gender'])
#Age
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.distplot(df.Age)
plt.title('Distribution of Age')

plt.subplot(1,2,2)
sns.boxplot(df.Age)
plt.title('spread of Age')
#Income 
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.distplot(df.Income)
plt.title('Distribution of Income')

plt.subplot(1,2,2)
sns.boxplot(df.Income)
plt.title('spread of Income')
#Score
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.distplot(df.Score)
plt.title('Distribution of Score')

plt.subplot(1,2,2)
sns.boxplot(df.Score)
plt.title('spread of Score')
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
sns.pairplot(data=df)
plt.figure(figsize=(26, 8))
plt.subplot(1,3,1)
plt1 = sns.scatterplot(x=df.Age,y=df.Income)
plt.title('spread of data b/w Age and Income')
plt1.set(xlabel = 'Age', ylabel='Income')

plt.subplot(1,3,2)
plt1 = sns.regplot(x=df.Age,y=df.Income)
plt.title('Regression plot b/w Age and income')
plt1.set(xlabel = 'Age', ylabel='Income')

plt.subplot(1,3,3)
plt1 = sns.scatterplot(x=df.Age,y=df.Income,hue=df.Gender)
plt.title('spread of data b/w age and income based on gender')
plt1.set(xlabel = 'age', ylabel='Income')

plt.show()
plt.figure(figsize=(26, 8))
plt.subplot(1,3,1)
plt1 = sns.scatterplot(x=df.Age,y=df.Score)
plt.title('spread of data b/w Age and score')
plt1.set(xlabel = 'Age', ylabel='Score')

plt.subplot(1,3,2)
plt1 = sns.regplot(x=df.Age,y=df.Score)
plt.title('Regression plot b/w Age and score')
plt1.set(xlabel = 'Age', ylabel='score')

plt.subplot(1,3,3)
plt1 = sns.scatterplot(x=df.Age,y=df.Score,hue=df.Gender)
plt.title('spread of data b/w age and score based on gender')
plt1.set(xlabel = 'age', ylabel='score')

plt.show()
plt.figure(figsize=(26, 8))
plt.subplot(1,3,1)
plt1 = sns.scatterplot(x=df.Income,y=df.Score)
plt.title('spread of data b/w income and score')
plt1.set(xlabel = 'income', ylabel='Score')

plt.subplot(1,3,2)
plt1 = sns.regplot(x=df.Income,y=df.Score)
plt.title('Regression plot b/w income and score')
plt1.set(xlabel = 'income', ylabel='score')

plt.subplot(1,3,3)
plt1 = sns.scatterplot(x=df.Income,y=df.Score,hue=df.Gender)
plt.title('spread of data b/w income and score based on gender')
plt1.set(xlabel = 'income', ylabel='score')

plt.show()
cls=['CustomerID','Gender']
df.drop(cls,axis=1,inplace=True)
df.head()
from sklearn.preprocessing import StandardScaler
num_cols=['Age','Income','Score']

scaler=StandardScaler()
scaler.fit(df[num_cols])
df[num_cols]=scaler.transform(df[num_cols])

df.head()
from sklearn.cluster import KMeans

sse=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=40)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    
plt.plot(range(1,11),sse)
plt.title('Elbow Method',fontsize=20)
plt.xlabel('No of Clusters')
plt.ylabel('SSE')
plt.show()
#Model Build
kmeansmodel = KMeans(n_clusters= 6, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(df)
y_preds=pd.DataFrame(y_kmeans)
type(y_preds)
y_preds.columns=['clusters']
y_preds

