import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.isnull().sum()
df['class'].unique()
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in df.columns:
    if len(df[col].value_counts())==2:
        
        df[col]=labelencoder.fit_transform(df[col])
    
df.head()

df=pd.get_dummies(df)
df.head()
X=df.drop(['class'],axis=1)
X.head()
Y=df['class']
Y.head()
Y=Y.to_frame()
Y.head()
X.describe()
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
clusters=kmeans.predict(X)
cluster_df = pd.DataFrame()

cluster_df['cluster'] = clusters
cluster_df['class'] = Y
cluster_df.head()
cluster0_df=cluster_df[cluster_df['cluster']==0]
cluster0_df.head()
cluster1_df=cluster_df[cluster_df['cluster']==1]
cluster1_df.head()
sns.countplot(x="class", data=cluster0_df)

sns.countplot(x='class',data=cluster1_df)
Y.describe()
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.20)
#K-Means Clustering with two clusters
kmeans = KMeans(n_clusters=2)

#Logistic Regression with no special parameters
logreg = LogisticRegression()
kmeans.fit(train_X)

logreg.fit(train_X, train_y)
kmeans_pred = kmeans.predict(test_X)

logreg_pred = logreg.predict(test_X)
test_y
#This DataFrame will allow us to visualize our results.
result_df = pd.DataFrame()

#The column containing the correct class for each mushroom in the test set, 'test_y'.
result_df['test_y'] = test_y['class'] 

#The predictions made by K-Means on the test set, 'test_X'.
result_df['kmeans_pred'] = kmeans_pred
#The column below will tell us whether each prediction made by our K-Means model was correct.
result_df['kmeans_correct'] = result_df['kmeans_pred'] == result_df['test_y']

#The predictions made by Logistic Regression on the test set, 'test_X'.
result_df['logreg_pred'] = logreg_pred
#The column below will tell us whether each prediction made by our Logistic Regression model was correct.
result_df['logreg_correct'] = result_df['logreg_pred'] == result_df['test_y']
result_df

sns.countplot(x=result_df['kmeans_correct'], order=[True,False]).set_title('K-Means Clustering')

sns.countplot(x=result_df['logreg_correct'], order=[True,False]).set_title('Logistic Regression')
