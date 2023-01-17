import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

import warnings

warnings.filterwarnings("ignore")

df=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head(5)
len(df)
df.isnull().sum()
import missingno as msno

msno.matrix(df)
df['Gender'].unique()
print(sum(df.duplicated()))

df = df.drop_duplicates()

ig, axes = plt.subplots(1,2, figsize=(21,6))

sns.distplot(df['Age'], ax=axes[0])

sns.distplot(df['Annual Income (k$)'], ax=axes[1])



sns.countplot(x='Gender', data=df, palette='viridis')
sns.stripplot(x='Gender', y = 'Spending Score (1-100)', data = df)
sns.boxplot( x= 'Gender', y = 'Annual Income (k$)', data = df )
x = df['Annual Income (k$)']

y = df['Age']

z = df['Spending Score (1-100)']



sns.lineplot(x, y, color = 'blue')

sns.lineplot(x, z, color = 'pink')

plt.title('Annual Income vs Age and Spending Score', fontsize = 20)

plt.show()

df['Gender'].replace({'Male': 0, 'Female': 1},inplace = True) 
df.head(5)
df.drop('CustomerID', axis=1, inplace = True)
df.head(5)
sns.heatmap(df.corr(), annot=True)
def impute_age(cols):

    spend=cols

    if spend > 55:

         return 1

    else:

         return 0

df['Spending Score (1-100)'] = df['Spending Score (1-100)'].apply(impute_age)

df.head(5)

    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(df.drop('Spending Score (1-100)',axis=1), df['Spending Score (1-100)'], test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(X_train,y_train)

pred=log.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, pred))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('Spending Score (1-100)',axis=1))

scaled_features = scaler.transform(df.drop('Spending Score (1-100)',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Spending Score (1-100)'],

                                                    test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors=3)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=3')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))