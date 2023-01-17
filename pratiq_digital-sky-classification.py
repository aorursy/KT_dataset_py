# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns    # plots



from sklearn.model_selection import train_test_split  # for train-test split



from sklearn.neighbors import KNeighborsClassifier  # KNN

from sklearn.ensemble import RandomForestClassifier  # Random Forest



from sklearn.metrics import classification_report,confusion_matrix  # model performance report



import warnings

warnings.filterwarnings('ignore')
# load data

df = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")

df.head()
# a look into the data

df.info()
df.describe()
# plot target class distribution

sns.countplot(x='class',data=df)
plt.figure(figsize=(10,6))

sns.boxplot(x='class',y='dec',data=df,palette='coolwarm')
plt.figure(figsize=(10,6))

sns.violinplot(x='class',y='dec',data=df,palette='plasma')
# chnange target class to numeric

num = {'STAR':1,'GALAXY':2,'QSO':3}

df.replace({'class':num},inplace=True)
df.head()
df.corr()['class'].drop('class')
plt.figure(figsize=(10,6))

sns.boxenplot(x='class',y='redshift',data=df,palette='winter')
plt.figure(figsize=(10,6))

sns.violinplot(x='class',y='mjd',data=df)
# an algorithm to find pairs with high correlation

# later we will drop one feature from each pair

features = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 

            'run', 'camcol', 'field', 'specobjid', 

            'redshift', 'plate', 'mjd', 'fiberid']

for i in features:

    for j in features:

        if (df.corr()[i][j] >= 0.9) & (i != j):

            print(i,'\t',j)

        else:

            pass
# we also found some triplets, good that we got more features to drop

df.drop(['i','r','specobjid','mjd','objid','rerun'],axis=1,inplace=True)
# a function to select features wrt threshold

correlation = df.corr()['class'].drop('class')

def feat_select(threshold):

    abs_corr = correlation.abs()

    feat = abs_corr[abs_corr>threshold].index.tolist()

    X = df[feat]

    return X
# using Random Forest

threshold = 0

rfc = RandomForestClassifier(n_estimators=100)

X = feat_select(threshold)

y = df['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=40)

rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)

print('Model Score = ',rfc.score(X_test,y_test)*100,' %')
# test for KNN with optimal K value

err = []

for i in range(1,20):

    knn=KNeighborsClassifier(n_neighbors=i,p=2)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    err.append(np.mean(y_test != pred_i))

    
plt.plot(range(1,20),err)