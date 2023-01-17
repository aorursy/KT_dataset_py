#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
data = pd.read_csv('../input/mushrooms.csv')
data.head()
#Checking for duplicates
tot = len(set(data.index))
last = data.shape[0]- tot
last
# checking for null values
data.isnull().sum()
# checking the shape of dataset
data.shape
# let see how the target variable is balance
print(data['class'].value_counts())
sns.countplot(x='class', data=data)
plt.show()
#Looking for categorical data
cat = data.select_dtypes(include=['object']).columns
cat
#details view of each columns
for c in cat:
    print(c)
    print('-'*50)
    print(data[c].value_counts())
    sns.countplot(x=c, data=data)
    plt.show()
    print('-'*50)
# we will rmove what all we think not important or less contribution to the target
data['cap-shape']=data[data['cap-shape']!= 'c']
data.dropna(inplace=True)
data.shape
data['cap-surface']=data[data['cap-surface']!= 'g']
data.dropna(inplace=True)
data.shape
data.drop('veil-type', axis=1, inplace = True)
cat = data.select_dtypes(include=['object']).columns
cat
# lets convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat:
    data[i]=le.fit_transform(data[i])
f,ax = plt.subplots(figsize=(20,15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, linecolor='red', fmt='.1f',ax=ax)
plt.show()
X = data.iloc[:,-1:]
X = X.values
y = data['class'].values
y
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.3, random_state=10)
algo = { 'LR' : LogisticRegression(),
         'DT' : DecisionTreeClassifier(),
         'RFC' : RandomForestClassifier(n_estimators=100),
         'SVM' : SVC(gamma=0.01),
         'KNN' : KNeighborsClassifier(n_neighbors=10)
}

for k, v in algo.items():
    model = v
    model.fit(X_train,  y_train)
    print('The accuracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100) + '%' )
