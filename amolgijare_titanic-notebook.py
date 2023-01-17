# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
train_dataset.head()
train_dataset.info()
test_dataset.head()
test_dataset.info()
train_dataset[train_dataset['Cabin'].isnull()]
test_dataset[test_dataset['Cabin'].isnull()]
X_train = train_dataset.drop(columns=['PassengerId', 'Survived', 'Cabin', 'Ticket'])
y_train = train_dataset['Survived']
X_test = test_dataset.drop(columns=['PassengerId', 'Cabin', 'Ticket'])
X_all = pd.concat([X_train,X_test], axis=0)
X_all.info()
X_all[X_all['Embarked'].isnull()]
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=X_all);
X_all['Embarked'] = X_all['Embarked'].fillna('C')
X_all[X_all['Fare'].isnull()]
X_all_3_S_median = X_all[(X_all['Embarked'] == 'S') & (X_all['Pclass'] == 3)]['Fare'].median()
print("Median value of class 3 and embarked s ==> " + str(X_all_3_S_median))
X_all['Fare'] = X_all['Fare'].fillna(X_all_3_S_median)
X_all.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_all.iloc[:,2] = le.fit_transform(X_all.iloc[:,2])
# Female = 0, Male = 1
le = LabelEncoder()
X_all.iloc[:,7] = le.fit_transform(X_all.iloc[:,7])
# C = 0, Q = 1, S = 2
X_all.head()
fig = plt.figure(figsize=(18,6))
train_dataset.Survived.value_counts().plot(kind='bar')
train_dataset.Survived.value_counts(normalize=True).plot(kind='bar')
X_all.hist(bins=10,figsize=(9,7))
import seaborn as sns
g = sns.FacetGrid(train_dataset, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",  color="purple");
g = sns.FacetGrid(train_dataset, hue="Survived", col="Pclass", row="Sex", margin_titles=True,
                  palette={1:"green", 0:"red"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
corr = train_dataset.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');
train_dataset.corr()['Survived']
X_all['Salutation'] = X_all.apply(lambda row: row['Name'].split()[1], axis=1)
X_all.iloc[:,8] = le.fit_transform(X_all.iloc[:,8])
X_all.head()
from sklearn.ensemble import RandomForestRegressor
#predicting missing values in age using Random Forest
def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Salutation']]
    # Split sets into train and test
    train  = age_df.loc[ (age_df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (age_df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (age_df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df

X_all=fill_missing_age(X_all)
X_all.info()
X_train = X_all.iloc[:891, [0,2,3,4,5,6,7,8]]
X_test = X_all.iloc[891:,[0,2,3,4,5,6,7,8]]
X_train.info()
X_test.info()
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0, n_jobs = -1)
lr_classifier.fit(X_train, y_train)
lr_y_pred = lr_classifier.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_jobs = -1)
knn_classifier.fit(X_train, y_train)
knn_y_pred = knn_classifier.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 2000, criterion='entropy', 
                                       n_jobs=-1, random_state = 100)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_test)
from sklearn.svm import SVC
sc_classifier = SVC(cache_size = 3000)
sc_classifier.fit(X_train, y_train)
svc_y_pred = sc_classifier.predict(X_test)
from statistics import mode
final_pred = []
for i in range(418):
    final_pred.append(mode([lr_y_pred[i],
                           knn_y_pred[i],
                           rf_y_pred[i],
                           dt_y_pred[i],
                           svc_y_pred[i]]))
final_pred