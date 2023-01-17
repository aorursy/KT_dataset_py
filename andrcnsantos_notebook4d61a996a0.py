import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.describe()
ax1 = df_train.plot.scatter(x='Fare',
                      y='Pclass',
                      c='DarkBlue')
ax1.set_xlim(0,100)
sum_df = df_train.groupby(['SibSp']).agg({'Survived': 'sum'})

count_df = df_train.groupby(['SibSp'])[['Survived']].count()

print(str(sum_df/count_df*100))
count_df
sum_df = df_train.groupby(['Parch']).agg({'Survived': 'sum'})

count_df = df_train.groupby(['Parch'])[['Survived']].count()

print(str(sum_df/count_df*100))
count_df
sum_column = (df_train["SibSp"] + df_train["Parch"])
df_train["Family"] = sum_column
df_train["Family"] = df_train["Family"].clip(upper=1)
sum_df = df_train.groupby(['Family']).agg({'Survived': 'sum'})

count_df = df_train.groupby(['Family'])[['Survived']].count()

print(str(sum_df/count_df*100))
count_df
df_train.dropna(axis='rows',inplace=True, subset=['Age'])

y = df_train['Survived']
df_train.drop(['SibSp', 'Fare','Parch','Name','Ticket','Cabin','Embarked','Survived'], axis=1, inplace=True)
df_train.describe()
bins= [0,4,13,20,55,150]
labels = ['Toddler','Kid','Teen','Adult','Elder']
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

X = df_train.iloc[:, :].values 


labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,5] = labelencoder_X.fit_transform(X[:,5])

ct = ColumnTransformer([("Class", OneHotEncoder(),[1]),("Sex", OneHotEncoder(),[2]),("AgeGroup", OneHotEncoder(),[5])], remainder="passthrough") 
X = ct.fit_transform(X) 


print(X)
df_train.drop(['Sex','Pclass','PassengerId','Age','AgeGroup'], axis=1, inplace=True)
df_train['1st']=X[:,0]
df_train['2nd']=X[:,1]
df_train['3rd']=X[:,2]
df_train['Female']=X[:,3]
df_train['Male']=X[:,4]
df_train['Adult']=X[:,5]
df_train['Elder']=X[:,6]
df_train['Child']=X[:,7]
df_train['Teen']=X[:,8]
df_train['Infant']=X[:,9]
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(df_train,y)
from sklearn import tree
import matplotlib.pyplot as plt

fn=['Family','1st','2nd','3rd','Female','Male','Adult','Elder','Child','Teen','Elder']
cn=['Died', 'Survived']

fig, ax = plt.subplots(figsize=(40, 20))
tree.plot_tree(clf,feature_names = fn, 
               class_names=cn,
               filled = True, max_depth=8, fontsize=10)

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test['Age'].fillna(25, inplace = True)

sum_column = (df_test["SibSp"] + df_test["Parch"])
df_test["Family"] = sum_column
df_test["Family"] = df_test["Family"].clip(upper=1)

df_test.drop(['SibSp', 'Fare','Parch','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)

bins= [0,4,13,20,55,150]
labels = ['Toddler','Kid','Teen','Adult','Elder']
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=bins, labels=labels, right=False)

X = df_test.iloc[:, :].values 


labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,5] = labelencoder_X.fit_transform(X[:,5])

ct = ColumnTransformer([("Class", OneHotEncoder(),[1]),("Sex", OneHotEncoder(),[2]),("AgeGroup", OneHotEncoder(),[5])], remainder="passthrough") 
X = ct.fit_transform(X)

Pid = df_test['PassengerId']

df_test.drop(['Sex','Pclass','PassengerId','Age','AgeGroup'], axis=1, inplace=True)
df_test['1st']=X[:,0]
df_test['2nd']=X[:,1]
df_test['3rd']=X[:,2]
df_test['Female']=X[:,3]
df_test['Male']=X[:,4]
df_test['Adult']=X[:,5]
df_test['Elder']=X[:,6]
df_test['Child']=X[:,7]
df_test['Teen']=X[:,8]
df_test['Infant']=X[:,9]
#Predict the response for test dataset
y_pred = clf.predict(df_test)
my_submission = pd.DataFrame({'Id': Pid, 'Survived': y_pred})

my_submission.to_csv('AS_Titanic_Submission.csv', index=False)
