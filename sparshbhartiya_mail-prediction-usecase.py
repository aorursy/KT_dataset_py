import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score)

train = pd.read_csv('../input/task_InputData.csv')

train.head()
train.info()


train.shape
target = 'label'

train[target].describe()
train[target].value_counts().plot.bar()
train.sports.isna().sum() 
train.describe()
sns.distplot(train['age'])

        
sns.distplot(train['earnings'])
train['lifestyle'].value_counts()
train['car'].value_counts()
train['sports'].value_counts()
train['family status'].value_counts()
train['Living area'].value_counts()
sns.countplot(x='lifestyle',data=train)
sns.countplot(x=train['car'],data=train)
sns.countplot(x=train['family status'],data=train)
sns.countplot(x=train['Living area'],data=train)
sns.countplot(x=train['sports'],data=train)
sns.boxplot(train['age'],train['label'])
sns.boxplot(train['earnings'],train['label'])
sns.countplot(x='lifestyle',hue='label',data=train)
sns.countplot(x=train['family status'],hue='label',data=train)
sns.countplot(x=train['car'],hue='label',data=train)
sns.countplot(x=train['sports'],hue='label',data=train)
sns.countplot(x=train['Living area'],hue='label',data=train)
num_cols=['age','earnings']

cat_cols = ['lifestyle','family status','car','sports','Living area']

for num_col in num_cols:

    fig = plt.figure(figsize = (35,8))

    k = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),k)

        sns.boxplot(y = train[num_col],x = train[cat_col], data = train)

        ax.set_xlabel(cat_col,fontsize=15)

        ax.set_ylabel(num_col,fontsize=15)

        plt.xticks(fontsize=15)

        plt.xticks(fontsize=15)

        k = k + 1
for num_col in num_cols:

    fig = plt.figure(figsize = (35,8))

    k = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),k)

        sns.boxplot(y = train[num_col],x = train[cat_col],data = train,hue= train['label'])

        ax.set_xlabel(cat_col,fontsize=15)

        ax.set_ylabel(num_col,fontsize=15)

        plt.xticks(fontsize=15)

        plt.xticks(fontsize=15)

        ax.legend('best')

        k += 1 
for i in range(len(cat_cols)):

    fig = plt.figure(figsize=(20,5))

    for j in range(0,len(cat_cols)):

        ax = fig.add_subplot(1,5,j+1)

        sns.countplot(train[cat_cols[i]],hue=train[cat_cols[j]])

for num in num_cols:

    j=1

    fig = plt.figure(figsize=(20,10))

    for col in cat_cols:

        ax = fig.add_subplot(1,5,j)

        sns.boxplot(train[col],train[num],hue=train['label'])

        j = j+1
sns.boxplot(train['earnings'],train['label'],hue=train['car'])
sns.boxplot(train['earnings'],train['car'],hue=train['Living area'])
sns.boxplot(train['age'],train['car'],hue=train['label'])
train['sports'].replace(np.nan,'unknown',inplace=True)
explore_data, validation_data = train_test_split(train, test_size = 0.2, random_state=20,stratify=train['label'])

explore_data.shape
validation_data.shape
explore_data.label.value_counts()
train_data, test_data = train_test_split(explore_data, test_size = 0.2, random_state=20)
train_data.shape
test_data.shape
X_train = train_data.drop(['name','zip code','label'],axis=1)

y_train = train_data['label']
X_train.head()
y_train.head()
y_train.shape
X_test = test_data.drop(['name','zip code','label'],axis=1)

y_test = test_data['label']

X_test.shape
X_val = validation_data.drop(['name','zip code','label'],axis=1)

y_val = validation_data['label']
y_test.head()
X_train = pd.get_dummies(X_train, prefix_sep='_', drop_first=True)

# X head

X_train.head()
X_train.shape
X_test = pd.get_dummies(X_test, prefix_sep='_', drop_first=True)

# X head

X_test.head()
X_val = pd.get_dummies(X_val, prefix_sep='_', drop_first=True)

# X head

X_val.head()
y_enc = LabelEncoder()

y_train = y_enc.fit_transform(y_train)

y_test = y_enc.transform(y_test)

y_val = y_enc.transform(y_val)
sc = StandardScaler()

X_train[num_cols] = sc.fit_transform(X_train[num_cols])

X_test[num_cols] = sc.transform(X_test[num_cols])

X_val[num_cols] = sc.transform(X_val[num_cols])

X_train.head()
sc.mean_
clf = LogisticRegression() #Creating logistic Regression object.

clf.fit(X_train,y_train) # Fitting training data to create a model
y_pred = clf.predict(X_test) # Testing the model using testing data.

y_pred
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred) #Checking accuracy of our model.
precision_score(y_test,y_pred) #Calulating precision score of model

 



recall_score(y_test,y_pred)  #Calculating recall score



f1_score(y_test,y_pred) # Calculating f1 score.
dt = DecisionTreeClassifier() #Creating logistic Regression object.





dt.fit(X_train,y_train) # Fitting training data to create a model



y_pred_dt = dt.predict(X_test) # Testing the model using testing data.

y_pred_dt



confusion_matrix(y_test,y_pred_dt)



accuracy_score(y_test,y_pred_dt) #Checking accuracy of our model.
precision_score(y_test,y_pred_dt)
recall_score(y_test,y_pred_dt)

f1_score(y_test,y_pred_dt)
rf = RandomForestClassifier() #Creating logistic Regression object.





rf.fit(X_train,y_train) # Fitting training data to create a model



y_pred_rf = dt.predict(X_test) # Testing the model using testing data.

y_pred_rf



confusion_matrix(y_test,y_pred_rf)

accuracy_score(y_test,y_pred_rf) #Checking accuracy of our model.
precision_score(y_test,y_pred_rf)
recall_score(y_test,y_pred_rf)

f1_score(y_test,y_pred_rf)