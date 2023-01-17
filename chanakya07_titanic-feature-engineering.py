# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt; # data visualization

import seaborn as sns # data visualtization 



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.preprocessing import StandardScaler

import re





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv');

test = pd.read_csv('../input/test.csv');

print(train.shape)

#train=train.dropna(axis=0)

print(test.shape)

train.dtypes
train1=train.drop(['Name', 'Ticket','Cabin','PassengerId','Embarked'], axis=1);

train1.head()

test1=test.drop(['Name', 'Ticket','Cabin','PassengerId','Embarked'], axis=1);

test1.head()



train1.Sex[train.Sex == 'male'] = 1

train1.Sex[train.Sex == 'female'] = 0

test1.Sex[train.Sex == 'male'] = 1

test1.Sex[train.Sex == 'female'] = 0



train1.head()
train_missing=train1.isna().sum();

print(train_missing)

test_missing=test1.isna().sum();

print(test_missing)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=test1,palette='winter')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Sex',y='Age',data=test1,palette='winter')
mean = test1.groupby(['Sex','Pclass'])['Age'].mean().values

print(mean)

median = test1.groupby(['Sex','Pclass'])['Age'].median().values

print(median)
def impute_age(cols):

    age = cols[0]

    sex = cols[1]

    pclass = cols[2]

    if pd.isnull(age):

        if sex == 0:

            if pclass == 1:

                return median[0]

            elif pclass == 2:

                return median[1]

            elif pclass == 3:

                return median[2]

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        elif sex == 1:

            if pclass == 1:

                return median[3]

            elif pclass == 2:

                return median[4]

            elif pclass == 3:

                return median[5]

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        else: print('error! sex should be female or male but it is '+sex+'!')

    else:

        return age

test1['Age']=test1[['Age','Sex','Pclass']].apply(impute_age,axis=1)

#train1['Age'] = train1[['Age','Pclass']].apply(impute_age,axis=1)

#test1['Age'] = test1[['Age','Sex','Pclass']].apply(impute_age,axis=1)

#train1.dropna(inplace=True);

#train1.head()

train_missing=test1.isna().sum();

train_missing
plt.figure(figsize=(12, 7))

sns.boxplot(x='Sex',y='Fare',data=test1,palette='winter')



plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Fare',data=test1,palette='winter')
mean = test1.groupby(['Sex','Pclass'])['Fare'].mean().values

print(mean)

median = test1.groupby(['Sex','Pclass'])['Fare'].median().values

print(median)
def impute_fare(cols):

    fare = cols[0]

    sex = cols[1]

    pclass = cols[2]

    if pd.isnull(fare):

        if sex == 0:

            if pclass == 1:

                return mean[0]

            elif pclass == 2:

                return mean[1]

            elif pclass == 3:

                return mean[2]

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        elif sex == 1:

            if pclass == 1:

                return mean[3]

            elif pclass == 2:

                return mean[4]

            elif pclass == 3:

                return mean[5]

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        else: print('error! sex should be female or male but it is '+sex+'!')

    else:

        return fare

test1['Fare']=test1[['Fare','Sex','Pclass']].apply(impute_fare,axis=1)

#train1['Age'] = train1[['Age','Pclass']].apply(impute_age,axis=1)

#test1['Age'] = test1[['Age','Sex','Pclass']].apply(impute_age,axis=1)

#train1.dropna(inplace=True);

#train1.head()

train_missing=test1.isna().sum();

train_missing


train1=train1.dropna(axis=0)

"""

mode_value=test1.mode()

print(mode_value)

"""
"""

for i in np.arange(0,6):

    mode_val = mode_value.iloc[0,i]

    test1.iloc[:,i].fillna(mode_val,inplace=True)

test_missing=test1.isna().sum();

print(test_missing)

"""
train1['family_size'] = train1.SibSp + train1.Parch+1

print(train1.head())

test1['family_size'] = test1.SibSp + test1.Parch+1

train1.corr()
plt.subplots(figsize = (15,8))

sns.heatmap(train1.corr(), annot=True,cmap="YlGnBu")

plt.title("Correlations Among Features", fontsize = 20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train1.drop('Survived',axis=1), 

                                                    train1['Survived'], test_size=0.30, 

                                                    random_state=42)

#y_train=train1[['Survived']]

#X_train=train1.drop(['Survived'],axis=1)

#X_train
print(X_train.head())

print(X_test.head())

scaler = StandardScaler(copy=False)

scaler.fit(X_train)

X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)

print(X_train)

print(X_test)


clf = RandomForestClassifier(random_state=42,n_estimators=10,max_depth=4,min_samples_split=100,min_samples_leaf = 1)

clf.fit(X_train,y_train)

print('Accuracy using the defualt gini impurity criterion...',clf.score(X_test,y_test))



"""clf = RandomForestClassifier(criterion="entropy",random_state=42,max_depth=10)

clf.fit(X_train,y_train)

print('Accuracy using the entropy criterion...',clf.score(X_test,y_test))"""



test_score = []

train_score = []

min_sample_split = np.arange(1,10,1)

for split in min_sample_split:

    clf = RandomForestClassifier(max_depth = split)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))

    

plt.figure(figsize = (8,8))   

plt.plot(min_sample_split,train_score)

plt.plot(min_sample_split, test_score)

plt.xlabel('Depth')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])

test_score = []

train_score = []

min_sample_split = np.arange(2,300,10)

for split in min_sample_split:

    clf = RandomForestClassifier(max_depth=4,min_samples_split = split)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))

    



max_x=min_sample_split[test_score.index(max(test_score))]  

print(max_x)    

plt.figure(figsize = (8,8))   

plt.plot(min_sample_split,train_score)

plt.plot(min_sample_split, test_score)

plt.xlabel('Min Sample Split')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])

test_score = []

train_score = []

min_sample_leaf = np.arange(1,16,1)

for leaf in min_sample_leaf:

    clf = RandomForestClassifier(max_depth=4,min_samples_split = split,min_samples_leaf = leaf)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))



max_x=min_sample_leaf[test_score.index(max(test_score))]  

print(max_x) 

plt.figure(figsize = (8,8))

plt.plot(min_sample_leaf,train_score)

plt.plot(min_sample_leaf, test_score)

plt.xlabel('Min Sample Leaf')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
test_score = []

train_score = []

n_estimators = np.arange(1,21)

for i in n_estimators:

    clf = RandomForestClassifier(random_state=42,max_depth=4,min_samples_split=100,min_samples_leaf =1,n_estimators=i)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))



max_x=n_estimators[test_score.index(max(test_score))]  

print(max_x) 

plt.figure(figsize = (8,8))

plt.plot(n_estimators,train_score)

plt.plot(n_estimators, test_score)

plt.xlabel('Iterations')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])


clf = RandomForestClassifier(random_state=42,n_estimators=10,max_depth=4,min_samples_split=100,min_samples_leaf = 1)

clf.fit(X_train,y_train)

print('Accuracy using the entropy criterion...',clf.score(X_test,y_test))

#parameters = {'criterion':('gini','entropy'),'max_depth':[2,3,4],'min_samples_split':[50,100,150],'min_samples_leaf':[4,8,12,16],'n_estimators':[100,200,300]}
"""

rf= RandomForestClassifier()

clf = GridSearchCV(rf, parameters, cv=5)

clf.fit(X_train,y_train)

print('Accuracy...',clf.score(X_test,y_test))

"""
test1=scaler.transform(test1)

print(test1)
predictions = clf.predict(test1)

predictions=predictions.astype(int)

predictions
#from sklearn.metrics import classification_report

#print(classification_report(y_test,predictions))







index=list(test['PassengerId'])

output_df=pd.DataFrame({'PassengerId':index,'Survived':predictions})

#output_df2=pd.DataFrame({'Survived':predictions})

#output_df.columns=['PassengerId','Survived']

#output_df1=output_df1.join(output_df2)

output_df

output_df.to_csv('output.csv', index=False)
output=pd.read_csv('output.csv')

output