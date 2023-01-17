import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

data.head()
data.info()
data.describe()
sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis',cbar=False)

#age and cabin has most null values
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=data)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=data)
sns.set_style('whitegrid')

sns.countplot(x='Pclass',hue='Sex',data=data)
plt.figure(figsize=(19, 8))

sns.boxplot(x='Pclass',y='Age',data=data,hue='Sex',color="cyan")
data.groupby(['Pclass','Sex'])['Age'].mean()
sns.distplot(data['Age'],kde=False)
data['Age']=data['Age'].fillna(0)

test['Age']=test['Age'].fillna(0)
def fill_AgeD(row):

    if(row['Sex']=='male' and row['Pclass']==1 and row['Age']==0):

        row['Age']=41

    elif(row['Sex']=='female' and row['Pclass']==1 and row['Age']==0):

        row['Age']=35

    elif(row['Sex']=='male' and row['Pclass']==2 and row['Age']==0):

        row['Age']=31

    elif(row['Sex']=='female' and row['Pclass']==2 and row['Age']==0):

        row['Age']=29

    elif(row['Sex']=='male' and row['Pclass']==3 and row['Age']==0):

        row['Age']=26

    elif(row['Sex']=='female' and row['Pclass']==3 and row['Age']==0):

        row['Age']=22

    return row

    

data=data.apply(fill_AgeD,axis=1)

test.groupby(['Pclass','Sex'])['Age'].mean()
def fill_AgeT(row):

    if(row['Sex']=='male' and row['Pclass']==1 and row['Age']==0):

        row['Age']=40

    elif(row['Sex']=='female' and row['Pclass']==1 and row['Age']==0):

        row['Age']=41

    elif(row['Sex']=='male' and row['Pclass']==2 and row['Age']==0):

        row['Age']=31

    elif(row['Sex']=='female' and row['Pclass']==2 and row['Age']==0):

        row['Age']=24

    elif(row['Sex']=='male' and row['Pclass']==3 and row['Age']==0):

        row['Age']=24

    elif(row['Sex']=='female' and row['Pclass']==3 and row['Age']==0):

        row['Age']=23

    return row

    

test=test.apply(fill_AgeT,axis=1)

test.head()
data.groupby(['Pclass','Sex'])['Age'].mean()
sns.distplot(data['Age'],kde=False)
sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis',cbar=False)
data.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
def age_cat(val):

    if val<1:

        return 'Infant'

    elif val>1 and val < 13:

        return 'Child'

    elif val>13 and val < 18:

        return 'Teen'

    elif val>18 and val < 65:

        return 'Adult'

    else:

        return 'Elderly'

data['Age_category']=data['Age'].apply(age_cat)
plt.figure(figsize=(15, 8))

sns.set_style('whitegrid')

sns.countplot(x='Age_category',hue='Survived',data=data)
res=data[data['Survived']==1].groupby('Sex')['Survived'].count()

res

res=np.array(res)

labels=['Female','Male']

plt.pie(res,autopct='%1.1f%%', shadow=True, startangle=140,labels=labels,explode = (0.1,0))

#most people survived from Embarked Region=Southampton

res=data[data['Survived']==1].groupby('Embarked')['Survived'].count()

res=np.array(res)

labels=['Cherbourg','Queenstown','Southampton']

plt.pie(res,autopct='%1.1f%%', shadow=True, startangle=140,labels=labels,explode = (0.1, 0, 0))

#most people survived from Embarked Region=Southampton

res=pd.get_dummies(data['Embarked'])

data=pd.concat([data,res],axis=1)

data.drop('Embarked',axis=1,inplace=True)

res=pd.get_dummies(test['Embarked'])

test=pd.concat([test,res],axis=1)

test.drop('Embarked',axis=1,inplace=True)

data.head()

data['Family_Size']=data['SibSp']+data['Parch']+1

data.drop(['SibSp','Parch'],axis=1,inplace=True)

test['Family_Size']=test['SibSp']+test['Parch']+1

test.drop(['SibSp','Parch'],axis=1,inplace=True)

data.head()
data['Sex']=data['Sex'].map({ 'male': 1, 'female': 0})

test['Sex']=test['Sex'].map({ 'male': 1, 'female': 0})

test.loc[test['Fare'].isna(),['Fare']]=13.67
X=data.iloc[:,[2,4,5,7,9,10,11,12]].values

y=data.iloc[:,1].values

test_X=test.iloc[:,[1,3,4,6,7,8,9,10]].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

test_X =sc.transform(test_X)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

# Creating odd list K for KNN

neighbors = list(range(1,30))

# empty list that will hold cv scores

cv_scores = [ ]

#perform 10-fold cross-validation

for K in neighbors:

    knn = KNeighborsClassifier(n_neighbors = K)

    scores = cross_val_score(knn,X_train,y_train,cv = 10,scoring =

    "accuracy")

    cv_scores.append(scores.mean())

# Changing to mis classification error

mse = [1-x for x in cv_scores]

# determing best k

optimal_k = neighbors[mse.index(min(mse))]



print("The optimal no. of neighbors is {}".format(optimal_k))
from sklearn.neighbors import KNeighborsClassifier

KNNclassifier=KNeighborsClassifier(n_neighbors=12)

KNNclassifier.fit(X_train,y_train)

y_pred = KNNclassifier.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy :",accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
from sklearn.svm import SVC

SVMlinear=SVC(kernel='linear')

SVMlinear.fit(X_train,y_train)

SVMlinear_predict=SVMlinear.predict(X_test)

y_pred = SVMlinear.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
from sklearn.svm import SVC

SVMrbf=SVC(kernel='rbf')

SVMrbf.fit(X_train,y_train)

SVMrbf_predict=SVMrbf.predict(X_test)

y_pred = SVMrbf.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
from sklearn.naive_bayes import GaussianNB

NB=GaussianNB()

NB.fit(X_train,y_train)

NB_predict=NB.predict(X_test)

y_pred = NB.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
from sklearn.tree import DecisionTreeClassifier

DecisionTree=DecisionTreeClassifier(criterion='entropy',random_state=0)

DecisionTree.fit(X_train,y_train)

DecisionTree_predict=DecisionTree.predict(X_test)

y_pred = DecisionTree.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

trees = list(range(1,20))

# empty list that will hold cv scores

cv_scores = [ ]

#perform 10-fold cross-validation

for n in trees:

    RFC = RandomForestClassifier(n_estimators = n,criterion='entropy',random_state=0)

    scores = cross_val_score(RFC,X_train,y_train,cv = 10,scoring =

    "accuracy")

    cv_scores.append(scores.mean())
# Changing to mis classification error

mse = [1-x for x in cv_scores]

# determing best n

optimal_n = trees[mse.index(min(mse))]

print("The optimal no. of trees is {}".format(optimal_n))
from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier(n_estimators=17,criterion='entropy',random_state=0)

RFC.fit(X_train,y_train)

RFC_predict=RFC.predict(X_test)

y_pred = RFC.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = RFC.predict(test_X)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output['PassengerId']=output['PassengerId'].astype(int)



output.to_csv('submission1.csv', index=False)