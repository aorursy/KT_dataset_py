import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
train=pd.read_csv('../input/train.csv')
train.head()
#Pairwise Plot of every feature.

sns.pairplot(train.drop(['Name','Ticket','Cabin'],axis=1).dropna())
plt.figure(figsize=(8,8))

#Scatter plot of Survived people vs their Age & differentiating the points on the basis of Sex.

sns.scatterplot(x='Survived',y='Age',data=train,hue='Sex')
plt.figure(figsize=(6,6))

#Heatmap of missing values.

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
plt.figure(figsize=(8,8))

#Boxplot of Passenger class of passengers vs their Age.

sns.boxplot(x='Pclass',y='Age',data=train)
#Sibsp stands for Sibling/Spouse.

#Parch stands for Parents/Children.

#Combining these two attributes as Family attribute.

# 1 if person is having family, otherwise 0.

family=[1 if i[6]>0 or i[7]>0 else 0 for i in train.values]

train['Family']=family
train.head()
plt.figure(figsize=(6,6))

fam_surv=train[['Family','Survived']].groupby('Family',as_index=False).mean()



fam_surv_x=fam_surv.iloc[:,0].values

fam_surv_y=fam_surv.iloc[:,1].values

#fam_surv_x is list containing 0 and 1.

# 1 if person is having family, otherwise 0.

#fam_surv_y is list containing average survival of a person with/without family.



plt.bar(['Alone','With Family'],fam_surv_y,color=['yellow','violet'])

plt.ylabel('Average Survived')
plt.figure(figsize=(20,6))

average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

average_age['Age']=average_age['Age'].apply(int)

sns.barplot(x='Age', y='Survived', data=average_age)
#Dividing category into child,male,female.

#If age<=16, then child. Else gender.

category=['child' if i[5]<=16 else i[4] for i in train.values]

train['Category']=category
plt.figure(figsize=(6,6))

category=train[['Category','Survived']].groupby('Category',as_index=False).mean()



category_x=category.iloc[:,0].values

category_y=category.iloc[:,1].values

#category_x is list containing child,male,female.

#category_y is list containing average survival of a category type.



plt.bar(['Child','Female','Male'],category_y,color=['green','blue','red'])

plt.ylabel('Average Survived')
train.head()
#Average Age of Different Passenger classes.

age_Pclass1=np.around(train[train['Pclass']==1]['Age'].mean())

age_Pclass2=np.around(train[train['Pclass']==2]['Age'].mean())

age_Pclass3=np.around(train[train['Pclass']==3]['Age'].mean())



def impute_age(x):         #Function to fill NaN values in the age data.

    if str(x[5]).lower()=='nan':

        if x[2]==1:

            return age_Pclass1

        elif x[2]==2:

            return age_Pclass2

        else:

            return age_Pclass3

    else:

        return x[5]

    

train['Age']=train.apply(impute_age,axis=1)
embarked_mode=train['Embarked'].mode()  #Mode of embarked attribute.



def impute_embarked(x):       #Function to fill NaN values in the embarked data.

    if str(x).lower()=='nan':

        return embarked_mode[0]

    else:

        return x

        

train['Embarked']=train['Embarked'].apply(impute_embarked)
plt.figure(figsize=(6,6))

#Heatmap of missing values.

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
#Generating dummy values for Embarked Attribute.

embarked_dummy=pd.get_dummies(train['Embarked'])

embarked_dummy.head()
#Generating dummy values for Sex Attribute.

category_dummy=pd.get_dummies(train['Category'])

category_dummy.head()
#Concatenating Dummy Embarked dataframe with main dataframe.

train=pd.concat([train,embarked_dummy],axis=1)

#Concatenating Dummy Sex dataframe with main dataframe.

train=pd.concat([train,category_dummy],axis=1)

train.head()
#PassengerID, Name & Ticket are useless attributes in prediction.

#Whereas Cabin contains a lot NaN values So it has to be dropped.

#Taking both Male & Female column is not necessary, so any one can be dropped.

#Same for Embarked column. Here 'Female' and 'S' is dropped.



X=train.iloc[:,[2,5,9,12,14,15,17,18]].values 

y=train.iloc[:,1].values
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()         #Instantiating Standard Scaler Object.
X=scale.fit_transform(X)       #Scaling every feature in X.
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
dtree=DecisionTreeClassifier()

ml_algo=AdaBoostClassifier(base_estimator=dtree,n_estimators=2500,learning_rate=0.01)
ml_algo.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,ml_algo.predict(X_test))    #Generating Confusion Matrix.
ml_algo.score(X_test,y_test)         #Test Score
ml_algo.score(X_train,y_train)       #Train Score
ml_algo.score(X,y)                   #Total Score
# from sklearn.model_selection import GridSearchCV

# gcv=GridSearchCV(AdaBoostClassifier(),param_grid={'base_estimator':[dtree],'n_estimators':[2500,3000,3500,4000],'learning_rate':[0.001,0.009,0.01,1]})

# gcv.fit(X,y)
# gcv.best_estimator_
y
ml_algo.fit(X,y)                     #Training the model with whole train set.
test=pd.read_csv('../input/test.csv')
test.head()
plt.figure(figsize=(6,6))

#Heatmap of missing values.

sns.heatmap(test.isnull(),yticklabels=False,cmap='viridis')
#Filling missing Fare value with Average of Fare.

test['Fare'].fillna(test['Fare'].mean(),inplace=True) 
#Average Age of Different Passenger classes.

age_Pclass1=np.around(test[test['Pclass']==1]['Age'].mean())

age_Pclass2=np.around(test[test['Pclass']==2]['Age'].mean())

age_Pclass3=np.around(test[test['Pclass']==3]['Age'].mean())



def impute_age(x):         #Function to fill NaN values in the age data.

    if str(x[4]).lower()=='nan':

        if x[1]==1:

            return age_Pclass1

        elif x[1]==2:

            return age_Pclass2

        else:

            return age_Pclass3

    else:

        return x[4]

    

test['Age']=test.apply(impute_age,axis=1)
plt.figure(figsize=(6,6))

#Heatmap of missing values.

sns.heatmap(test.isnull(),yticklabels=False,cmap='viridis')
#Sibsp stands for Sibling/Spouse.

#Parch stands for Parents/Children.

#Combining these two attributes as Family attribute.

# 1 if person is having family, otherwise 0.

family=[1 if i[5]>0 or i[6]>0 else 0 for i in test.values]

test['Family']=family
#Dividing category into child,male,female.

#If age<=16, then child. Else gender.

category=['child' if i[4]<=16 else i[3] for i in test.values]

test['Category']=category
#Generating dummy values for Embarked Attribute.

embarked_dummy=pd.get_dummies(test['Embarked'])

embarked_dummy.head()
#Generating dummy values for Category Attribute.

category_dummy=pd.get_dummies(test['Category'])

category_dummy.head()
#Concatenating Dummy Embarked dataframe with main dataframe.

test=pd.concat([test,embarked_dummy],axis=1)

#Concatenating Dummy Category dataframe with main dataframe.

test=pd.concat([test,category_dummy],axis=1)

test.head()
#PassengerID, Name & Ticket are useless attributes in prediction.

#Whereas Cabin contains a lot NaN values So it has to be dropped.

#Taking both Male & Female column is not necessary, so any one can be dropped.

#Same for Embarked column. Here 'Female' and 'S' is dropped.

X_TEST=test.iloc[:,[1,4,8,11,13,14,16,17]].values
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()         #Instantiating Standard Scaler Object.
X_TEST=scale.fit_transform(X_TEST)       #Scaling every feature in X.
X_TEST
pred=ml_algo.predict(X_TEST).reshape(-1,1)  #Predictions
id=test.iloc[:,0].values.reshape(-1,1)  #Passenger ID of Test Data

output=np.concatenate((id,pred),axis=1)
submission=pd.DataFrame(output,columns=['PassengerId','Survived'])        #Dataframe of submissions
submission.to_csv('Submission.csv',index=False)    #Generating submission file.