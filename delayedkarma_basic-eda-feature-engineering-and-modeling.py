# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import libraries
import pandas as pd
pd.set_option("display.max_rows",200)
pd.set_option("display.max_columns",200)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
from scipy import stats

# Import train and test datasets and check lengths
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print(len(df_train),len(df_test))
# Check the 1st 5 rows
df_train.head()
df_test.head()
# We have to predict the Survived column (Target)
df_train.info()
# Have to change columns with object values to numerical values

# Explore the features in the df_train set
# Should drop Name from train but check to see if
# names have any significance, perhaps an honorific implies a higher survival chance?

plt.subplot(2,2,1)
df_train[df_train['Name'].str.contains('Mr. ')]['Survived'].value_counts().plot(kind='bar',\
                                                                               title='Survived by Mr')


plt.xticks(rotation=0);

plt.subplot(2,2,2)
df_train[df_train['Name'].str.contains('Mrs. ')]['Survived'].value_counts().plot(kind='bar',\
                                                                                title=\
                                                                        'Survived by Mrs');

plt.xticks(rotation=0);

plt.subplot(2,2,3)
df_train[df_train['Name'].str.contains('Miss. ')]['Survived'].value_counts().plot(kind='bar',\
                                                                    title='Survived by Miss');

plt.xticks(rotation=0);

plt.subplot(2,2,4)
df_train[df_train['Name'].str.contains('Master. ')]['Survived'].value_counts().plot(kind='bar',\
                                                            title='Survived by Master');
plt.tight_layout()
plt.xticks(rotation=0);

len(df_train[df_train['Name'].str.contains('Dr. ')])
len(df_train[df_train['Name'].str.contains('Sir. ')])
len(df_train[df_train['Cabin'].isnull()])
# Since most of the values for Cabin are null we can safely drop it
# Drop columns that we won't use as features from train and test
df_train.drop(['Name','PassengerId','Ticket','Cabin'],axis=1,inplace=True)
df_train.head()
# Exploring features
# 1. Embarked

# Any missing values?
df_train[df_train['Embarked'].isnull()]
df_train['Embarked'].unique()

df_train['Embarked'].value_counts()
# So the port S is most common, so it would make sense to fill in
# the missing values with 'S'
df_train['Embarked'].fillna('S',inplace=True)
df_train[df_train['Embarked'].isnull()]
# Check survival rates by port of embarkation
sns.factorplot('Embarked','Survived',data=df_train,aspect=3);
# Port C has highest Survival Rate, so a case could be made for 
# filling in missing Embarked vals with 'C' instead of 'S'
# More checks on dependence of survival on embarkation port
fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked',data=df_train,ax=axis1);
sns.countplot(x='Survived',hue='Embarked',data=df_train,order=[0,1],ax=axis2);
# Plot mean survival rate for each port of embarkation
embark_grp_mean = df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked',y='Survived',data=embark_grp_mean,order=['S','C','Q'],ax=axis3);
# Get unique vals of embarked
embarked_vals = sorted(df_train['Embarked'].unique())
# Generate mapping of embarked from strings to numbers
embarked_map = dict(zip(embarked_vals,range(0,len(embarked_vals)+1)))
embarked_map

# A numerical ordering for Embarked makes no sense
# so change this to dummy variables instead
df_train = pd.concat([df_train,pd.get_dummies(df_train['Embarked'],prefix='Embarked')],axis=1)
df_train.drop('Embarked',axis=1,inplace=True)
df_train.head()

# Check passenger class and sex based on embarkation port
plt.subplot(2,1,1)
df_train[df_train['Embarked_C']==1.0]['Sex'].value_counts().plot(kind='bar',\
                                                                title='Cherbourg')
plt.xticks(rotation=0);
plt.subplot(2,1,2)
df_train[df_train['Embarked_C']==1.0]['Pclass'].value_counts().plot(kind='bar')
                                                                
plt.xticks(rotation=0);
# Check passenger class and sex based on embarkation port
plt.subplot(2,1,1)
df_train[df_train['Embarked_Q']==1.0]['Sex'].value_counts().plot(kind='bar',\
                                                                title='Queenstown')
plt.xticks(rotation=0);
plt.subplot(2,1,2)
df_train[df_train['Embarked_Q']==1.0]['Pclass'].value_counts().plot(kind='bar')
                                                                
plt.xticks(rotation=0);

# Check passenger class and sex based on embarkation port
plt.subplot(2,1,1)
df_train[df_train['Embarked_S']==1.0]['Sex'].value_counts().plot(kind='bar',\
                                                                title='Southampton')
plt.xticks(rotation=0);
plt.subplot(2,1,2)
df_train[df_train['Embarked_S']==1.0]['Pclass'].value_counts().plot(kind='bar')
                                                                
plt.xticks(rotation=0);

# Exploring features
# 2. Fare

# Check for null values
df_train[df_train['Fare'].isnull()]
# Look at relationship between Fare and survival rate

fare_notsurv = df_train[df_train['Survived']==0]['Fare'] #Fare of passengers who didn't survive
fare_surv = df_train[df_train['Survived']==1]['Fare'] #Fare of passengers who survived
max_fare = df_train['Fare'].max()
plt.hist([fare_notsurv,fare_surv],stacked=True,bins=int(max_fare/50),range=(1,max_fare));
plt.xlabel('Fare');plt.ylabel('Count');
plt.legend(['Dead','Survived'],loc='best');

avg_fare = pd.DataFrame([fare_notsurv.mean(),fare_surv.mean()])
std_fare = pd.DataFrame([fare_notsurv.std(),fare_surv.std()])
print("Mean fare for not survived is {} and survived is {}".format(fare_notsurv.mean(),\
                                                                   fare_surv.mean()))
avg_fare.plot(yerr=std_fare,kind='bar',legend=None)
plt.ylabel('Fare')
plt.xlabel('Survived')
plt.xticks(rotation=0);
#Exploring features
#3. Age

fig,(axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original age values')
axis2.set_title('New age values')

df_train['Age'].dropna().hist(bins=70,ax=axis1);
# The age distribution seems skewed, so using median value to fillna
df_train['Age'].fillna(df_train['Age'].dropna().median(),inplace=True)
df_train['Age'].hist(bins=70,ax=axis2);
# Basic nature of distribution seems to be unchanged; this statement can be investigated
df_train[df_train['Age'].isnull()]
# Look at relationship between Age and survival rate

age_notsurv = df_train[df_train['Survived']==0]['Age'] #Fare of passengers who didn't survive
age_surv = df_train[df_train['Survived']==1]['Age'] #Age of passengers who survived
max_age = df_train['Age'].max()
plt.hist([age_notsurv,age_surv],stacked=True,bins=int(max_age/10),range=(1,max_age));
plt.xlabel('Age');plt.ylabel('Count');
plt.legend(['Dead','Survived'],loc='best');

avg_age = pd.DataFrame([age_notsurv.mean(),age_surv.mean()])
std_age = pd.DataFrame([age_notsurv.std(),age_surv.std()])
print("Mean age for not survived is {} and survived is {}".format(age_notsurv.mean(),\
                                                                   age_surv.mean()))

avg_age.plot(yerr=std_age,kind='bar',legend=None)
plt.ylabel('Age')
plt.xlabel('Survived')
plt.xticks(rotation=0);
# Hard to tell a dependency from this plot
# Check age dependencies and survival of various other features
df_train_survived = df_train[df_train['Survived']==1]

plt.subplot(2,2,1)
df_train_survived['Age'].hist(bins=int(max_age/10),range=(1,max_age))
plt.title('Age of survivors')

plt.subplot(2,2,2)
df_train_survived[df_train_survived['Sex']=='female']['Age'].hist(bins=int(max_age/10),range=(1,max_age))
plt.title('Age of female survivors')

plt.subplot(2,2,3)
df_train_survived[df_train_survived['Pclass']==1]['Age'].hist(bins=int(max_age/10),range=(1,max_age))
plt.title('Age of 1st class passengers who survived')

plt.subplot(2,2,4)
df_train_survived[df_train_survived['Fare']>50]['Age'].hist(bins=int(max_age/10),range=(1,max_age))
plt.title('Age of survivors who paid > 50 fare')
plt.tight_layout()

# In all cases, most survivors seem to be between 20 and 30 years of age

passenger_classes = sorted(df_train['Pclass'].unique())
for p in passenger_classes:
    df_train[df_train['Pclass']==p]['Age'].plot(kind='kde')
plt.title('Age density by passenger class')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(('1st class','2nd class','3rd class'),loc='best');
# 1st class passengers tend to be the oldest

for p in passenger_classes:
    df_train_survived[df_train_survived['Pclass']==p]['Age'].plot(kind='kde')
plt.title('Age density by passenger class (survivors)')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(('1st class','2nd class','3rd class'),loc='best');
# Difference between this and above plot seems most marked for 2nd and 3rd class passengers
# Exploring features
# 4. Family

# Create one family column
df_train['Family'] = df_train['Parch'] + df_train['SibSp']
df_train.drop(['Parch','SibSp'],axis=1,inplace=True)
# Make family a categorical variable, family member present = 1, else = 0
df_train['Family'].loc[df_train['Family']>1]=1
df_train['Family'].loc[df_train['Family']==0]=0
df_train['Family'].value_counts()
family_surv = df_train[['Family','Survived']].groupby(['Family'],as_index=False).mean()
ax=sns.barplot(x='Family',y='Survived',data=family_surv,order=[1,0])
ax.set_xticklabels(['With family','No family'],rotation=0);

# Exploring features
# 5. Sex

# 'Women' and 'children' first so divide into male, female and child categories (child age is < 12)

def get_person(passenger):
    age,sex=passenger
    return 'child' if age < 12 else sex

df_train['Person'] = df_train[['Age','Sex']].apply(get_person,axis=1)
df_train.drop('Sex',axis=1,inplace=True)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Person',data=df_train,ax=ax1)

# Mean survival rate for male, female, child
person_mean = df_train[['Person','Survived']].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person',y='Survived',data=person_mean,ax=ax2,order=['male','female','child']);
# As we would expect, females and children have a high rate of survival
# Make 'Person' a categorical variable
df_train = pd.concat([df_train,pd.get_dummies(df_train['Person'],prefix='Person')],axis=1)
df_train.drop(['Person'],axis=1,inplace=True)
# Exploring features
# 6. Passenger Class

sns.factorplot(x='Pclass',y='Survived',data=df_train,order=[1,2,3],size=5,aspect=3);
# Very high survival rate for first class passengers, Pclass is definitely important for survival predictions
df_train.head()
df_train.info()
# All columns now have numeric entries
# Go through all the formatting steps for the test data set

df_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True) #Can't drop PassengerId here, has to be associated 
#  to binary survival output

df_test['Embarked'].fillna('S',inplace=True)
df_test = pd.concat([df_test,pd.get_dummies(df_test['Embarked'],prefix='Embarked')],axis=1)
df_test.drop('Embarked',axis=1,inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(),inplace=True)

df_test['Age'].fillna(df_test['Age'].dropna().median(),inplace=True)

df_test['Family'] = df_test['Parch'] + df_test['SibSp']
df_test.drop(['Parch','SibSp'],axis=1,inplace=True)
df_test['Family'].loc[df_test['Family']>1]=1
df_test['Family'].loc[df_test['Family']==0]=0

def get_person(passenger):
    age,sex=passenger
    return 'child' if age < 12 else sex
df_test['Person'] = df_test[['Age','Sex']].apply(get_person,axis=1)
df_test.drop('Sex',axis=1,inplace=True)
df_test = pd.concat([df_test,pd.get_dummies(df_test['Person'],prefix='Person')],axis=1)
df_test.drop(['Person'],axis=1,inplace=True)

df_test.head()
# Define training and testing sets

features_train = df_train.drop(['Survived'],axis=1) #Train_features
target_train= df_train['Survived'] #Train_target

features_test = df_test.drop(['PassengerId'],axis=1) #Test_feature
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
train_x,test_x,train_y,test_y = train_test_split(features_train,target_train,test_size=0.2,random_state=42)
#train test split for evaluating model accuracy
# 1. Gaussian NB
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(features_train,target_train)
target_test_nb = clf_nb.predict(features_test)
df_test['Survived'] = target_test_nb
df_test[['PassengerId','Survived']].to_csv('gaussnb-kaggle.csv',index=False) #Kaggle Submission
# Evaluate model accuracy
clf_nb.fit(train_x,train_y)
pred_gnb_y = clf_nb.predict(test_x)
print('Accuracy score of Gaussian NB is {}'.format(metrics.accuracy_score(pred_gnb_y,test_y)))
# 2. SVM
from sklearn.svm import SVC
svc = SVC(kernel='rbf',class_weight='balanced')
param_grid_svm = {'C': [1, 5, 10, 50,100],
              'gamma': [0.0001, 0.0005, 0.001, 0.005,0.01]}
grid_svm = GridSearchCV(estimator=svc, param_grid=param_grid_svm)
grid_svm.fit(train_x,train_y)
grid_svm.best_params_
clf_svm = grid_svm.best_estimator_
clf_svm.fit(features_train,target_train)
target_test_svm = clf_svm.predict(features_test)
df_test['Survived'] = target_test_svm
df_test[['PassengerId','Survived']].to_csv('svm-kaggle.csv',index=False) #Kaggle Submission
#Evaluate model accuracy
clf_svm.fit(train_x,train_y)
pred_svm_y = clf_svm.predict(test_x)
print('Accuracy score of SVM is {}'.format(metrics.accuracy_score(pred_svm_y,test_y)))
# 3. RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy')
param_grid_rf = {'n_estimators':[10,100,250,500,1000],
               'max_features':['sqrt','log2'],'min_samples_split':[2,5,10,50,100]}
grid_rf = GridSearchCV(estimator=rf,param_grid=param_grid_rf)
grid_rf.fit(train_x,train_y)
grid_rf.best_params_
clf_rf = grid_rf.best_estimator_
clf_rf.fit(features_train,target_train)
target_test_rf = clf_rf.predict(features_test)
df_test['Survived'] = target_test_rf
df_test[['PassengerId','Survived']].to_csv('rf-kaggle.csv',index=False) #Kaggle Submission
#Evaluate model accuracy
clf_rf.fit(train_x,train_y)
pred_rf_y = clf_rf.predict(test_x)
print('Accuracy score of RF is {}'.format(metrics.accuracy_score(pred_rf_y,test_y)))
### Try out a basic averaging

target_avg = 0.2 * target_test_nb + 0.3 * target_test_svm + 0.5 * target_test_rf
df_test['Survived'] = target_test_rf
df_test[['PassengerId','Survived']].to_csv('avg-kaggle.csv',index=False) #Kaggle Submission
