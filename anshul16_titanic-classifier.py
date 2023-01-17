
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')

#Read the data
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
data_df=train_df.append(test_df)

#explore the data
print(train_df.shape)
print("---------------")
print(test_df.shape)

print("\n Train Data Details")
train_df.info()
print("\n Test Data Details")
test_df.info()
print("--------------------------\n")
print("Null values correponding to each feature in train data")
print(pd.isnull(train_df).sum())
print("\n Null values correponding to each feature in test data")
print(pd.isnull(test_df).sum())
# visualize the data
train_df.head(10)
# drop Ticket, PassengerId, Name
train_df=train_df.drop(['PassengerId','Ticket'], axis=1)
test_df=test_df.drop(['Ticket'], axis=1)
# visualise filtered train data
train_df.head()
# visualize filtered test data
test_df.head()
# Dealing with missing values in embarked.
# finding counts of each categorial value in embarked column'
train_df['Embarked'].value_counts()

#Thus filling missing embarked values with 'S' the most frequent values and then visualizing the effect on survival rates.

#train_df['Embarked'].fillna('S')

#Visualizing embarked and survival 
fig, axes = plt.subplots(1,3,figsize=(15,5))

# Plot showing Distribution of Embarked
sns.countplot(x='Embarked',data=train_df,ax=axes[0])

#Plot Showing Survival rates corresponding to each Embarked category
sns.countplot(x='Survived',hue='Embarked',data=train_df,ax=axes[1],order=[1,0])

# Calculate Mean survival % based on each embarked category
embarked_perc=train_df[['Embarked','Survived']].groupby('Embarked',as_index=False).mean()
print(embarked_perc)
sns.barplot(x='Embarked',y='Survived',data=embarked_perc,ax=axes[2])

# From the plots it can be interpreted that 'S' has a lower survival impact as compared to 'C' or 'Q'.
# drop'S' and consider only 'C' or 'Q' as they have good survival rates.
# ref-https://en.wikiversity.org/wiki/Dummy_variable_(statistics)
embark_dummies_train  = pd.get_dummies(train_df['Embarked'])
embark_dummies_train.drop(['S'], axis=1, inplace=True)

embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

train_df=train_df.join(embark_dummies_train)
test_df=test_df.join(embark_dummies_test)

train_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

print(train_df.head(10))
print(test_df.head(10))
# Examining Cabin
# Cabin has lot of NULL values, thus we need to drop it. 
# Reference (https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/)
# The below figure indicates more than 77 % data in cabin section of train is missing, thus removing it
print(float(train_df['Cabin'].isnull().sum())/891.0)

train_df.drop(['Cabin'],inplace=True,axis=1)
test_df.drop(['Cabin'],inplace=True,axis=1)
print(train_df.head(10))
print(test_df.head(10))
# Dealing with family size

# instead of having two features SibSp and Parch we can have a feature with family of Alone.

train_df['Family']=train_df['SibSp']+train_df['Parch']
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

test_df['Family']=test_df['SibSp']+test_df['Parch']
test_df['Family'].loc[test_df['Family']>0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0


# drop Parch and SibSp
train_df.drop(['SibSp','Parch'],axis=1,inplace=True)
test_df.drop(['SibSp','Parch'],axis=1,inplace=True)

# Visulaize Survival rate vs compared to family
fig, axes = plt.subplots(1,2,figsize=(10,5))
# Plot to show count of people with or without family
sns.countplot(x='Family',data=train_df,ax=axes[0],order=[1,0])

# Plot to show mean survival rate of people with or without family
family_mean=train_df[['Family','Survived']].groupby('Family',as_index=False).mean()
sns.barplot(x='Family',y='Survived',data=family_mean,ax=axes[1],order=[1,0])

print(train_df.head(10))
print(test_df.head(10))
# Dealing with Age
train_null_ages=(train_df['Age'].isnull().sum())/891.0
print("Null Age percentage %.10f" %train_null_ages)
# Thus not dropping ages since, filling ages with numbers in the range of mean+std_dev to mean-std_dev 
# This is chosen to keep the generated ages out of oulier section of data



# for training data
train_age_mean=train_df['Age'].mean()
train_age_std_dev=train_df['Age'].std()
train_null_ages=train_df['Age'].isnull().sum()

# for test data
test_age_mean=test_df['Age'].mean()
test_age_std_dev=test_df['Age'].std()
test_null_ages=test_df['Age'].isnull().sum()

# genetate random ages
train_rand_ages=np.random.randint(train_age_mean-train_age_std_dev,train_age_mean+train_age_std_dev,size=train_null_ages)
test_rand_ages=np.random.randint(test_age_mean-test_age_std_dev,test_age_mean+test_age_std_dev,size=test_null_ages)

train_df['Age'][np.isnan(train_df['Age'])]=train_rand_ages
test_df['Age'][np.isnan(test_df['Age'])]=test_rand_ages

print(train_df.head(10))
print(test_df.head(10))
# Fare
print("Missing Fare Values in train %d" %train_df['Fare'].isnull().sum())
print("Missing Fare Values in test %d" %test_df['Fare'].isnull().sum())

#Only one value is missing in Fare section of test data. Fill it with median
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)
# converting fare values to intervals and assigning them Integer Labels 
# ref-https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
#https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-interval-variables/
train_df['CategoricalFare'] = pd.qcut(train_df['Fare'], 4)
train_fare_mean=train_df[['CategoricalFare','Survived']].groupby(['CategoricalFare'],as_index=False).mean()
print(train_fare_mean)

# Mapping Fare in train data
train_df.loc[ train_df['Fare'] <= 7.91, 'Fare']= 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare']   = 2
train_df.loc[ train_df['Fare'] > 31, 'Fare'] = 3
train_df['Fare'] = train_df['Fare'].astype(int)

# Mapping Fare in test data
test_df.loc[ train_df['Fare'] <= 7.91, 'Fare']= 0
test_df.loc[(train_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
test_df.loc[(train_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare']   = 2
test_df.loc[ train_df['Fare'] > 31, 'Fare'] = 3
test_df['Fare'] = test_df['Fare'].astype(int)

train_df.drop(['CategoricalFare'],axis=1,inplace=True)

print(train_df.head(10))
print(test_df.head(10))
# Pclass
fig ,(ax1,ax2) = plt.subplots(1,2, figsize=(15,4))

sns.countplot(x='Pclass',data=train_df,ax=ax1)
Pclass_survived_mean=train_df[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()
print(Pclass_survived_mean)
sns.barplot(x="Pclass",y='Survived',data=Pclass_survived_mean,ax=ax2)

# # Since 3rd class has lowest survival average we can drop it and retain remaining.

# Pclass_train_dummies=pd.get_dummies(train_df['Pclass'])
# Pclass_train_dummies.columns=['C1','C2','C3']
# Pclass_train_dummies.drop(['C3'],inplace=True,axis=1)

# Pclass_test_dummies=pd.get_dummies(test_df['Pclass'])
# Pclass_test_dummies.columns=['C1','C2','C3']
# Pclass_test_dummies.drop(['C3'],inplace=True,axis=1)

# train_df.drop(['Pclass'],inplace=True,axis=1)
# test_df.drop(['Pclass'],inplace=True,axis=1)

# train_df=train_df.join(Pclass_train_dummies)
# test_df=test_df.join(Pclass_test_dummies)
print(train_df.head(10))
print(test_df.head(10))


# Sex

# Sex category has close relation with Age and in all related to Survival.
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age']  = test_df['Age'].astype(int)

train_age_mean=train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean()

fig, ax=plt.subplots(1,1,figsize=(18,4))
sns.barplot(x='Age',y='Survived',data=train_age_mean)
# Here we see that age < 16 have high survival rates, thus we can group sex into three sub categories,
# We create a new column with 3 categorical values Male, Female and Children.

def get_person(person):
    age,sex = person
    return 'child' if age < 16 else sex

train_df['Person']= train_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']=test_df[['Age','Sex']].apply(get_person,axis=1)

person_mean_survival=train_df[['Person','Survived']].groupby(['Person'],as_index=False).mean()
fig, ax=plt.subplots(1,1,figsize=(18,4))
sns.barplot(x="Person",y="Survived",data=person_mean_survival)
# We drop the male category as it has the lowest survival rate

train_person_dummies=pd.get_dummies(train_df['Person'])
train_person_dummies.drop(['male'],axis=1,inplace=True)
train_df=train_df.join(train_person_dummies)

test_person_dummies=pd.get_dummies(test_df['Person'])
test_person_dummies.drop(['male'],axis=1,inplace=True)
test_df=test_df.join(test_person_dummies)

train_df.drop(['Sex','Person'],axis=1,inplace=True)
test_df.drop(['Sex','Person'],axis=1,inplace=True)
print(train_df.head(10))
print(test_df.head(10))
# Converting Age into intervals and categorical data.
train_df['CategoricalAge']=pd.qcut(train_df['Age'],4)
train_age_mean=train_df[['CategoricalAge','Survived']].groupby(['CategoricalAge'],as_index=False).mean()
print(train_age_mean)

train_df.loc[ train_df['Age'] <= 21, 'Age'] = 0
train_df.loc[(train_df['Age'] > 21) & (train_df['Age'] <= 28), 'Age'] = 1
train_df.loc[(train_df['Age'] > 28) & (train_df['Age'] <= 37), 'Age'] = 2
train_df.loc[(train_df['Age'] > 37) & (train_df['Age'] <= 80), 'Age'] = 3 

test_df.loc[test_df['Age'] <= 21, 'Age'] = 0
test_df.loc[(test_df['Age'] > 21) & (test_df['Age'] <= 28), 'Age'] = 1
test_df.loc[(test_df['Age'] > 28) & (test_df['Age'] <= 37), 'Age'] = 2
test_df.loc[(test_df['Age'] > 37) & (test_df['Age'] <= 80), 'Age'] = 3


train_df.drop(['CategoricalAge'],inplace=True,axis=1)
print(train_df.head(10))
print(test_df.head(10))

# Processing Name
import re
def get_title(name):
    # match characters from [a-zA-Z] followed by .
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

train_df['Title']=train_df['Name'].apply(get_title)
test_df['Title']=test_df['Name'].apply(get_title)

train_title_survive=train_df[['Title','Survived']].groupby(['Title'],as_index=False).mean()
print(train_title_survive)

# Replace titles other than Master,Miss,Mr,Mrs with 'Rare'
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
# train_df['Title'] = train_df['Title'].map(title_mapping)
# train_df['Title'] = train_df['Title'].fillna(0)
# test_df['Title'] = test_df['Title'].map(title_mapping)
# test_df['Title'] = test_df['Title'].fillna(0)

train_title_survive_2=train_df[['Title','Survived']].groupby(['Title'],as_index=False).mean()
print(train_title_survive_2)
# dropping Mr. as it has less survival rate and assigning one hot encoding to others.

train_title_dummies=pd.get_dummies(train_df['Title'])
train_title_dummies.drop(['Mr'],inplace=True,axis=1)
train_df=train_df.join(train_title_dummies)

test_title_dummies=pd.get_dummies(test_df['Title'])
test_title_dummies.drop(['Mr'],inplace=True,axis=1)
test_df=test_df.join(test_title_dummies)

train_df.drop(['Name','Title'],axis=1,inplace=True)
test_df.drop(['Name','Title'],axis=1,inplace=True)

print(train_df.head(10))
print(test_df.head(10))
# Split data sets into features and labels
features_train=train_df.drop(['Survived'],axis=1)
labels_train=train_df['Survived']

pids=test_df['PassengerId']
features_test=test_df.drop(['PassengerId'],axis=1)
sns.countplot(x='Survived',data=train_df)

# Random Forest with K fold cross validation
# from sklearn.model_selection import cross_val_score,train_test_split
# from sklearn.ensemble import RandomForestClassifier as rfc

# # n_estimators has been set to 170 by trying out different values.
# clf_rfc=rfc(n_estimators=170)

# scores=cross_val_score(clf_rfc, features_train, labels_train, cv=7)
# print("For K = %d cross_val_score = %.10f" %(7,scores.mean()))
# Init models and use Stratified K-Fold Cross Val Score

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import f1_score, accuracy_score

models_list=[
    RandomForestClassifier(n_estimators=250),
    AdaBoostClassifier(RandomForestClassifier(n_estimators=250)),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=3),
    SVC(),
    LogisticRegression()
]

log_cols = ["Classifier", "F1-Score"]
log = pd.DataFrame(columns=log_cols)

f1_score_dict={}
n_folds=10
sss=StratifiedShuffleSplit(n_splits=n_folds,test_size=0.1,random_state=42)

X=features_train.values
y=labels_train.values
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    
    for model in models_list:
        # to extract only the name of the class excluding modules.
        clf_name=model.__class__.__name__
        model.fit(X_train,y_train)
        predictions=model.predict(X_test)
        f1=f1_score(y_test,predictions,
                          average='binary'
                         )
        
        if clf_name in f1_score_dict:
            f1_score_dict[clf_name]+=f1
        else:
            f1_score_dict[clf_name]=f1

for clf in f1_score_dict:
    f1_score_dict[clf] = f1_score_dict[clf] / float(n_folds)
    log_entry = pd.DataFrame([[clf, f1_score_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)


fig, axis = plt.subplots(1,1,figsize=(15,4))
plt.xlabel('Classifiers')
plt.ylabel('F1-Score(Mean)')
plt.title('Classifiers vs F1-Score')
log=log.sort_values('F1-Score')
sns.barplot(x='F1-Score',y='Classifier',data=log)
clf_final=AdaBoostClassifier(RandomForestClassifier(n_estimators=250))
clf_final.fit(features_train,labels_train)
predictions=clf_final.predict(features_test)
submission = pd.DataFrame({
        "PassengerId": pids,
        "Survived": predictions
    })
submission.to_csv('submission_final.csv', index=False)


