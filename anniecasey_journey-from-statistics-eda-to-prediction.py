import numpy as np

import scipy as sc

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statistics as stat

%matplotlib inline

sns.set_style('whitegrid')



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Normalizer



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.describe()
test.describe()
train.describe(include=['O'])
train.describe(include=[np.object])

## [np.number], [np.object], ['category'], 'all'
test.describe(include=['O'])
train.head()
f,ax = plt.subplots(3,4,figsize=(20,16))

sns.countplot('Pclass',data=train,ax=ax[0,0])

sns.countplot('Sex',data=train,ax=ax[0,1])

sns.boxplot(x='Pclass',y='Age',data=train,ax=ax[0,2])

sns.countplot('SibSp',hue='Survived',data=train,ax=ax[0,3],palette='husl')

sns.distplot(train['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')

sns.countplot('Embarked',data=train,ax=ax[2,2])



sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1,0],palette='husl')

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1,1],palette='husl')

sns.distplot(train[train['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)

sns.distplot(train[train['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)

sns.countplot('Parch',hue='Survived',data=train,ax=ax[1,3],palette='husl')

sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train,palette='husl',ax=ax[2,1])

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[2,3],palette='husl')



ax[0,0].set_title('Total Passengers by Class')

ax[0,1].set_title('Total Passengers by Gender')

ax[0,2].set_title('Age Box Plot By Class')

ax[0,3].set_title('Survival Rate by SibSp')

ax[1,0].set_title('Survival Rate by Class')

ax[1,1].set_title('Survival Rate by Gender')

ax[1,2].set_title('Survival Rate by Age')

ax[1,3].set_title('Survival Rate by Parch')

ax[2,0].set_title('Fare Distribution')

ax[2,1].set_title('Survival Rate by Fare and Pclass')

ax[2,2].set_title('Total Passengers by Embarked')

ax[2,3].set_title('Survival Rate by Embarked')
train['Cabin'].value_counts(dropna=True)

## Did anyone with NA Cabin survive?

train[train.Survived == 1]

## yes, some NA Cabin survived, so they should be counted
train['Cabin'].value_counts().head()
g = sns.FacetGrid(col='Embarked',data=train)

g.map(sns.pointplot,'Pclass','Survived','Sex',palette='viridis',hue_order=['male','female'])

g.add_legend()
f,ax = plt.subplots(1,2,figsize=(15,3))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
train.isnull().any()

## Another way to find out which columns contain NaN values
train.isnull().sum()

## Another way to find out which columns contain NaN values and how many
f,ax = plt.subplots(1,2,figsize=(15,6))

sns.boxplot(x='Pclass',y='Age',data=train,ax=ax[0])

sns.boxplot(x='Pclass',y='Age',data=test,ax=ax[1])
# Find test dataset average age according to Pclass



cols=['Pclass','Age']

test[cols]

class1 = test[cols].loc[test['Pclass'] == 1] #picks rows of dataset with Pclass = 1

class2 = test[cols].loc[test['Pclass'] == 2]

class3 = test[cols].loc[test['Pclass'] == 3]



round_vect = np.vectorize(lambda x: round(x))

test_means = [class1['Age'].mean(), class2['Age'].mean(), class3['Age'].mean()]

test_means = round_vect(test_means)

# a=round(class1['Age'].mean())

# b=round(class2['Age'].mean())

# c=round(class3['Age'].mean())

# print(a,b,c)

print(test_means)



# Note: Slightly varies from original found age info
# Find train dataset average age according to Pclass



cols=['Pclass','Age']

train[cols]

class1 = train[cols].loc[train['Pclass'] == 1] #picks rows of dataset with Pclass = 1

class2 = train[cols].loc[train['Pclass'] == 2]

class3 = train[cols].loc[train['Pclass'] == 3]



train_means = [class1['Age'].mean(), class2['Age'].mean(), class3['Age'].mean()]

train_means = round_vect(train_means)

# a=round(class1['Age'].mean())

# b=round(class2['Age'].mean())

# c=round(class3['Age'].mean())

# print(a,b,c)

print(train_means[0])



# Note: Would it be more or less accurate to round up or down?
def fill_age_train(cols):

    Age = cols[0]

    PClass = cols[1]

    

    if pd.isnull(Age):

        if PClass == 1:

            return train_means[0]

        elif PClass == 2:

            return train_means[1]

        else:

            return train_means[2]

    else:

        return Age



def fill_age_test(cols):

    Age = cols[0]

    PClass = cols[1]

    

    if pd.isnull(Age):

        if PClass == 1:

            return test_means[0]

        elif PClass == 2:

            return test_means[1]

        else:

            return test_means[2]

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(fill_age_train,axis=1)

test['Age'] = test[['Age','Pclass']].apply(fill_age_test,axis=1)
test['Fare'].fillna(stat.mode(test['Fare']),inplace=True)

train['Embarked'].fillna('S',inplace=True)

train['Cabin'].fillna('No Cabin',inplace=True)

test['Cabin'].fillna('No Cabin',inplace=True)
f,ax = plt.subplots(1,2,figsize=(15,3))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
train.drop('Ticket',axis=1,inplace=True)

test.drop('Ticket',axis=1,inplace=True)
train.head()
#combine dataset 1st for easier Feature Engineering

train['IsTrain'] = 1

test['IsTrain'] = 0

df = pd.concat([train,test])

df.head()
#Scaler Initiation

scaler = MinMaxScaler()

print(scaler)
df['Title'] = df['Name'].str.split(', ').str[1].str.split('.').str[0]

df['Title'].value_counts()
df['Title'].replace('Mme','Mrs',inplace=True)

df['Title'].replace(['Ms','Mlle'],'Miss',inplace=True)

df['Title'].replace(['Dr','Rev','Col','Major','Dona','Don','Sir','Lady','Jonkheer','Capt','the Countess'],'Others',inplace=True)

df['Title'].value_counts()
df.drop('Name',axis=1,inplace=True)

df.head()
sns.distplot(df['Age'],bins=5)
df['AgeGroup'] = df['Age']

df.loc[df['AgeGroup']<=12, 'AgeGroup'] = 0

df.loc[(df['AgeGroup']>12) & (df['AgeGroup']<=18), 'AgeGroup'] = 1

df.loc[(df['AgeGroup']>18) & (df['AgeGroup']<=30), 'AgeGroup'] = 2

df.loc[(df['AgeGroup']>30) & (df['AgeGroup']<=40), 'AgeGroup'] = 3

df.loc[df['AgeGroup']>40, 'AgeGroup'] = 4

sns.countplot(x='AgeGroup',hue='Survived',data=df[df['IsTrain']==1],palette='husl')
df.drop('Age',axis=1,inplace=True)

df.head()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 #himself

df['IsAlone'] = 0

df.loc[df['FamilySize']==1, 'IsAlone'] = 1
#checking correlation with survival rate

f,ax = plt.subplots(1,2,figsize=(15,6))

sns.countplot(df[df['IsTrain']==1]['FamilySize'],hue=train['Survived'],ax=ax[0],palette='husl')

sns.countplot(df[df['IsTrain']==1]['IsAlone'],hue=train['Survived'],ax=ax[1],palette='husl')
df.drop(['SibSp','Parch','FamilySize'],axis=1,inplace=True)

df.head()
df.head()
df['Deck'] = df['Cabin']

df.loc[df['Deck']!='No Cabin','Deck'] = df[df['Cabin']!='No Cabin']['Cabin'].str.split().apply(lambda x: np.sort(x)).str[0].str[0]

df.loc[df['Deck']=='No Cabin','Deck'] = 'N/A'
sns.countplot(x='Deck',hue='Survived',data=df[df['IsTrain']==1],palette='husl')
df.loc[df['Deck']=='N/A', 'Deck'] = 0

df.loc[df['Deck']=='G', 'Deck'] = 1

df.loc[df['Deck']=='F', 'Deck'] = 2

df.loc[df['Deck']=='E', 'Deck'] = 3

df.loc[df['Deck']=='D', 'Deck'] = 4

df.loc[df['Deck']=='C', 'Deck'] = 5

df.loc[df['Deck']=='B', 'Deck'] = 6

df.loc[df['Deck']=='A', 'Deck'] = 7

df.loc[df['Deck']=='T', 'Deck'] = 0
df.drop('Cabin',axis=1,inplace=True)

df.head()
df[['Fare','Pclass','Deck']] = scaler.fit_transform(df[['Fare','Pclass','Deck']])
df.head()
def process_dummies(df,cols):

    for col in cols:

        dummies = pd.get_dummies(df[col],prefix=col,drop_first=True)

        df = pd.concat([df.drop(col,axis=1),dummies],axis=1)

    return df
df = process_dummies(df,['Embarked','Sex','Title','AgeGroup'])
df.head()
dataset = df[df['IsTrain']==1]

dataset.drop(['IsTrain','PassengerId'],axis=1,inplace=True)

holdout = df[df['IsTrain']==0]

test_id = holdout['PassengerId']

holdout.drop(['IsTrain','PassengerId','Survived'],axis=1,inplace=True)
class_one_total = int(np.sum(dataset['Survived']))

class_zero_counter = 0

indices_to_remove = []



for i in range(dataset.shape[0]):

    if(dataset['Survived'].iloc[i] == 0):

        class_zero_counter += 1

        if(class_zero_counter > class_one_total):

            indices_to_remove.append(i)



#dataset.drop(dataset.index[indices_to_remove],inplace=True)
int(np.sum(dataset['Survived'])), dataset.shape[0]
df.to_csv('titanic_dataset_preprocessed.csv',index=False)
X = dataset.drop(['Survived'],axis=1)

y = dataset['Survived'].astype('int')

# print(y)

#X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
model = RandomForestClassifier()

kf = KFold(n_splits=10,shuffle=True,random_state=101)

score = 0

train_indices, validation_indices = [],[]

for curr_train_indices, curr_validation_indices in kf.split(X):

    result = model.fit(X.iloc[curr_train_indices], y.iloc[curr_train_indices])

    curr_score = result.score(X.iloc[curr_validation_indices],y.iloc[curr_validation_indices])

    print(curr_score)

    

    if(curr_score > score):

        score = curr_score

        train_indices = curr_train_indices

        validation_indices = curr_validation_indices

print('Best Score: ',score)    
# for i,j in kf.split(X):

#  print(i, j)
param_grid = [

  {'n_estimators':[1,10,20,50,100,1000,3000], 'min_samples_leaf':[1,2,3,4,5], 'max_features':[3,5,7,9,10,'auto']},

 ]

grid = GridSearchCV(model,param_grid,n_jobs=4)

grid.fit(X, y)
grid.cv_results_
grid.best_params_, grid.best_score_
predictions = grid.predict(holdout)

len(predictions)
submission = pd.DataFrame({

    'PassengerId': test_id,

    'Survived': predictions

})

submission.to_csv('submission.csv',index=False)