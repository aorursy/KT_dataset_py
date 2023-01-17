import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()



plt.style.use('seaborn')



import warnings

warnings.filterwarnings("ignore", category = DeprecationWarning)

warnings.filterwarnings("ignore", category = FutureWarning)



from sklearn.utils.testing import ignore_warnings
train_raw = pd.read_csv('../input/titanic/train.csv')

test_raw = pd.read_csv('../input/titanic/test.csv')



passengerid = test_raw['PassengerId']
train_raw.sample(4)
test_raw.sample(4)
train_copy = train_raw.copy(deep = True)

test_copy = test_raw.copy(deep = True)
plt.figure(figsize = (4,4))

sns.countplot(x = 'Survived',data = train_copy)

plt.title("1 Feature")

plt.show()
male_df = train_copy.loc[train_copy['Sex']=='male']

female_df = train_copy.loc[train_copy['Sex']=='female']



fig, axes = plt.subplots(1,2, figsize = (9,4))

sns.countplot(x = 'Survived', data = male_df, ax = axes[0])

sns.countplot(x = 'Survived',data = female_df, ax = axes[1])

plt.show()
plt.figure(figsize = (6,4))

sns.countplot(x = 'Survived',data = train_copy, hue = 'Sex')

plt.title("Survival by Sex")

plt.show()
sns.catplot(x = 'Survived', kind = 'count',col = 'Sex', data = train_copy, height = 3.3, aspect = 1.2)

plt.title("Survival by Sex")

plt.show()
sns.catplot(x = 'Survived', kind = 'count',row = 'Sex',col = 'Embarked', data = train_copy,margin_titles = True, 

            height = 2.6, aspect = 1.2)

plt.show()
n_rows = 2

n_cols = 2



fig, axes = plt.subplots(n_rows, n_cols,figsize=(12,8))



sns.countplot(x = 'Embarked',hue = 'Survived',data = train_copy,ax = axes[0][0])

sns.countplot(x = 'Pclass',hue = 'Survived',data = train_copy,ax = axes[0][1])

sns.countplot(x = 'SibSp',hue = 'Survived',data = train_copy,ax = axes[1][0])

sns.countplot(x = 'Parch',hue = 'Survived',data = train_copy,ax = axes[1][1])



axes[1][0].legend(loc = 'upper right')

axes[1][1].legend(loc = 'upper right')



plt.show()
fig, axes = plt.subplots(figsize=(9,5))

sns.heatmap(train_copy.isnull(), cbar = False, cmap = 'magma',ax = axes)

plt.title("Missing values in train data")

plt.show()
fig, axes = plt.subplots(figsize=(9,5))

sns.heatmap(test_copy.isnull(), cbar = False, cmap = 'magma',ax = axes)

plt.title("Missing values in test data")

plt.show()
women_survived = train_copy.loc[train_copy['Sex']=='female']['Survived'].sum()

total_women = train_copy.loc[train_copy['Sex']=='female'].shape[0]



men_survived = train_copy.loc[train_copy['Sex']=='male']['Survived'].sum()

total_men = train_copy.loc[train_copy['Sex']=='male'].shape[0]



fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize = (14,6))



ax1.pie([women_survived,total_women-women_survived],

        labels = ['Survived','Dead'],

        autopct = '%1.1f%%',

        startangle = 20,

        explode = [0,0.2]

       )

ax1.set_title("Female")



ax2.pie([men_survived,total_men-men_survived],

        labels = ['Survived','Dead'],

        autopct = '%1.1f%%',

        startangle = 180,

        explode = [0,0.2]

       )

ax2.set_title("Male")



plt.show()
sns.catplot(x = 'Pclass',y='Survived', kind = 'point',data = train_copy, hue = 'Sex',

           height = 4,aspect = 2)

plt.title("Pointplot : Survival vs Pclass")

plt.show()
sns.catplot(x = 'Pclass', col = 'Survived', kind = 'count',data = train_copy)

plt.show()
sns.catplot(x = 'Embarked', y= 'Survived', kind = 'bar',data = train_copy)

plt.title("Survival vs Embarked")

plt.show()
sns.catplot(x = 'Embarked', col = 'Survived', kind = 'count', data = train_copy,

           height = 5, aspect = 1)

plt.show()
sns.catplot(x = 'Embarked',y = 'Survived',kind = 'point', hue = 'Sex',data = train_copy,height = 4, aspect = 2)

plt.title("Survived vs Embarked by Sex")

plt.show()
sns.catplot(x='Embarked', kind = 'count', col = 'Pclass', data=train_copy)

plt.show()
sns.catplot(x = 'Embarked', kind = 'point',y = 'Survived',col = 'Pclass', data = train_copy,margin_titles = True)

plt.show()
sns.catplot(x = 'Embarked',y = 'Survived',kind = 'point',col = 'Pclass', hue = 'Sex', data = train_copy)

plt.show()
train_copy.loc[(train_copy['Sex']=='female') & (train_copy['Embarked']=='S') & (train_copy['Pclass']==1)&(train_copy['Survived']==0)]
plt.figure(figsize = (16,9))

sns.pointplot(x = 'Fare',y = 'Survived',data = train_copy)

plt.title("Bear Grylls isn't happy with this chart for sure")

plt.xticks(rotation = 'vertical')

plt.show()
fig, axes = plt.subplots(figsize = (13,7))

sns.swarmplot(y = 'Age', x = 'Pclass', hue = 'Survived',dodge = True,data = train_copy, ax = axes)

plt.title("Swarmplot : Age vs Survival by Pclass")

plt.show()
sns.catplot(kind = 'swarm',x = 'Pclass', y = 'Age', col = 'Sex', hue = 'Survived',dodge = True,data = train_copy)

plt.show()
fig, axes = plt.subplots(figsize = (13,7))

sns.swarmplot(x = 'Pclass', y = 'Fare', hue = 'Survived',dodge = True,data = train_copy, ax = axes)

plt.show()
sns.catplot(kind = 'swarm',x = 'Pclass', y = 'Fare', col = 'Sex', hue = 'Survived',dodge = True,data = train_copy)

plt.show()
common_titles = ['Mr.','Miss.','Mrs.','Master.'] #From prev analysis



for df in [train_copy, test_copy]:

    

    df['FamilySize'] = df['Parch']+df['SibSp']+1

    

    df['IsAlone']=0

    df.loc[(df.FamilySize==1),'IsAlone'] = 1

    

    df['NameLen'] = df['Name'].apply(lambda x : len(x))

    

    df['Title'] = df['Name'].apply(lambda x : x.split(',')[1].strip().split()[0])

    df['Title'] = df['Title'].apply( lambda x : x if x in common_titles else 'Misc.')
plt.figure(figsize = (22,10))

sns.lineplot(x = 'NameLen', y = 'Survived',data = train_copy, marker = 'o')

plt.xticks(train_copy['NameLen'])

plt.title("Survival by NameLen")

plt.show()
train_copy.loc[train_copy['NameLen']==54]
sns.catplot(kind = 'bar', x = 'NameLen',y = 'Survived', data = train_copy, height = 7, aspect = 2)

plt.show()
train_copy['NameLenBin'],train_namelen_bins = pd.cut(train_copy['NameLen'].astype(int),20,retbins = True)
sns.countplot(data = train_copy, x = 'NameLenBin')

plt.xticks(rotation = 'vertical')

plt.title("Count of passengers by NameLen")

plt.show()
sns.barplot(data = train_copy, x = 'NameLenBin', y = 'Survived')

plt.xticks(rotation = 'vertical')

plt.title("Survival by NameLenBin")

plt.show()
fig, axes = plt.subplots(1,2, figsize=(11,5))



sns.countplot(x = 'Title', data = train_copy, ax = axes [0])

sns.barplot(x = 'Title', y = 'Survived', data = train_copy, ax = axes[1])

plt.show()
fig, axes = plt.subplots(1,2, figsize = (15,6))



sns.countplot(x = 'FamilySize', data = train_copy, ax = axes [0])

sns.barplot(x = 'FamilySize', y = 'Survived', data = train_copy, ax = axes[1])

plt.show()
fig, saxis = plt.subplots(1, 2,figsize=(12,5))



sns.pointplot(x = 'IsAlone', y = 'Survived',  kind = 'point', data = train_copy, ax = saxis[0])

sns.pointplot(x = 'IsAlone', y = 'Survived', hue = 'Sex', kind = 'point', data = train_copy, ax = saxis[1])

plt.show()
#Filling missing ages by title



titlewise_grouping = train_copy.groupby('Title')

titlewise_agemean = titlewise_grouping.mean()



titlewise_agemean['Age'].to_frame()
train_copy['Age'] = titlewise_grouping['Age'].apply(lambda x: x.fillna(x.mean()))
#Same for test now



titlewise_grouping = test_copy.groupby('Title')

titlewise_agemean = titlewise_grouping.mean()



test_copy['Age'] = titlewise_grouping['Age'].apply(lambda x: x.fillna(x.mean()))
#Filling Embarked and Fare and compiling them into bins



for dataset in [train_copy, test_copy]:

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
#NameLenBins already visualized above

train_copy['AgeBin'],train_age_bins = pd.cut(train_copy['Age'],8,retbins = True)

train_copy['FareBin'],train_fare_bins = pd.qcut(train_copy['Fare'],8,retbins = True)
test_age_bins = np.concatenate(([-np.inf], train_age_bins[1:-1], [np.inf]))

test_namelen_bins = np.concatenate(([-np.inf], train_namelen_bins[1:-1], [np.inf]))



test_copy['AgeBin'] = pd.cut(test_copy['Age'], test_age_bins)

test_copy['NameLenBin'] = pd.cut(test_copy['NameLen'], test_namelen_bins)



test_fare_bins = np.concatenate(([-np.inf], train_fare_bins[1:-1], [np.inf]))

test_copy['FareBin'] = pd.cut(test_copy['Fare'], test_fare_bins)
print("Done. Final Check")

print(train_copy.isnull().sum())

print("-"*25)

print(test_copy.isnull().sum())
train_copy.sample(5)
test_copy.sample(5)
for dataset in [train_copy,test_copy]:

    dataset = dataset.drop(['Cabin','PassengerId','Ticket','Name','FareBin','SibSp','Parch','NameLenBin','Age'],axis = 'columns',inplace = True)
train_copy = pd.get_dummies(train_copy,columns = ['Embarked'],drop_first = False)

test_copy = pd.get_dummies(test_copy, columns = ['Embarked'],drop_first = False)
for dataframe in [train_copy,test_copy]:

    

    dataframe['Sex'] = dataframe['Sex'].map( {'male':0, 'female': 1 } )

    dataframe['Title'] = dataframe['Title'].map( {'Mr.':1, 'Misc.': 0, 'Master.':2, 'Miss.': 3, 'Mrs.': 4 } )

    dataframe['Pclass'] = dataframe['Pclass'].map( {1:3,2:2,3:1} )
from sklearn.preprocessing import LabelEncoder



labeler = LabelEncoder()



train_copy['AgeBin'] = labeler.fit_transform(train_copy['AgeBin'])

test_copy['AgeBin'] = labeler.fit_transform(test_copy['AgeBin'])
train_copy
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



#TRAIN

scaler.fit(train_copy.drop(['Survived'],axis=1)) 

#As inplace = True has not been written the scaler will get train_copy without 'Survived', but train_copy is unaffected



scaled_features = scaler.transform(train_copy.drop(['Survived'],axis=1))

train_copy_scaled = pd.DataFrame(scaled_features)



#TEST

test_copy.fillna(test_copy.mean(), inplace=True)



scaled_features = scaler.transform(test_copy)

test_copy_scaled = pd.DataFrame(scaled_features)
train_copy_scaled.sample(5)
X_train_scaled = train_copy_scaled

y_train_scaled = train_copy['Survived']

X_test_scaled = test_copy_scaled
X_train = train_copy.drop('Survived',axis = 1)

y_train = train_copy['Survived']

X_test = test_copy.copy(deep = True)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV
svc = SVC(C=100, gamma=0.01)

scores_rfc = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy')

print("Mean score for 10 folds is ",scores_rfc.mean()*100)
svc = SVC(C=100, gamma=0.01)

scores_rfc = cross_val_score(svc, X_train_scaled, y_train_scaled, cv=10, scoring='accuracy')

print("Mean score for 10 folds is ",scores_rfc.mean()*100)
def get_best_score(model):

    

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_
param_grid = {'C': [0.1,10, 100, 1000,5000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



best_svc = GridSearchCV(SVC(), param_grid, cv=10, refit=True, verbose=0)

best_svc.fit(X_train_scaled,y_train_scaled)
pred_all_svc = best_svc.predict(X_test_scaled)



sub_svc = pd.DataFrame()



sub_svc['PassengerId'] = passengerid

sub_svc['Survived'] = pred_all_svc



sub_svc.to_csv('bear_grylls.csv',index=False)