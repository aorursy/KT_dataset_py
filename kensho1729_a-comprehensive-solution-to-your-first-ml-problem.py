import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #Visualization
import matplotlib.pyplot as plt #Visualization
%matplotlib inline 
# ^^ Will make your plot outputs appear and be stored within the notebook.

from itertools import chain #For ironing out lists - Can be avoided. 
                            #Using it as it'll be useful in Python for Data analyses in general

#Classifiers
from sklearn.ensemble import RandomForestClassifier
    
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Train set size:",train.shape,"|| Test set size:",test.shape)

train.info()
train.head(3)
train.describe()#Numerical Variables
train.describe(include=['O'])  #Categorical Variables
categ_vars = ['Survived','Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
fig, ax = plt.subplots(nrows = 2, ncols = 3 ,figsize=(20,10))
fig.subplots_adjust(wspace=0.4, hspace=0.4)
ax = list(chain.from_iterable(ax)) #Change ax from matrix to a list for iteration 
for i in range(len(categ_vars)):
    sns.countplot(train[categ_vars[i]], hue=train['Survived'], ax=ax[i])

train['Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
tab = pd.crosstab(train['Title'],train['Survived'])
print(tab)
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
tab_prop.plot(kind="bar", stacked=True)
train['Name_Len'] = train['Name'].apply(lambda x: len(x))
print(train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean(),
      pd.qcut(train['Name_Len'],5).value_counts())
train['Deck'] = train['Cabin'].apply(lambda x: str(x)[0])
sns.countplot(train['Deck'], hue=train['Survived'])
train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
sns.countplot(train['Ticket_Lett'], hue=train['Survived'])
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
sns.countplot(train['Ticket_Len'], hue=train['Survived'])
train['Age'].fillna(-1,inplace=True)
fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(20,8))
age = sns.distplot(train['Age'].dropna(), label='Total',bins=12,kde =False,ax=ax[0])
age = sns.distplot(train[train['Survived']==1].Age.dropna(), label='Survived',bins=12,kde =False,ax=ax[0])
age.legend()

fare = sns.distplot(train['Fare'], label='Total',bins=12,kde =False,ax=ax[1])
fare = sns.distplot(train[train['Survived']==1].Fare, label='Survived',bins=12,kde =False,ax=ax[1])
fare.legend()
train.loc[train.Fare.argmax()]
#sns.pairplot(train[train.columns[train.columns!='Survived'] ])
sns.pairplot(train.drop(['Survived','PassengerId'],axis=1))
sns.factorplot(x="Pclass", hue="Survived", col="Sex",data=train, kind="count");
sns.factorplot(x="Embarked", hue="Survived", col="Sex", data=train, kind="count");
sns.factorplot(x="Embarked", hue="Pclass", col="Survived", data=train, kind="count");
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

predictor = train["Survived"]

all_data = (train.drop(["Survived"] , axis=1)).append(test);
all_data.isnull().sum()
all_data['Fare'].fillna(all_data['Fare'].mean(), inplace = True)
all_data['Embarked'].fillna('S', inplace = True) #all_data['Embarked'].mode() doesn't work because of NAs
all_data['FamSz'] = all_data['SibSp'] + all_data['Parch']
all_data['FamSz'] = np.where(all_data['FamSz'] == 0 , 'Alone',
                           np.where(all_data['FamSz'] <= 3,'Midsize', 'Large'))
#Name
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
all_data['NameLen'] = all_data['Name'].apply(lambda x: len(x))

#Title
all_data['TicketLett'] = all_data['Ticket'].apply(lambda x: str(x)[0])
all_data['TicketLen'] = all_data['Ticket'].apply(lambda x: len(x))

#Cabin
all_data['Deck'] = all_data['Cabin'].apply(lambda x: str(x)[0])

#Class
all_data['Pclass'] = all_data['Pclass'].astype(str)

all_data.drop(['Cabin','Ticket','Name','SibSp','Parch','PassengerId'],axis=1,inplace=True)

all_data['AgeNull'] = all_data['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
data = all_data.groupby(['Title', 'Pclass'])['Age']
all_data['Age'] = data.transform(lambda x: x.fillna(x.mean()))
all_data['Age'].fillna(all_data['Age'].mean(),inplace=True)

print(all_data.isnull().sum()) #Sanity check that all values are filled
all_data['Title'] = np.where((all_data['Title']).isin(['Col.', 'Mlle.', 'Ms.','Major.','the','Lady.','Jonkheer.', 'Sir.',
                                    'Capt.','Don.','Dona.' ,'Mme.']), 'Rare',all_data['Title'])
all_data['TicketLett']= all_data['TicketLett'].apply(lambda x: str(x))
all_data['TicketLett'] = np.where((all_data['TicketLett']).isin(['W', '7', 'F','4', '6', 'L', '5', '8','9']), 'Rare',
                                            all_data['TicketLett'])

for_encoding = list(all_data.select_dtypes(include=["object"]))
remaining_cols = list(all_data.select_dtypes(exclude=["object"]))
numerical = all_data[remaining_cols]
encoded = pd.get_dummies(all_data[for_encoding])
all_data_new = pd.concat([numerical,encoded],axis=1)
print(len(all_data_new.columns))
train_new = all_data_new[0:len(train)]
test_new = all_data_new[len(train)::]
print(train_new.shape,test_new.shape)
print(all_data_new.columns)
print(train_new.dtypes)
rf1 = RandomForestClassifier(criterion= 'gini',
                             n_estimators=100,
                             min_samples_split=4,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf1.fit(train_new, predictor)
print("%.4f" % rf1.oob_score_)
importances = pd.DataFrame({'feature':train_new.columns,'importance':np.round(rf1.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
importances[0:15].plot.bar()
predictions = rf1.predict(test_new)
output = pd.DataFrame({ 'PassengerId': range(892,1310),
                            'Survived': predictions  })
output.to_csv('submission_madhu_rf.csv',index_label=False,index=False)
