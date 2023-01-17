import os

import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib as mpl

import matplotlib.pyplot as plt



from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder

from datetime import datetime
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
categorical = len(train.select_dtypes(include=['object']).columns)

numerical = len(train.select_dtypes(include=['int64','float64']).columns)

print('Total number of variables= ', categorical, 'Categorical', '+',

      numerical, 'Numerical', '=>', categorical+numerical, 'variables')

train.head(10)
train.describe()
sns.barplot(x=train['Sex'], y=train['Survived']) 





sns.catplot(x="Age", y="Sex", 

            hue="Survived", kind="violin", figsize=(25, 25),

            split=True, inner="stick",

            palette={0: "b", 1: "g"},

            bw=.15, cut=0,

            data=train)

#The graphs above contain some useful information. First, men are much more likely to die than women. Second, young men are more likely to survive. To be specific, chances of survival for men under the age of 15 are very high, but not for women. Third, men between the ages of 18 and 30 are more likely to die. Fourth, age in general does not appear to have a direct impact on women's survival. These results show that the principle of priority for women and children in war and disaster was applied to the Titanic.



cols = ['Embarked', 'Pclass', 'SibSp', 'Parch']

n_rows = 2

n_cols = 2



fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7,n_rows*7))



for row in range(0,n_rows):

    for column in range(0,n_cols):  

        i = row*n_cols+ column       

        ax = axs[row][column]

        sns.countplot(train[cols[i]], hue=train["Survived"], ax=ax)

        ax.set_title(cols[i])

        ax.legend(title="Survived", loc='upper right') 

        

plt.tight_layout() 
grid = sns.catplot(x="Fare", y="Survived", row="Pclass",

                kind="box", orient="h", height=1.5, aspect=4,

                data=train.query("Fare > 0"))
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
comb = train.append(test)

comb.shape



comb.isnull().sum()/comb.isnull().count()*100 
comb['IsFemale'] = np.where(comb['Sex']=='female', 1, 0 )

sns.barplot(x='IsFemale', y='Survived', data=comb) 
Titles = set()

for name in comb['Name']:

    Titles.add(name.split(',')[1].split('.')[0].strip())

Dict = {"Capt": "Special", "Col": "Special", "Major": "Special", "Jonkheer": "Special","Don": "Special","Dona": "Special",

    "Sir" : "Special", "Dr": "Special", "Rev": "Special", "the Countess":"Special", "Mme": "Mrs", "Mlle": "Miss",

    "Ms": "Mrs", "Mr" : "Mr", "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master", "Lady" : "Special"}

def Titles():

    comb['Title'] = comb['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    comb['Title'] = comb.Title.map(Dict)

    return comb

comb = Titles()



sns.barplot(x='Title', y='Survived', data=comb) 

comb['IsMr'] = np.where(comb['Title']=='Mr', 1,0)

sns.barplot(x='IsMr', y='Survived', data=comb) 
#Extract ticket type from Ticket

import string

TicketType = []

for i in range(len(comb.Ticket)):

    ticket = comb.Ticket.iloc[i]

    for c in string.punctuation:

                ticket = ticket.replace(c,"")

                splited_ticket = ticket.split(" ")   

    if len(splited_ticket) == 1:

                TicketType.append('NO')

    else: 

                TicketType.append(splited_ticket[0])

comb['TicketType'] = TicketType

comb['TicketType'] = np.where((comb.TicketType!='NO') & (comb.TicketType!='PC') & 

                                (comb.TicketType!='CA') & (comb.TicketType!='A5') & 

                                (comb.TicketType!='SOTONOQ'),'OT',comb.TicketType)





#Extract Crew from Ticket.

comb['Crew'] = np.where(comb.Fare==0, 1, 0)

comb.Crew.value_counts()



#sns.catplot(x='Crew', y='Survived', kind='bar', data=comb)

#Extract LastName from Name

comb['LastName'] = comb['Name'].str.extract('([A-Za-z]+),', expand=False)



#Create a new variable by cencatinatinig TicketType, Embarked, LastName 

comb['Group'] = comb['TicketType'].astype(str) + comb['Embarked'].astype(str) + comb['LastName'].astype(str) 

le = LabelEncoder()

comb['GroupID']  = le.fit_transform(comb['Group'])
comb['TicketFreq'] = comb.groupby('Ticket')['Ticket'].transform('count')
cols = ['IsMr', 'IsFemale', 'Pclass', 'TicketFreq']



n_rows = 2

n_cols = 2

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7,n_rows*7))



for row in range(0,n_rows):

    for column in range(0,n_cols):  

        i = row*n_cols+ column      

        ax = axs[row][column] 

        sns.countplot(comb[cols[i]], hue=comb["Survived"], ax=ax)

        ax.set_title(cols[i])

        ax.legend(title="Survived", loc='upper right') 

        

plt.tight_layout() 
# Drop unwanted features

comb.drop(['Ticket', 'Sex', 'Title', 'Name', 'Fare', 'Age', 'TicketType', 'LastName', 'Parch', 'SibSp', 'Crew', 

           'Cabin', 'Embarked', 'Group'], axis=1, inplace=True) 

comb.head(5)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, cross_val_predict

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix



#Before we run the models we need to split the combined dataset into training and testing datasets.

n=len(train)

train_df = comb.iloc[:n] 

test_df = comb.iloc[n:]



y_train = train_df['Survived'].astype(int)

X_train = train_df.drop(['Survived', 'PassengerId'], axis=1) 

X_test = test_df.drop(['Survived', 'PassengerId'], axis=1)


def stats(models, X_train, y_train):

    stats = {}

    for name, inst in models.items():

        mscores = []

        model_pipe = make_pipeline(StandardScaler(), inst)

        model_pipe.fit(X_train, y_train)

        acc=round(model_pipe.score(X_train, y_train)* 100, 2)

        mscores.append(acc)

        scores = cross_val_score(model_pipe, X_train, y_train, cv=10, scoring='accuracy')

        acccv=round(scores.mean()* 100, 2)

        mscores.append(acccv)

        mscores.append(scores.std())

        y_train_cv_pred = cross_val_predict(model_pipe, X_train, y_train, cv=10)

        p = round(precision_score(y_train, y_train_cv_pred)* 100, 2)

        s = round(recall_score(y_train, y_train_cv_pred)* 100, 2)

        mscores.append(p)

        mscores.append(s)

        f1 = 2*((p*s)/(p+s))

        mscores.append(f1)

        roc_auc_cvs = round(cross_val_score(model_pipe, X_train, y_train, cv=10, scoring='roc_auc').mean()* 100, 2)

        mscores.append(roc_auc_cvs)

        stats[name] = mscores

    colnames = ['Accuracy','AccuracyCv','StandardDeviation', 'Precision','Sensitivity','F1Score', 'RocAucCv']

    df_ms = pd.DataFrame.from_dict(stats, orient='index', columns=colnames)

    df_msr = df_ms.sort_values(by='RocAucCv', ascending=False)

    return df_msr



models = {'AdaBoostClassifier': AdaBoostClassifier(),

          'GradientBoostingClassifier': GradientBoostingClassifier(),

          'LogisticRegression': LogisticRegression(),

          'SupportVectorMachines': SVC(random_state=0),

          'RandomForestClassifier': RandomForestClassifier(),

          'KNeighborsClassifier': KNeighborsClassifier(),

          'DecisionTreeClassifier': DecisionTreeClassifier()}

df_all_models = stats(models, X_train,y_train)



df_all_models
vote = VotingClassifier(estimators=[

    ('AdaBoostClassifier', AdaBoostClassifier()),

    ('GradientBoostingClassifier', GradientBoostingClassifier()),

    ('LogisticRegression', LogisticRegression()),

    ('SupportVectorMachines', SVC()),

    ('RandomForest', RandomForestClassifier()),

    ('KNN', KNeighborsClassifier())

],

          voting='hard', n_jobs=15)

vote.fit(X_train,y_train)

prediction_vote=vote.predict(X_test)

print ('Voting Classifier Accuracy Score = {:.2f}%'.format(round(vote.score(X_train,y_train, sample_weight=None)* 100, 2)), 'on', datetime.now())
output= pd.DataFrame (pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": prediction_vote}))

output.head()

output.to_csv('FinalSubmission.csv', index=False)