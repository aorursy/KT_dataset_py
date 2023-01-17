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
#some more imports

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV



import seaborn as sns

import matplotlib.pyplot as plt

import re



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
def print_na(train, test, ftr):

    train_na = len(train[pd.isna(train[ftr])])

    print('Amount of NaN values in train: {0} ({1:.1f}%)'.format(train_na,

                                                             train_na*100.0/len(train)))

    test_na = len(test[pd.isna(test[ftr])])

    print('Amount of NaN values in test: {0} ({1:.1f}%)'.format(test_na,

                                                           test_na*100.0/len(test)))

print_na(train, test, 'Sex')
#sex

train.Sex.value_counts()
def barplot_survived(train, ftr):

    gr = train.groupby([ftr,'Survived']).PassengerId.count().reset_index()

    gr_total = train.groupby(ftr).PassengerId.count().reset_index()

    sns.barplot(data = gr_total, x = ftr, y = 'PassengerId',alpha = .25)

    sns.barplot(data = gr,x = ftr,y = 'PassengerId', hue = 'Survived')

barplot_survived(train,'Sex')
#transforming Sex into digital values

train['Sex'] = train.Sex.apply(lambda x: 1 if x=='male' else 0)

test['Sex'] = test.Sex.apply(lambda x: 1 if x=='male' else 0)
print_na(train, test, 'Cabin')
train.Cabin.dropna().head(10)
#Cabin type fill

def cabin_type(x):

    try:

        return x[0]

    except:

        return 'unknown'

train['Cabin'] = train.Cabin.apply(cabin_type)

test['Cabin'] = test.Cabin.apply(cabin_type)

#na fill

train['Cabin'] = train.Cabin.fillna('unknown')

test['Cabin'] = train.Cabin.fillna('unknown')
barplot_survived(train,'Cabin')
print_na(train,test,'Ticket')
train.Ticket.dropna().head(10)
def ticket_type(x):

    try:

        ticket = int(x)

        return 'unknown'

    except:

        ticket = x.split(' ')[0].strip().lower()

        ticket_type = re.sub(r'[\/\d\.\,]','',ticket)

        if ticket_type in ['pc','ca','a']:

            return ticket_type

        else:

            return 'unknown'

train['TicketType'] = train.Ticket.apply(ticket_type)

test['TicketType'] = test.Ticket.apply(ticket_type)
barplot_survived(train,'TicketType')
print_na(train,test,'Embarked')
train['Embarked'] = train.Embarked.fillna(train.Embarked.mode()[0])
barplot_survived(train,'Embarked')
print_na(train,test,'Pclass')
barplot_survived(train,'Pclass')
print_na(train,test,'Age')
def distplot_survived(train, ftr):

    f,ax = plt.subplots(figsize = (10,5))

    sns.distplot(train[train.Survived==1][ftr].dropna(),ax= ax, label = 'Survived')

    sns.distplot(train[train.Survived==0][ftr].dropna(),ax= ax, label = 'Died')

    ax.legend()

distplot_survived(train,'Age')
train[train.Survived==1].Age.mode()[0]
print("Mode for survived: {0}, Average for survived: {1:.1f}".format(train[train.Survived==1].Age.mode()[0],

                                                                 train[train.Survived==1].Age.mean()))

print("Mode for died: {0}, Average for died: {1:.1f}".format(train[train.Survived==0].Age.mode()[0],

                                                    train[train.Survived==0].Age.mean()))

print("Mode for all train: {0}, Average for all train: {1:.1f}".format(train.Age.mode()[0],

                                                    train.Age.mean()))

train['Age'] = train.Age.fillna(train.Age.mode()[0])

test['Age'] = test.Age.fillna(test.Age.mode()[0])
print_na(train,test,'Fare')
test['Fare'] = test.Fare.fillna(test.Fare.mode()[0])
distplot_survived(train,'Fare')
print('Mean fare: {0:.1f}, mode fare: {1}'.format(train.Fare.mean(),

                                             train.Fare.mode()[0]))
print_na(train,test,'SibSp')
barplot_survived(train,'SibSp')
print_na(train,test,'Parch')
barplot_survived(train,'Parch')
f, axs = plt.subplots(2,3,figsize = (16,9))

index = 0

for ftr in ['Sex','Age','Fare','Pclass','Parch','SibSp']:    

    ax = axs[index//3,index%3]

    sns.distplot(train[train.Survived==1][ftr].dropna(),ax= ax, label = 'Survived')

    sns.distplot(train[train.Survived==0][ftr].dropna(),ax= ax, label = 'Died')

    ax.legend()

    index +=1
#last name selection

train['LastName'] = train.Name.apply(lambda x: x.split(',')[0].lower().strip())

test['LastName'] = test.Name.apply(lambda x: x.split(',')[0].lower().strip())
#status selection

train['Status'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0].lower().strip())

test['Status'] = test.Name.apply(lambda x: x.split(',')[1].split('.')[0].lower().strip())
def married(df):

    try:

        c = lastname[df['LastName']]['mrs']

    except:

        c = 0

    if df['Sex']==0  and df['Status']=='mrs' and df['SibSp']>=1:

        return 1

    elif df['Sex']==1 and df['SibSp']>=1 and c>=1:

        return 1

    else:

        return 0

def status_update(df):

    if df['Status'] in ['mr','miss','mrs','master']:

        return df['Status']

    elif df['Sex']==1:

        return 'mr'

    else:

        return 'miss'

train['Status'] = train.apply(status_update, axis = 1)

test['Status'] = test.apply(status_update, axis = 1)
lastname = train.groupby(['LastName','Status'])['PassengerId'].count()

train['Married'] = train.apply(married, axis = 1)

lastname = test.groupby(['LastName','Status'])['PassengerId'].count()

test['Married'] = test.apply(married, axis = 1)
train['NameLength'] = train.Name.apply(lambda x: len(x))

test['NameLength'] = test.Name.apply(lambda x:len(x))
features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked', 'TicketType', 'Married','Status','NameLength']

y = train.Survived

X_train_ftr = pd.get_dummies(train[features])

X_test_ftr = pd.get_dummies(test[features])

#X_train_ftr.index = train.PassengerId

#X_test_ftr.index = test.PassengerId

train.Married.value_counts()
corr = pd.concat((X_train_ftr,y),axis = 1).corr()



mask = np.zeros_like(np.abs(corr), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))





# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(np.abs(corr), mask=mask,  vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(pd.concat((X_train_ftr[['Sex','Fare','Age','NameLength']],y),axis = 1), hue = 'Survived')
ftr_list = ['Pclass','Sex','Fare','Married','Cabin_unknown']
from sklearn.model_selection import cross_val_score,KFold



def scoring(clf, X, y):

    kf = KFold(5, shuffle=True, random_state=19).get_n_splits(X)

    cvs = cross_val_score(clf, X, y, cv=kf,scoring='accuracy')

    return cvs
from sklearn.feature_selection import VarianceThreshold
all_data = pd.concat((X_train_ftr,X_test_ftr))

selected  = VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(all_data)

X_train_sel = selected[0:len(train)]

X_test_sel = selected[len(train):]
#X_train = RobustScaler(quantile_range=(10,90)).fit_transform(X_train_ftr)

#X_test = RobustScaler(quantile_range=(10,90)).fit_transform(X_test_ftr)
from sklearn import svm
'''kf = KFold(5, shuffle=True, random_state=19).get_n_splits(X_train_sel)

svc = GridSearchCV(svm.SVC(random_state = 42), cv = kf,

                   param_grid = {

                       'kernel' : ['linear'],

                       'C': [100],

                   })

'''
svc = svm.SVC(random_state = 42, kernel = 'linear')
#svc.fit(X_train_sel,y)
#svc.best_estimator_
scr = scoring(svc, X_train_sel, y)
print('Mean: {0:.4f}, std: {1:.4f}'.format(scr.mean(),scr.std()))
svc.fit(X_train_sel,y)

prediction = svc.predict(X_test_sel)

res = pd.DataFrame()

res['PassengerId'] = test.PassengerId

res['Survived'] = prediction

res['Survived'] = res.Survived.astype(int)

res.to_csv('submission.csv', index = False)