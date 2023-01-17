import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random

import seaborn as sns
dataset_train = pd.read_csv('../input/titanic/train.csv')

dataset_test = pd.read_csv('../input/titanic/test.csv')
train = dataset_train.copy()

test = dataset_test.copy()
combine = [train,test]
train.head()
test.head()
train.describe()
train.info()
col_list, lst_str = [], ['object','int64','float64']# different 

i=0

for string in lst_str:

    lst = train.select_dtypes(include=[string]).columns

    print(lst_str[i],'\n',lst)

    col_list.append(lst)

    i+=1
train.pop('Ticket') # this is not necessary because we have PClass and Fare 

train.pop('Cabin') # similar reasoning to ticket

test.pop('Ticket')

test.pop('Cabin')

print('Unecessary Columns Deleted')
for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# show the titles and distribution of titles and gender

pd.crosstab(combine[0]['Title'], combine[0]['Sex'])
neon = ['#df0772','#fe546f','#ff9e7d','#ffd080','#01cbcf','#0188a5','#3e3264']

neon_text = ['#352a55','#01cbcf','#fffdff','#dfc907']
train['Sex']
plt.rcParams['text.color'] = neon_text[2]

plt.rcParams['xtick.color'] = neon_text[2]

plt.rcParams['ytick.color'] = neon_text[2]

plt.rcParams['axes.edgecolor'] = neon_text[2]

plt.rcParams['axes.facecolor'] = neon_text[0]

width = 0.3
perished = train[train['Survived']==0].groupby('Sex')['Survived'].count()

survived = train[train['Survived']==1].groupby('Sex')['Survived'].count()



fig = plt.figure(figsize=(12,4), dpi=80, facecolor=neon_text[0],edgecolor=neon_text[2])



ax1 = fig.add_subplot(121)

plt.title('Perished')

perished.plot(kind='bar', ax=ax1, legend=False,color=[neon[1],neon[5]])

ax1.xaxis.set_label_text('')

plt.xticks(rotation=0)



ax2 = fig.add_subplot(122)

plt.title('Survived')

survived.plot(kind='bar', ax=ax2, legend=False,color=[neon[1],neon[5]])

ax2.xaxis.set_label_text('')

plt.xticks(rotation=0)
perished_a = train[train['Survived']==0][['Age','Sex','Survived']].groupby('Sex')['Age']

survived_a = train[train['Survived']==1][['Age','Sex','Survived']].groupby('Sex')['Age']



fig = plt.figure(figsize=(12,4), dpi=80, facecolor=neon_text[0],edgecolor=neon_text[2])



ax1 = fig.add_subplot(121)

plt.title('Perished')

ax1.set_ylim((0, 60))

perished_a.plot.hist(ax=ax1,color=neon[1],alpha=0.5,bins=20)

ax1.yaxis.set_label_text('')



ax2 = fig.add_subplot(122)

plt.title('Survived')

ax2.set_ylim((0, 60))

survived_a.plot.hist(ax=ax2,color=neon[5],alpha=0.5,bins=20)

ax2.yaxis.set_label_text('')
perished_fsp = train[(train['Sex']=='female')&(train['Survived']==0)].groupby('Parch')['Parch'].count()

survived_fsp = train[(train['Sex']=='female')&(train['Survived']==1)].groupby('Parch')['Parch'].count()

perished_msp = train[(train['Sex']=='male')&(train['Survived']==0)].groupby('Parch')['Parch'].count()

survived_msp = train[(train['Sex']=='male')&(train['Survived']==1)].groupby('Parch')['Parch'].count()



fig = plt.figure(figsize=(12,6), dpi=80, facecolor=neon_text[0],edgecolor=neon_text[2])

ax1 = fig.add_subplot(221)

plt.title('Perished Females by # of Parch')

perished_fsp.plot(kind='bar', ax=ax1, legend=False,color=neon[1])

ax1.xaxis.set_label_text('')

plt.xticks(rotation=0)



ax2 = fig.add_subplot(222)

plt.title('Survived Females by # of Parch')

survived_fsp.plot(kind='bar', ax=ax2, legend=False,color=neon[1])

ax2.xaxis.set_label_text('')

plt.xticks(rotation=0)



ax3 = fig.add_subplot(223)

plt.title('Perished Males by # of Parch')

perished_msp.plot(kind='bar', ax=ax3, legend=False,color=neon[5])

ax3.xaxis.set_label_text('')

plt.xticks(rotation=0)



ax4 = fig.add_subplot(224)

plt.title('Survived Males by # of Parch')

survived_msp.plot(kind='bar', ax=ax4, legend=False,color=neon[5])

ax4.xaxis.set_label_text('')

plt.xticks(rotation=0)
for dataset in combine:

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Sir' if (x == 'Sir') | (x == 'Col') | (x == 'Jonkheer') | (x == 'Capt') | (x == 'Don') | (x == 'Major') else x)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Miss' if (x == 'Miss') | (x == 'Ms') | (x == 'Mlle') else x)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Mrs' if (x == 'Mrs') | (x == 'Mme') else x)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Lady' if (x == 'Lady') | (x == 'Countess') | (x == 'Dona') else x)

combine[0]['Title'].unique()
title_survival = pd.DataFrame()

for title in combine[0]['Title'].unique():

    count = combine[0][combine[0]['Title'] == title]['Survived'].sum()

    survived = round( count / combine[0][combine[0]['Title'] == title]['Survived'].shape[0],2)

    perished = 1 - survived

    title_survival.at[0,title] = survived

    title_survival.at[1,title] = perished
title_survival
fig = plt.figure(figsize=(12,4), dpi=80, facecolor=neon_text[0],edgecolor=neon_text[2])



ax1 = fig.add_subplot(121)

plt.title('Perished')

title_survival.loc[1].plot(kind='bar', ax=ax1, legend=False,color=neon[1])

ax1.xaxis.set_label_text('')

plt.xticks(rotation=0)



ax2 = fig.add_subplot(122)

plt.title('Survived')

title_survival.loc[0].plot(kind='bar', ax=ax2, legend=False,color=neon[5])

ax2.xaxis.set_label_text('')

plt.xticks(rotation=0)
embarked = train[['Embarked','Survived']]

perished_e = embarked[embarked['Survived']==0].groupby('Embarked').count()

survived_e = embarked[embarked['Survived']==1].groupby('Embarked').count()

for p,s in zip(perished_e['Survived'],survived_e['Survived']):

    p_ = round(p / (p + s)*100,2)

    perished_e[perished_e['Survived']==p] = p_ 

    s_ = round(s / (s + p)*100,2)

    survived_e[survived_e['Survived']==s] = s_

    

fig = plt.figure(figsize=(12,4), dpi=80, facecolor=neon_text[0],edgecolor=neon_text[2])



ax1 = fig.add_subplot(121)

plt.title('% Perished')

perished_e.plot(kind='bar', ax=ax1, legend=False,color=neon[1])

ax1.yaxis.set_label_text('')

ax1.set_ylim((0, 100))

plt.xticks(rotation=0)



ax2 = fig.add_subplot(122)

plt.title('% Survived')

survived_e.plot(kind='bar', ax=ax2, legend=False,color=neon[5])

ax2.yaxis.set_label_text('')

ax2.set_ylim((0, 100))

plt.xticks(rotation=0)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(0)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: 1 if x == 'S' else 2 if x == 'Q' else 3)
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].apply(lambda x: 1 if x == 'female' else 0)
for dataset in combine:

    dataset['FareGroups'] = pd.qcut(dataset['Fare'],5,labels=False) # sort into Quantile buckets for encoding

    dataset['Fare'] = dataset['FareGroups']

    dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)

combine[0]['Fare'].unique()
for dataset in combine:

    # 0% survived of the following

    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x == 'Rev' else x)

    # 16 % survived of the following

    dataset['Title'] = dataset['Title'].apply(lambda x: 1 if x == 'Mr' else x)

    # 38% survived of the following

    dataset['Title'] = dataset['Title'].apply(lambda x: 2 if x == 'Sir' else x)

    # 43% survived of the following

    dataset['Title'] = dataset['Title'].apply(lambda x: 3 if x == 'Dr' else x)

    # 57 % of the following survived

    dataset['Title'] = dataset['Title'].apply(lambda x: 4 if x == 'Master' else x)

    # 70% of the following survived

    dataset['Title'] = dataset['Title'].apply(lambda x: 5 if x == 'Miss' else x)

    # 79% survived of the following

    dataset['Title'] = dataset['Title'].apply(lambda x: 6 if x == 'Mrs' else x)

    # 100% survived of the following

    dataset['Title'] = dataset['Title'].apply(lambda x: 7 if x == 'Lady' else x)

combine[0]['Title'].unique()

for dataset in combine:

    age_guess = []

    for sex,p,t in zip(dataset['Sex'],dataset['Pclass'],dataset['Title']):

        #print(sex,p)

        age_group = dataset[(dataset['Sex']==sex) & (dataset['Pclass']==p) & (dataset['Title']==t)]['Age'].dropna()

        median_age = int(age_group.median())

        age_guess.append(median_age)

    dataset['AgeGuess'] = age_guess

    dataset['Age'] = dataset['Age'].fillna(dataset['AgeGuess'])
for dataset in combine:

    dataset['AgeBand'] = pd.qcut(dataset['Age'],5,labels=False) # sort into Quantile buckets for encoding
for dataset in combine:

    dataset['Alone'] = dataset['SibSp'] + dataset['Parch']

    dataset['Alone'] = dataset['Alone'].apply(lambda x: 1 if x > 0 else 0)
for dataset in combine:

    dataset['Age'] = dataset['AgeBand'].astype(int)

    dataset.pop('AgeGuess')

    dataset.pop('AgeBand')

    dataset.pop('FareGroups')

    #dataset.pop('PassengerId')

    dataset.pop('Name')

    dataset['AgeClass'] = dataset['Pclass'] + dataset['Age']

combine[0].head()
combine[0].describe()
combine[0].corr()
#importing libraries from sklearn

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn import metrics



# import algorithm modules

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support as score
train.info()
test.info()
X_train = train.copy()

X_train.pop('PassengerId')

y_train = X_train.pop('Survived')

X_test = test.copy()
X_train.head()
# PassengerId is not required for testing

X_test.pop('PassengerId') # popping this column is a simpler method than dropping it

# you could save it into a variable if you wanted to --- passenger_id = X_test.pop('PassengerId')

X_test.head()
results_array = [] # array to store all the scores for the many models that will be created

import time

def train_rf(n_est, depth,m_leaf):

    rf = RandomForestClassifier(n_estimators=n_est,

                                max_depth=depth,

                                min_samples_leaf = m_leaf,

                                n_jobs=-1)

    

    start = time.time()

    rf_model = rf.fit(X_train,y_train)

    end = time.time()

    fit_time = round((end - start),3)

    start = time.time()

    y_pred = rf_model.predict(X_test)

    end = time.time()

    pred_time = round((end - start),3)

    rf_model.score(X_train, y_train)

    rfs = round(rf_model.score(X_train, y_train) * 100, 2)

    #precision, recall, fscore, support = score(y_test, y_pred,average='macro') # pass the y labels to the score, the predicted, positive label; what we are predicting

    results_array.append([rfs,n_est,m_leaf,depth,fit_time,pred_time])
for n_est in [5,10,15,20,25,30,35,40,50,60,70,80]:

    for depth in [5,10,15,20,25,30,35,50,75,100,None]:

        for m_leaf in [0.25,0.5,1,2]:

            train_rf(n_est,depth,m_leaf)
final = pd.DataFrame(results_array,columns=['score','estimators','min samples','depth','fit time','pred time'])
# sorting by score, using ascending False to place the top scores at the top of the DataFrame

final = final.sort_values(by=['score'],ascending=False)

final.head()
# let's now grab the top 10 results, sort those by the time it takes for the model to make the prediciton

# reset the index and take the top result for our model

final = final.head(10)

final = final.sort_values(by=['pred time'])

final = final.reset_index()

final.pop('index')

final
# this section gets the parameters from the top performing model based on criteria above

estimators,leaf,depth = final.at[0,'estimators'],final.at[0,'min samples'],final.at[0,'depth']

if np.isnan(depth):

    depth = None

if leaf >= 1:

    leaf = int(leaf)

print('Final Model Parameters:\n Estimators:',estimators,'\n Samples:',leaf,'\nDepth:',depth)
# best score to date 0.76076 with the following

estimators,leaf,depth = 70,1,25
#Random Forest Regression

def forestRegression_1(x,y):

    model = RandomForestClassifier(n_estimators = estimators,

                                   min_samples_leaf = leaf,

                                   max_depth = depth,

                                   n_jobs=-1)

    model.fit(x, y)

    return model



rf = forestRegression_1(X_train, y_train)
print('Random Forest',round(rf.score(X_train, y_train)*100,2))
def predict_outcome(model):

    

    def probability_of_survival(num,model):

        columns = X_train.columns # get the columns from the training set so we drop what we don't need from the test set

        p = test[test['PassengerId'] == num][columns].values[0] # this way we can get PassengerId without using it

        e = model.predict_proba([p]).flatten()

        return e

    

    def predict_survival(num,model):

        columns = X_train.columns

        p = test[test['PassengerId'] == num][columns].values[0]

        e = model.predict([p]).flatten()

        return e[0]

    

    inference = []

    passengers = test['PassengerId']

    for i in passengers:

        prob = probability_of_survival(i,model)

        pred = predict_survival(i,model)

        inference.append([i,pred,prob])

    dz = pd.DataFrame(inference)

    return dz
r1 = predict_outcome(rf)
submission = pd.DataFrame()

submission['PassengerId'] = r1[0]

submission['Survived'] = r1[1]
submission.describe()
submission.info()
print('Test Set Prediction:\nPerished:',submission[submission['Survived']==0]['Survived'].count(),'\nSurvived:',submission[submission['Survived']==1]['Survived'].count())
submission.head()
submission.to_csv('gender_submission.csv', index=False)