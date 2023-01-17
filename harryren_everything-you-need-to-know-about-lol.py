import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import math

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import Ridge,RidgeCV

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.metrics import accuracy_score

import scipy

from scipy import stats

import plotly

import seaborn as sns

from scipy.stats import pearsonr

from sklearn.preprocessing import Binarizer



Rawdata = pd.read_csv('../input/league-of-legends/games.csv')



# Any results you write to the current directory are saved as output.
df = pd.DataFrame(Rawdata)

df.sample(5)
data = df[['gameDuration', 'winner', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon'

         , 'firstRiftHerald', 't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills'

         , 't2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']]
def pltDuration(data):

    plt.figure(figsize = (15, 10))

    Duration_plot = plt.hist(data['gameDuration'], bins = 200)

    my_x_ticks = np.arange(0, 4200, 300)

    plt.xticks(my_x_ticks)

    plt.xlabel("gameDuration (s)", fontsize = 13)

    plt.ylabel('Frequency', fontsize = 13)

    plt.title('GameDuration Distribution', fontsize = 15)

    plt.show()

pltDuration(data)
data.info()
data = data[data['gameDuration'] >= 900]

data.info()

data.sample(5)
pltDuration(data)
n = data.count()

Team1_win = data[data['winner'] == 1].count()

Team2_win = data[data['winner'] == 2].count()

Team1_win_percent = Team1_win/n * 100

Team2_win_percent = Team2_win/n * 100

T1 = Team1_win_percent['winner']

T2 = Team2_win_percent['winner']

plt.pie((T1,T2), labels = ('Team1 Win', 'Team2 Win'), startangle = 90, autopct='%.2f%%')

plt.axis('equal')

plt.title('Winning Rate for 2 Teams')

plt.show()
data['winner'].replace(2,0,inplace=True)

data.describe()
true_mu = 0.5

t_test = scipy.stats.ttest_1samp(data['winner'], true_mu)

t_value = t_test[0]

p_value = t_test[1]

print("t value: " + str(t_value) + " ,p value: "+ str(p_value))
scipy.stats.t.interval(0.95, len(data['winner'])-1, loc=np.mean(data['winner']), scale=scipy.stats.sem(data['winner']))
p_firstBlood = data[(data['firstBlood'] == 1) & (data['winner'] == 1)].count()/data[data['firstBlood'] == 1].count()

p_firstTower = data[(data['firstTower'] == 1) & (data['winner'] == 1)].count()/data[data['firstTower'] == 1].count()

p_firstInhibitor = data[(data['firstInhibitor'] == 1) & (data['winner'] == 1)].count()/data[data['firstInhibitor'] == 1].count()

p_firstBaron = data[(data['firstBaron'] == 1) & (data['winner'] == 1)].count()/data[data['firstBaron'] == 1].count()

p_firstDragon = data[(data['firstDragon'] == 1) & (data['winner'] == 1)].count()/data[data['firstDragon'] == 1].count()

p_firstRiftHerald = data[(data['firstRiftHerald'] == 1) & (data['winner'] == 1)].count()/data[data['firstRiftHerald'] == 1].count()
labels = ('firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald')

probabilities = [p_firstBlood['gameDuration'], p_firstTower['gameDuration'], p_firstInhibitor['gameDuration'], p_firstBaron['gameDuration']

                ,p_firstDragon['gameDuration'] ,p_firstRiftHerald['gameDuration']]

probabilities = [i * 100 for i in probabilities]

y_pos = np.arange(len(labels))

plt.figure(figsize=(25,15))

plt.bar(y_pos, probabilities, align='center', alpha=1)

plt.xticks(y_pos, labels, fontsize = 30)

plt.yticks(fontsize = 30)

plt.ylabel('Probability(%)', fontsize = 30)

plt.title('Winning Probability when Team 1 got FirstXXX', fontsize = 40)

for a,b in zip(y_pos, probabilities):

    plt.text(a, b, '%.3f'%b+'%', ha='center', va= 'bottom',fontsize=30)

plt.show()
p_firstBlood1 = data[(data['firstBlood'] == 1) & (data['winner'] == 1)].count()/data[data['winner'] == 1].count()

p_firstTower1 = data[(data['firstTower'] == 1) & (data['winner'] == 1)].count()/data[data['winner'] == 1].count()

p_firstInhibitor1 = data[(data['firstInhibitor'] == 1) & (data['winner'] == 1)].count()/data[data['winner'] == 1].count()

p_firstBaron1 = data[(data['firstBaron'] == 1) & (data['winner'] == 1)].count()/data[data['winner'] == 1].count()

p_firstDragon1 = data[(data['firstDragon'] == 1) & (data['winner'] == 1)].count()/data[data['winner'] == 1].count()

p_firstRiftHerald1 = data[(data['firstRiftHerald'] == 1) & (data['winner'] == 1)].count()/data[data['winner'] == 1].count()
labels1 = ('firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald')

probabilities1 = [p_firstBlood1['gameDuration'], p_firstTower1['gameDuration'], p_firstInhibitor1['gameDuration'], p_firstBaron1['gameDuration']

                ,p_firstDragon1['gameDuration'] ,p_firstRiftHerald1['gameDuration']]

probabilities1 = [i * 100 for i in probabilities1]

y_pos1 = np.arange(len(labels1))

plt.figure(figsize=(25,15))

plt.bar(y_pos1, probabilities1, align='center', alpha=1)

plt.xticks(y_pos1, labels1, fontsize = 30)

plt.yticks(fontsize = 30)

plt.ylabel('Probability(%)', fontsize = 30)

plt.title('FirstXXX if Team 1 Win', fontsize = 40)

for a,b in zip(y_pos1, probabilities1):

    plt.text(a, b, '%.3f'%b+'%', ha='center', va= 'bottom',fontsize=30)

plt.show()




dummy_data = data.copy()

dummy_data = pd.concat([dummy_data,pd.get_dummies(data['firstBlood'],prefix = 'firstBlood'),

                       pd.get_dummies(data['firstTower'],prefix = 'firstTower'),

                       pd.get_dummies(data['firstInhibitor'],prefix = 'firstInhibitor'),

                       pd.get_dummies(data['firstBaron'],prefix = 'firstBaron'),

                       pd.get_dummies(data['firstDragon'],prefix = 'firstDragon'),

                       pd.get_dummies(data['firstRiftHerald'],prefix = 'firstRiftHerald')], axis = 1)

dummy_data.info()
X = dummy_data[['t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills',

                           't2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills',

                           'firstBlood_1','firstTower_1','firstTower_2','firstInhibitor_1','firstInhibitor_2',

                           'firstBaron_1','firstBaron_2','firstDragon_1','firstDragon_2','firstRiftHerald_1','firstRiftHerald_2',

                           'gameDuration']]

y = dummy_data['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=0)
lr_model= LogisticRegression(C = 1,

                              penalty = 'l2',solver = 'liblinear')



lr_model.fit(X_train,y_train)

pred_train = lr_model.predict(X_train)

print(classification_report(y_train, pred_train > 0.5))



pred_test= lr_model.predict(X_test)

print(classification_report(y_test, pred_test > 0.5))

print(lr_model.coef_)

print(lr_model.intercept_)

scores = cross_val_score(lr_model,X,y,cv=7)

print('The cross validation score is {0}'.format(scores.mean()))
p = []

t = []

X1 = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

      [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,900],

     [2,0,0,0,0,3,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1200],

     [5,0,0,1,0,4,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1500],

     [6,1,1,2,0,4,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1800],

     [9,2,1,3,0,5,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,2100]]

pre = lr_model.predict_proba(X1)

print(pre)

for i in pre:

    p.append(i[1])

for j in X1:

    t.append(j[-1])

fig = plt.figure(figsize = (10,5))

plt.plot(t,p)

plt.xlabel('Time(Seconds)')

plt.ylabel('Winning Probability')

plt.title('Team 1 Winning Prediction')

plt.show()
pred= lr_model.predict_proba(X_test)

result= pd.DataFrame(pred)

result.index= X_test.index

result.columns= ['0', 'winning_probability']

print(result.head(5))
y_score = lr_model.decision_function(X_test)

fpr,tpr,threshold = roc_curve(y_test, y_score)

roc_auc = auc(fpr,tpr)

print("AUC: " + str(roc_auc))
plt.figure()

lw = 2

plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
decile_data = pd.concat([result['winning_probability'],y_test],axis = 1)

decile_data['Decile_rank']=pd.qcut(decile_data['winning_probability'],10,labels=False)

print(decile_data.head(5))
decile = pd.DataFrame(columns = ['Decile/global_mean'])

global_mean = decile_data['winner'].mean()

for i in range(9,-1,-1):

    decile.loc[i] = decile_data[decile_data['Decile_rank'] == i].mean()['winning_probability']/global_mean

print(decile)
plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features',y=1.05,size=15)

sns.heatmap(X.astype(float).corr(),linewidths=0.1,vmax=1.0,

            square=True,linecolor='white',annot=True)

plt.xticks(rotation=90)

plt.yticks(rotation=360)

plt.show()
transformer = SelectKBest(chi2, k=10)

transformer.fit(X,y)

X_new = transformer.transform(X)
transformer.get_support(True)
X.columns
def multivariate_pearsonr(X,y):

    scores,pvalues = [],[]

    for column in range(X.shape[1]):

        cur_score,cur_p = pearsonr(X[:,column],y)

        scores.append(abs(cur_score))

        pvalues.append(cur_p)

    return (np.array(scores), np.array(pvalues))
transformer = SelectKBest(score_func=multivariate_pearsonr, k=10) 

Xt_pearson = transformer.fit_transform(X, y) 

print(transformer.scores_)
transformer.get_support(True)
X.columns
Xt_pearson
lr_model= LogisticRegression(C = 1,

                              penalty = 'l2',solver = 'liblinear')

scores = cross_val_score(lr_model,Xt_pearson,y,cv=7)

print('The cross validation score is {0}'.format(scores.mean()))
def sigmoid(inX): 

    res = np.array(inX)

    res = 1.0/(1+np.exp(-res))

    return res
model = Ridge(alpha=1)

#model = RidgeCV(alphas=[0.1, 1.0, 10.0])

model.fit(X, y)
model.coef_
#model.alpha_
pred_train_r = model.predict(X_train)

print(max(pred_train_r))

print(min(pred_train_r))

print(classification_report(y_train, pred_train_r > 0.5))

pred_test_r= model.predict(X_test)

print(classification_report(y_test, pred_test_r > 0.5))

scores = cross_val_score(model,X,y,cv=7)

print('The cross validation score is {0}'.format(scores.mean()))
X1 = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

      [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,900],

     [2,0,0,0,0,3,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1200],

     [5,0,0,1,0,4,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1500],

     [6,1,1,2,0,4,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1800],

     [9,2,1,3,0,5,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,2100]]

pre_r = model.predict(X1)

print(pre_r)

fig = plt.figure(figsize = (10,5))

plt.plot(t,pre_r)

plt.xlabel('Time(Seconds)')

plt.ylabel('Winning Probability')

plt.title('Team 1 Winning Prediction')

plt.show()
neigh = KNeighborsClassifier(n_neighbors=2, weights = 'uniform')

neigh.fit(X, y)
neigh_pred = neigh.predict(X_test)

print(classification_report(y_test, neigh_pred > 0.5))
X1 = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

      [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,900],

     [2,0,0,0,0,3,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1200],

     [5,0,0,1,0,4,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1500],

     [6,1,1,2,0,4,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1800],

     [9,2,1,3,0,5,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,2100]]

print(neigh.predict_proba(X1))

p = neigh.predict(X1)

fig = plt.figure(figsize = (10,5))

plt.plot(t,p)

plt.xlabel('Time(Seconds)')

plt.ylabel('Winning Probability')

plt.title('Team 1 Winning Prediction')

plt.show()