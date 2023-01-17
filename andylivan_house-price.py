import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train





train.info()
train[['MSSubClass', 'SalePrice']].groupby('MSSubClass', as_index=False).mean()
plt.bar(train['MSSubClass'], train['SalePrice'])

plt.show()
dic = { 20 : 0, 30 : 1, 40 : 2, 45 : 3, 50 : 4,

        60 : 5, 70 : 6, 75 : 7, 80 : 8, 85 : 9,

        90 : 10, 120 : 11, 160 : 12, 180 : 13, 190 : 14 }
train['MSSubClass'] = train['MSSubClass'].map(dic).astype(int)
test['MSSubClass'] = test['MSSubClass'].map(dic).fillna(15).astype(int)
train.head()
train[['MSZoning', 'SalePrice']].groupby('MSZoning', as_index=False).mean()
plt.bar(train['MSZoning'], train['SalePrice'])

plt.show()
train['MSZoning'] = train['MSZoning'].map({'C (all)':0, 'FV':1, 'RH':2, 'RL':3, 'RM':4}).astype(int)
train.head()
test['MSZoning'] = test['MSZoning'].map({'C (all)':0, 'FV':1, 'RH':2, 'RL':3, 'RM':4}).fillna(5).astype(int)
test.head()
train[['LotFrontage', 'SalePrice']].groupby('LotFrontage', as_index=False).mean()
plt.bar(train['LotFrontage'], train['SalePrice'])

plt.show()
train['LotFrontageBand'] = pd.cut(train['LotFrontage'], 5)

train[['LotFrontageBand', 'SalePrice']].groupby('LotFrontageBand', as_index=False).mean()
train.loc[train['LotFrontage'] <= 79.4 , 'LotFrontage'] = 1

train.loc[(train['LotFrontage'] > 79.4) & (train['LotFrontage'] <= 137.8), 'LotFrontage'] = 2

train.loc[(train['LotFrontage'] > 137.8) & (train['LotFrontage'] <= 196.2), 'LotFrontage'] = 3

train.loc[train['LotFrontage'] > 196.2 , 'LotFrontage'] = 4

train['LotFrontage'] = train['LotFrontage'].fillna(0).astype(int)
test.loc[test['LotFrontage'] <= 79.4 , 'LotFrontage'] = 1

test.loc[(test['LotFrontage'] > 79.4) & (test['LotFrontage'] <= 137.8), 'LotFrontage'] = 2

test.loc[(test['LotFrontage'] > 137.8) & (test['LotFrontage'] <= 196.2), 'LotFrontage'] = 3

test.loc[test['LotFrontage'] > 196.2 , 'LotFrontage'] = 4

test['LotFrontage'] = test['LotFrontage'].fillna(0).astype(int)

train = train.drop('LotFrontageBand', axis=1)
train.head()
train[['LotArea', 'SalePrice']].groupby('LotArea', as_index=False).mean()
train['LotAreaBand'] = pd.cut(train['LotArea'], 10)

train[['LotAreaBand', 'SalePrice']].groupby('LotAreaBand', as_index=False).count()

# train.loc[train['LotArea'] <= 44089.0 , 'LotArea'] = 0

# train.loc[(train['LotArea'] > 44089.0) & (train['LotArea'] <= 86878.0), 'LotFrontage'] = 1

# train.loc[(train['LotArea'] > 86878.0) & (train['LotArea'] <= 129667.0), 'LotFrontage'] = 2

# train.loc[(train['LotArea'] > 129667.0) & (train['LotArea'] <= 172456.0), 'LotFrontage'] = 3

# train.loc[train['LotArea'] > 172456.0 , 'LotArea'] = 4
train.head()
# train[['LotArea', 'SalePrice']].groupby('LotArea', as_index=False).mean()

plt.bar(train['Street'], train['SalePrice'])

plt.xticks(train['Street'])

plt.show()
train[['Street', 'SalePrice']].groupby('Street', as_index=False).mean()
train['Street'] = train['Street'].map({'Grvl':0, 'Pave':1}).astype(int)
test['Street'] = test['Street'].map({'Grvl':0, 'Pave':1}).astype(int)
train.head()
train = train.drop('Alley', axis=1)
test = test.drop('Alley', axis=1)
train.head()
plt.bar(train['LotShape'], train['SalePrice'])

plt.show()
train[['LotShape', 'SalePrice']].groupby('LotShape', as_index=False).mean()
train['LotShape'] = train['LotShape'].map({'IR1':0, 'IR2':1, 'IR3':2, 'Reg':4,})
test['LotShape'] = test['LotShape'].map({'IR1':0, 'IR2':1, 'IR3':2, 'Reg':4,})
train.head()
plt.bar(train['LandContour'], train['SalePrice'])

plt.show()
train[['LandContour', 'SalePrice']].groupby('LandContour', as_index=False).mean()
train['LandContour'] = train['LandContour'].map({'Bnk':0, 'HLS':1, 'Low':2, 'Lvl':3}).astype(int)
test['LandContour'] = test['LandContour'].map({'Bnk':0, 'HLS':1, 'Low':2, 'Lvl':3}).astype(int)
train.head()
plt.bar(train['Utilities'], train['SalePrice'])

plt.show()
train[['Utilities', 'SalePrice']].groupby('Utilities', as_index=False).mean()
train['Utilities'] = train['Utilities'].map({'AllPub':0, 'NoSeWa':1}).astype(int)
test['Utilities'] = test['Utilities'].map({'AllPub':0, 'NoSeWa':1}).fillna(2).astype(int)
test.head()

train[['LotConfig', 'SalePrice']].groupby('LotConfig', as_index=False).mean()
train['LotConfig'] = train['LotConfig'].map({'Corner':0, 'CulDSac':1, 'FR2':2, 'FR3':3, 'Inside':4}).astype(int)
test['LotConfig'] = test['LotConfig'].map({'Corner':0, 'CulDSac':1, 'FR2':2, 'FR3':3, 'Inside':4}).astype(int)
train.head()
train[['LandSlope', 'SalePrice']].groupby('LandSlope', as_index=False).mean()
plt.scatter(train['LandSlope'], train['SalePrice'])

plt.show()
train['LandSlope'] = train['LandSlope'].map({'Gtl':0, 'Mod':1, 'Sev':2,}).astype(int)
test['LandSlope'] = test['LandSlope'].map({'Gtl':0, 'Mod':1, 'Sev':2,}).astype(int)
grNbh = train[['Neighborhood', 'SalePrice']].groupby('Neighborhood', as_index=False).mean()

grNbh
plt.scatter(train['Neighborhood'], train['SalePrice'])

plt.show()
dicNbh={}

num = 0

for line in grNbh['Neighborhood']:

    dicNbh.update({line : num})

    num += 1
train['Neighborhood'] = train['Neighborhood'].map(dicNbh).astype(int)

test['Neighborhood'] = test['Neighborhood'].map(dicNbh).astype(int)
train[['Neighborhood', 'SalePrice']].groupby('Neighborhood', as_index=False).mean()

grC1 = train[['Condition1', 'SalePrice']].groupby('Condition1', as_index=False).mean()

grC1

plt.scatter(train['Condition1'], train['SalePrice'])

plt.show()
dicC1={}

num = 0

for line in grC1['Condition1']:

    dicC1.update({line : num})

    num += 1
train['Condition1'] = train['Condition1'].map(dicC1).astype(int)
test['Condition1'] = test['Condition1'].map(dicC1).astype(int)
grC2 = train[['Condition2', 'SalePrice']].groupby('Condition2', as_index=False).mean()

grC2

plt.scatter(train['Condition2'], train['SalePrice'])

plt.show()
dicC2={}

num = 0

for line in grC2['Condition2']:

    dicC2.update({line : num})

    num += 1
train['Condition2'] = train['Condition2'].map(dicC2).astype(int)
test['Condition2'] = test['Condition2'].map(dicC2).astype(int)
grBt = train[['BldgType', 'SalePrice']].groupby('BldgType', as_index=False).mean()

grBt
plt.scatter(train['BldgType'], train['SalePrice'])

plt.show()
dicBt={}

num = 0

for line in grBt['BldgType']:

    dicBt.update({line : num})

    num += 1
train['BldgType'] = train['BldgType'].map(dicBt).astype(int)
test['BldgType'] = test['BldgType'].map(dicBt).astype(int)
grHs = train[['HouseStyle', 'SalePrice']].groupby('HouseStyle', as_index=False).mean()

grHs
plt.scatter(train['HouseStyle'], train['SalePrice'])

plt.show()
dicHs={}

num = 0

for line in grHs['HouseStyle']:

    dicHs.update({line : num})

    num += 1
train['HouseStyle'] = train['HouseStyle'].map(dicHs).astype(int)
test['HouseStyle'] = test['HouseStyle'].map(dicHs).astype(int)
train.head()
train[['OverallQual', 'SalePrice']].groupby('OverallQual', as_index=False).mean()
plt.scatter(train['OverallQual'], train['SalePrice'])

plt.show()
train.info()
trMv = train[['MiscVal', 'SalePrice']].groupby('MiscVal', as_index=False).mean()

trMv
plt.scatter(train['MiscVal'], train['SalePrice'])
train.head()
train[['MoSold', 'SalePrice']].groupby('MoSold', as_index=False).mean()
plt.scatter(train['MoSold'], train['SalePrice'])

plt.show()
train[['YrSold', 'SalePrice']].groupby('YrSold', as_index=False).mean()
plt.scatter(train['YrSold'], train['SalePrice'])

plt.show()
grSt = train[['SaleType', 'SalePrice']].groupby('SaleType', as_index=False).mean()

grSt

plt.scatter(train['SaleType'], train['SalePrice'])

plt.show()
def get_dict(data):

    dic={}

    num = 0

    for line in data:

        dic.update({line : num})

        num += 1

    return dic
dicSt = get_dict(grSt['SaleType'])
train['SaleType'] = train['SaleType'].map(dicSt).astype(int)
test['SaleType'] = test['SaleType'].map(dicSt).fillna(9).astype(int)
grSc = train[['SaleCondition', 'SalePrice']].groupby('SaleCondition', as_index=False).mean()

grSc
plt.scatter(train['SaleCondition'], train['SalePrice'])

plt.show()
dicSc = get_dict(grSc['SaleCondition'])
train['SaleCondition'] = train['SaleCondition'].map(dicSc).astype(int)
test['SaleCondition'] = test['SaleCondition'].map(dicSc).astype(int)
test.head()
X_train = train[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'SaleType', 'SaleCondition']]

y_train = train['SalePrice']

X_test = test[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'SaleType', 'SaleCondition']]

y_test = sample_submission['SalePrice']
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

logreg.score(X_train, y_train)
dtr = DecisionTreeClassifier()

dtr.fit(X_train, y_train)

dtr.predict(X_test)

dtr.score(X_train, y_train)
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

rfc.score(X_train, y_train)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

gnb.score(X_train, y_train)
svc = LinearSVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

svc.score(X_train, y_train)
prc = Perceptron()

prc.fit(X_train, y_train)

y_pred = prc.predict(X_test)

prc.score(X_train, y_train)
knc = KNeighborsClassifier(n_neighbors = 3)

knc.fit(X_train, y_train)

y_pred = knc.predict(X_test)

knc.score(X_train, y_train)