import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv')

df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \

                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

print(df.head())

print(df.isnull().values.sum())
df.info()
dfFraud = df[df.isFraud == 1]

print(f'Number of frauds: {len(dfFraud)}')

print(f'% of frauds among transactions: {len(dfFraud)/len(df)*100}%')

print(f'Number of frauds, by type: {list(dict(dfFraud.type.value_counts()).keys())[0]}: {list(dict(dfFraud.type.value_counts()).values())[0]}\n{list(dict(dfFraud.type.value_counts()).keys())[1]}: {list(dict(dfFraud.type.value_counts()).values())[1]}')

print(f'Flagged as fraud: {len(df[df.isFlaggedFraud==1])}')
sns.pairplot(df[df.isFlaggedFraud==1])
print(df.shape)

df.drop(columns="isFlaggedFraud", inplace=True)

print(df.shape)

print(df.columns)
df['type_nameOrig'] = df.nameOrig.apply(lambda x: x[0])

print(df.type_nameOrig.value_counts())

df['type_nameDest'] = df.nameDest.apply(lambda x: x[0])

print(df.type_nameDest.value_counts())
df.drop('type_nameOrig', axis=1, inplace=True)

df.drop('nameOrig', axis=1, inplace=True)

df.drop('nameDest', axis=1, inplace=True)
dfFraud = df[df.isFraud == 1]

print('type_nameDest')

print(f"% among all: {len(df[df.type_nameDest == 'C'])/len(df)*100}%")

print(f"% among frauds: {len(dfFraud[dfFraud.type_nameDest == 'C'])/len(dfFraud)*100}%")
dummies = pd.get_dummies(df['type_nameDest'], prefix='type_nameDest', drop_first=True)

df = pd.concat([df, dummies], axis=1)

df.drop('type_nameDest', axis=1, inplace=True)

df.info()
dfFraud = df[df.isFraud == 1]  

print(dfFraud.type.value_counts())

dfFraud = []
X = df[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

y = X['isFraud']

Xfraud = X.loc[y == 1]

X['type'] = X.type.apply(lambda x: 1 if x == 'CASH_OUT' else 0)

print(X.type.value_counts())
X.type_nameDest_M.value_counts()
X.drop('type_nameDest_M', axis=1, inplace=True)
limit = len(X)



def plotStrip(x, y, hue, figsize = (14, 9)):

    

    fig = plt.figure(figsize = figsize)

    colours = plt.cm.tab10(np.linspace(0, 1, 9))

    with sns.axes_style('ticks'):

        ax = sns.stripplot(x, y, \

             hue = hue, jitter = 0.4, marker = '.', \

             size = 4, palette = colours)

        ax.set_xlabel('')

        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)

        for axis in ['top','bottom','left','right']:

            ax.spines[axis].set_linewidth(2)



        handles, labels = ax.get_legend_handles_labels()

        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), \

               loc=2, borderaxespad=0, fontsize = 16);

    return ax
ax = plotStrip(y, np.log(X.amount), X.type)

ax.set_ylabel('log_amount', size = 25)
maxtds = X.amount.max()

maxfraud = X[X.isFraud == 1].amount.max()



print(f"Max amount among all: {maxtds}\nMax amount among frauds: {maxfraud}")

print(f'Equal to max à among all: {len(X[(X.amount == maxtds)])}\nEqual to max among frauds: {len(Xfraud[Xfraud.amount == maxfraud])}\n')

print()

print(f'Equal to 10.000.000 among all: {len(X[X.amount==10000000])}')

print(f'Equal to 10.000.000 among frauds: {len(Xfraud[Xfraud.amount==10000000])}')

print()

print(f'% of 10.000.000 among all: {len(X[X.amount==10000000])/len(X)*100}%')

print(f'% of 10.000.000 among frauds: {len(Xfraud[X.amount==10000000])/len(Xfraud)*100}%')



X['amount10'] = X.amount.apply(lambda x: 1 if x == 10000000 else 0)
plt.hist(X[X.isFraud==1].oldBalanceDest, range=[0, 1000000], bins=20, color='r')
plt.hist(X[X.isFraud==0].oldBalanceDest, range=[0, 1000000], bins=20, color='b')
plt.hist(X[X.isFraud==1].newBalanceDest, range=[0, 5000000], bins=20, color='r')
plt.hist(X[X.isFraud==0].newBalanceDest, range=[0, 5000000], bins=20, color='b')
plt.hist(np.log1p(X[X.isFraud==1].newBalanceOrig), range=[0, 19], bins=14, color='r')
plt.hist(np.log1p(X[X.isFraud==0].newBalanceOrig), range=[0, 19], bins=14, color='b')
plt.hist(np.log1p(X[X.isFraud==1].oldBalanceOrig), range=[-5, 19], bins=14, color='r')
plt.hist(np.log1p(X[X.isFraud==0].oldBalanceOrig), range=[-5, 19], bins=14, color='b')
xis = X.loc[(np.log1p(X.newBalanceOrig)>10) & (np.log1p(X.newBalanceOrig)<20)]

print(f'% among all entre todos: {len(xis)/len(X)*100}%')

xis = Xfraud.loc[(np.log1p(Xfraud.newBalanceOrig)>10) & (np.log1p(Xfraud.newBalanceOrig)<20)]

print(f'% among frauds: {len(xis)/len(Xfraud)*100}%')

xis = 0
sns.scatterplot(np.log1p(X.oldBalanceOrig), np.log1p(X.amount), hue=X.isFraud)
sns.scatterplot(np.log1p(X.loc[X.amount == X.oldBalanceOrig].oldBalanceOrig),

                np.log1p(X.loc[X.amount == X.oldBalanceOrig].amount), alpha=0.05, hue=X.isFraud)
sns.scatterplot(np.log1p(X.newBalanceDest),np.log1p(X.oldBalanceDest), hue=X.isFraud)
sns.scatterplot(np.log1p(X.loc[X.newBalanceDest == X.oldBalanceDest].newBalanceDest),

                np.log1p(X.loc[X.newBalanceDest == X.oldBalanceDest].oldBalanceDest), alpha=0.9, hue=X.isFraud)
X['newBalanceDest0'] = X.newBalanceDest.apply(lambda x: 1 if x == 0 else 0)

X['oldBalanceDest0'] = X.oldBalanceDest.apply(lambda x: 1 if x == 0 else 0)



X.loc[(df.oldBalanceDest == 0) & (df.newBalanceDest == 0), 'newoldBalanceDest0'] = 1

X.loc[(df.oldBalanceDest != 0) | (df.newBalanceDest != 0), 'newoldBalanceDest0'] = 0



X.loc[(X.newBalanceDest == X.oldBalanceDest), 'balanceDestEqual'] = 1

X.loc[(X.newBalanceDest != X.oldBalanceDest), 'balanceDestEqual'] = 0



X.loc[(X.amount == X.oldBalanceOrig), 'amountOldBalanceOrigEqual'] = 1

X.loc[(X.amount != X.oldBalanceOrig), 'amountOldBalanceOrigEqual'] = 0



X.loc[(X.oldBalanceOrig == 0), 'oldBalanceOrig0'] = 1

X.loc[(X.oldBalanceOrig != 0), 'oldBalanceOrig0'] = 0

X.loc[(X.newBalanceOrig == 0), 'newBalanceOrig0'] = 1

X.loc[(X.newBalanceOrig != 0), 'newBalanceOrig0'] = 0





Xfraud = X.loc[y == 1]



li = ['newBalanceDest0','oldBalanceDest0', 'newoldBalanceDest0',  'balanceDestEqual', 'amountOldBalanceOrigEqual', 'oldBalanceOrig0', 'newBalanceOrig0']

for l in li:

    print(l)

    print(f'% among all: {X[l].mean()*100}%')

    print(f'% among frauds: {Xfraud[l].mean()*100}%')

    print()
del X['newBalanceOrig0']
X['erroOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig

X['erroDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
ax = plotStrip(y, X.erroDest, X.type)

ax.set_ylabel('erroDest', size = 25)
X['erroDest<0'] = X.erroDest.apply(lambda x: 1 if x == 0 else 0)

Xfraud = X[X.isFraud == 1]

print(f"% among all: {X['erroDest<0'].mean()*100}%")

print(f"% among frauds: {Xfraud['erroDest<0'].mean()*100}%")
ax = plotStrip(y, X.erroOrig, X.type)

ax.set_ylabel('erroOrig', size = 25)
print(f'% among all: {len(X[X.erroOrig == 0])/len(X)*100}%')

print(f'% among frauds: {len(Xfraud[Xfraud.erroOrig == 0])/len(Xfraud)*100}%')

X['erroOrig0'] = X.erroOrig.apply(lambda x: 1 if x==0 else 0)
numeric_feats = X.dtypes[X.dtypes != "object"].index



skewed_feats = X[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

print("\nSkewness: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(20)
nolog = ['amount10','isFlaggedFraud','erroDest<0', 'newBalanceDest0',

          'erroDest', 'type', 'step', 'isFraud', 'balanceDestEqual', 'amountOldBalanceOrigEqual',

          'oldBalanceDest0', 'newoldBalanceDest0', 'oldBalanceOrig0', 'newBalanceOrig0', 'erroOrig0']



for c in X.columns:

    if c not in nolog:

        X[f'log_{c}'] = np.log1p(X[c])

        X[f'standard_log_{c}'] = (X[f'log_{c}'] - X[f'log_{c}'].mean()) / X[f'log_{c}'].std()

        del X[c]

        del X[f'log_{c}']



numeric_feats = X.dtypes[X.dtypes != "object"].index



skewed_feats = X[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

print("\nSkewness: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(20)
X[X.columns[0:]].corr()['isFraud'][:-1]
X.drop(columns="isFraud", inplace=True)
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split 



sm = SMOTE(random_state = 33)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 38) 

X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

y_train = pd.Series(y_train_new)

X_train = pd.DataFrame(X_train_new)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics



clf = GradientBoostingClassifier(random_state = 0)



# Fitting the model

clf.fit(X_train, y_train)

print()

print('****Results****')

# Making Predictions

y_pred = clf.predict(X_test)

# Printing metrics

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:   ", metrics.recall_score(y_test, y_pred))

print("F1:       ", metrics.f1_score(y_test, y_pred))

print("AUPRC:    ", metrics.average_precision_score(y_test, y_pred))
#AUPRC CURVE



precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

no_skill = len(y_test[y_test==1]) / len(y_test)

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.plot(recall, precision, marker='.', label='Classifier')

# axis labels

plt.xlabel('Recall')

plt.ylabel('Precision')

# show the legend

plt.legend()

# show the plot

plt.show()
# FEATURE IMPORTANCE

feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = feature_importance / feature_importance.max()

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()