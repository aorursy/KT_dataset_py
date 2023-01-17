import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
print(df.head())
df.isna().sum()
df.isnull().values.any()
df.columns
df.type.unique()
## there are two kind of fraud transactions here, either through cashout/transfer. 

fraudTransfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]
fraudCashout = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]

print(len(fraudTransfer))
print(len(fraudCashout))

# fraud occurs only in cashout/transfer type of transactions, so lets filter it out.

x = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
x
sns.countplot(df.isFraud)

df.isFraud.value_counts()

y = x.isFraud
y
x = x.drop(columns = ['isFraud'])
x
np.random.seed(3)
x
x = x.drop(columns = ['nameOrig', 'nameDest', 'isFlaggedFraud'])
x.head()
# converting categorical into numerical
# convert column type having TRANSFER and CASH_OUT into 0 & 1 encoding

x.type = pd.get_dummies(x.type)
x.head()
# incase of fraud transaction the old and new balance is not updated, only the transaction shows the amount
# incase of non fraud transaction both old and new balance are updated, lets have a look

xfraud = x.loc[y == 1]
xnonfraud = x.loc[y == 0]

fraudpercent = len(xfraud.loc[(xfraud.oldBalanceDest == 0 ) &  (xfraud.newBalanceDest == 0) * xfraud.amount])/len(xfraud)
nonfraudpercent = len(xnonfraud.loc[(xnonfraud.oldBalanceDest == 0 ) &  (xnonfraud.newBalanceDest == 0) * xnonfraud.amount])/len(xnonfraud)


print("% of fraud transactions where old ad new balance doesn't get updated is : ",fraudpercent*100)
print("% of nonfraud transactions where old ad new balance doesn't get updated is : ", nonfraudpercent*100)
x.loc[(x.oldBalanceDest == 0) & (x.newBalanceDest == 0) & (x.amount != 0), ['oldBalanceDest', 'newBalanceDest']] = -1
x
y
x.loc[(x.oldBalanceOrig == 0) & (x.newBalanceOrig == 0) & (x.amount != 0), ['oldBalanceOrig', 'newBalanceOrig']] = np.nan
x
x['errorBalanceOrig'] = x.newBalanceOrig + x.amount - x.oldBalanceOrig
x['errorBalanceDest'] = x.oldBalanceDest + x.amount - x.newBalanceDest
x
# split into training and testing


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 3)
# Long computation in this cell (~1.8 minutes)
weights = (y == 0).sum() / (1.0 * (y == 1).sum())

clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)

probabilities = clf.fit(xtrain, ytrain).predict_proba(xtest)
print('AUPRC = {}'.format(average_precision_score(ytest, probabilities[:, 1])))
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(clf, height = 1, color = colours, grid = False, \
                     show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);
ypred = clf.predict(xtest)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

cmXGBoost = confusion_matrix(ytest, ypred)
cmXGBoost
sns.heatmap(cmXGBoost, annot = True)
acc = accuracy_score(ytest, ypred)
prec = precision_score(ytest, ypred)
rec = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)
results = pd.DataFrame([['XGBoost', acc, prec, rec, f1]],
            columns = ["Model", 'accuray', 'precision' , 'recall', 'f1 score'])
results
xtrain.isna().sum()

xtrain = xtrain.fillna(0)
xtest = xtest.fillna(0)
lr = LogisticRegression(random_state = 0, penalty = 'l2')
lr.fit(xtrain, ytrain)
ypred = lr.predict(xtest)
acc = accuracy_score(ytest, ypred)
prec = precision_score(ytest, ypred)
rec = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)

model_results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
            columns = ["Model", 'accuray', 'precision' , 'recall', 'f1 score'])

results = results.append(model_results, ignore_index = True)
#results = results.loc[(results.index !=1)]
results
cmLR = confusion_matrix(ytest, ypred)
cmLR
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators= 20)
rf.fit(xtrain, ytrain)
ypred = rf.predict(xtest)

acc = accuracy_score(ytest, ypred)
prec = precision_score(ytest, ypred)
rec = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)

model_results = pd.DataFrame([['Random Forest', acc, prec, rec, f1]],
            columns = ["Model", 'accuray', 'precision' , 'recall', 'f1 score'])

results = results.append(model_results, ignore_index = True)
results
cmRF = confusion_matrix(ytest, ypred)
cmRF
plt.figure(figsize=(24,8))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(1,3,1)
plt.title("XGBoost Confusion Matrix")
sns.heatmap(cmXGBoost,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,3,2)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cmLR,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,3,3)
plt.title("Random forest Confusion Matrix")
sns.heatmap(cmRF,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.show()
results
