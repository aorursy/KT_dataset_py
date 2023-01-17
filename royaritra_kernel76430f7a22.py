import pandas as pd

import numpy as np

data = pd.read_csv('../input/creditcard/creditcard.csv')
data.head(5)
data.tail()
data.info()
data.shape
data = data.drop_duplicates()
data.shape
data.isnull().sum()
data.nunique()
data['Time_Hr'] = data['Time']/3600

data
data['Class'].value_counts()
def bar_graph(data,predictor):

    grouped=data.groupby(predictor)

    chart=grouped.size().plot.bar(rot=0, title='Bar Chart showing the size of different '+str(predictor))

    chart.set_xlabel(predictor)

    
bar_graph(data=data,predictor='Class')
%matplotlib inline

import matplotlib.pyplot as plt

%matplotlib inline

data.hist(['V1','V2','V3','V4','V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'],figsize=(20,15))
data.boxplot(by='Class', column='Amount')
data= data[data['Amount']<20000]
data['Amount'][data['Amount']>12000]=12000
data.boxplot(by='Class', column='Amount')
data_corr=data.corr()

data_corr
def anova_test(data,target,predictor):

    data1=data.groupby(predictor)[target].apply(list)

    from scipy.stats import f_oneway

    AnovaResults = f_oneway(*data1)

    print('P-Value for Anova is: ', AnovaResults[1])

anova_test(data=data,target='Amount',predictor='Class')
anova_test(data=data,target='V1',predictor='Class')
anova_test(data=data,target='V2',predictor='Class')
anova_test(data=data,target='V3',predictor='Class')
anova_test(data=data,target='V4',predictor='Class')
anova_test(data=data,target='V5',predictor='Class')
anova_test(data=data,target='V6',predictor='Class')
anova_test(data=data,target='V7',predictor='Class')
anova_test(data=data,target='V8',predictor='Class')
anova_test(data=data,target='V9',predictor='Class')
anova_test(data=data,target='V10',predictor='Class')
anova_test(data=data,target='V11',predictor='Class')
anova_test(data=data,target='V12',predictor='Class')
anova_test(data=data,target='V13',predictor='Class')
anova_test(data=data,target='V14',predictor='Class')
anova_test(data=data,target='V15',predictor='Class')
anova_test(data=data,target='V16',predictor='Class')
anova_test(data=data,target='V17',predictor='Class')
anova_test(data=data,target='V18',predictor='Class')
anova_test(data=data,target='V19',predictor='Class')
anova_test(data=data,target='V20',predictor='Class')
anova_test(data=data,target='V21',predictor='Class')
anova_test(data=data,target='V22',predictor='Class')
anova_test(data=data,target='V23',predictor='Class')
anova_test(data=data,target='V24',predictor='Class')
anova_test(data=data,target='V25',predictor='Class')
anova_test(data=data,target='V26',predictor='Class')
anova_test(data=data,target='V27',predictor='Class')
anova_test(data=data,target='V28',predictor='Class')
y=data['Class'].values

x=data[['Time_Hr', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',

       'V23', 'V24', 'V26', 'V27', 'V28', 'Amount']].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=457)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from collections import Counter

from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,roc_auc_score

from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler

rus= RandomUnderSampler(random_state=457)

x_under,y_under= rus.fit_resample(x_train,y_train)

print('Resampled dataset shape %s' % Counter(y_under))
from imblearn.over_sampling import RandomOverSampler

ros= RandomOverSampler(random_state=457)

x_over,y_over= ros.fit_resample(x_train,y_train)

print('Resampled dataset shape %s' % Counter(y_over))
from imblearn.over_sampling import SMOTE

smk=SMOTE(random_state=457)

x_smote,y_smote=smk.fit_sample(x_train,y_train)

print('Resampled dataset shape %s' % Counter(y_smote))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 1, 10]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(x_train, y_train)

log_reg = grid_log_reg.best_estimator_

print(log_reg)
from sklearn.tree import DecisionTreeClassifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(3,4,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(x_train, y_train)





tree_clf = grid_tree.best_estimator_

print(tree_clf)
from sklearn.neighbors import KNeighborsClassifier

knears_params = {"n_neighbors": list(range(4,7,1))}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(x_train, y_train)



knears_neighbors = grid_knears.best_estimator_



print(knears_neighbors)
from sklearn.ensemble import RandomForestClassifier

forest_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "n_estimators":list(range(200,1000,100))}

grid_tree = GridSearchCV(RandomForestClassifier(), tree_params)

grid_tree.fit(x_train, y_train)





tree_clf = grid_tree.best_estimator_

print(forest_params)
lr = LogisticRegression(C=1)

lr.fit( x_train, y_train )





y_pred = lr.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = lr.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
lr = LogisticRegression(C=1)

lr.fit( x_under, y_under )





y_pred = lr.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = lr.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
lr = LogisticRegression(C=1)

lr.fit( x_over, y_over )





y_pred = lr.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = lr.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
lr = LogisticRegression(C=1)

lr.fit( x_smote, y_smote )





y_pred = lr.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = lr.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
dte = DecisionTreeClassifier(criterion='entropy', max_depth=3)

dte.fit( x_train, y_train )





y_pred = dte.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = dte.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
dte = DecisionTreeClassifier(criterion='entropy', max_depth=3)

dte.fit( x_under, y_under )





y_pred = dte.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = dte.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
dte = DecisionTreeClassifier(criterion='entropy', max_depth=3)

dte.fit( x_over, y_over )





y_pred = dte.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = dte.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
dte = DecisionTreeClassifier(criterion='entropy', max_depth=3)

dte.fit( x_smote, y_smote )





y_pred = dte.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = dte.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
knc = KNeighborsClassifier(n_neighbors=3)

knc.fit( x_train, y_train )





y_pred = knc.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = knc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
knc = KNeighborsClassifier(n_neighbors=3)

knc.fit( x_under, y_under )





y_pred = knc.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = knc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
knc = KNeighborsClassifier(n_neighbors=3)

knc.fit( x_over, y_over )





y_pred = knc.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = knc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
knc = KNeighborsClassifier(n_neighbors=3)

knc.fit( x_smote, y_smote )





y_pred = knc.predict(x_test)





print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = knc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)

rfc.fit( x_train, y_train )



y_pred = rfc.predict(x_test)



print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = rfc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)

rfc.fit( x_under, y_under )



y_pred = rfc.predict(x_test)



print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = rfc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)

rfc.fit( x_over, y_over )



y_pred = rfc.predict(x_test)



print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = rfc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)

rfc.fit( x_smote, y_smote )



y_pred = rfc.predict(x_test)



print('Accuracy:',accuracy_score(y_pred , y_test))



y_pred_proba = rfc.predict_proba(x_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))