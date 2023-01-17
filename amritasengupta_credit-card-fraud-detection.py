import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
df= pd.read_csv('../input/credit-card-fraud-detection-data/creditcard.csv')
df.head()
df.tail()
df.info()
df.duplicated().sum()

df=df.drop_duplicates()
df.shape
df.isnull().sum()
df=df[df['Amount']>0]
df['Class'].value_counts(normalize=True)
def bar_graph(data,predictor):
    grouped=data.groupby(predictor)
    chart=grouped.size().plot.bar(rot=0, title='Bar Chart showing the size of different '+str(predictor))
    chart.set_xlabel(predictor)
bar_graph(data=df,predictor='Class')
df.describe().T
df.nunique()
df['Hour']=df['Time']/3600
df
%matplotlib inline
df.hist(['V1','V2','V3','V4','V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'],figsize=(25,20))
def anova_test(data,target,predictor):
    data1=data.groupby(predictor)[target].apply(list)
    from scipy.stats import f_oneway
    AnovaResults = f_oneway(*data1)
    print('P-Value for Anova is: ', AnovaResults[1])

anova_test(data=df,target='Amount',predictor='Class')
anova_test(data=df,target='Hour',predictor='Class')
anova_test(data=df,target='V1',predictor='Class')
anova_test(data=df,target='V2',predictor='Class')
anova_test(data=df,target='V3',predictor='Class')
anova_test(data=df,target='V5',predictor='Class')
anova_test(data=df,target='V7',predictor='Class')
anova_test(data=df,target='V8',predictor='Class')
anova_test(data=df,target='V9',predictor='Class')
anova_test(data=df,target='V10',predictor='Class')
anova_test(data=df,target='V11',predictor='Class')
anova_test(data=df,target='V12',predictor='Class')
anova_test(data=df,target='V13',predictor='Class')
anova_test(data=df,target='V14',predictor='Class')
anova_test(data=df,target='V15',predictor='Class')
anova_test(data=df,target='V16',predictor='Class')
anova_test(data=df,target='V17',predictor='Class')
anova_test(data=df,target='V18',predictor='Class')
anova_test(data=df,target='V19',predictor='Class')
anova_test(data=df,target='V20',predictor='Class')
anova_test(data=df,target='V21',predictor='Class')
anova_test(data=df,target='V22',predictor='Class')
anova_test(data=df,target='V23',predictor='Class')
anova_test(data=df,target='V24',predictor='Class')
anova_test(data=df,target='V25',predictor='Class')
anova_test(data=df,target='V26',predictor='Class')
anova_test(data=df,target='V27',predictor='Class')
anova_test(data=df,target='V28',predictor='Class')
df.drop(columns=['V13', 'V15', 'V25','Time'],inplace=True)
df_corr=df.corr()
df_corr
df_corr['V28'][abs(df_corr['V28']) > 0.5 ]
df.boxplot(by='Class', column='Amount')
df= df[df['Amount']<15000]
df['Amount'].sort_values(ascending=False).head(10)
df['Amount'][df['Amount']>12000]=11898.09
df.boxplot(by='Class', column='Amount')
y=df['Class'].values
x=df[['Hour', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
       'V23', 'V24', 'V26', 'V27', 'V28', 'Amount']].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
np.unique(y_train, return_counts=True)
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
smk=SMOTE(random_state=42)
x_smote,y_smote=smk.fit_sample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_smote))
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=42)
x_over,y_over= ros.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_over))
from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=42)
x_under,y_under= rus.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_under))
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 1, 10]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(x_train, y_train)
log_reg = grid_log_reg.best_estimator_
print(log_reg)
knears_params = {"n_neighbors": list(range(2,5,1))}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(x_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

print(knears_neighbors)
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(x_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_
print(tree_clf)
forest_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "n_estimators":list(range(200,1000,100))}
grid_tree = GridSearchCV(RandomForestClassifier(), tree_params)
grid_tree.fit(x_train, y_train)

# tree best estimator
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
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)
rfc.fit( x_train, y_train )

y_pred = rfc.predict(x_test)

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

y_pred_proba = rfc.predict_proba(x_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
import copy
df2=copy.deepcopy(df)
df2.drop(columns='Class',inplace=True)
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
std_scale_fit=std_scaler.fit(df2)
StandardizedFullData = std_scale_fit.transform(df2)
df2_sd=pd.DataFrame(StandardizedFullData,columns=df2.columns)
y=df['Class'].values
x=df2_sd[['Hour', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
       'V23', 'V24', 'V26', 'V27', 'V28', 'Amount']].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from imblearn.over_sampling import SMOTE
smk=SMOTE(random_state=42)
x_smote,y_smote=smk.fit_sample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_smote))
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=42)
x_over,y_over= ros.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_over))
from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=42)
x_under,y_under= rus.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_under))
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
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)
rfc.fit( x_train, y_train )

y_pred = rfc.predict(x_test)

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

y_pred_proba = rfc.predict_proba(x_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
import copy
df3=copy.deepcopy(df)
df3.drop(columns='Class')
from sklearn import preprocessing
minmax_scaler = preprocessing.MinMaxScaler()
minmax_scaler_fit=minmax_scaler.fit(df3)
Normalized= minmax_scaler_fit.transform(df3)
df3_normal=pd.DataFrame(Normalized,columns=df3.columns)
y=df['Class'].values
x=df3_normal[['Hour', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
       'V23', 'V24', 'V26', 'V27', 'V28', 'Amount']].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from imblearn.over_sampling import SMOTE
smk=SMOTE(random_state=42)
x_smote,y_smote=smk.fit_sample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_smote))
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=42)
x_over,y_over= ros.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_over))
from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=42)
x_under,y_under= rus.fit_resample(x_train,y_train)
print('Resampled dataset shape %s' % Counter(y_under))
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
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit( x_over, y_over)


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
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)
rfc.fit( x_train, y_train )

y_pred = rfc.predict(x_test)

print(accuracy_score(y_pred , y_test))

y_pred_proba = rfc.predict_proba(x_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
rfc = RandomForestClassifier(criterion='entropy', max_depth=2, n_estimators=200)
rfc.fit( x_smote, y_smote)

y_pred = rfc.predict(x_test)

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

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

print(accuracy_score(y_pred , y_test))

y_pred_proba = rfc.predict_proba(x_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
pd.crosstab(pd.Series(y_pred,name='Predicted'),pd.Series(y_test,name='Actual'))
