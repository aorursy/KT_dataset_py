import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import LeavePOut

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn import tree

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/bank-marketing.csv')
df.head()
df.tail()
df.info()
df.dtypes
df.isnull().sum()
df.describe()
df
df['pdays']
df['pdays'].value_counts().sum()
df['pdays'].value_counts()
df['pdays'].describe()
df['pdays'].describe()
x = df.groupby(['education'], as_index=False)['balance'].median()



fig = plt.figure(figsize=(12,8))



sns.barplot(x="balance", y="education", data=x,

            label="Total", palette="magma")

plt.show()
df['education'].value_counts()
plt.boxplot(df['pdays'])

plt.show()
df.head()
cat = ['job', 'marital', 'education', 'targeted',   'default', 'housing', 'loan', 'contact', 'month','poutcome']



fig, axis = plt.subplots(4, 3,  figsize=(25, 20))



counter = 0

for items in cat:

    value_counts = df[items].value_counts()

    

    trace_x = counter // 3

    trace_y = counter % 3

    x_pos = np.arange(0, len(value_counts))

    my_colors = 'rgbkymc'

    

    axis[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index,color=my_colors)

    

    axis[trace_x, trace_y].set_title(items)

    

    for tick in axis[trace_x, trace_y].get_xticklabels():

        tick.set_rotation(90)

    

    counter += 1



plt.tight_layout()

plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



fig, axis = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))



counter = 0

for items in num:

    

    trace_x = counter // 3

    trace_y = counter % 3

    

    

    axis[trace_x, trace_y].hist(df[items])

    

    axis[trace_x, trace_y].set_title(items)

    

    counter += 1



plt.tight_layout()

plt.show()
plt.figure(figsize=(25, 6))







plt.subplot(1,2,1)

plt1 = df.age.value_counts().plot('bar')

plt.title('Age')





plt.subplot(1,2,2)

plt1 = df.job.value_counts().plot('bar')

plt.title('Job')



plt.figure(figsize=(25, 6))



plt.subplot(1,2,1)

plt1 = df.salary.value_counts().plot('bar')

plt.title('Salary')

plt.show()



plt.subplot(1,2,2)

plt1 = df.education.value_counts().plot('bar')

plt.title('Education')

plt.show()





plt.figure(figsize=(25, 6))



plt.subplot(1,2,1)

plt1 = df.targeted.value_counts().plot('bar')

plt.title('targeted')

plt.show()



plt.subplot(1,2,2)

plt1 = df.default.value_counts().plot('bar')

plt.title('default')

plt.show()



















plt.tight_layout()

plt.show()
def plot_count(x,fig):

    plt.subplot(4,2,fig)

   

    sns.countplot(df[x],palette=("magma"))

    plt.subplot(4,2,(fig+1))

    

    sns.boxplot(x=df[x], y=df.response, palette=("magma"))

    

plt.figure(figsize=(15,20))



plot_count('age', 1)

plot_count('salary', 3)

plot_count('day', 5)

plot_count('campaign', 7)





plt.tight_layout()

plt.show()
df.head()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,6))

    sns.distplot(df[items])

    

plt.tight_layout()




num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']





for item in num:

    plt.figure(figsize=(10,8))

    sns.violinplot(df[item],df["response"])

    

    plt.xlabel(item,fontsize=12)

    plt.ylabel("Response",fontsize=12)

    plt.show()
df.corr()
plt.figure(figsize=(15,12))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['salary'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['balance'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['day'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['duration'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['campaign'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['pdays'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
num = ['age','salary','balance', 'day','duration', 'campaign', 'pdays', 'previous']



for items in num:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = df[items],y = df['previous'],kind='reg')

    plt.xlabel(items,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
df
sns.pairplot(df)

plt.show()
df.head()
df.drop(['campaign','pdays','previous'],axis=1,inplace=True)
df
def binary_map(x):

    return x.map({'yes': 1, "no": 0})
cols_yn = ['targeted', 'default', 'housing', 'loan', 'response']
df1 = df.copy()
df1[cols_yn] = df1[cols_yn].apply(binary_map)
df1
df1 = pd.get_dummies(df1, drop_first=True)
df1.head()
df1.info()
df1
X = df1.drop('response',axis=1)
X.head()
y = df1[['response']]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
log = LogisticRegression()
scalar = MinMaxScaler()
log.fit(X_train,y_train)
log.classes_
log.coef_
pred = log.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
# lets select all the features onces just forget about pvalues
import statsmodels.api as sm
log1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

log1.fit().summary()
from sklearn.feature_selection import RFE
rfe = RFE(log, 25)
rfe.fit(X_train,y_train)
rfe.ranking_
rfe.support_
a=X_train.columns[rfe.support_]
a
len(a)
X_train[a].describe()
log1 = sm.GLM(y_train,(sm.add_constant(X_train[a])), family = sm.families.Binomial())

log1.fit().summary()
# the rfe has selected features but some featues are high value but just build a model
log.fit(X_train[a],y_train)
log.classes_
log.coef_
pred = log.predict(X_test[a])
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
# the model done well but in that there were few high p values
# lets check with another approach ie.. vif
# vif with all the values of X_train
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 25)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# now looking to the various factors such as vif and rfe lets take features manually and build a model
b = ['housing','loan','job_retired','marital_married','education_secondary','education_tertiary','education_unknown','contact_telephone',

     'contact_unknown','month_aug','month_dec','month_feb','month_jan','month_jul','month_mar','month_may','month_nov','month_oct',

    'month_sep','poutcome_other','poutcome_success' ]
log2 = sm.GLM(y_train,(sm.add_constant(X_train[b])), family = sm.families.Binomial())

log2.fit().summary()
log.fit(X_train[b],y_train)
log.classes_
log.coef_
predf = log.predict(X_test[b])
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predf)
from sklearn.metrics import confusion_matrix

from sklearn import metrics
confusion_matrix(y_test,predf)
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 4))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
draw_roc(y_test,predf)
from sklearn.metrics import classification_report
print(classification_report(y_test,predf))
X.head(n=3)
y.head(n=3)
# Holdout Validation Approach - Train and Test Set Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=100)

log = LogisticRegression()

log.fit(X_train,y_train)

result = log.score(X_train,y_train)

print("Accuracy: %.2f%%" % (result*100.0))
# Stratified K-fold Cross-Validation
skfold = StratifiedKFold(n_splits=3, random_state=100)

model_skfold = LogisticRegression()

results_skfold = model_selection.cross_val_score(model_skfold, X, y, cv=skfold)

print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
# K-fold Cross-Validation
kfold = model_selection.KFold(n_splits=5, random_state=100)

model_kfold = LogisticRegression()

results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
# Repeated Random Test-Train Splits
kfold2 = model_selection.ShuffleSplit(n_splits=5, test_size=0.30, random_state=100)

model_shufflecv = LogisticRegression()

results_4 = model_selection.cross_val_score(model_shufflecv, X, y, cv=kfold2)

print("Accuracy: %.2f%% (%.2f%%)" % (results_4.mean()*100.0, results_4.std()*100.0))
from sklearn.model_selection import cross_val_score
log = LogisticRegression()
cross_val_score(log,X,y,cv=5)
cross_val_score(log,X,y,cv=5).mean()
standardizer = StandardScaler()

log = LogisticRegression()

pipeline = make_pipeline(standardizer, log)
kf = KFold(n_splits=10, shuffle=True, random_state=100)
cv_results = cross_val_score(pipeline, 

                             X, 

                             y, 

                             cv=kf, 

                             scoring="accuracy", 

                             n_jobs=-1) 
cv_results.mean()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train,y_train)
print('Number of Trees used : ', model.n_estimators)
 
predict_train = model.predict(X_train)

print('\nTarget on train data',predict_train) 





accuracy_train = accuracy_score(y_train,predict_train)

print('\naccuracy_score on train dataset : ', accuracy_train)
predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 





accuracy_test = accuracy_score(y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
kfold = model_selection.KFold(n_splits=5, random_state=100)

model_kfold = RandomForestClassifier()

results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
print(classification_report(y_test,predict_test))
from sklearn.model_selection import GridSearchCV
param_grid = {

    'max_depth': [1, 2, 5, 10, 20],

    'max_features': [1, 2, 3],

    'min_samples_leaf': [1, 3, 4, 5],

#    'min_samples_split': [8, 10, 12],

    'n_estimators': [10, 30, 50, 100, 200]

}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 

                          cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
grid_search.fit(X_train,y_train)
grid_search.best_estimator_
best_model = grid_search.best_estimator_
pred= best_model.predict(X_test)
confusion_matrix(y_test,pred)
#XGB-Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))