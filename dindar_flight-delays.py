import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import thinkstats2
import thinkplot
from sklearn.feature_selection import VarianceThreshold
plt.style.use('fivethirtyeight')
from scipy import stats
train = pd.read_csv('G:/Desktop/datasets/flight-delays-spring-2018/flight_delays_train.csv')
test = pd.read_csv('G:/Desktop/datasets/flight-delays-spring-2018/flight_delays_test.csv')
train.head()
train.shape, test.shape
train.info(memory_usage='deep')
test.info(memory_usage='deep')
data = pd.concat([train,test])
data.info(memory_usage='deep')
data.shape
data.columns
data.dtypes
data.describe()
data.isnull().sum()

data = data.fillna(np.nan)
data['dep_delayed_15min'].isnull().sum()
data['dep_delayed_15min'].unique()
d = {'N':0, 'Y':1}
data['dep_delayed_15min'] = data['dep_delayed_15min'].map(d)
data['dep_delayed_15min'].value_counts(normalize=True)
plt.figure(figsize=(10, 7))
sns.countplot(data['dep_delayed_15min'])


data['DayOfWeek'].head()
data['DayOfWeek'].dtypes
data['DayOfWeek'].isnull().any()
data['DayOfWeek'].unique()
d = {'c-7':7, 'c-3':3, 'c-5':5, 'c-6':6, 'c-4':4, 'c-2':2, 'c-1':1}
data['DayOfWeek'] = data['DayOfWeek'].map(d)
data['DayOfWeek'].dtypes
data['DayOfWeek'].value_counts()
pmf = thinkstats2.Pmf(data['DayOfWeek'])
pmf
plt.figure(figsize=(10, 7))
thinkplot.Pmf(pmf)
thinkplot.Show(xlabel='Day of Week', ylabel='PMF')

data['week_sin'] = np.sin(data['DayOfWeek']*(2.*np.pi/7))
data['week_cos'] = np.cos(data['DayOfWeek']*(2.*np.pi/7))

yes_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 1.0]['DayOfWeek'], label='1')
no_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 0.0]['DayOfWeek'],  label='0')
plt.figure(figsize=(10, 7))
thinkplot.Pmfs([yes_pmf, no_pmf])
thinkplot.Show(xlabel='Week days', ylabel='PMF')
plt.figure(figsize=(12, 9))
sns.factorplot('DayOfWeek', 'dep_delayed_15min', data=data)
data.drop('DayOfWeek', axis=1, inplace=True)
data['Month'].head()
data['Month'].isnull().any()
data['Month'].unique()
d = {'c-8':8, 'c-4':4, 'c-9':9, 'c-11':11, 'c-10':10, 'c-1':1, 'c-7':7, 'c-6':6, 'c-5':5,
       'c-3':3, 'c-12':12, 'c-2':2}
data['Month'] = data['Month'].map(d)
data['Month'].value_counts(normalize=True).sort_index()
yes_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 1.0]['Month'], label='1')
no_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 0.0]['Month'],  label='0')
plt.figure(figsize=(10, 7))
thinkplot.Pmfs([yes_pmf, no_pmf])
thinkplot.Show(xlabel='Month', ylabel='PMF')
sns.factorplot('Month', 'dep_delayed_15min', data=data, size=8)
data['month_sin'] = np.sin((data['Month'])*(2.*np.pi/12))
data['month_cos'] = np.cos((data['Month'])*(2.*np.pi/12))
data.drop('Month', axis=1, inplace=True)


data['DayofMonth'].head()
data['DayofMonth'].dtypes
data['DayofMonth'].isnull().any()
data['DayofMonth'].unique()
d = {'c-21':21, 'c-20':20, 'c-2':2, 'c-25':25, 'c-7':7, 'c-3':3, 'c-27':27, 'c-29':29,
       'c-28':28, 'c-5':5, 'c-6':6, 'c-10':10, 'c-19':19, 'c-26':26, 'c-14':14, 'c-22':22,
       'c-9':9, 'c-31':31, 'c-4':4, 'c-16':16, 'c-18':18, 'c-30':30, 'c-1':1, 'c-24':24,
       'c-15':15, 'c-17':17, 'c-8':8, 'c-12':12, 'c-13':13, 'c-11':11, 'c-23':23}
data['DayofMonth'] = data['DayofMonth'].map(d)
data['DayofMonth'].value_counts(normalize=True).sort_index()
pmf1 = thinkstats2.Pmf(data['DayofMonth'])
plt.figure(figsize=(10, 7))
thinkplot.Pmf(pmf1)
thinkplot.Show(xlabel='Day of Month', ylabel='PMF')
yes_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 1.0]['DayofMonth'], label='1')
no_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 0.0]['DayofMonth'],  label='0')
plt.figure(figsize=(10, 7))
thinkplot.Pmfs([yes_pmf, no_pmf])
thinkplot.Show(xlabel='Days', ylabel='PMF')
sns.factorplot('DayofMonth', 'dep_delayed_15min', data=data, size = 10)
data['day_sin'] = np.sin(data['DayofMonth']*(2.*np.pi/31))
data['day_cos'] = np.cos(data['DayofMonth']*(2.*np.pi/31))
data.drop('DayofMonth', axis=1, inplace=True)

data['DepTime'].head()
data['DepTime'].dtypes
data['DepTime'].describe()
plt.figure(figsize=(10,7))
sns.distplot(data['DepTime'])
indexes = data[data['DepTime'] > 2360]['DepTime'].index
len(indexes)
d = data[(data['DepTime'] > 2361)&(data['DepTime'] < 2461)]
d1 = data[data['DepTime'] < 61]
d.shape[0], d1.shape[0]
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.countplot(d['dep_delayed_15min'], ax = ax[0])
sns.countplot(d1['dep_delayed_15min'], ax = ax[1])
ax[0].set_title(' > 2361 & < 2461 ')
ax[1].set_title(' < 61')
d2 = data[(data['DepTime'] > 2461)&(data['DepTime'] < 2561)]
d3 = data[(data['DepTime'] > 61)&(data['DepTime'] < 161)]
d2.shape[0], d3.shape[0]
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.countplot(d2['dep_delayed_15min'], ax = ax[0])
sns.countplot(d3['dep_delayed_15min'], ax = ax[1])
ax[0].set_title(' > 2461 & < 2561 ')
ax[1].set_title(' > 61 & < 161 ')
data.loc[(data['DepTime'] > 2361)&(data['DepTime'] < 2461),'DepTime'] = 59
data.loc[(data['DepTime'] > 2461)&(data['DepTime'] < 2561),'DepTime'] = 159
data['depTime']=0
data.loc[data['DepTime']<=61,'depTime']=1
data.loc[(data['DepTime']>61)&(data['DepTime']<=161),'depTime']=2
data.loc[(data['DepTime']>161)&(data['DepTime']<261),'depTime']=3
data.loc[(data['DepTime']>261)&(data['DepTime']<=361),'depTime']=4
data.loc[(data['DepTime']>361)&(data['DepTime']<=461),'depTime']=5
data.loc[(data['DepTime']>461)&(data['DepTime']<=561),'depTime']=6
data.loc[(data['DepTime']>561)&(data['DepTime']<=661),'depTime']=7
data.loc[(data['DepTime']>661)&(data['DepTime']<=761),'depTime']=8
data.loc[(data['DepTime']>761)&(data['DepTime']<=861),'depTime']=9
data.loc[(data['DepTime']>861)&(data['DepTime']<=961),'depTime']=10
data.loc[(data['DepTime']>961)&(data['DepTime']<=1061),'depTime']=11
data.loc[(data['DepTime']>1061)&(data['DepTime']<=1161),'depTime']=12
data.loc[(data['DepTime']>1161)&(data['DepTime']<=1261),'depTime']=13
data.loc[(data['DepTime']>1261)&(data['DepTime']<=1361),'depTime']=14
data.loc[(data['DepTime']>1361)&(data['DepTime']<=1461),'depTime']=15
data.loc[(data['DepTime']>1461)&(data['DepTime']<=1561),'depTime']=16
data.loc[(data['DepTime']>1561)&(data['DepTime']<=1661),'depTime']=17
data.loc[(data['DepTime']>1661)&(data['DepTime']<=1761),'depTime']=18
data.loc[(data['DepTime']>1761)&(data['DepTime']<=1861),'depTime']=19
data.loc[(data['DepTime']>1861)&(data['DepTime']<=1961),'depTime']=20
data.loc[(data['DepTime']>1961)&(data['DepTime']<=2061),'depTime']=21
data.loc[(data['DepTime']>2061)&(data['DepTime']<=2161),'depTime']=22
data.loc[(data['DepTime']>2161)&(data['DepTime']<=2261),'depTime']=23
data.loc[(data['DepTime']>2261)&(data['DepTime']<=2361),'depTime']=24
data['depTime'].value_counts().sort_index()
plt.figure(figsize=(10, 7))
pmf = thinkstats2.Pmf(data['depTime'])
thinkplot.Pmf(pmf)
data.groupby('depTime').sum()['dep_delayed_15min']
yes_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 1.0]['depTime'], label='1')
no_pmf = thinkstats2.Pmf(data[data['dep_delayed_15min'] == 0.0]['depTime'],  label='0')
plt.figure(figsize=(10, 7))
thinkplot.Pmfs([yes_pmf, no_pmf])
thinkplot.Show(xlabel='depTime', ylabel='PMF')
data.drop('DepTime', axis=1, inplace=True)
data['Dest'].head(), data['Origin'].head()
data['Dest'].dtypes, data['Origin'].dtypes
data['Dest'].unique(), data['Origin'].unique()
len(data['Dest'].unique()), len(data['Origin'].unique())
data['Dest'].value_counts(), data['Origin'].value_counts()
data = pd.get_dummies(data=data, columns=['Dest', 'Origin'])





data['UniqueCarrier'].unique()
data.groupby('UniqueCarrier').sum()['dep_delayed_15min']
plt.figure(figsize=(10, 7))
sns.countplot(x='UniqueCarrier', data=data, order=data['UniqueCarrier'].value_counts().index)
plt.figure(figsize=(10, 7))
sns.countplot(x='UniqueCarrier', data=data, hue = 'dep_delayed_15min', order=data['UniqueCarrier'].value_counts().index)

data = pd.get_dummies(data=data, columns=['UniqueCarrier'])
data['Distance'].head()
data['Distance'].dtypes
data['Distance'].describe()
plt.figure(figsize=(10, 7))
sns.distplot(data['Distance'], fit=stats.norm)
plt.figure(figsize=(10, 7))
stats.probplot(data['Distance'], plot=plt)
plt.figure(figsize=(10, 7))
sns.distplot(data['Distance'], fit=stats.lognorm)
plt.figure(figsize=(10, 7))
sns.distplot(np.log(data['Distance']), fit=stats.norm)
plt.figure(figsize=(10, 7))
stats.probplot(np.log(data['Distance']), plot=plt)
plt.figure(figsize=(10, 7))
sns.distplot(data['Distance'], fit=stats.expon)
cdf = thinkstats2.Cdf(data['Distance'])
plt.figure(figsize=(10, 7))
thinkplot.Cdf(cdf)
plt.figure(figsize=(10, 7))
thinkplot.Cdf(cdf, complement=True)
thinkplot.Config(yscale='log', loc='upper right')
plt.figure(figsize=(10, 7))
sns.distplot(data['Distance'], fit=stats.pareto)

data['Distance'] = np.log(data['Distance'])
pdf = thinkstats2.EstimatedPdf(data['Distance'])
plt.figure(figsize=(10, 7))
thinkplot.Pdf(pdf)
yes_cdf = thinkstats2.Cdf(data[data['dep_delayed_15min'] == 1.0]['Distance'], label='1')
no_cdf = thinkstats2.Cdf(data[data['dep_delayed_15min'] == 0.0]['Distance'],  label='0')
plt.figure(figsize=(10, 7))
thinkplot.Cdfs([yes_cdf, no_cdf])


x = data.drop('dep_delayed_15min', axis = 1)
y = data['dep_delayed_15min']
x = x[:100000]
plt.figure(figsize=(10,7))
print(y.value_counts(normalize=True))
sns.countplot(y)
x.describe()
x = (x - x.mean()) / (x.std()) 
x.var().describe()
a = []
for i in x.columns:
    if x[i].isnull().sum() > 0:
        a.append(i)
a = []
b = x.var().mean()
for i in x.columns:
    if x[i].var() < b:
        a.append(i)
x = x.drop(a, axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y[:100000], test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# find best scored 5 features
select_feature = SelectKBest(f_classif, k=5).fit(x_train, y_train)
print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)
x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)
print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()
'Dest_EWR', 'Dest_IAH', 'Origin_EWR', 'Origin_ORD', 'Origin_SLC'
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y[:100000], test_size=0.3, random_state=42)
x_train.columns[indices]
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')








xx = data[['Origin_ORD', 'UniqueCarrier_MQ', 'Dest_EWR']]
y = y[:100000]


X = data.drop('dep_delayed_15min', axis=1)
y = data['dep_delayed_15min'][:100000]
inputs = data[:100000]
test = data[10000:]
X = inputs.drop('dep_delayed_15min', axis=1)
y = inputs['dep_delayed_15min']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(xx, y, test_size=0.3, random_state=42)
data[data['week_sin'] < 0]['week_sin'].any()
minus = []
for i in X.columns:
    if X[X[i] < 0][i].any() == True:
        minus.append(i)
plus = []
for i in X.columns:
    if X[X[i] < 0][i].any() == False:
        plus.append(i)
XX = X[minus]
from sklearn.feature_selection import SelectKBest, f_classif, chi2
X_new=SelectKBest(score_func=chi2,k=45).fit_transform(X[plus],y)

mask = X_new.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, X[plus].columns):
    if bool:
        new_features.append(feature)
new_features  = new_features + minus
train = X[new_features]
test = X[new_features]






X = pd.merge(pd.DataFrame(X_new), XX, left_index=True, right_index=True)


data.dtypes

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
lr=LogisticRegression()
%%time
cross_val_score(lr, X, y, cv=5).mean()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV, StratifiedKFold

%%time
extra = ExtraTreesClassifier()
extra = extra.fit(X_train,y_train)
extra.score(X_test, y_test)



parameters = {'max_features': [2, 4, 7, 9], 'max_samples': [0.5, 0.7, 0.9], 
              "base_estimator__C": [0.0001, 0.001, 0.01, 1, 10]}
%%time
bg = BaggingClassifier(XGBClassifier(),
                       n_estimators=100, n_jobs=-1, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
r_grid_search = GridSearchCV(bg, parameters, cv=skf)
%%time
r_grid_search = r_grid_search.fit(X_train, y_train)
r_grid_search.score(X_test, y_test)
r_grid_search.best_estimator_
bb = BaggingClassifier(base_estimator=XGBClassifier(C=0.0001, base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1),
         bootstrap=True, bootstrap_features=False, max_features=9,
         max_samples=0.7, n_estimators=100, n_jobs=-1, oob_score=False,

                       random_state=42, verbose=0, warm_start=False)
%%time
bb.fit(X_train, y_train)
bb.score(X_test, y_test)

%%time
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('RBF',SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('xgb',XGBClassifier()),
                                              ('svm',SVC(kernel='linear',probability=True)),
                                              ('ada',AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)),
                                              ('gbc',GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1))
                                             ], 
                       voting='soft').fit(X_train,y_train)
ensemble_lin_rbf.score(X_test, y_test)
ensemble_lin_rbf.score(X_test, y_test)
ensemble_lin_rbf.score(X_test, y_test)
%%time
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
ada.fit(X_train, y_train)
ada.score(X_test, y_test)
xgb = XGBClassifier()
%%time
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)
model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=100, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)
%%time
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["auc","error"]
model.fit(X_train, y_train)
model.score(X_test, y_test)
parameters = {'learning_rate' : [0.1, 0.01],
             'n_estimators':[50, 100],
             'max_depth' : [1, 3, 5, 9, 15, 19],
             'subsample' : [0.75, 0.8, 0.9, 1.0],
             'colsample_bytree' : [0.3, 0.4, 0.6, 0.8],
             'gamma' : [0.1, 5]}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
r_grid_search = GridSearchCV(xgb, parameters, cv=skf)
%%time
r_grid_search.fit(X_train, y_train)
print('aaa')
r_grid_search.score(X_test, y_test)
r_grid_search.score(X_test, y_test)
r_grid_search.best_estimator_
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=5,
       learning_rate=0.1, max_delta_step=0, max_depth=9,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1.0, verbosity=1)
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
%%time
grad.fit(X_train, y_train)
grad.score(X_test, y_test)

rf = RandomForestClassifier()
%%time
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
knn = KNeighborsClassifier()
%%time
knn.fit(X_train, y_train)
%%time
knn.score(X_test, y_test)


%%time
C=[0.05,0.1,0.2,0.4,0.6,0.7,1]
gamma=[0.1,0.3,0.5,0.6,0.8,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=SVC(),param_grid=hyper,verbose=True)
gd.fit(X_train, y_train)
gd.score(X_test, y_test)
gd.best_estimator_
sss = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.3, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
sss.fit(X_train, y_train)
sss.score(X_test, y_test)




svc = SVC()
%%time
svc.fit(X_train, y_train)
%%time
svc.score(X_test, y_test)






