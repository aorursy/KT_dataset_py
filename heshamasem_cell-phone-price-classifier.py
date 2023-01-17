import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier 
train = pd.read_csv('/kaggle/input/train.csv')

test = pd.read_csv('/kaggle/input/test.csv')
train.head()
train.shape
test.head()
test.shape
train.describe()
train.info()
sns.countplot(x="price_range", data=train,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))
train.columns
print('number of unique values for attery power is : {}'.format(len(train.battery_power.unique())))

train.battery_power.unique()
train['battery code'] = round(train['battery_power']/100)
print('number of unique values for attery power is : {}'.format(len(train['battery code'].unique())))

train['battery code'].unique()
sns.jointplot("battery code", "price_range", train, kind='kde')
sns.barplot(x="battery code", y="price_range", data=train)
train.drop(['battery code'], axis=1, inplace=True)
sns.barplot(x="dual_sim", y="price_range", data=train)
train.drop(['dual_sim'], axis=1, inplace=True)
sns.barplot(x="blue", y="price_range", data=train)
train.drop(['blue'], axis=1, inplace=True)
sns.jointplot("clock_speed", "price_range", train, kind='kde')
sliced_train = train.loc[:,['price_range','battery_power','clock_speed'] ]
sliced_train.head()
sns.heatmap(sliced_train.corr(), annot=True, linewidths=.5, fmt= '.1f')
sliced_train = train.loc[:,['price_range','fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores']]   
sliced_train.head()
sns.heatmap(sliced_train.corr(), annot=True, linewidths=.5, fmt= '.1f')
sliced_train = train.loc[:,['price_range', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w']]   
sliced_train.head()
sns.heatmap(sliced_train.corr(), annot=True, linewidths=.5, fmt= '.1f')
sliced_train = train.loc[:,['price_range', 'talk_time', 'three_g', 'touch_screen', 'wifi']]   
sliced_train.head()
sns.heatmap(sliced_train.corr(), annot=True, linewidths=.5, fmt= '.1f')
X_data = train.drop(['price_range'], axis=1, inplace=False)

y_data = train['price_range']
X_data.head()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=44, shuffle =True)
print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
SelectedModel = SVC(gamma='auto_deprecated')

SelectedParameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5]}





GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
print('All Results are :\n', GridSearchResults )

print('===========================================')

print('Best Score is :', GridSearchModel.best_score_)

print('===========================================')

print('Best Parameters are :', GridSearchModel.best_params_)

print('===========================================')

print('Best Estimator is :', GridSearchModel.best_estimator_)
SVCModel =  GridSearchModel.best_estimator_

SVCModel.fit(X_train, y_train)
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))

print('----------------------------------------------------')





y_pred = SVCModel.predict(X_test)

print('Predicted Value for SVCModel is : ' , y_pred[:10])
SelectedModel = LogisticRegression(penalty='l2' , solver='sag',random_state=33)

SelectedParameters = {'C':[1,2,3,4,5]}





GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 4,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
print('All Results are :\n', GridSearchResults )

print('===========================================')

print('Best Score is :', GridSearchModel.best_score_)

print('===========================================')

print('Best Parameters are :', GridSearchModel.best_params_)

print('===========================================')

print('Best Estimator is :', GridSearchModel.best_estimator_)
DTModel_ = DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state = 33)

GaussianNBModel_ = GaussianNB()

BernoulliNBModel_ = BernoulliNB(alpha = 0.1)

MultinomialNBModel_= MultinomialNB(alpha = 0.1)

SGDModel_ = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)
#loading Voting Classifier

VotingClassifierModel = VotingClassifier(estimators=[('DTModel',DTModel_),('GaussianNBModel',GaussianNBModel_),

                                                     ('BernoulliNBModel',BernoulliNBModel_),

                                                     ('MultinomialNBModel',MultinomialNBModel_),

                                                     ('SGDModel',SGDModel_)], voting='hard')

VotingClassifierModel.fit(X_train, y_train)
#Calculating Details

print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))

print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))

print('----------------------------------------------------')
SVCModel =  SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3,

                gamma='auto_deprecated',kernel='linear', max_iter=-1, probability=False, random_state=None,

                shrinking=True, tol=0.001, verbose=False)

SVCModel.fit(X_train, y_train)



print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))

print('----------------------------------------------------')
test.head()
test.drop(['id','blue','dual_sim'], axis=1, inplace=True)
print('Test Dimension is {}'.format(test.shape))

test.head()
print('X_train Dimension is {}'.format(X_train.shape))

X_train.head()
final_result = SVCModel.predict(test)

final_result
test.insert(18,'Expected Price',final_result)
test.head(30)