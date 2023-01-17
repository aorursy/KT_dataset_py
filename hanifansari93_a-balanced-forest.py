#import the usual suspects

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Ignore warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
#importing the data set

pd.set_option('display.max_columns', None)

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

data.head()
#Distribution of the target variable

plt.figure(figsize = (7,4))

sns.set(style="darkgrid")



percentage = lambda i: len(i) / float(len(data['Class'])) * 100

ax = sns.barplot(x= data['Class'], y=data['Class'],  estimator=percentage)

ax.set(ylabel="Percent")

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 1,'{:1.2f}'.format(height),ha="center")

plt.show()
#Summary Statistics for Amount and Class

data[['Amount','Class']].describe()
#Any null values in the dataset ?

data.isnull().sum().sum()
#Distribution of Time

plt.figure(figsize = (10,4))

sns.distplot(data['Time'], color='g', bins = 100)

plt.show()
#Box plot for Amount

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data, palette="PRGn",showfliers=False)

plt.show()
#Splitting data set

X = data.drop('Class', axis = 1)

y = data['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()



#Fitting the model

dtree.fit(X_train, y_train)



#Prediction

y_score = dtree.predict_proba(X_test)

pred = dtree.predict(X_test)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred))



#Recall Score

from sklearn.metrics import recall_score

print('Recall Score: ',recall_score(y_test, pred))
#Precision-Recall curve

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_score[:,-1])



from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



precision, recall, _ = precision_recall_curve(y_test, y_score[:,-1])



plt.step(recall, precision, color='b', alpha=0.2, where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

plt.show()
#Cross validation score

#Stratified because we need balanced samples

from sklearn.model_selection import StratifiedKFold, cross_val_score

SKfold = StratifiedKFold(n_splits=5, random_state=42)

scores = cross_val_score(dtree, X, y, cv=SKfold, scoring='recall')

scores.mean()
%%time



#Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV



param_dist = {

    'max_depth':[1,2,3,4,5],

    'min_samples_leaf':[1,2,3,4,5],

    'min_samples_split':[2,3,4,5],

    'criterion':['gini','entropy']

}



random_search = RandomizedSearchCV(estimator=dtree, param_distributions=param_dist, scoring='recall', n_jobs=-1, cv=SKfold, n_iter=100)

random_search.fit(X_train, y_train)



print('Best Score: ', random_search.best_score_)

print('Best Params:', random_search.best_params_)
#Checking the SD of the best score

std_dev = pd.DataFrame(random_search.cv_results_)

std_dev = std_dev[std_dev['rank_test_score'] == 1]

print(std_dev['std_test_score'].unique())
#Prediction and evaluation using Decision Tree

pred = random_search.predict(X_test)



from sklearn.metrics import confusion_matrix, recall_score

print(confusion_matrix(y_test, pred))

print('Recall Score: ',recall_score(y_test, pred))
%%time



#SMOTE within CV

from imblearn.pipeline import Pipeline, make_pipeline

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import RandomizedSearchCV



param_dist = {

    'max_depth':[1,2,3,4,5],

    'min_samples_leaf':[1,2,3,4,5],

    'min_samples_split':[2,3,4,5],

    'criterion':['entropy']

}



param_dist = {'decisiontreeclassifier__' + key: param_dist[key] for key in param_dist}

imba_pipeline = make_pipeline(SMOTE(random_state=42), dtree)

random_search = RandomizedSearchCV(imba_pipeline, param_distributions=param_dist, cv=SKfold, scoring='recall', n_iter=50)

random_search.fit(X_train, y_train)



print('Best Score: ', random_search.best_score_)

print('Best Params:', random_search.best_params_)
#Checking the SD of the best score

std_dev = pd.DataFrame(random_search.cv_results_)

std_dev = std_dev[std_dev['rank_test_score'] == 1]

print(std_dev['std_test_score'].unique())
#Prediction and evaluation using Decision Tree with SMOTE

pred = random_search.predict(X_test)



from sklearn.metrics import confusion_matrix, recall_score

print(confusion_matrix(y_test, pred))

print('Recall Score: ',recall_score(y_test, pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfm = RandomForestClassifier()



#Cross validation score

from sklearn.model_selection import StratifiedKFold, cross_val_score

SKfold = StratifiedKFold(n_splits=5, random_state=42)

scores = cross_val_score(rfm, X, y, cv=SKfold, scoring='recall')

print('Mean score: ', scores.mean())

print('Standard deviation: ', scores.std())
%%time



#Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV



param_dist = {'n_estimators': [5, 25, 50],

              'max_features': ['log2', 'sqrt'],

              'max_depth': [int(x) for x in np.linspace(10, 50, num = 5)],

              'min_samples_split': [2, 5, 10],

              'min_samples_leaf': [1, 3, 5]

}



random_search = RandomizedSearchCV(estimator=rfm, param_distributions=param_dist, scoring='recall', n_jobs=-1, cv=SKfold, n_iter=50)

random_search.fit(X_train, y_train)



print('Best Score: ', random_search.best_score_)

print('Best Params:', random_search.best_params_)
#Checking the SD of the best score

std_dev = pd.DataFrame(random_search.cv_results_)

std_dev = std_dev[std_dev['rank_test_score'] == 1]

std_dev['std_test_score']
#Prediction and evaluation using Random Forest

pred = random_search.predict(X_test)



from sklearn.metrics import confusion_matrix, recall_score

print(confusion_matrix(y_test, pred))

print('Recall Score: ',recall_score(y_test, pred))
%%time



#Balanced Random Forest

from imblearn.ensemble import BalancedRandomForestClassifier

brfm = BalancedRandomForestClassifier(sampling_strategy=0.1)



#Cross validation score

from sklearn.model_selection import StratifiedKFold, cross_val_score

SKfold = StratifiedKFold(n_splits=5, random_state=42)

scores = cross_val_score(brfm, X, y, cv=SKfold, scoring='recall')

print('Mean score: ', scores.mean())

print('Standard deviation: ', scores.std())
%%time



#Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV



param_dist = {'sampling_strategy': [0.1, 0.15, 0.2],

              'max_features': ['log2', 'sqrt'],

              'max_depth': [int(x) for x in np.linspace(10, 50, num = 5)],

              'min_samples_split': [2, 5, 10, 25],

              'min_samples_leaf': [1, 3, 5]

}



random_search = RandomizedSearchCV(estimator=brfm, param_distributions=param_dist, scoring='recall', n_jobs=-1, cv=SKfold, n_iter=100)

random_search.fit(X_train, y_train)



print('Best Score: ', random_search.best_score_)

print('Best Params:', random_search.best_params_)
#Checking the SD of the best score

std_dev = pd.DataFrame(random_search.cv_results_)

std_dev = std_dev[std_dev['rank_test_score'] == 1]

print(std_dev['std_test_score'].unique())
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))