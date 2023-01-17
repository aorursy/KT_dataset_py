import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings ('ignore')



%matplotlib inline
train= pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')
train.head()
sns.distplot(train['Age'].dropna(), bins = 30)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))

sns.countplot(data=train, x='Survived', hue='Sex', ax=ax1)

sns.countplot(data=train, x='Survived', hue='Pclass', ax=ax2)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))

sns.countplot(data=train, x='Survived', hue='Parch', ax=ax1)

sns.countplot(data=train, x='Survived', hue='SibSp', ax=ax2)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')



#Find missing values

total_miss_train = train.isnull().sum()

perc_miss_train = total_miss_train/train.isnull().count()*100

missing_data_train = pd.DataFrame(({'Total missing train':total_miss_train,

                            '% missing':perc_miss_train}))

missing_data_train.sort_values(by='Total missing train',ascending=False).head(2)
sns.set(style="darkgrid")

plt.figure (figsize=(12,7))

sns.boxplot(data=train, y='Age', x='Pclass')
def impute_age (cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else: return Age
train['Age']= train[['Age', 'Pclass']].apply(impute_age, axis=1)
sex=pd.get_dummies(train['Sex'], drop_first=True)

embark = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train,sex,embark],axis=1)
train.head()
sns.set(style="darkgrid")

plt.figure (figsize=(12,7))

sns.boxplot(data=test, y='Age', x='Pclass')
def impute_age (cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 42

        elif Pclass==2:

            return 27

        else:

            return 25

    else: return Age
test['Age']= test[['Age', 'Pclass']].apply(impute_age, axis=1)
sex=pd.get_dummies(test['Sex'], drop_first=True)

embark = pd.get_dummies(test['Embarked'], drop_first=True)

test = pd.concat([test,sex,embark],axis=1)
test.head()
features = ['Pclass','Age','SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']

target = ['Survived']
# Check for missing values in the test dataset

train[features].isnull().sum()
# Check for missing values in the test dataset

test[features].isnull().sum()
# Fill them in with 0

test[features]=test[features].replace(np.NAN, 0)

#test["Survived"] = ""
data_correlation = train.corr()

mask = np.array(data_correlation)

mask[np.tril_indices_from(mask)] = False

fig = plt.subplots(figsize=(20,10))

sns.heatmap(data_correlation, mask=mask, vmax=1, square=True, annot=True)
# Model Selection

from sklearn.model_selection import train_test_split



X = train[features]

y = train[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101)

#X_train = train[features]

#y_train = train[target]

#X_test = test[features]
#Selection of algorithm 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier 

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection



models = [LogisticRegression(),

          SGDClassifier(),

          DecisionTreeClassifier(), 

          GradientBoostingClassifier(),

          RandomForestClassifier(),

          BaggingClassifier(),

          svm.SVC(),

          GaussianNB()]



def test_algorithms(model):

    kfold = model_selection.KFold(n_splits=10, random_state=101)

    predicted = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')

    print(predicted.mean())

    

for model in models:

    test_algorithms(model)
from sklearn.metrics import roc_curve, auc

learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]

train_results = []

test_results = []

for eta in learning_rates:

   model = GradientBoostingClassifier(learning_rate=eta)

   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(learning_rates, train_results, 'b', label='Train AUC')

line2, = plt.plot(learning_rates, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('learning rate')

plt.show()

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 300, 500]

train_results = []

test_results = []

for estimator in n_estimators:

   model = GradientBoostingClassifier(n_estimators=estimator)

   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')

line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()

max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []

test_results = []

for max_depth in max_depths:

   model = GradientBoostingClassifier(max_depth=max_depth)

   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators':[10,20,30,50, 100, 200, 300],'max_depth':[3,5,7,9,11,13,14], 'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3]}

grid_rf = GridSearchCV(GradientBoostingClassifier(),param_grid,cv=10,scoring='roc_auc').fit(X_train,y_train)

print('Best parameter: {}'.format(grid_rf.best_params_))

print('Best score: {:.2f}'.format((grid_rf.best_score_)))
gbc = GradientBoostingClassifier(max_depth=3, n_estimators=20,learning_rate=0.15)

gbc.fit(X_train,y_train)

y_pred = gbc.predict(X_test)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve, auc, roc_auc_score

# predict probabilities

probs = gbc.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

# show the plot

plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(GradientBoostingClassifier(max_depth=3, n_estimators=20, learning_rate=0.15), X_train, y_train, cv=3, scoring='roc_auc')



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

valid_scores_mean = np.mean(valid_scores, axis=1)

valid_scores_std = np.std(valid_scores, axis=1)



plt.figure()

plt.plot(train_sizes,valid_scores_mean,label='valid')

plt.plot(train_sizes,train_scores_mean,label='train')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3,color="g")

plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.3, color="b")

plt.xlabel('Number of samples')

plt.ylabel('ROC_AUC')

plt.legend()
gbc = GradientBoostingClassifier(max_depth=3, n_estimators=20,learning_rate=0.15)

gbc.fit(train[features],train[target])

y_pred = gbc.predict(test[features])
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)