import numpy as np # linear algebra

import pandas as pd # data processing, CSV fIle I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from time import time

from sklearn.model_selection import GridSearchCV

import seaborn as sns

import warnings; warnings.simplefilter('ignore')



print(os.listdir("../input"))
df = pd.read_csv("../input/heart.csv")

df.head()
df.shape
categoricals = ['cp','restecg','slope','thal']

df[categoricals].head(2)
numerics = np.setdiff1d(df.columns.tolist(),categoricals)

df[numerics].head(3)
# There aren't any null values

df.isnull().sum()
df.describe()
plt.scatter(df['target'],df['chol']);
#We do see this individual as an extreme outlier, but I will leave this in the model.

#I am curious to see their other attributes however next to the rest of the distributions shown in the "describe" method.

df[df['chol']==564]
df.describe()
df['thal'].max()
pd.get_dummies(df[categoricals].astype(str)).head()
skew_calcs = df[numerics].skew()

skew_calcs[abs(skew_calcs)>0.7]
skewed_features = skew_calcs[skew_calcs.abs()>0.7].index.tolist()

skewed_features
df[skewed_features].hist();
np.log1p(df[skewed_features]).hist();
df.head()
modeldata = pd.merge(df[numerics],pd.get_dummies(df[categoricals].astype(str)),left_index = True, right_index = True,how = 'inner')

modeldata.head()
from sklearn.model_selection import train_test_split

X, y = modeldata.drop(['target'],axis = 1), modeldata['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import fbeta_score, accuracy_score
def train_predict_with_parameters(learner, X_train, y_train, X_test, y_test,parameters = {}): 

    results = {}

    start = time() # Get start time

    if parameters != {}:

        clf = GridSearchCV(learner, parameters)

        learner = clf.fit(X_train,y_train)

        results['best_params'] = learner.best_params_

    else:

        learner = learner.fit(X_train,y_train)

        results['best_params'] = ''

    end = time() # Get end time

    results['train_time'] = end - start

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train)

    end = time() # Get end time

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train,predictions_train)

    results['acc_test'] = accuracy_score(y_test,predictions_test)

    results['f_train'] = fbeta_score(y_train,predictions_train,.8)

    results['f_test'] = fbeta_score(y_test,predictions_test,.8)

    #results['best_params'] = learner.best_params_

    return results
learners= []



from sklearn.linear_model import LogisticRegression

learners.append([LogisticRegression(),{}])



from sklearn.tree import DecisionTreeClassifier

learners.append([DecisionTreeClassifier(),{}])



from sklearn.neighbors import KNeighborsClassifier

learners.append([KNeighborsClassifier(),{}])



from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier 

learners.append([GradientBoostingClassifier(),{ 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] }])

learners.append([AdaBoostClassifier(),{'n_estimators': [16, 32]}])



learners.append([BaggingClassifier(),{}])

learners.append([RandomForestClassifier(),

                 {'bootstrap': [True, False],

                 'max_depth': [3, 5, 10, 20, None],

                 'max_features': ['auto', 'sqrt'],

                 'min_samples_leaf': [1, 2, 4],

                 'min_samples_split': [2, 5, 10]}

                ])



learners.append([RandomForestClassifier(),{}])



from sklearn.svm import LinearSVC, SVC

learners.append([SVC(),{'kernel': ['linear', 'rbf'],'C':[1, 10]}])

learners.append([LinearSVC(),{}])



from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

learners.append([BernoulliNB(),{}])

learners.append([GaussianNB(),{}])

learners.append([MultinomialNB(),{}])
results = {}

for clf in learners:

    #No Params

    clf_name = clf[0].__class__.__name__

    results[clf_name] = train_predict_with_parameters(clf[0], X_train, y_train, X_test, y_test)

    #GridSearchCV

    if clf[1] != {}:

        clf_name = clf[0].__class__.__name__

        results[str(clf_name) + '_gridsearch'] = train_predict_with_parameters(clf[0], X_train, y_train, X_test, y_test, clf[1])
pd.DataFrame.from_dict(results).transpose().sort_values(['f_test','acc_test'], ascending = [False,False])
from sklearn.ensemble import VotingClassifier
clf1 = BernoulliNB()

clf2 = RandomForestClassifier()

clf3 = AdaBoostClassifier()

eclf1 = VotingClassifier(estimators=[('bnb', clf1), ('rf', clf2), ('ab', clf3)], voting='hard')

#If ‘hard’, uses predicted class labels for majority rule voting. Else if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.

clf_name = eclf1.__class__.__name__

results[clf_name] = train_predict_with_parameters(eclf1, X_train, y_train, X_test, y_test)

pd.DataFrame.from_dict(results).transpose().sort_values(['f_test','acc_test'], ascending = [False,False])
#We'll be using this function a few times to show feature importances



def show_importances(importances,vars = 8):

    indices = np.argsort(importances)[::-1]

    columns = X_train.columns.values[indices[:vars]]

    values = importances[indices][:vars]

    fig = plt.figure(figsize = (9,5))

    plt.title("Normalized Weights for Most Predictive Features", fontsize = 16)

    plt.bar(np.arange(vars), values, width = 0.6, align="center", color = '#00A000', \

          label = "Feature Weight")

    plt.bar(np.arange(vars) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \

          label = "Cumulative Feature Weight")

    plt.xticks(np.arange(vars), columns)

    plt.ylabel("Weight", fontsize = 12)

    plt.xlabel("Feature", fontsize = 12)

    plt.legend(loc = 'upper center')

    plt.tight_layout()

    plt.show()
clf = BaggingClassifier()

clf.fit(X, y)



importances = np.mean([

    tree.feature_importances_ for tree in clf.estimators_

], axis=0)



show_importances(importances,8)
### Let's distinguish by target to see the distributions of the features by target

target1 = modeldata[modeldata['target']==1]

target0 = modeldata[modeldata['target']==0]
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,4))



ax1.hist(target1['oldpeak'],int(modeldata['oldpeak'].max()),alpha = 0.5, label = 'with heart disease')

ax1.hist(target0['oldpeak'],int(modeldata['oldpeak'].max()),alpha = 0.5, label = 'without heart disease')

ax1.title.set_text('oldpeak')

ax1.legend(loc=0)



ax2.hist(target1['ca'],int(modeldata['ca'].max()),alpha = 0.5, label = 'with heart disease')

ax2.hist(target0['ca'],int(modeldata['ca'].max()),alpha = 0.5, label = 'without heart disease')

ax2.title.set_text('ca')

ax2.legend(loc=0);
modeldata.corr()['target'].sort_values().head(5)
cp0 = modeldata[['cp_0','target']]

cp0['combined'] = modeldata['cp_0'] * modeldata['target']



print('{}% of all had heart disease%'.format(round(df['target'].sum()/df.shape[0]*100,0)))

print('{}% of those with cp_0 had heart disease%'.format(round(39/143*100,0)))

cp0.sum()
pd.merge(pd.get_dummies(df['thal']), df[['target']], left_index = True, right_index = True, how = 'inner').corr()['target']
modeldata.corr()['target'].sort_values().head(5)
print('{}% of those with thal_2 had heart disease%'.format(round(130/166*100,0)))

thal2 = modeldata[['thal_2','target']]

thal2['combined'] = modeldata['thal_2'] * modeldata['target']

thal2.sum()
modeldata.corr()['target'].sort_values(ascending = False).head(10)
pd.merge(pd.get_dummies(df['thal']), df[['target']], left_index = True, right_index = True, how = 'inner').corr()['target']
clf = RandomForestClassifier()

clf.fit(X,y)



importances = np.mean([

    tree.feature_importances_ for tree in clf.estimators_

], axis=0)



show_importances(importances,8)
sns.kdeplot(data = target1['thalach'],color = 'red',shade = True)#,bw=True)

sns.kdeplot(data = target0['thalach'],color = 'blue',shade = True)#, bw = True)

plt.show()    
clf = AdaBoostClassifier()

clf.fit(X,y)



importances = np.mean([

    tree.feature_importances_ for tree in clf.estimators_

], axis=0)

    

show_importances(importances,6)    
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16,4))



sns.kdeplot(data = target1['age'],color = 'red',shade = True, ax = ax1)#,bw=True)

sns.kdeplot(data = target0['age'],color = 'blue',shade = True, ax = ax1)#, bw = True)



sns.kdeplot(data = target1['chol'],color = 'red',shade = True, ax = ax2)#,bw=True)

sns.kdeplot(data = target0['chol'],color = 'blue',shade = True, ax = ax2)#, bw = True)



sns.kdeplot(data = target1['trestbps'],color = 'red',shade = True, ax = ax3)#,bw=True)

sns.kdeplot(data = target0['trestbps'],color = 'blue',shade = True, ax = ax3)#, bw = True)

plt.show()    
modeldata[['target','age','chol','trestbps']].corr()['target'].sort_values()
eclf1.fit(X, y)



importances = np.mean([

    tree.feature_importances_ for tree in clf.estimators_

], axis=0)



show_importances(importances,8)
modeldata[['target','age','chol','trestbps']].corr()['target'].sort_values()