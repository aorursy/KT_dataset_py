import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_validate

from sklearn.utils import shuffle

from sklearn import neighbors

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn import tree

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn import ensemble

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

pd.set_option('display.max_colwidth', -1)
train_df = pd.read_csv("/kaggle/input/mse-3-bb-ds-ws19-congressional-voting/CongressionalVotingID.shuf.train.csv")

test_df = pd.read_csv("/kaggle/input/mse-3-bb-ds-ws19-congressional-voting/CongressionalVotingID.shuf.test.csv")
test_df.head()
train_df.head()
test_id = pd.DataFrame(test_df['ID'])

test_df = test_df.drop(['ID'], axis=1)

target = pd.DataFrame(train_df['class'])

train_df = train_df.drop(['ID', 'class'], axis=1)
target.describe()

#60% democrat votes
train_df.describe()
test_df.describe()
classifiers = [

    neighbors.KNeighborsClassifier(5),

    neighbors.KNeighborsClassifier(10),

    neighbors.KNeighborsClassifier(15),

    GaussianNB(),

    Perceptron(tol=1e-3),

    tree.DecisionTreeClassifier(),

    tree.DecisionTreeClassifier(max_depth=3),

    ensemble.RandomForestClassifier(),

    ensemble.RandomForestClassifier(n_estimators=300),

    svm.SVC(),

    svm.LinearSVC()

]



classifiers_name = [

    '2-NN',

    '7-NN',

    '15-NN',

    'Naive Bayes',

    'Perceptron',

    'Full Decision Tree',

    'Decision Tree (max depth=3)',

    'Random Forest (n=100)',

    'Random Forest (n=300)',

    'SVC',

    'LinearSVC'

]
le = LabelEncoder()

train_encoded = train_df.apply(le.fit_transform)

test_encoded = test_df.apply(le.fit_transform)

target_encoded = target.apply(le.fit_transform)

train_encoded.head()
df_results = pd.DataFrame([],columns = ['Classifier', 'Accuracy', 'std'])

scoring = ['precision_micro', 'balanced_accuracy']

    #X_train, X_test, y_train, y_test = train_test_split(train_encoded, target_encoded, test_size=0.33)

for indexClassifier, classifier in enumerate(classifiers):

    scores = cross_validate(classifier, train_encoded, target_encoded.values.ravel(), cv=10, scoring=scoring)

    # Compute metrics

    acc = scores['test_balanced_accuracy']

    pre = scores['test_precision_micro']

    accuracy_str = str(round(acc.mean(),4)) 

    precision_str = str(round(pre.mean(),4))

    acc_std = acc.std() * 2

    df = pd.DataFrame([(classifiers_name[indexClassifier], accuracy_str, acc_std)], columns = ['Classifier', 'Accuracy', 'std'])

    df_results = df_results.append(df)
# Best Results per Iteration

#df_results[df_results['Accuracy'] == df_results.groupby(['Test'])['Accuracy'].transform('max')]

df_results
# Search for best score in Linear SVC

loss = ['hinge', 'squared_hinge']

C= list(np.arange(0.001,1, 0.01))

param_grid = { 

    'C': C,

    'loss': loss

}



cl = svm.LinearSVC(max_iter=10000)

CV_rfc = GridSearchCV(estimator=cl, param_grid=param_grid, cv= 10)

CV_rfc.fit(train_encoded, target_encoded.values.ravel())

print (CV_rfc.best_params_)

print (CV_rfc.best_score_)
pd.DataFrame(CV_rfc.cv_results_).loc[:, ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score').head()
# Search for best score in Linear SVC

loss = ['hinge', 'squared_hinge']

C= list(np.arange(1,100, 1))

param_grid = { 

    'C': C,

    'loss': loss

}



cl = svm.LinearSVC(max_iter=10000)

CV_rfc = GridSearchCV(estimator=cl, param_grid=param_grid, cv= 10)

CV_rfc.fit(train_encoded, target_encoded.values.ravel())

print (CV_rfc.best_params_)

print (CV_rfc.best_score_)
pd.DataFrame(CV_rfc.cv_results_).loc[:, ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score').head()
#Search for best scores in Random Forest

criterion = ['gini', 'entropy']

n_estimators= list(np.arange(10,500, 10))

max_features = ['auto', 'log2' ]

param_grid2 = { 

    'n_estimators': n_estimators,

    'criterion': criterion,

    'max_features': max_features

}

cl2 = ensemble.RandomForestClassifier()

CV_rfc2 = GridSearchCV(estimator=cl2, param_grid=param_grid2, cv= 10)

CV_rfc2.fit(train_encoded, target_encoded.values.ravel())

print (CV_rfc2.best_params_)

print (CV_rfc2.best_score_)
pd.DataFrame(CV_rfc2.cv_results_).loc[:, ['mean_test_score','std_test_score','rank_test_score', 'params']].sort_values(by='rank_test_score').head()
#Search for best scores in Decision Tree

criterion = ['gini', 'entropy']

max_features = ['auto', 'log2' ]

max_depth = list(np.arange(10,100, 10))

param_grid3 = { 

    'criterion': criterion,

    'max_features': max_features,

    'max_depth': max_depth

}

cl3 = tree.DecisionTreeClassifier()

CV_rfc3 = GridSearchCV(estimator=cl3, param_grid=param_grid3, cv= 10)

CV_rfc3.fit(train_encoded, target_encoded.values.ravel())

print (CV_rfc3.best_params_)

print (CV_rfc3.best_score_)
pd.DataFrame(CV_rfc3.cv_results_).loc[:, ['mean_test_score', 'std_test_score','rank_test_score', 'params']].sort_values(by='rank_test_score').head()
train_encoded_avg = train_encoded.replace(1, np.nan)

test_encoded_avg = test_encoded.replace(1, np.nan)

train_encoded_avg = train_encoded_avg.fillna(train_encoded_avg.mean())

test_encoded_avg = test_encoded_avg.fillna(test_encoded_avg.mean())
df_results_avg = pd.DataFrame([],columns = ['Classifier', 'Accuracy', 'std'])

scoring = ['precision_micro', 'balanced_accuracy']

for indexClassifier, classifier in enumerate(classifiers):

    scores = cross_validate(classifier, train_encoded_avg, target_encoded.values.ravel(), cv=10, scoring=scoring)

    # Compute metrics

    acc = scores['test_balanced_accuracy']

    pre = scores['test_precision_micro']

    accuracy_str = str(round(acc.mean(),4)) 

    precision_str = str(round(pre.mean(),4))

    acc_std = acc.std() * 2

    df = pd.DataFrame([(classifiers_name[indexClassifier], accuracy_str, acc_std)], columns = ['Classifier', 'Accuracy', 'std'])

    df_results_avg = df_results_avg.append(df)
df_results_avg
# Test with LinearSVC 1 - Default Parameters

c = svm.LinearSVC()

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted1 = c.predict(test_encoded)
# Test with LinearSVC 2 - C=0.05 & 10000 iterations

c = svm.LinearSVC(max_iter=10000, C=0.05)

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted2 = c.predict(test_encoded)
# Test with LinearSVC 3 - C=3

c = svm.LinearSVC(max_iter=10000, C=3)

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted10 = c.predict(test_encoded)
# Test with Random Forest 1 - Default Parameters

c = ensemble.RandomForestClassifier()

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted3 = c.predict(test_encoded)
# Test with Random Forest 2 - Gini & 30 trees & log2

c = ensemble.RandomForestClassifier(criterion='gini', n_estimators=30, max_features='log2')

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted4 = c.predict(test_encoded)
# Test with Random Forest 3 - Gini & 50 trees & auto

c = ensemble.RandomForestClassifier(criterion='gini', n_estimators=50, max_features='auto')

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted5 = c.predict(test_encoded)
# Test with Random Forest 4 -Information Gain & 10 trees & auto

c = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=10, max_features='auto')

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted6 = c.predict(test_encoded)
# Test with Random Forest 5 - Information Gain & 90 trees & log2

c = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=90, max_features='log2')

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted7= c.predict(test_encoded)
# Test with Decision Tree - Information Gain & Depth 50 & auto

c = tree.DecisionTreeClassifier(max_depth=50, criterion='entropy', max_features='auto')

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted8= c.predict(test_encoded)
# Test with Decision Tree - Information Gain & Depth 10 & log2

c = tree.DecisionTreeClassifier(max_depth=10, criterion='entropy', max_features='log2')

c.fit(train_encoded, target_encoded.values.ravel())

# predict

y_test_predicted9= c.predict(test_encoded)
#Test with Linear SVC and Average

c = svm.LinearSVC()

c.fit(train_encoded_avg, target_encoded.values.ravel())

# predict

y_test_predicted11= c.predict(test_encoded_avg)
df_result1 = pd.DataFrame(le.inverse_transform(y_test_predicted1))

df_result2 = pd.DataFrame(le.inverse_transform(y_test_predicted2))

df_result3 = pd.DataFrame(le.inverse_transform(y_test_predicted3))

df_result4 = pd.DataFrame(le.inverse_transform(y_test_predicted4))

df_result5 = pd.DataFrame(le.inverse_transform(y_test_predicted5))

df_result6 = pd.DataFrame(le.inverse_transform(y_test_predicted6))

df_result7 = pd.DataFrame(le.inverse_transform(y_test_predicted7))

df_result8 = pd.DataFrame(le.inverse_transform(y_test_predicted8))

df_result9 = pd.DataFrame(le.inverse_transform(y_test_predicted9))

df_result10 = pd.DataFrame(le.inverse_transform(y_test_predicted10))

df_result11 = pd.DataFrame(le.inverse_transform(y_test_predicted11))
merge1 = pd.concat([test_id, df_result1], axis=1)

merge1.columns = ['ID', 'class']

merge2 = pd.concat([test_id, df_result2], axis=1)

merge2.columns = ['ID', 'class']

merge3 = pd.concat([test_id, df_result3], axis=1)

merge3.columns = ['ID', 'class']

merge4 = pd.concat([test_id, df_result4], axis=1)

merge4.columns = ['ID', 'class']

merge5 = pd.concat([test_id, df_result5], axis=1)

merge5.columns = ['ID', 'class']

merge6 = pd.concat([test_id, df_result6], axis=1)

merge6.columns = ['ID', 'class']

merge7 = pd.concat([test_id, df_result7], axis=1)

merge7.columns = ['ID', 'class']

merge8 = pd.concat([test_id, df_result8], axis=1)

merge8.columns = ['ID', 'class']

merge9 = pd.concat([test_id, df_result9], axis=1)

merge9.columns = ['ID', 'class']

merge10 = pd.concat([test_id, df_result10], axis=1)

merge10.columns = ['ID', 'class']

merge11 = pd.concat([test_id, df_result11], axis=1)

merge11.columns = ['ID', 'class']
merge1.to_csv('LinearSVC_default.csv', index=False)

merge2.to_csv('LinearSVC_C0.5.csv', index=False)

merge3.to_csv('RandomForest_default.csv', index=False)

merge4.to_csv('RandomForest_Gini_30_log.csv', index=False)

merge5.to_csv('RandomForest_Gini_50_auto.csv', index=False)

merge6.to_csv('RandomForest_IG_10_auto.csv', index=False)

merge7.to_csv('RandomForest_IG_90_log.csv', index=False)

merge8.to_csv('DecisionTree_IG_50_auto.csv', index=False)

merge9.to_csv('DecisionTree_IG_10_log.csv', index=False)

merge10.to_csv('LinearSVC_C_3.csv', index=False)

merge11.to_csv('LinearSVC_avg.csv', index=False)
merge11.to_csv('LinearSVC_avg.csv', index=False)