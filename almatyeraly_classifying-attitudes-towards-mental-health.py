import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from collections import defaultdict



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, chi2



data = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')

data.head()
data = data.drop(columns=['self_employed'])

data = data[data.Country.map(data.Country.value_counts()) >= 10]
data2 = data[data.mental_health_consequence != 'Maybe']



y = data.mental_health_consequence

y2 = data2.mental_health_consequence



data = data.drop(columns=['mental_health_consequence', 'Timestamp', 'state','comments', 'Age', 'Gender'])

data2 = data2.drop(columns=['mental_health_consequence', 'Timestamp', 'state','comments', 'Age', 'Gender'])



X = data

X2 = data2
f,ax = plt.subplots(3,2,figsize=(10,10))



X2.Country.value_counts().plot(kind='bar', ax=ax[0,0])

ax[0,0].set_title('Frequency of Countries')



y.value_counts().plot(kind='bar', ax=ax[0,1])

ax[0,1].set_title('Do you think that discussing a mental health issue \n with your employer would have negative consequences?')



X2.supervisor.value_counts().plot(kind='bar', ax=ax[1,0])

ax[1,0].set_title('Would you be willing to discuss a mental health issue \n with your direct supervisor(s)?')



X2.coworkers.value_counts().plot(kind='bar', ax=ax[1,1])

ax[1,1].set_title('Would you be willing to discuss a mental health issue \n with your coworkers?')



X2.obs_consequence.value_counts().plot(kind='bar', ax=ax[2,0])

ax[2,0].set_title('Have you heard of or observed negative consequences for coworkers \n with mental health conditions in your workplace?')

ax[2,0].title.set_size(10)



X2.anonymity.value_counts().plot(kind='bar', ax=ax[2,1])

ax[2,1].set_title('Is your anonymity protected if you choose to take advantage of \n mental health or substance abuse treatment resources?')

ax[2,1].title.set_size(10)



f.tight_layout()
dict1 = {'No':0,

            'Maybe':1,

             'Yes':2

            }

y = y.map(dict1)



dict2 = {'No':0,

            'Yes':1,

            }

y2 = y2.map(dict2)



X.work_interfere = X.work_interfere.fillna('Never')

encoder_dict = defaultdict(LabelEncoder)

X = X.apply(lambda a: encoder_dict[a.name].fit_transform(a))

X2 = X2.apply(lambda a: encoder_dict[a.name].fit_transform(a.astype(str)))
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=20)

fit = bestfeatures.fit(X2,y2)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(20,'Score'))  #print 20 best features
X2 = X2.drop(columns=featureScores.Feature[featureScores.Score < 10].values)
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=1)

print('Train', X_train.shape, y_train.shape)

print('Test', X_test.shape, y_test.shape)
model = OneVsRestClassifier(SVC())

model.fit(X_train, y_train)

predict = model.predict(X_test)

acc1 = accuracy_score(y_test, predict)



print(confusion_matrix(y_test, predict))

print('Accuracy: ', acc1)
model = OneVsOneClassifier(LinearSVC())

model.fit(X_train, y_train)

y_pred1 = model.predict(X_test)

acc = accuracy_score(y_test, y_pred1)



print(confusion_matrix(y_test, y_pred1))

print('Accuracy: ', acc)
model = LinearSVC(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)



print(confusion_matrix(y_test, y_pred))

print('Accuracy: ', acc)
# generate binary values using get_dummies

Xdum = pd.get_dummies(X2, columns=X2.columns )



X_train, X_test, y_train, y_test = train_test_split(Xdum, y2, test_size=0.25, random_state=1)



model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)



print(confusion_matrix(y_test, y_pred))

print('Accuracy:', acc)
max_depths = np.linspace(1, 15, 15, endpoint=True)

accs = []

for maxd in max_depths:

    model = DecisionTreeClassifier(max_depth = maxd)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accs.append(accuracy_score(y_test, pred))



print('Best max_depth: ', max_depths[accs.index(max(accs))])

print('Accuracy with best max_depth: ', max(accs))
X_train, X_test, y_train, y_test = train_test_split(Xdum, y2, test_size=0.25, random_state=1)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, pred))
estimators = np.linspace(1, 200, 10, endpoint=True, dtype='int')

accs_est = []

for n in estimators:

    model = RandomForestClassifier(n_estimators=n)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accs_est.append(accuracy_score(y_test, pred))

    

print('Best estimator: ', estimators[accs_est.index(max(accs_est))])

print('Accuracy with best estimator: ', max(accs_est))
max_depths = np.linspace(1, 32, 32, endpoint=True)

accs_maxd = []

for n in max_depths:

    model = RandomForestClassifier(max_depth=n, n_estimators = estimators[accs_est.index(max(accs_est))])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accs_maxd.append(accuracy_score(y_test, pred))

    

print('Best max_depth: ', max_depths[accs_maxd.index(max(accs_maxd))])

print('Accuracy with best estimator and max_depth: ', max(accs_maxd))
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=20)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(20,'Score'))  #print 20 best features
X = X.drop(columns=featureScores.Feature[featureScores.Score < 10].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print('Train', X_train.shape, y_train.shape)

print('Test', X_test.shape, y_test.shape)
model = OneVsRestClassifier(SVC())

model.fit(X_train, y_train)

predict = model.predict(X_test)

acc5 = accuracy_score(y_test, predict)



print(confusion_matrix(y_test, predict))

print('Accuracy: ', acc5)
model = OneVsOneClassifier(LinearSVC())

model.fit(X_train, y_train)

y_pred1 = model.predict(X_test)

acc7 = accuracy_score(y_test, y_pred1)



print(confusion_matrix(y_test, y_pred1))

print('Accuracy: ', acc7)
model = LinearSVC(random_state=0)

model.fit(X_train, y_train)

y_pred2 = model.predict(X_test)

acc4 = accuracy_score(y_test, y_pred2)



print(confusion_matrix(y_test, y_pred2))

print('Accuracy: ', acc4)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pr = model.predict(X_test)

a = accuracy_score(y_test, y_pr)



print(confusion_matrix(y_test, y_pr))

print('Accuracy:', a)
max_depths = np.linspace(1, 15, 15, endpoint=True)

accs = []

for maxd in max_depths:

    model = DecisionTreeClassifier(max_depth = maxd)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accs.append(accuracy_score(y_test, pred))



print('Best max_depth: ', max_depths[accs.index(max(accs))])

print('Accuracy with best max_depth: ', max(accs))
model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, pred))
estimators = np.linspace(1, 200, 10, endpoint=True, dtype='int')

accs_est = []

for n in estimators:

    model = RandomForestClassifier(n_estimators=n)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accs_est.append(accuracy_score(y_test, pred))

    

print('Best estimator: ', estimators[accs_est.index(max(accs_est))])

print('Accuracy with best estimator: ', max(accs_est))
max_depths = np.linspace(1, 32, 32, endpoint=True)

accs_maxd = []

for n in max_depths:

    model = RandomForestClassifier(max_depth=n, n_estimators = estimators[accs_est.index(max(accs_est))])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accs_maxd.append(accuracy_score(y_test, pred))

    

print('Best max_depth: ', max_depths[accs_maxd.index(max(accs_maxd))])

print('Accuracy with best estimator and max_depth: ', max(accs_maxd))