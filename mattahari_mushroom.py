# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from visualization_utils import confussion_pies
df=pd.read_csv("../input/mushroom-classification/mushrooms.csv")

df.head()
df.isnull().sum()
df.columns
from sklearn import preprocessing

#Uncomment the next two lines for using label enconder on the dataset

#label_encoder = preprocessing.LabelEncoder()

#df = df.apply(label_encoder.fit_transform)



#Dummies

new_df = pd.DataFrame(df['class'])

for column in df.columns:

    if column != 'class':

        new_df = pd.concat([new_df, pd.get_dummies(df[column], prefix=column)],axis=1)



new_df['class'] = new_df['class'].map(lambda x : 1 if x == 'p' else 0)

df = new_df

# end of Dummies, comment these lines if you are using label encoders



df.head()
'''correlation_matrix =  df.corr().round(2)

plt.figure(figsize=(20,20))

sns.heatmap(data=correlation_matrix, annot=True)'''
from sklearn.model_selection import train_test_split



X = df.drop(['class'], axis=1)

y = pd.DataFrame(df['class'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)



y_pred  = clf.predict(X_test)

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g')
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(average_precision))
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
good_features = featureScores.nlargest(12,'Score').Specs.tolist()



best_X_train = X_train[good_features]

best_X_test = X_test[good_features]
best_clf = MultinomialNB()

best_clf.fit(best_X_train, y_train.values.ravel())

print('Score : ' + str(best_clf.score(best_X_train, y_train.values.ravel())))



best_y_pred  = best_clf.predict(best_X_test)



average_precision = average_precision_score(y_test, best_y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



cm = confusion_matrix(y_test, best_y_pred)

sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g')



confussion_pies(y_test['class'], best_y_pred)
from sklearn.naive_bayes import BernoulliNB

bernoulli = BernoulliNB()

bernoulli.fit(best_X_train, y_train.values.ravel())

print('Score : ' + str(bernoulli.score(best_X_train, y_train.values.ravel())))



bernoulli_y_pred  = bernoulli.predict(best_X_test)



average_precision = average_precision_score(y_test, bernoulli_y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



cm = confusion_matrix(y_test, bernoulli_y_pred)

sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g')



confussion_pies(y_test['class'], bernoulli_y_pred)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=4, max_depth=2,

                             random_state=0)

random_forest.fit(best_X_train, y_train.values.ravel())

print('Score : ' + str(random_forest.score(best_X_train, y_train.values.ravel())))



rf_y_pred  = random_forest.predict(best_X_test)



average_precision = average_precision_score(y_test, rf_y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



cm = confusion_matrix(y_test, rf_y_pred)

sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g')



confussion_pies(y_test['class'], rf_y_pred)
from sklearn import svm

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc = svm.SVC(gamma="scale")

gs_svc = GridSearchCV(svc, parameters, cv=5)

gs_svc.fit(best_X_train,  y_train.values.ravel())

print('Score : ' + str(gs_svc.score(best_X_train, y_train.values.ravel())))



gs_svc_pred  = random_forest.predict(best_X_test)

average_precision = average_precision_score(y_test, gs_svc_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



cm = confusion_matrix(y_test, gs_svc_pred)

sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g')



confussion_pies(y_test['class'], gs_svc_pred)