# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from IPython.display import HTML



import warnings

warnings.filterwarnings('ignore')



class bold:

   START = '\033[1m'

   END = '\033[0m'



print('\t'+'\t'+bold.START+ 'What Is Your Myers Briggs Personality Type?' + bold.END)



# Youtube

HTML("<iframe width='560' height='315' src='https://www.youtube.com/embed/M4YLO-2Tb2w/' frameborder='0' allowfullscreen></iframe>")



#Source from: https://gist.github.com/christopherlovell/e3e70880c0b0ad666e7b5fe311320a62
#Importing the required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#Libraries for data processing

from sklearn.feature_extraction.text import TfidfVectorizer



#Libraries for modeling. We will assess the performance of various models and choose the best one

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier



#Libraries for model assessment

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report



import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')
#Exploring the shape and content of the data

print(df_train.shape)

print(df_train.head())
#Visualising the distribution of the personality types in our dataset

mbti_types = df_train[['type', 'posts']].groupby('type').count()

mbti_types.sort_values('posts', ascending=False).plot(kind='bar')
#Obtain the categorical indicators from the personality types

df_train['Mind'] = df_train['type'].str[0]

df_train['Energy'] = df_train['type'].str[1]

df_train['Nature'] = df_train['type'].str[2]

df_train['Tactics'] = df_train['type'].str[3]



#Map the categorical indicators to either a 1 or 0 according to the submission format

df_train = df_train.replace({'Mind' : {'I':0, 'E':1},

                             'Energy' : {'S':0, 'N':1},

                             'Nature' : {'F':0, 'T':1},

                             'Tactics' : {'P':0, 'J':1}})



#Remove the type column from df_train as this information is now encoded into the four different categories

df_train = df_train.drop('type', axis = 1)
#Visualising the distribution of each category

df_train[['Mind','Energy','Nature','Tactics']].sum().plot(kind='bar')



#Determining the proportion of the classes in each category

print(df_train['Mind'].sum()/len(df_train))

print(df_train['Energy'].sum()/len(df_train))

print(df_train['Nature'].sum()/len(df_train))

print(df_train['Tactics'].sum()/len(df_train))
df_test = pd.read_csv('../input/test.csv')



#This variable will be used to keep track of the train and test datasets in the combined dataset

len_train = len(df_train)



#Combining the train and test datasets

df_all = pd.concat([df_train, df_test], sort=False)
#Confirming that the datasets were correctly combined

print(df_all.shape)
#Create a TFIDF sparce matrix of the top 10 000 features. ngrams of 2 are also included in the sparce matrix. 

tfv = TfidfVectorizer(stop_words='english',max_features=10000,lowercase=True,max_df=0.75,ngram_range=(1,2))

X = tfv.fit_transform(df_all['posts'])
#Extracting the training set from the sparce matrix

train_X = X[:len_train]
#Create predictor variables for each category

y_mind = df_train['Mind']

y_energy = df_train['Energy']

y_nature = df_train['Nature']

y_tactics = df_train['Tactics']
#Initializing model classifiers

lr = LogisticRegression(n_jobs = -1)

rf = RandomForestClassifier(n_jobs=-1)

adb = AdaBoostClassifier()

knn = KNeighborsClassifier(n_jobs=-1)
#Evaluating Accuracy scores



##Logistic Regression

print('Logistic Regression\n')

scores = cross_val_score(lr, train_X, y_mind, cv=5)

print("Accuracy (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_energy, cv=5)

print("Accuracy (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_nature, cv=5)

print("Accuracy (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_tactics, cv=5)

print("Accuracy (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##Random Forest

print('-'*30)

print('Random Forest\n')

scores = cross_val_score(rf, train_X, y_mind, cv=5)

print("Accuracy (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_energy, cv=5)

print("Accuracy (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_nature, cv=5)

print("Accuracy (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_tactics, cv=5)

print("Accuracy (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##AdaBoost

print('-'*30)

print('AdaBoost\n')

scores = cross_val_score(adb, train_X, y_mind, cv=5)

print("Accuracy (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_energy, cv=5)

print("Accuracy (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_nature, cv=5)

print("Accuracy (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_tactics, cv=5)

print("Accuracy (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##KNN

print('-'*30)

print('KNN\n')

scores = cross_val_score(knn, train_X, y_mind, cv=5)

print("Accuracy (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_energy, cv=5)

print("Accuracy (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_nature, cv=5)

print("Accuracy (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_tactics, cv=5)

print("Accuracy (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Evaluating F1 scores



##Logistic Regression

print('Logistic Regression\n')

scores = cross_val_score(lr, train_X, y_mind, cv=5, scoring='f1_macro')

print("F1 (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_energy, cv=5, scoring='f1_macro')

print("F1 (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_nature, cv=5, scoring='f1_macro')

print("F1 (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_tactics, cv=5, scoring='f1_macro')

print("F1 (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##Random Forest

print('-'*30)

print('Random Forest\n')

scores = cross_val_score(rf, train_X, y_mind, cv=5, scoring='f1_macro')

print("F1 (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_energy, cv=5, scoring='f1_macro')

print("F1 (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_nature, cv=5, scoring='f1_macro')

print("F1 (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_tactics, cv=5, scoring='f1_macro')

print("F1 (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##AdaBoost

print('-'*30)

print('AdaBoost\n')

scores = cross_val_score(adb, train_X, y_mind, cv=5, scoring='f1_macro')

print("F1 (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_energy, cv=5, scoring='f1_macro')

print("F1 (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_nature, cv=5, scoring='f1_macro')

print("F1 (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_tactics, cv=5, scoring='f1_macro')

print("F1 (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##KNN

print('-'*30)

print('KNN\n')

scores = cross_val_score(knn, train_X, y_mind, cv=5, scoring='f1_macro')

print("F1 (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_energy, cv=5, scoring='f1_macro')

print("F1 (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_nature, cv=5, scoring='f1_macro')

print("F1 (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_tactics, cv=5, scoring='f1_macro')

print("F1 (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Evaluating Nagative Log Loss scores



##Logistic Regression

print('Logistic Regression\n')

scores = cross_val_score(lr, train_X, y_mind, cv=5, scoring='neg_log_loss')

print("Log Loss (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_energy, cv=5, scoring='neg_log_loss')

print("Log Loss (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_nature, cv=5, scoring='neg_log_loss')

print("Log Loss (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_tactics, cv=5, scoring='neg_log_loss')

print("Log Loss (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##Random Forest

print('-'*30)

print('Random Forest\n')

scores = cross_val_score(rf, train_X, y_mind, cv=5, scoring='neg_log_loss')

print("Log Loss (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_energy, cv=5, scoring='neg_log_loss')

print("Log Loss (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_nature, cv=5, scoring='neg_log_loss')

print("Log Loss (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_tactics, cv=5, scoring='neg_log_loss')

print("Log Loss (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##AdaBoost

print('-'*30)

print('AdaBoost\n')

scores = cross_val_score(adb, train_X, y_mind, cv=5, scoring='neg_log_loss')

print("Log Loss (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_energy, cv=5, scoring='neg_log_loss')

print("Log Loss (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_nature, cv=5, scoring='neg_log_loss')

print("Log Loss (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_tactics, cv=5, scoring='neg_log_loss')

print("Log Loss (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##KNN

print('-'*30)

print('KNN\n')

scores = cross_val_score(knn, train_X, y_mind, cv=5, scoring='neg_log_loss')

print("Log Loss (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_energy, cv=5, scoring='neg_log_loss')

print("Log Loss (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_nature, cv=5, scoring='neg_log_loss')

print("Log Loss (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_tactics, cv=5, scoring='neg_log_loss')

print("Log Loss (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Evaluating AUC scores



##Logistic Regression

print('Logistic Regression\n')

scores = cross_val_score(lr, train_X, y_mind, cv=5, scoring='roc_auc')

print("AUC (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_energy, cv=5, scoring='roc_auc')

print("AUC (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_nature, cv=5, scoring='roc_auc')

print("AUC (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(lr, train_X, y_tactics, cv=5, scoring='roc_auc')

print("AUC (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##Random Forest

print('-'*30)

print('Random Forest\n')

scores = cross_val_score(rf, train_X, y_mind, cv=5, scoring='roc_auc')

print("AUC (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_energy, cv=5, scoring='roc_auc')

print("AUC (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_nature, cv=5, scoring='roc_auc')

print("AUC (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(rf, train_X, y_tactics, cv=5, scoring='roc_auc')

print("AUC (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##AdaBoost

print('-'*30)

print('AdaBoost\n')

scores = cross_val_score(adb, train_X, y_mind, cv=5, scoring='roc_auc')

print("AUC (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_energy, cv=5, scoring='roc_auc')

print("AUC (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_nature, cv=5, scoring='roc_auc')

print("AUC (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(adb, train_X, y_tactics, cv=5, scoring='roc_auc')

print("AUC (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##KNN

print('-'*30)

print('KNN\n')

scores = cross_val_score(knn, train_X, y_mind, cv=5, scoring='roc_auc')

print("AUC (Mind): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_energy, cv=5, scoring='roc_auc')

print("AUC (Energy): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_nature, cv=5, scoring='roc_auc')

print("AUC (Nature): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn, train_X, y_tactics, cv=5, scoring='roc_auc')

print("AUC (Tactics): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Extracting the test set from the combined vectorized set

test_X = X[len_train:]
#Initializing Logistic Regression estimators for the four different categories

lr_mind = LogisticRegression(n_jobs=-1)

lr_energy = LogisticRegression(n_jobs=-1)

lr_nature = LogisticRegression(n_jobs=-1)

lr_tactics = LogisticRegression(n_jobs=-1)
#Fitting the training data to the models for each cetegory

lr_mind.fit(train_X,y_mind)

lr_energy.fit(train_X,y_energy)

lr_nature.fit(train_X,y_nature)

lr_tactics.fit(train_X,y_tactics)
#Putting the predictions into a dataframe

submission = pd.DataFrame({'id': df_test.id.values, 'mind': lr_mind.predict(test_X), 'energy': lr_energy.predict(test_X), 'nature': lr_nature.predict(test_X), 'tactics': lr_tactics.predict(test_X)})
#Visualising the distributions of the predictions for the four categories

submission[['mind','energy','nature','tactics']].sum().plot(kind='bar')



#Determining the proportion of the predicted classes in each category

print(submission['mind'].sum()/len(df_test))

print(submission['energy'].sum()/len(df_test))

print(submission['nature'].sum()/len(df_test))

print(submission['tactics'].sum()/len(df_test))
#Predicting the probabilities for each user for each category and determining the average probablility for the different categories

mind_prob = lr_mind.predict_proba(test_X)

mind_t = mind_prob[:,0].mean()

energy_prob = lr_energy.predict_proba(test_X)

energy_t = energy_prob[:,0].mean()

nature_prob = lr_nature.predict_proba(test_X)

nature_t = nature_prob[:,0].mean()

tactics_prob = lr_tactics.predict_proba(test_X)

tactics_t = tactics_prob[:,0].mean()



#The average probability for the categories are used to determine the threshold for each category

mind_threshold = np.percentile(mind_prob[:,1], mind_t*100,)

energy_threshold = np.percentile(energy_prob[:,1], energy_t*100,)

nature_threshold = np.percentile(nature_prob[:,1], nature_t*100,)

tactics_threshold = np.percentile(tactics_prob[:,1], tactics_t*100,)



#The thresholds are used to scale the predictions

mind_pred_threshold = (mind_prob[:,1] > mind_threshold).astype(int)

energy_pred_threshold = (energy_prob[:,1] > energy_threshold).astype(int)

nature_pred_threshold = (nature_prob[:,1] > nature_threshold).astype(int)

tactics_pred_threshold = (tactics_prob[:,1] > tactics_threshold).astype(int)
#Putting the predictions into a dataframe

submission = pd.DataFrame({'id': df_test.id.values, 'mind': mind_pred_threshold, 'energy': energy_pred_threshold, 'nature': nature_pred_threshold, 'tactics': tactics_pred_threshold})
#Visualising the distributions of the scaled predictions for the four categories

submission[['mind','energy','nature','tactics']].sum().plot(kind='bar')



#Determining the proportion of the scaled predicted classes in each category

print(submission['mind'].sum()/len(df_test))

print(submission['energy'].sum()/len(df_test))

print(submission['nature'].sum()/len(df_test))

print(submission['tactics'].sum()/len(df_test))
#Creating file for submission

submission.to_csv('submission.csv', index=False)