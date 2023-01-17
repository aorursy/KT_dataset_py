### DataFrame ####

import numpy as np

import pandas as pd



### Visualization ####

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



### Scikit ####

from sklearn.model_selection import train_test_split

np.random.seed(42)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale, StandardScaler

from sklearn.decomposition import PCA

from sklearn.datasets import fetch_openml

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score



### Others ###

import time

import warnings

warnings.filterwarnings('ignore')

# for kaggle kernel

train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")
# copy the datasets to avoid corrupting the orginaldataset. 



train = train_set.copy()

test = test_set.copy()
# train shape

print("train shape:",train.shape)



# Normalising data by dividing it by 255 should improve activation functions performance

y_train = train['label'].values

X_train = train.drop(columns=['label']).values/255

print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)
# test shape

print("test shape:",test.shape)



# y_test, to predict this ?

X_test = test.values/255

print("X_train shape:", X_test.shape)
train.describe()
test.describe()
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Model

rf = RandomForestClassifier(warm_start=True, n_jobs=-1, random_state=42)

rf
train_scores_a = []

train_best_score_a = 0

train_best_estimators_a = 0



# Time

start_time = time.time()



# estimator range 1,5,10....100

estimator_range = range(1, 100, 5)

for n_estimators in estimator_range:

    rf.n_estimators = n_estimators

    

    #fit the model

    rf.fit(X_train, y_train)

    

    #Train score

    train_score_a = rf.score(X_train, y_train)

    train_scores_a.append(train_score_a)

    if train_score_a>train_best_score_a:

        train_best_score_a = train_score_a

        train_best_estimators_a = n_estimators



print("Time taken:--- %s seconds ---" % (time.time() - start_time))

print("--------------------for Scenario A----------------------------------------")

print(" Train- best score :%s" %train_best_score_a)

print(" Train- Estimator :%s " %train_best_estimators_a)

# run the Model with best estimators

rft = RandomForestClassifier(warm_start=True, n_jobs=-1,n_estimators=train_best_estimators_a, random_state=42)

rft
# Time

start_time = time.time()

rft.fit(X_train, y_train)
# quick check on the accuracy score.

y_predict_train = rft.predict(X_train)



# time taken

print("Time taken:--- %s seconds ---" % (time.time() - start_time))

print(accuracy_score(y_train, y_predict_train))
y_predict_A = rft.predict(X_test) 
# let create our own kaggle accurancy score index.



# Submission csv by a topper in this competition. He/she accurracy is 1.Great job !

# y_test_kaggle = pd.read_csv('y_test_Kaggle.csv')

# y_test_kag_values = y_test_kaggle['Label'].values



# what this scenario's accurracy score when compared to the topper's submission?

# accuracy_score(y_predict_A,y_test_kag_values)
#sub = pd.read_csv('sample_submission.csv')

#sub['Label'] = y_predict_A

#sub.to_csv('submission_A.csv',index=False)
pca = PCA()

pca.fit(X_train)
# PCA features

features = range(pca.n_components_)

features
# number of intrinsic dimensions

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95)+1

d
# total variance of the intrinsic dimensions

np.sum(pca.explained_variance_ratio_)
# datasets after dimensionality reduction



pca = PCA(n_components=d)

pca.fit(X_train)

X_train_reduced = pca.transform(X_train)

X_test_reduced = pca.transform(X_test)



print(X_train_reduced.shape)

print(X_test_reduced.shape)
# Model

rf_pca = RandomForestClassifier(warm_start=True, n_jobs=-1, random_state=42)

rf_pca
train_scores_b = []

train_best_score_b = 0

train_best_estimators_b = 0



# Time

start_time = time.time()



# estimator range 1,5,10....100

estimator_range = range(1, 100, 5)

for n_estimators in estimator_range:

    rf_pca.n_estimators = n_estimators

    

    #fit the model

    rf_pca.fit(X_train_reduced, y_train)

    

    #Train score

    train_score_b = rf_pca.score(X_train_reduced, y_train)

    train_scores_b.append(train_score_b)

    if train_score_b>train_best_score_b:

        train_best_score_b = train_score_b

        train_best_estimators_b = n_estimators



print("Time taken:--- %s seconds ---" % (time.time() - start_time))

print("------------------for Scenario B------------------------------------------")

print(" Train- best score :%s" %train_best_score_b)

print(" Train- Estimator:%s " %train_best_estimators_b)
# Model with best estimators

rft_pca = RandomForestClassifier(warm_start=True, n_jobs=-1,n_estimators=train_best_estimators_b, random_state=42)

rft_pca
# Time

start_time = time.time()



rft_pca.fit(X_train_reduced, y_train)
# Qucik check on accuracy

y_predict_trainB = rft_pca.predict(X_train_reduced)

print("Time taken:--- %s seconds ---" % (time.time() - start_time))

print(accuracy_score(y_train, y_predict_trainB))
y_predict_B = rft_pca.predict(X_test_reduced) 
# let create our own kaggle accurancy score index.

# accuracy_score(y_predict_B,y_test_kag_values)
#sub = pd.read_csv('sample_submission.csv')

#sub['Label'] = y_predict_B

#sub.to_csv('submission_B.csv',index=False)
# accurancy score

accuracy_score(y_predict_A, y_predict_B)
plt.plot(estimator_range, train_scores_a, label="train scores A")

plt.plot(estimator_range, train_scores_b, label="train scores B")

plt.ylabel("accuracy")

plt.xlabel("n_estimators")

plt.legend()