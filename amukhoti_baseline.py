import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV,cross_val_score

from sklearn.feature_selection import SelectFromModel

import math as m

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import figure

import matplotlib.patches as patches

from scipy import interp

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')



df_train.isnull().sum()
test_index=df_test['Unnamed: 0'] #copying test index for later

df_train.head()

df_test.head()



train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']

test_X = df_test.loc[:, 'V1':'V16']

print(df_train['Class'].value_counts())



cvtrain_X,cvtest_X,cvtrain_y,cvtest_y = train_test_split(train_X,train_y,test_size=0.01,random_state = 2)

# To account for imbalanced dataset.Did not lead to any improment

smote = SMOTE(random_state = 0)

rus = RandomUnderSampler(random_state = 0)

ntrain_X,ntrain_y = smote.fit_sample(train_X,train_y)

mtrain_X,mtrain_y = rus.fit_sample(train_X,train_y)

print(ntrain_X.shape)

print(type(ntrain_y))

ntrain_X = pd.DataFrame(data = ntrain_X[0:,0:],

                        index = [i for i in range(ntrain_X.shape[0])],

                        columns = ['V'+str(i+1) for i in range(ntrain_X.shape[1])])

ntrain_y = pd.DataFrame(data = ntrain_y[:],

                        columns = ['Class'])



xgb = XGBClassifier(eta= 0.02, max_depth= 7, subsample= 0.4, colsample_bytree = 0.7, objective= 'binary:logistic',n_jobs = -1)





results = cross_val_score(xgb,train_X,train_y)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

xgb.fit(train_X,train_y)

predict = xgb.predict(cvtest_X)

print(confusion_matrix(predict,cvtest_y))

print(classification_report(predict,cvtest_y))

print(xgb)

#Attempt with Random Forest

"""

rftrain = RandomForestClassifier(n_estimators = 300,max_features = 4,oob_score = True, min_samples_leaf = 80,n_jobs = -1,random_state = 10)

rftrain.fit(cvtrain_X,cvtrain_y)

print(rftrain)

cvtest_predict = rftrain.predict(cvtest_X)

print(accuracy_score(cvtest_y,cvtest_predict))

print(classification_report(cvtest_y,cvtest_predict))

print(confusion_matrix(cvtest_predict,cvtest_y))



from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,confusion_matrix



#rf = RandomForestClassifier()

#random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 600, num = 15)]

               'max_features': ['auto', 'sqrt']

               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]

               'min_samples_split':  [2, 5, 10]

               'min_samples_leaf': [1, 2, 4,8,16,32,64,128]

               'bootstrap': [True]}

#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 70, cv = 5, verbose=2, random_state=0, n_jobs = -1)# Fit the random search model

#rf_random.fit(train_X, train_y)

#print(rf_random.best_params_)

#print(rf_random)

"""





       

        

    

    

# Tried to plot n_estimators vs accuracy graph

"""

feat_labels = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16']

rf.fit(train_X, train_y)

predictions = []

for tree in rf.estimators_:

    predictions.append(tree.predict_proba(cvtest_X)[None,:])

predictions = np.vstack(predictions)

mean = np.cumsum(predictions, axis = 0)/np.arange(1, predictions.shape[0] +1)[:,None,None]

scores = []

for pred in mean:

    scores.append(accuracy_score(cvtest_y,np.argmax(pred,axis = 1)))

figure.Figure(figsize =(10,2))

plt.plot(scores,linewidth = 3)

plt.xlabel('num_trees')

plt.ylabel('accuracy')



"""







df_test = df_test.loc[:, 'V1':'V16']

pred = xgb.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)