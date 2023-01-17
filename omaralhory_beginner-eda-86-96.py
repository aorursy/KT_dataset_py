import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import missingno

import time

from sklearn import model_selection

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

import pickle
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

pd.options.display.max_columns = None
data.shape
cat_features = data.select_dtypes('object').columns

cat_features = list(cat_features)

cat_features
print(data['AGE_PERCENTIL'].unique())

print('\n',data['WINDOW'].unique())

data.fillna(0.0, inplace=True)
data.head()
data.drop('PATIENT_VISIT_IDENTIFIER', axis=1, inplace=True)
data_one_hot = pd.get_dummies(data,columns=cat_features)
data_one_hot.head()
X = data_one_hot.drop('ICU', axis=1)

Y = data_one_hot['ICU']
X = (X - np.min(X)) / np.std(X)
X.head(1)
def fit_algo(algo, x, y, cv):

    #Fit the model

    model = algo.fit(x, y)

    

    #Check its score

    acc = round(model.score(x, y) *100, 2)

    y_pred = model_selection.cross_val_predict(algo, x, y, cv=cv, n_jobs = -1)

    

    acc_cv = round(metrics.accuracy_score(Y,y_pred)*100, 2)

    

    return y_pred, acc, acc_cv, model
from sklearn.linear_model import LogisticRegression

start_time = time.time()

pred_now, acc_lr, acc_cv_lr, lr = fit_algo(LogisticRegression(C=0.1)

                                        , X, Y, 10)



lr_time = (time.time() - start_time)



print("Accuracy: %s" % acc_lr)

print("Accuracy of CV: %s" % acc_cv_lr)

print("Execution time: %s" % lr_time)
def feature_plot(imp):

    global X

    fimp = pd.DataFrame({'Feature': X.columns, 'Importance' : np.round(imp)})

    fimp =fimp.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10,10))

    plt.plot(fimp['Feature'], fimp['Importance'])

    plt.xticks(rotation=90);
feature_plot(lr.coef_[0])
fimp_lr = pd.DataFrame({'Feature': X.columns, 'Importance' : np.round(lr.coef_[0])})

fimp_lr =fimp_lr.sort_values(by='Importance', ascending=False)

fimp_lr
from sklearn.tree import DecisionTreeClassifier

start_time = time.time()

pred_now, acc_dt, acc_cv_dt, dt = fit_algo(DecisionTreeClassifier(random_state = 1)

                                        , X, Y, 10)



dt_time = (time.time() - start_time)



print("Accuracy: %s" % acc_dt)

print("Accuracy of CV: %s" % acc_cv_dt)

print("Execution time: %s" % dt_time)
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

pred_now, acc_rf, acc_cv_rf, rf = fit_algo(RandomForestClassifier(n_estimators = 100)

                                        , X, Y, 10)



rf_time = (time.time() - start_time)



print("Accuracy: %s" % acc_rf)

print("Accuracy of CV: %s" % acc_cv_rf)

print("Execution time: %s" % rf_time)

from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

pred_now, acc_rf2, acc_cv_rf2, rf2 = fit_algo(RandomForestClassifier(n_estimators = 100, criterion='entropy')

                                        , X, Y, 10)



rf2_time = (time.time() - start_time)



print("Accuracy: %s" % acc_rf2)

print("Accuracy of CV: %s" % acc_cv_rf2)

print("Execution time: %s" % rf2_time)
from sklearn.neural_network import MLPClassifier



start_time = time.time()

pred_now, acc_nn, acc_cv_nn, nn = fit_algo(MLPClassifier(hidden_layer_sizes = (50,10), activation='relu', solver='adam')

                                        , X, Y, 5)



nn_time = (time.time() - start_time)



print("Accuracy: %s" % acc_nn)

print("Accuracy of CV: %s" % acc_cv_nn)

print("Execution time: %s" % nn_time)
from sklearn.naive_bayes import GaussianNB

start_time = time.time()



pred_now, acc_gnb, acc_cv_gnb, gnb= fit_algo(GaussianNB()

                                        ,X,Y,5)



gnb_time = (time.time() - start_time)



print("Accuracy: %s" % acc_gnb)

print("Accuracy of CV: %s" % acc_cv_gnb)

print("Execution time: %s" % gnb_time)
from sklearn.ensemble import GradientBoostingClassifier

start_time = time.time()



pred_now, acc_gbt, acc_cv_gbt, gbt= fit_algo(GradientBoostingClassifier()

                                        , X, Y, 10)



gbt_time = (time.time() - start_time)



print("Accuracy: %s" % acc_gbt)

print("Accuracy of CV: %s" % acc_cv_gbt)

print("Execution time: %s" % gbt_time)
from sklearn.svm import LinearSVC

start_time = time.time()



pred_now, acc_svc, acc_cv_svc, svc= fit_algo(LinearSVC()

                                        ,X,Y,10)



svc_time = (time.time() - start_time)



print("Accuracy: %s" % acc_svc)

print("Accuracy of CV: %s" % acc_cv_svc)

print("Execution time: %s" % svc_time)
algo_name = ['Log. Reg.', 'Decision Tree', 'RandomForest Gini', 'RandomForest IG', 'Neural Network', 'Gaussian NB', 'GBC', 'SVM']

acc_df = pd.DataFrame({'Algorithm' : algo_name, 'Accuracy %' : [acc_cv_lr, acc_cv_dt, acc_cv_rf, acc_cv_rf2, acc_cv_nn, acc_cv_gnb, acc_cv_gbt, acc_cv_svc] })

acc_df = acc_df.sort_values(by='Accuracy %', ascending = False)

acc_df = acc_df.reset_index(drop=True)

acc_df
fimp_rf = pd.DataFrame({'Feature' : X.columns, 'Importance' : (gbt.feature_importances_*100).astype(float)})

fimp_rf = fimp_rf.sort_values(by='Importance', ascending=False)

fimp_rf
feature_plot(gbt.feature_importances_*100)
filename = 'gbt_covid_icu.sav'

pickle.dump(rf2, open(filename, 'wb'))