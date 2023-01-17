#Loading necessary libraries

import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import normalize

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import time

import os

import matplotlib.pyplot as plt

import seaborn as sns
#Loading the data

os.chdir('/kaggle/input/world-values-survey-wave-6')

wvs = pd.read_csv("wvs.csv.bz2",sep="\t")

wvs
abortion = wvs[(wvs.V204 > 0)]

abortion.V204.describe()
#Removing negative values from V2 as well from abortion, and drop missing values (NA) from entire dataset

wvs_ab = abortion[(abortion.V2 > 0)].dropna()

wvs_ab.shape
wvs_ab['abortion'] = [1 if x > 3 else 0 for x in wvs_ab['V204']]

wvs_ab.abortion.describe()
wvs_ab_corr_all = wvs_ab.corr(method="pearson")[['abortion']].sort_values(by='abortion',ascending=False)

wvs_ab_corr_all
wvs_ab_corr = wvs_ab_corr_all[(wvs_ab_corr_all.abortion > 0.4) | (wvs_ab_corr_all.abortion < -0.4)]

#Not seeing the first and second row related to abortion

wvs_ab_corr[2::]
wvs_ab = wvs_ab.rename(columns={"V2":"country"})

wvs_ab_d = pd.get_dummies(wvs_ab,columns=['country']) #This step removes the country variable

wvs_ab_d.shape
#Number of columns with names starting with country - dummies we created == Unique countries in original dataset

wvs_ab_d[wvs_ab_d.columns[pd.Series(wvs_ab_d.columns).str.startswith('country')]].shape[1] == wvs_ab.country.unique().size
wvs_dummy = wvs_ab_d.drop(['country_887','V204'],axis=1)

#Also dropping the V204 column - responses for abortion - from the dataframe

wvs_dummy.shape
def kcv(k,unfit_m,X,y):

    indices = X.index.values

    i_shuffle = shuffle(indices)

    f1=[]

    accuracy=[]

    rmse=[]

    for i in np.arange(k):

        v = i_shuffle[i::k]

        X_valid = X.loc[v,:]

        X_train = X[~X.index.isin(X_valid.index)]

        y_valid = y.loc[v]

        y_train = y[~y.index.isin(y_valid.index)]

        m = unfit_m.fit(X_train,y_train)

        y_predict = m.predict(X_valid)

        f1.append(f1_score(y_valid,y_predict,average='weighted'))

        accuracy.append(accuracy_score(y_valid,y_predict))

        rmse.append(np.sqrt(np.mean([np.square(m - n) for m,n in zip(y_valid,y_predict)])))

    return (np.mean(f1),np.mean(accuracy),np.mean(rmse))
#Picking a sample of 7000 observations to avoid forever run

wvs_sample = wvs_dummy.sample(n=10000,random_state=1)

X_sample = wvs_sample.loc[:, wvs_sample.columns != 'abortion']

y_sample = wvs_sample['abortion']

#X and y for the entire dataset

X = wvs_dummy.loc[:, wvs_dummy.columns != 'abortion']

y = wvs_dummy['abortion']
#Create a structure to store accuracy and F-scores

mycolumns = ['model','accuracy','f-score','RMSE','runtime']

models = pd.DataFrame(columns=mycolumns)

models.set_index('model')
k = 5

start_time = time.clock()

knn_5 = KNeighborsClassifier(n_neighbors=k)

#5 fold cross validation for sample of original data

f1_knn_5,accuracy_knn_5,rmse_knn_5 = kcv(5,knn_5,X_sample,y_sample)

print("F1-score :",f1_knn_5)

print("Accuracy :",accuracy_knn_5)

models.loc[len(models)] = ['knn, k=5',accuracy_knn_5,f1_knn_5,rmse_knn_5,time.clock() - start_time]
k = 3

start_time = time.clock()

knn = KNeighborsClassifier(n_neighbors=k)

#5 fold cross validation for original data

f1_knn_3,accuracy_knn_3,rmse_knn_3 = kcv(5,knn,X_sample,y_sample)

print("F1-score :",f1_knn_3)

print("Accuracy :",accuracy_knn_3)

models.loc[len(models)] = ['knn, k=3',accuracy_knn_3,f1_knn_3,rmse_knn_3,time.clock() - start_time]
k = 7

start_time = time.clock()

knn = KNeighborsClassifier(n_neighbors=k)

#5 fold cross validation for original data

f1_knn_7,accuracy_knn_7,rmse_knn_7 = kcv(5,knn,X_sample,y_sample)

print("F1-score :",f1_knn_7)

print("Accuracy :",accuracy_knn_7)

models.loc[len(models)] = ['knn, k=7',accuracy_knn_7,f1_knn_7,rmse_knn_7,time.clock() - start_time]
k = 9

start_time = time.clock()

knn = KNeighborsClassifier(n_neighbors=k)

#5 fold cross validation for original data

f1_knn_9,accuracy_knn_9,rmse_knn_9 = kcv(5,knn,X_sample,y_sample)

print("F1-score :",f1_knn_9)

print("Accuracy :",accuracy_knn_9)

models.loc[len(models)] = ['knn, k=9',accuracy_knn_9,f1_knn_9,rmse_knn_9,time.clock() - start_time]
k = 13

start_time = time.clock()

knn = KNeighborsClassifier(n_neighbors=k)

#5 fold cross validation for original data

f1_knn_13,accuracy_knn_13,rmse_knn_13 = kcv(5,knn,X_sample,y_sample)

print("F1-score :",f1_knn_13)

print("Accuracy :",accuracy_knn_13)

models.loc[len(models)] = ['knn, k=13',accuracy_knn_13,f1_knn_13,rmse_knn_13,time.clock() - start_time]
models.sort_values(by=['accuracy','f-score'],ascending=False)
start_time = time.clock()

logreg = LogisticRegression(random_state=0)

#5 fold cross validation

f1_log,accuracy_log,rmse_log = kcv(5,logreg,X_sample,y_sample)

print("F1-score :",f1_log)

print("Accuracy :",accuracy_log)

models.loc[len(models)] = ['logistic regression',accuracy_log,f1_log,rmse_log,time.clock() - start_time]
start_time = time.clock()

svm_linear = SVC(kernel='linear', gamma='auto')

#5 fold cross validation

f1_svm_lin,accuracy_svm_lin,rmse_svm_lin = kcv(5,svm_linear,X_sample,y_sample)

print("F1-score :",f1_svm_lin)

print("Accuracy :",accuracy_svm_lin)

models.loc[len(models)] = ['svm, linear',accuracy_svm_lin,f1_svm_lin,rmse_svm_lin,time.clock() - start_time]
start_time = time.clock()

#Rbf kernel with gamma=5

svm_radial = SVC(kernel='rbf', gamma=5)

#5 fold cross validation

f1_svm_rad_5,accuracy_svm_rad_5,rmse_svm_rad = kcv(5,svm_radial,X_sample,y_sample)

print("F1-score :",f1_svm_rad_5)

print("Accuracy :",accuracy_svm_rad_5)

models.loc[len(models)] = ['svm, radial, y=5',accuracy_svm_rad_5,f1_svm_rad_5,rmse_svm_rad,time.clock() - start_time]
start_time = time.clock()

#Rbf kernel with gamma=10

svm_radial = SVC(kernel='rbf', gamma=10)

#5 fold cross validation

f1_svm_rad_10,accuracy_svm_rad_10,rmse_svm_rad_10 = kcv(5,svm_radial,X_sample,y_sample)

print("F1-score :",f1_svm_rad_10)

print("Accuracy :",accuracy_svm_rad_10)

models.loc[len(models)] = ['svm, radial, y=10',accuracy_svm_rad_10,f1_svm_rad_10,rmse_svm_rad_10,time.clock() - start_time]
start_time = time.clock()

#Polynomial kernel with degree=2

svm_poly_2 = SVC(kernel='poly', gamma='auto',degree=2)

#5 fold cross validation

f1_svm_poly_2,accuracy_svm_poly_2,rmse_svm_poly_2 = kcv(5,svm_poly_2,X_sample,y_sample)

print("F1-score :",f1_svm_poly_2)

print("Accuracy :",accuracy_svm_poly_2)

models.loc[len(models)] = ['svm, polynomial, d=2',accuracy_svm_poly_2,f1_svm_poly_2,rmse_svm_poly_2,time.clock() - start_time]
start_time = time.clock()

#Polynomial kernel with degree=3

svm_poly_3 = SVC(kernel='poly', gamma='auto',degree=3)

#5 fold cross validation

f1_svm_poly_3,accuracy_svm_poly_3,rmse_svm_poly_3 = kcv(5,svm_poly_3,X_sample,y_sample)

print("F1-score :",f1_svm_poly_3)

print("Accuracy :",accuracy_svm_poly_3)

models.loc[len(models)] = ['svm, polynomial, d=3',accuracy_svm_poly_3,f1_svm_poly_3,rmse_svm_poly_3,time.clock() - start_time]
start_time = time.clock()

#Polynomial kernel with degree=8

svm_poly_8 = SVC(kernel='poly', gamma='auto',degree=8)

#5 fold cross validation

f1_svm_poly_8,accuracy_svm_poly_8,rmse_svm_poly_8 = kcv(5,svm_poly_8,X_sample,y_sample)

print("F1-score :",f1_svm_poly_8)

print("Accuracy :",accuracy_svm_poly_8)

models.loc[len(models)] = ['svm, polynomial, d=8',accuracy_svm_poly_8,f1_svm_poly_8,rmse_svm_poly_8,time.clock() - start_time]
start_time = time.clock()

#Sigmoid kernel with gamma=5

svm_sig_5 = SVC(kernel='sigmoid', gamma=5)

#5 fold cross validation

f1_svm_sig_5,accuracy_svm_sig_5,rmse_svm_sig_5 = kcv(5,svm_sig_5,X_sample,y_sample)

print("F1-score :",f1_svm_sig_5)

print("Accuracy :",accuracy_svm_sig_5)

models.loc[len(models)] = ['svm, sigmoid, y=5',accuracy_svm_sig_5,f1_svm_sig_5,rmse_svm_sig_5,time.clock() - start_time]
start_time = time.clock()

rdf = RandomForestClassifier(n_estimators=100)

#5 fold cross validation

f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X_sample,y_sample)

print("F1-score :",f1_rf)

print("Accuracy :",accuracy_rf)

models.loc[len(models)] = ['random forest',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]
models.sort_values(by=['accuracy','f-score'],ascending=False)
start_time = time.clock()

rdf = RandomForestClassifier(n_estimators=100)

#5 fold cross validation

f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X,y)

print("F1-score :",f1_rf)

print("Accuracy :",accuracy_rf)

models.loc[len(models)] = ['random forest - all',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]
rdf.fit(X,y)

feature_imp = pd.Series(rdf.feature_importances_,index=X.columns).sort_values(ascending=False)
%matplotlib inline

# Creating a bar plot

sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
feature_imp.sort_values(ascending=False).head(12)
X_imp = X[['V205','V203','V206','V203A','V207','V207A','V152','V210','V202','V9','V145','V200']]

start_time = time.clock()

rdf = RandomForestClassifier(n_estimators=100)

#5 fold cross validation

f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X_imp,y)

print("F1-score :",f1_rf)

print("Accuracy :",accuracy_rf)

models.loc[len(models)] = ['random forest - important',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]
#Drop the columns starting with country_ from sample of X

X_nocountry = X[X.columns.drop(list(X.filter(regex='country_')))]
start_time = time.clock()

rdf = RandomForestClassifier(n_estimators=100)

#5 fold cross validation

f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X_nocountry,y)

print("F1-score :",f1_rf)

print("Accuracy :",accuracy_rf)

models.loc[len(models)] = ['random forest - no country',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]