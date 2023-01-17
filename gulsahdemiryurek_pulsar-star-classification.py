# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/pulsar_stars.csv")
data.info()
data = data.rename(columns={' Mean of the integrated profile':"mean_integrated_profile",

       ' Standard deviation of the integrated profile':"std_deviation_integrated_profile",

       ' Excess kurtosis of the integrated profile':"kurtosis_integrated_profile",

       ' Skewness of the integrated profile':"skewness_integrated_profile", 

        ' Mean of the DM-SNR curve':"mean_dm_snr_curve",

       ' Standard deviation of the DM-SNR curve':"std_deviation_dm_snr_curve",

       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dm_snr_curve",

       ' Skewness of the DM-SNR curve':"skewness_dm_snr_curve",

       })
data.head()
f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,linecolor="blue",fmt=".2f",ax=ax)

plt.show()
g = sns.pairplot(data, hue="target_class",palette="husl",diag_kind = "kde",kind = "scatter")
y = data["target_class"].values

x_data = data.drop(["target_class"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

lr_prediction = lr.predict(x_test)
from sklearn.metrics import mean_squared_error

mse_lr=mean_squared_error(y_test,lr_prediction)



from sklearn.metrics import confusion_matrix,classification_report

cm_lr=confusion_matrix(y_test,lr_prediction)

cm_lr=pd.DataFrame(cm_lr)

cm_lr["total"]=cm_lr[0]+cm_lr[1]

cr_lr=classification_report(y_test,lr_prediction)

from sklearn.metrics import cohen_kappa_score

cks_lr= cohen_kappa_score(y_test, lr_prediction)



score_and_mse={"model":["logistic regression"],"Score":[lr.score(x_test,y_test)],"Cohen Kappa Score":[cks_lr],"MSE":[mse_lr]}

score_and_mse=pd.DataFrame(score_and_mse)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors =13) # n_neighbors = k

knn.fit(x_train,y_train)

knn_prediction = knn.predict(x_test)
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
mse_knn=mean_squared_error(y_test,knn_prediction)

cm_knn=confusion_matrix(y_test,knn_prediction)

cm_knn=pd.DataFrame(cm_knn)

cr_knn=classification_report(y_test,knn_prediction)

cm_knn["total"]=cm_knn[0]+cm_knn[1]
from sklearn.metrics import cohen_kappa_score

cks_knn= cohen_kappa_score(y_test, knn_prediction)

score_and_mse = score_and_mse.append({'model': "knn classification","Score":knn.score(x_test,y_test),"Cohen Kappa Score":cks_knn,"MSE":mse_knn}, ignore_index=True)
from sklearn.svm import SVC

svm=SVC(random_state=1)

svm.fit(x_train,y_train)

svm_prediction=svm.predict(x_test)
mse_svm=mean_squared_error(y_test,svm_prediction)

svm_cm=confusion_matrix(y_test,svm_prediction)

cm_svm=pd.DataFrame(svm_cm)

cm_svm["total"]=cm_svm[0]+cm_svm[1]



cr_svm=classification_report(y_test,svm_prediction)

cks_svm= cohen_kappa_score(y_test, svm_prediction)

score_and_mse = score_and_mse.append({'model': "svm classification","Score":svm.score(x_test,y_test),"Cohen Kappa Score":cks_svm,"MSE":mse_svm}, ignore_index=True)
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

prediction_nb=nb.predict(x_test)
nb_mse=mean_squared_error(y_test,prediction_nb)

nb_cm=confusion_matrix(y_test,prediction_nb)

nb_cm=pd.DataFrame(nb_cm)

nb_cm["total"]=nb_cm[0]+nb_cm[1]



cr_nb=classification_report(y_test,prediction_nb)

cks_nb= cohen_kappa_score(y_test, prediction_nb)

score_and_mse = score_and_mse.append({'model': "naive bayes classification","Score":nb.score(x_test,y_test),"Cohen Kappa Score":cks_nb,"MSE":nb_mse}, ignore_index=True)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

prediction_dt=dt.predict(x_test)
dt_mse=mean_squared_error(y_test,prediction_dt)

dt_cm=confusion_matrix(y_test,prediction_dt)

dt_cm=pd.DataFrame(dt_cm)

dt_cm["total"]=dt_cm[0]+dt_cm[1]



cr_dt=classification_report(y_test,prediction_dt)

cks_dt= cohen_kappa_score(y_test, prediction_dt)

score_and_mse = score_and_mse.append({'model': "decision tree classification","Score":dt.score(x_test,y_test),"Cohen Kappa Score":cks_dt, "MSE":dt_mse}, ignore_index=True)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=1)

rf.fit(x_train,y_train)



prediction_rf=rf.predict(x_test)
rf_mse=mean_squared_error(y_test,prediction_rf)

rf_cm=confusion_matrix(y_test,prediction_rf)

rf_cm=pd.DataFrame(rf_cm)

rf_cm["total"]=rf_cm[0]+rf_cm[1]



cr_rf=classification_report(y_test,prediction_rf)

cks_rf= cohen_kappa_score(y_test, prediction_rf)
score_and_mse = score_and_mse.append({'model': "random forest classification","Score":rf.score(x_test,y_test),"Cohen Kappa Score":cks_rf,"MSE":rf_mse}, ignore_index=True)
print('Classification report for Logistic Regression: \n',cr_lr)

print('Classification report for KNN Classification: \n',cr_knn)

print('Classification report for SVM Classification: \n',cr_svm)

print('Classification report for Naive Bayes Classification: \n',cr_nb)

print('Classification report for Decision Tree Classification: \n',cr_dt)

print('Classification report for Random Forest Classification: \n',cr_rf)
f, axes = plt.subplots(2, 3,figsize=(18,12))

g1 = sns.heatmap(cm_lr,annot=True,fmt=".1f",cmap="flag",cbar=False,ax=axes[0,0])

g1.set_ylabel('y_true')

g1.set_xlabel('y_head')

g1.set_title("Logistic Regression")

g2 = sns.heatmap(cm_knn,annot=True,fmt=".1f",cmap="flag",cbar=False,ax=axes[0,1])

g2.set_ylabel('y_true')

g2.set_xlabel('y_head')

g2.set_title("KNN Classification")

g3 = sns.heatmap(cm_svm,annot=True,fmt=".1f",cmap="flag",ax=axes[0,2])

g3.set_ylabel('y_true')

g3.set_xlabel('y_head')

g3.set_title("SVM Classification")

g4 = sns.heatmap(nb_cm,annot=True,fmt=".1f",cmap="flag",cbar=False,ax=axes[1,0])

g4.set_ylabel('y_true')

g4.set_xlabel('y_head')

g4.set_title("Naive Bayes Classification")

g5 = sns.heatmap(dt_cm,annot=True,fmt=".1f",cmap="flag",cbar=False,ax=axes[1,1])

g5.set_ylabel('y_true')

g5.set_xlabel('y_head')

g5.set_title("Decision Tree Classification")

g6 = sns.heatmap(rf_cm,annot=True,fmt=".1f",cmap="flag",ax=axes[1,2])

g6.set_ylabel('y_true')

g6.set_xlabel('y_head')

g6.set_title("Random Forest Classification")



from sklearn.metrics import roc_curve

fpr_lr, tpr_lr, thresholds = roc_curve(y_test, lr_prediction)

plt.plot([0, 1], [0, 1], 'k--',color="grey")

plt.plot(fpr_lr, tpr_lr,color="red")

plt.title('Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
score_and_mse