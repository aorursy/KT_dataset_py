# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
cc_data = pd.read_csv(r"/kaggle/input/creditcardfraud/creditcard.csv")



cc_data.head()
cc_data.describe()
cc_data.shape
from sklearn.preprocessing import RobustScaler



rob_scaler = RobustScaler()



cc_data['Amount_Scaled'] = rob_scaler.fit_transform(cc_data['Amount'].values.reshape(-1,1))

cc_data['Time_Scaled_'] = rob_scaler.fit_transform(cc_data['Time'].values.reshape(-1,1))



cc_data.head()
fraud = cc_data[cc_data['Class'] == 1] 

valid = cc_data[cc_data['Class'] == 0] 

 

print('Number of Fraud Cases: {}'.format(len(cc_data[cc_data['Class'] == 1]))) 

print('Number of Valid Transactions: {}'.format(len(cc_data[cc_data['Class'] == 0])))
corrmat = cc_data.corr()

fig = plt.figure(figsize = (15, 10))



sns.heatmap(corrmat, vmax = .75, square = True)

plt.show()
data = cc_data.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_data = data.loc[data['Class'] == 1]

non_fraud_data = data.loc[data['Class'] == 0][:492]



data1 = pd.concat([fraud_data, non_fraud_data])



# Shuffle dataframe rows

new_data = data1.sample(frac=1, random_state=42)



new_data.drop(['Time','Amount'], axis=1, inplace=True)



new_data.head()
print(new_data['Class'].value_counts())
X=new_data.drop(['Class'], axis=1)

Y=new_data["Class"]



X_data=X.values

Y_data=Y.values
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)
from sklearn.metrics import classification_report, accuracy_score, precision_score,recall_score,f1_score,matthews_corrcoef

from sklearn.linear_model import  LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score
#Logistic Regression

lr_model = LogisticRegression(max_iter=500)



lr_model.fit(X_train, Y_train)

lr_y_pred = lr_model.predict(X_test)



lr_acc=accuracy_score(Y_test,lr_y_pred)

print(f'Accuracy is {(lr_acc)}')

lr_roc_auc=roc_auc_score(Y_test,lr_y_pred)

print(f'The ROC-AUC is {(lr_roc_auc)}')

lr_prec= precision_score(Y_test,lr_y_pred)

print(f"The precision is {(lr_prec)}")

lr_rec= recall_score(Y_test,lr_y_pred)

print(f"The recall is {(lr_rec)}")

lr_f1= f1_score(Y_test,lr_y_pred)

print(f"The F1-Score is {(lr_f1)}")

lr_MCC=matthews_corrcoef(Y_test,lr_y_pred)

print(f"The Matthews correlation coefficient is {(lr_MCC)}")
#Gaussian Naive Bayes

gau_model = GaussianNB()



gau_model.fit(X_train, Y_train)

gau_y_pred = gau_model.predict(X_test)



gau_acc=accuracy_score(Y_test,gau_y_pred)

print(f'Accuracy is {(gau_acc)}')

gau_roc_auc=roc_auc_score(Y_test,gau_y_pred)

print(f'The ROC_AUC score is {(gau_acc)}')

gau_prec= precision_score(Y_test,gau_y_pred)

print(f"The precision is {(gau_prec)}")

gau_rec= recall_score(Y_test,gau_y_pred)

print(f"The recall is {(gau_rec)}")

gau_f1= f1_score(Y_test,gau_y_pred)

print(f"The F1-Score is {(gau_f1)}")

gau_MCC=matthews_corrcoef(Y_test,gau_y_pred)

print(f"The Matthews correlation coefficient is {(gau_MCC)}")
#Random Forest

rfc = RandomForestClassifier()

rfc.fit(X_train,Y_train)



rfc_y_pred = rfc.predict(X_test)



rfc_acc=accuracy_score(Y_test,rfc_y_pred)

print(f'Accuracy is {(rfc_acc)}')

rfc_roc_auc=roc_auc_score(Y_test,rfc_y_pred)

print(f'The ROC-AUC score is {(rfc_roc_auc)}')

rfc_prec= precision_score(Y_test,rfc_y_pred)

print(f"The precision is {(rfc_prec)}")

rfc_rec= recall_score(Y_test,rfc_y_pred)

print(f"The recall is {(rfc_rec)}")

rfc_f1= f1_score(Y_test,rfc_y_pred)

print(f"The F1-Score is {(rfc_f1)}")

rfc_MCC=matthews_corrcoef(Y_test,rfc_y_pred)

print(f"The Matthews correlation coefficient is {(rfc_MCC)}")
#KNearest Neighbours

knn = KNeighborsClassifier()

knn.fit(X_train,Y_train)



knn_y_pred = knn.predict(X_test)



knn_acc=accuracy_score(Y_test,knn_y_pred)

print(f'The accuracy is {(knn_acc)}')

knn_roc_auc=roc_auc_score(Y_test,knn_y_pred)

print(f'The ROC-AUC score is {(knn_roc_auc)}')

knn_prec= precision_score(Y_test,knn_y_pred)

print(f"The precision is {(knn_prec)}")

knn_rec= recall_score(Y_test,knn_y_pred)

print(f"The recall is {(knn_rec)}")

knn_f1= f1_score(Y_test,knn_y_pred)

print(f"The F1-Score is {(knn_f1)}")

knn_MCC=matthews_corrcoef(Y_test,knn_y_pred)

print(f"The Matthews correlation coefficient is {(knn_MCC)}")
from sklearn.metrics import confusion_matrix



from matplotlib import cm



lr_cm = confusion_matrix(Y_test, lr_y_pred)

gau_cm = confusion_matrix(Y_test, gau_y_pred)

rfc_cm = confusion_matrix(Y_test, rfc_y_pred)

knn_cm = confusion_matrix(Y_test, knn_y_pred)



fig, ax = plt.subplots(2, 2,figsize=(22,12))





sns.heatmap(lr_cm , ax=ax[0][0], annot=True, cmap=plt.cm.get_cmap('YlGnBu'))

ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)

ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(gau_cm, ax=ax[0][1], annot=True, cmap=plt.cm.get_cmap('YlGnBu'))

ax[0][1].set_title("Gaussian Naive-Bayes \n Confusion Matrix", fontsize=14)

ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(rfc_cm, ax=ax[1][0], annot=True, cmap=plt.cm.get_cmap('YlGnBu'))

ax[1][0].set_title("Random Forest Classifier \n Confusion Matrix", fontsize=14)

ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(knn_cm, ax=ax[1][1], annot=True, cmap=plt.cm.get_cmap('YlGnBu'))

ax[1][1].set_title("K-Nearest Neighbours \n Confusion Matrix", fontsize=14)

ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)





plt.show()