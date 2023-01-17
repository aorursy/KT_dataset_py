#importing all the necessary libraries



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve,auc,precision_recall_fscore_support

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score

import csv
df=pd.read_csv("../input/train.csv")

#dropping the ID code, which is useless for our classification task

df=df.drop(['ID_code'],axis=1)



#Y is a list in which all the target values are stored

Y=[]



for i in range(len(df)):

    Y.append(df.iloc[i]['target'])
df['target'].value_counts().plot.bar()

#dropping the target, because it is not an explainatory feature

df=df.drop(['target'],axis=1)



#printing the correlation matrix among the features

print(df.corr())
#getting all the column headers of the dataset

index_list=df.columns.values



X=[]

aux_list=[]



for i in range(0,len(df)):

    for j in range(len(index_list)):

        aux_list.append(df.iloc[i][index_list[j]])

    X.append(aux_list)

    aux_list=[]

    

#X is a list of lists where all the values of explainatory variables are stored. It contains all the values

#for each row of our training set
#splitting into train and validation set

X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.33,shuffle=False)



#scaling the features using the standard scaler (so that variables have mean=0 and std=1)

scaler=StandardScaler()

X_train_std=scaler.fit_transform(X_train)

X_val_std=scaler.transform(X_val)
clf_nb=GaussianNB()

clf_nb.fit(X_train_std,y_train)
clf_mlp=MLPClassifier()

clf_mlp.fit(X_train_std,y_train)
#clf_svm=SVC(kernel='rbf',probability=True)

#clf_svm.fit(X_train_std,y_train)
clf_rf=RandomForestClassifier()

clf_rf.fit(X_train_std,y_train)
y_scores_nb=clf_nb.predict_proba(X_val_std)

y_scores_mlp=clf_mlp.predict_proba(X_val_std)

y_scores_rf=clf_rf.predict_proba(X_val_std)



#y_scores_svm=clf_svm.predict_proba(X_val_std)
false_positive_rate_nb, true_positive_rate_nb, thresholds_nb = roc_curve(y_val,y_scores_nb[:,1],pos_label=1)

false_positive_rate_mlp, true_positive_rate_mlp, thresholds_mlp = roc_curve(y_val,y_scores_mlp[:,1],pos_label=1) 

false_positive_rate_rf, true_positive_rate_rf, thresholds_rf = roc_curve(y_val,y_scores_rf[:,1],pos_label=1)



#false_positive_rate_svm, true_positive_rate_svm, thresholds_svm = roc_curve(y_val,y_scores_svm[:,1],pos_label=1)
roc_auc_nb=auc(false_positive_rate_nb,true_positive_rate_nb)

roc_auc_mlp=auc(false_positive_rate_mlp,true_positive_rate_mlp)

roc_auc_rf=auc(false_positive_rate_rf,true_positive_rate_rf)



#roc_auc_svm=auc(false_positive_rate_svm,true_positive_rate_svm)
print("Area under curve of Naive Bayes:",roc_auc_nb)

print(" ")

print("Area under curve of MultiLayer Perceptron:",roc_auc_mlp)

print(" ")

print("Area under curve of Random Forest:",roc_auc_rf)



#print("Area under curve of Support Vector Machine:",roc_auc_svm)

#print(" ")
plt.title('Receiver Operating Characteristic Comparison')

plt.plot(false_positive_rate_nb,true_positive_rate_nb, 'b', label = 'AUC GNB = %0.2f' % roc_auc_nb)

plt.plot(false_positive_rate_mlp,true_positive_rate_mlp, 'r', label = 'AUC MLP = %0.2f' % roc_auc_mlp)

plt.plot(false_positive_rate_rf,true_positive_rate_rf, 'g', label = 'AUC RF = %0.2f' % roc_auc_rf)



#plt.plot(false_positive_rate_svm,true_positive_rate_svm, 'y', label = 'AUC SVM = %0.2f' % roc_auc_svm)



plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

df_test=pd.read_csv("../input/test.csv")



id_codes=df_test['ID_code'].tolist()



df_test=df_test.drop(['ID_code'],axis=1)
index_list_new=df_test.columns.values



X_test=[]

aux_list_new=[]



for i in range(len(df_test)):

    for j in range(len(index_list_new)):

        aux_list_new.append(df_test.iloc[i][index_list_new[j]])

    X_test.append(aux_list_new)

    aux_list_new=[]



X_test_std=scaler.transform(X_test)
y_scores_test_nb=clf_nb.predict_proba(X_test_std)[:,1]

y_test_pred=clf_nb.predict(X_test_std)
submission = pd.DataFrame({"ID_code": id_codes})

submission["target"] = y_scores_test_nb

submission.to_csv("Sample_Submission.csv",index=False)