##Import the Libraries and classifiers as needed for various models

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler as ss

from sklearn.decomposition import PCA
###import sklearn metrics to determine model characteristcs



from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support
#Read Dataset and check for any missing variables across columns

df = pd.read_csv("../input/data.csv")
df.shape  ## (569,33)
df.isnull().sum()
#Drop columns which are not needed based on the summary from above



df = df.drop(['id','Unnamed: 32'], axis=1)
df
#####Seperating the features/Predictors



X = df.loc[:, df.columns != 'diagnosis']
X.shape
# 3.1 Separating out the target



y = df.loc[:,['diagnosis']].values
y.shape         # (569,1)
## Map values in ' y ' (target) from 'M' and 'B' to 1 and 0, as integers,



df['diagnosis'].value_counts()

y[y=='M'] = 1

y[y=='B'] = 0
y=y.astype('int64')
y
####Scale all numerical features in X  using sklearn's StandardScaler class

scale = ss()

X = scale.fit_transform(X)
X.shape
#####Perform PCA on numeric features, X. Use sklearn's PCA class. Only retain as many principal components (PCs) 

##as needed to achieve 95% variance

pca = PCA()

out = pca.fit_transform(X)
out.shape
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()
###Based on cumsum, first 10 PCs result in 95% Variance, hence we can drop rest of the columns to get new_X

X = out[:,:10]
y = y.ravel() # for appropriate sizing
####Split X and y into test and train datasets at 80:20 ratio  

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size = 0.2,

                                                    shuffle = True

                                                    )
X_train.shape
X_test.shape
###Create the Default Classifiers using various models



dt = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=100)

et = ExtraTreesClassifier(n_estimators=100)

gbm = GradientBoostingClassifier()

xg = XGBClassifier(learning_rate=0.5,

                   reg_alpha= 5,

                   reg_lambda= 0.1)

knn = KNeighborsClassifier()
#fit the train data into each of these models



dt1 = dt.fit(X_train,y_train)

rf1 = rf.fit(X_train,y_train)

et1 = et.fit(X_train,y_train)

gbm1 = gbm.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)

knn1 = knn.fit(X_train,y_train)
####Use these trained models on actual test data and check the predictions

y_pred_dt = dt1.predict(X_test)

y_pred_rf = rf1.predict(X_test)

y_pred_et = et1.predict(X_test)

y_pred_gbm = gbm1.predict(X_test)

y_pred_xg = xg1.predict(X_test)

y_pred_knn = knn1.predict(X_test)
###Probability Values for each of these models

y_pred_dt_prob = dt1.predict_proba(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_et_prob = et1.predict_proba(X_test)

y_pred_gbm_prob = gbm1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)

y_pred_knn_prob = knn1.predict_proba(X_test)
#Calucate accuracy of these models for comparision

print("The Accuracy Score of Decision Tree Classifier is")

print(accuracy_score(y_test,y_pred_dt))

print("The Accuracy Score of Random Forest Classifier is")

print(accuracy_score(y_test,y_pred_rf))

print("The Accuracy Score of Extra Trees Classifier is")

print(accuracy_score(y_test,y_pred_et))

print("The Accuracy Score of Gradient Boosting Machine Classifier is")

print(accuracy_score(y_test,y_pred_gbm))

print("The Accuracy Score of XG Boost Classifier is")

print(accuracy_score(y_test,y_pred_xg))

print("The Accuracy Score of KNeighbors Classifier is")

print(accuracy_score(y_test,y_pred_knn))
#Determine the Confusion matrix for these models

print("The Confusion Matrix of Decision Tree Classifier is")

print(confusion_matrix(y_test,y_pred_dt))

print("The Confusion Matrix of Random Forest Classifier is")

print(confusion_matrix(y_test,y_pred_rf))

print("The Confusion Matrix of Extra Trees Classifier is")

print(confusion_matrix(y_test,y_pred_et))

print("The Confusion Matrix of Gradient Boosting Machine Classifier is")

print(confusion_matrix(y_test,y_pred_gbm))

print("The Confusion Matrix of XG Boost Classifier is")

print(confusion_matrix(y_test,y_pred_xg))

print("The Confusion Matrix of KNeighbors Classifier is")

print(confusion_matrix(y_test,y_pred_knn))
# Determine and print the Precision, Recall & F-score values

p_dt,r_dt,f_dt,_ = precision_recall_fscore_support(y_test,y_pred_dt)

print(" Precision, Recall and F-Score values of Decision Tree Classifier are ", p_dt,r_dt,f_dt)

p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf)

print(" Precision, Recall and F-Score values of Random Forest Classifier are ", p_rf,r_rf,f_rf)

p_et,r_et,f_et,_ = precision_recall_fscore_support(y_test,y_pred_et)

print(" Precision, Recall and F-Score values of Extra Trees Classifier are ", p_et,r_et,f_et)

p_gbm,r_gbm,f_gbm,_ = precision_recall_fscore_support(y_test,y_pred_gbm)

print(" Precision, Recall and F-Score values of Gradient Boosting Machine Classifier are ", p_gbm,r_gbm,f_gbm)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg)

print(" Precision, Recall and F-Score values of XG Boost Classifier are ", p_xg,r_xg,f_xg)

p_knn,r_knn,f_knn,_ = precision_recall_fscore_support(y_test,y_pred_knn)

print(" Precision, Recall and F-Score values of KNeighbors Classifier are ", p_knn,r_knn,f_knn)
# Plotting Graph



Classifier_models = [(dt, "decisiontree"), (rf, "randomForest"), (et, "extratrees"), (gbm, "gradientboost"), (xg,"xgboost"), (knn,"Kneighbors")]



#  Plotting the ROC curve





fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111)



# Connecting diagonals and specifying Labels,Title



ax.plot([0, 1], [0, 1], ls="--")

ax.set_title('ROC curve for models')

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')





# Setting x,y axes limits



ax.set_xlim([0.0, 1.0])



ax.set_ylim([0.0, 1.0])



AUC = []

for clf,name in Classifier_models:

    clf.fit(X_train,y_train)

    y_pred_prob = clf.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test,

                                     y_pred_prob[: , 1],

                                     pos_label= 1

                                     )

    AUC.append((auc(fpr,tpr)))

    ax.plot(fpr, tpr, label = name)



ax.legend(loc="lower right")

plt.show()
print("The AUC values for DT Classifier, RF Classifier, ET Classifier, GBM Classifier, XG Boost Classifier, KNeighbors Classifier respectively are :",AUC)
print(" Gradient Boost Classifier model has the best performance")
print("Observation made from plot : The slower the increase in False Positive Rate from 0, the larger is the Area Under Curve")