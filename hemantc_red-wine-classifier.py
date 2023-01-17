import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from pandas.plotting import scatter_matrix

print('Import Complete')

#wine_data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
#wine_data  = pd.read_csv('D:\\Hemant\\Machine_Learning_Self_Study\\datasets\\red-wine-quality-cortez-et-al-2009\\winequality-red.csv')

wine_data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine_data.head()
wine_data.info() # to check number of missing values
wine_data.columns.to_list()
wine_data.plot.bar(x = 'fixed acidity', y = 'quality')
attributes = wine_data.columns.to_list()
scatter_matrix(wine_data[attributes], figsize=(30,30))
wine_data['quality'].value_counts()
from sklearn.model_selection import train_test_split
attr = wine_data.columns.to_list()[:-1]
X = np.array(wine_data.iloc[:,:-1])

y = np.array(wine_data.iloc[:,-1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 23)
y_train
y_train_qual = (y_train >= 7)

y_test_qual = (y_test >=7)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=23)
sgd_clf.fit(X_train, y_train_qual)
from sklearn.tree import DecisionTreeClassifier
dct_clf = DecisionTreeClassifier(random_state = 23)

dct_clf.fit(X_train, y_train_qual)
from sklearn.ensemble import RandomForestClassifier
rndm_frst_clf = RandomForestClassifier(random_state = 23)

rndm_frst_clf.fit(X_train, y_train_qual)
from sklearn.svm import LinearSVC

svm_clf = LinearSVC(random_state = 23)

svm_clf.fit(X_train, y_train_qual)

from sklearn.linear_model import LogisticRegression



log_reg_clf = LogisticRegression(random_state = 23)

log_reg_clf.fit(X_train, y_train_qual)
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

X_scaled = std_scaler.fit_transform(X)
print(X)

print(X_scaled)
X_train_svm,X_test_svm, y_train, y_test = train_test_split(X_scaled, y, random_state = 23, test_size = 0.2)



svm_clf_scaled = LinearSVC(random_state = 23)
svm_clf_scaled.fit(X_train_svm, y_train_qual)
from sklearn.metrics import confusion_matrix
#COnfusion Matrix for Train

print("*************Training Accuracies************")

print("Confusion Matrix of SGDClassifier:\n",confusion_matrix(y_train_qual, sgd_clf.predict(X_train)), end = '\n')

print("Confusion Matrix of DecisionTreeClassifer:\n",confusion_matrix(y_train_qual, dct_clf.predict(X_train)), end = '\n')

print("Confusion Matrix of RandomForestClassifier:\n",confusion_matrix(y_train_qual, rndm_frst_clf.predict(X_train)), end = '\n')

print("Confusion Matrix of LinearSVC:\n",confusion_matrix(y_train_qual, svm_clf.predict(X_train)), end = '\n')

print("Confusion Matrix of LinearSVC_Scaled:\n",confusion_matrix(y_train_qual, svm_clf_scaled.predict(X_train_svm)), end = '\n')

print("Confusion Matrix of LogisticRegression:\n",confusion_matrix(y_train_qual, log_reg_clf.predict(X_train)), end = '\n')







print("Train Precision: Accuracy on Train for SGDClassifier")

print("The Precision is {}".format(precision_score(y_train_qual, sgd_clf.predict(X_train))))

print("The Recall is {}".format(recall_score(y_train_qual, sgd_clf.predict(X_train))))

print("The F1 Score is {}".format(f1_score(y_train_qual, sgd_clf.predict(X_train))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_train_qual, sgd_clf.predict(X_train))), end = '\n\n')



print("Train Precision: Accuracy on Train for DecisionTreeClassifer")

print("The Precision is {}".format(precision_score(y_train_qual, dct_clf.predict(X_train))))

print("The Recall is {}".format(recall_score(y_train_qual, dct_clf.predict(X_train))))

print("The F1 Score is {}".format(f1_score(y_train_qual, dct_clf.predict(X_train))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_train_qual, dct_clf.predict(X_train))), end = '\n\n')





print("Train Precision: Accuracy on Train for RandomForestClassifier")

print("The Precision is {}".format(precision_score(y_train_qual, rndm_frst_clf.predict(X_train))))

print("The Recall is {}".format(recall_score(y_train_qual, rndm_frst_clf.predict(X_train))))

print("The F1 Score is {}".format(f1_score(y_train_qual, rndm_frst_clf.predict(X_train))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_train_qual, rndm_frst_clf.predict(X_train))), end = '\n\n')



print("Train Precision: Accuracy on Train for Linear SVC -- Unscaled")

print("The Precision is {}".format(precision_score(y_train_qual, svm_clf.predict(X_train))))

print("The Recall is {}".format(recall_score(y_train_qual, svm_clf.predict(X_train))))

print("The F1 Score is {}".format(f1_score(y_train_qual, svm_clf.predict(X_train))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_train_qual, svm_clf.predict(X_train))), end = '\n\n')



print("Train Precision: Accuracy on Train for Linear SVC -- Scaled")

print("The Precision is {}".format(precision_score(y_train_qual, svm_clf_scaled.predict(X_train_svm))))

print("The Recall is {}".format(recall_score(y_train_qual, svm_clf_scaled.predict(X_train_svm))))

print("The F1 Score is {}".format(f1_score(y_train_qual, svm_clf_scaled.predict(X_train_svm))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_train_qual, svm_clf_scaled.predict(X_train_svm))), end = '\n\n')



print("Train Precision: Accuracy on Train for LogisticRegression")

print("The Precision is {}".format(precision_score(y_train_qual, log_reg_clf.predict(X_train_svm))))

print("The Recall is {}".format(recall_score(y_train_qual, log_reg_clf.predict(X_train_svm))))

print("The F1 Score is {}".format(f1_score(y_train_qual, log_reg_clf.predict(X_train_svm))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_train_qual, log_reg_clf.predict(X_train_svm))), end = '\n\n')

y_pred_sgd = sgd_clf.predict(X_test)

y_pred_dct = dct_clf.predict(X_test)

y_pred_rndm_frst = rndm_frst_clf.predict(X_test)

y_pred_svm_unscaled = svm_clf.predict(X_test)

y_pred_svm_scaled = svm_clf_scaled.predict(X_test_svm)

#COnfusion Matrix for Test

print("*************Test Accuracies************")

print("Confusion Matrix of SGDClassifier:\n",confusion_matrix(y_test_qual, sgd_clf.predict(X_test)), end = '\n')

print("Confusion Matrix of DecisionTreeClassifer:\n",confusion_matrix(y_test_qual, dct_clf.predict(X_test)), end = '\n')

print("Confusion Matrix of RandomForestClassifier:\n",confusion_matrix(y_test_qual, rndm_frst_clf.predict(X_test)), end = '\n')

print("Confusion Matrix of LinearSVC:\n",confusion_matrix(y_test_qual, svm_clf.predict(X_test)), end = '\n')

print("Confusion Matrix of LinearSVC_Scaled:\n",confusion_matrix(y_test_qual, svm_clf_scaled.predict(X_test_svm)), end = '\n')

print("Confusion Matrix of LogisticRegression:\n",confusion_matrix(y_test_qual, log_reg_clf.predict(X_test)), end = '\n')







print("Test Precision: Accuracy on Test for SGDClassifier")

print("The Precision is {}".format(precision_score(y_test_qual, sgd_clf.predict(X_test))))

print("The Recall is {}".format(recall_score(y_test_qual, sgd_clf.predict(X_test))))

print("The F1 Score is {}".format(f1_score(y_test_qual, sgd_clf.predict(X_test))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_test_qual, sgd_clf.predict(X_test))), end ='\n\n')





print("Test Precision: Accuracy on Test for DecisionTreeClassifer")

print("The Precision is {}".format(precision_score(y_test_qual, dct_clf.predict(X_test))))

print("The Recall is {}".format(recall_score(y_test_qual, dct_clf.predict(X_test))))

print("The F1 Score is {}".format(f1_score(y_test_qual, dct_clf.predict(X_test))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_test_qual, dct_clf.predict(X_test))), end ='\n\n')



print("Test Precision: Accuracy on Test for RandomForestClassifier")

print("The Precision is {}".format(precision_score(y_test_qual, rndm_frst_clf.predict(X_test))))

print("The Recall is {}".format(recall_score(y_test_qual, rndm_frst_clf.predict(X_test))))

print("The F1 Score is {}".format(f1_score(y_test_qual, rndm_frst_clf.predict(X_test))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_test_qual, rndm_frst_clf.predict(X_test))), end ='\n\n')



print("Test Precision: Accuracy on Test for LinearSVC - Unscaled")

print("The Precision is {}".format(precision_score(y_test_qual, svm_clf.predict(X_test))))

print("The Recall is {}".format(recall_score(y_test_qual, svm_clf.predict(X_test))))

print("The F1 Score is {}".format(f1_score(y_test_qual, svm_clf.predict(X_test))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_test_qual, svm_clf.predict(X_test))), end ='\n\n')



print("Test Precision: Accuracy on Test for LinearSVC - Scaled")

print("The Precision is {}".format(precision_score(y_test_qual, svm_clf_scaled.predict(X_test_svm))))

print("The Recall is {}".format(recall_score(y_test_qual, svm_clf_scaled.predict(X_test_svm))))

print("The F1 Score is {}".format(f1_score(y_test_qual, svm_clf_scaled.predict(X_test_svm))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_test_qual, svm_clf_scaled.predict(X_test_svm))), end ='\n\n')



print("Test Precision: Accuracy on Test for LogisticRegression")

print("The Precision is {}".format(precision_score(y_test_qual, log_reg_clf.predict(X_test))))

print("The Recall is {}".format(recall_score(y_test_qual, log_reg_clf.predict(X_test))))

print("The F1 Score is {}".format(f1_score(y_test_qual, log_reg_clf.predict(X_test))))

print("The ROC AUC Score is {}".format(roc_auc_score(y_test_qual, log_reg_clf.predict(X_test))), end ='\n\n')

def plot_roc_curve(fpr, tpr, label = None):

    plt.plot(fpr, tpr, label = label)

    plt.xlabel("False Postive Rate")

    plt.ylabel("True Positive Rate")

    plt.plot([0,1],[0,1], 'g--')

    plt.legend(loc = "lower right")

    
# None Scaled except for SVM

fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train_qual, sgd_clf.predict(X_train))

plot_roc_curve(fpr_sgd, tpr_sgd,"SGDClassifier")



fpr_dct, tpr_dct, thresholds_dct = roc_curve(y_train_qual, dct_clf.predict(X_train))

plot_roc_curve(fpr_dct, tpr_dct,"DecisionTreeClassifier")



fpr_rndm_frst, tpr_rndm_frst, thresholds_rndm_frst = roc_curve(y_train_qual, rndm_frst_clf.predict(X_train))

plot_roc_curve(fpr_rndm_frst, tpr_rndm_frst,"RandomForestClassifier")



fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_train_qual, svm_clf.predict(X_train))

plot_roc_curve(fpr_svm, tpr_svm,"LinearSVC  Unscaled")



fpr_svm_scaled, tpr_svm_scaled, thresholds_svm_scaled = roc_curve(y_train_qual, svm_clf_scaled.predict(X_train_svm))

plot_roc_curve(fpr_svm_scaled, tpr_svm_scaled,"LinearSVC  Scaled")



fpr_log, tpr_log, thresholds_log = roc_curve(y_train_qual, log_reg_clf.predict(X_train))

plot_roc_curve(fpr_log, tpr_log,"Logistics Regression")

# ALL scaled

sgd_clf.fit(X_train_svm,y_train_qual)

dct_clf.fit(X_train_svm,y_train_qual)

rndm_frst_clf.fit(X_train_svm,y_train_qual)

log_reg_clf.fit(X_train_svm,y_train_qual)









fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train_qual, sgd_clf.predict(X_train_svm))

plot_roc_curve(fpr_sgd, tpr_sgd,"SGDClassifier")



fpr_dct, tpr_dct, thresholds_dct = roc_curve(y_train_qual, dct_clf.predict(X_train_svm))

plot_roc_curve(fpr_dct, tpr_dct,"DecisionTreeClassifier")



fpr_rndm_frst, tpr_rndm_frst, thresholds_rndm_frst = roc_curve(y_train_qual, rndm_frst_clf.predict(X_train_svm))

plot_roc_curve(fpr_rndm_frst, tpr_rndm_frst,"RandomForestClassifier")



# fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_train_qual, svm_clf.predict(X_train_svm))

# plot_roc_curve(fpr_svm, tpr_svm,"LinearSVC  Unscaled")



fpr_svm_scaled, tpr_svm_scaled, thresholds_svm_scaled = roc_curve(y_train_qual, svm_clf_scaled.predict(X_train_svm))

plot_roc_curve(fpr_svm_scaled, tpr_svm_scaled,"LinearSVC  Scaled")



fpr_log, tpr_log, thresholds_log = roc_curve(y_train_qual, log_reg_clf.predict(X_train_svm))

plot_roc_curve(fpr_log, tpr_log,"Logistics Regression")

# Test ROC Curves -- NO scaling except for SVM

sgd_clf.fit(X_train,y_train_qual)

dct_clf.fit(X_train,y_train_qual)

rndm_frst_clf.fit(X_train,y_train_qual)

svm_clf.fit(X_train, y_train_qual)

svm_clf_scaled.fit(X_train_svm, y_train_qual)

log_reg_clf.fit(X_train,y_train_qual)





fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_test_qual, sgd_clf.predict(X_test))

plot_roc_curve(fpr_sgd, tpr_sgd,"SGDClassifier")



fpr_dct, tpr_dct, thresholds_dct = roc_curve(y_test_qual, dct_clf.predict(X_test))

plot_roc_curve(fpr_dct, tpr_dct,"DecisionTreeClassifier")



fpr_rndm_frst, tpr_rndm_frst, thresholds_rndm_frst = roc_curve(y_test_qual, rndm_frst_clf.predict(X_test))

plot_roc_curve(fpr_rndm_frst, tpr_rndm_frst,"RandomForestClassifier")



fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test_qual, svm_clf.predict(X_test))

plot_roc_curve(fpr_svm, tpr_svm,"LinearSVC  Unscaled")



fpr_svm_scaled, tpr_svm_scaled, thresholds_svm_scaled = roc_curve(y_test_qual, svm_clf_scaled.predict(X_test_svm))

plot_roc_curve(fpr_svm_scaled, tpr_svm_scaled,"LinearSVC  Scaled")



fpr_log, tpr_log, thresholds_log = roc_curve(y_test_qual, log_reg_clf.predict(X_test))

plot_roc_curve(fpr_log, tpr_log,"Logistics Regression")
# Test ROC Curves -- All Scaled

sgd_clf.fit(X_train_svm,y_train_qual)

dct_clf.fit(X_train_svm,y_train_qual)

rndm_frst_clf.fit(X_train_svm,y_train_qual)

#svm_clf.fit(X_train_svm, y_train_qual)

svm_clf_scaled.fit(X_train_svm, y_train_qual)

log_reg_clf.fit(X_train_svm,y_train_qual)





fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_test_qual, sgd_clf.predict(X_test_svm))

plot_roc_curve(fpr_sgd, tpr_sgd,"SGDClassifier")



fpr_dct, tpr_dct, thresholds_dct = roc_curve(y_test_qual, dct_clf.predict(X_test_svm))

plot_roc_curve(fpr_dct, tpr_dct,"DecisionTreeClassifier")



fpr_rndm_frst, tpr_rndm_frst, thresholds_rndm_frst = roc_curve(y_test_qual, rndm_frst_clf.predict(X_test_svm))

plot_roc_curve(fpr_rndm_frst, tpr_rndm_frst,"RandomForestClassifier")



# fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test_qual, svm_clf.predict(X_test))

# plot_roc_curve(fpr_svm, tpr_svm,"LinearSVC  Unscaled")



fpr_svm_scaled, tpr_svm_scaled, thresholds_svm_scaled = roc_curve(y_test_qual, svm_clf_scaled.predict(X_test_svm))

plot_roc_curve(fpr_svm_scaled, tpr_svm_scaled,"LinearSVC  Scaled")



fpr_log, tpr_log, thresholds_log = roc_curve(y_test_qual, log_reg_clf.predict(X_test_svm))

plot_roc_curve(fpr_log, tpr_log,"Logistics Regression")