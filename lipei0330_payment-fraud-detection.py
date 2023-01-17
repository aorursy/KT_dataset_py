import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #to create validation data set
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import confusion_matrix, recall_score,accuracy_score,f1_score, roc_curve, auc, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import graphviz 
import itertools

%matplotlib inline
sns.set_style("whitegrid")

payment = pd.read_csv("../input/payment.csv")
payment.head(5)
payment.describe()
print(payment.keys())
def nan_table(data):
    print(pd.isnull(data).sum())

nan_table(payment)
fig = plt.figure(figsize = (16,8))
sns.barplot(x = "step", y = "fraud", data = payment)
plt.title("Distribution of Fraud  based on step")
plt.show()
fig = plt.figure(figsize = (16,6))
sns.barplot(x = "age", y = "fraud", data = payment)
plt.title("Distribution of Fraud based on Age")
plt.show()
fig = plt.figure(figsize = (12,6))
sns.barplot(x = "gender", y = "fraud", data = payment)
plt.title("Distribution of Fraud based on Gender")
plt.show()
fig = plt.figure(figsize = (16,6))
sns.barplot(x="age", y="fraud", hue="gender", data=payment)
plt.title("Distribution of Fraud Based on Gender and Age")
plt.show()
fig = plt.figure(figsize = (16,6))
sns.barplot(x = "category", y = "fraud", data = payment)
plt.title("Distribution of Fraud based on Category")
plt.xticks(rotation = 90)
plt.show()
fig = plt.figure(figsize = (16,6))
sns.barplot(x = "merchant", y = "fraud", data = payment)
plt.title("Distribution of Fraud based on Merchant")
plt.xticks(rotation = 90)
plt.show()
payment.dtypes

# build a function to check each feature's unique values
# for the number of unique values is more than 20, we won't print each value
check_cols = payment.select_dtypes(include = ["object"])
def check_uniques(data):
    for col in check_cols:
        if len(data[col].unique().tolist()) <= 20:
            print(col, " : ", len(data[col].unique().tolist()), " : ", data[col].unique())
        else:
            print(col, " : ", len(data[col].unique().tolist()))
    
check_uniques(payment)
# use the labelEncoder to convert category values into numerical values
# need_trans_features = ["gender", "age", "category", "merchant"]
labelencoder_feature = LabelEncoder()
payment["gender"] = labelencoder_feature.fit_transform(payment["gender"])
payment["age"] = labelencoder_feature.fit_transform(payment["age"])
payment["merchant"] = labelencoder_feature.fit_transform(payment["merchant"])
payment["category"] = labelencoder_feature.fit_transform(payment["category"])
# check the changed data types
payment.dtypes
# drop the columns of zipcodeOri and zipMerchant, because they have only one constant value
# we also drop the column of "customer" which not used in our predictive model
payment_copy = payment.copy()
payment_copy.drop(labels = ["customer", "zipcodeOri", "zipMerchant"], axis = 1, inplace = True)
# transform the amount into category type,
# 5 types
amount_rank = ["vlow",
              "low",
              "mid",
              "high",
              "vhigh"]
payment_copy["amountRank"] = pd.qcut(x = payment_copy["amount"],q =len(amount_rank),labels = amount_rank)
payment_copy["stepRank"] = pd.qcut(x = payment_copy["step"],q =len(amount_rank),labels = amount_rank)
# drop columns of step and amount
payment_copy.drop(labels = ["step","amount"], axis = 1, inplace = True)
# change the category datatype into int
# vlow -- vhigh: 1-5
rank_dict = {"vlow":1, "low":2, "mid":3, "high":4, "vhigh":5}
payment_copy["stepRank"] = payment_copy["stepRank"].map(rank_dict)
payment_copy["amountRank"] = payment_copy["amountRank"].map(rank_dict)

payment_copy.head()
# To decide which features of the data to include in our predictive churn model
# We'll examine the correlation between churn and each customer feature

corr = payment_copy.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(16,10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
# Display the distribution of the sample
fig = plt.figure(figsize=(12, 4))
count_fraud = pd.value_counts(payment["fraud"], sort = True).sort_index()
print(count_fraud)
count_fraud.plot(kind = "bar")
plt.title("Fraud histogram")
plt.xlabel("Fraud")
plt.ylabel("Frequency")
plt.show()
# Split the training and testing dataset
features = ["age", "gender", "category", "stepRank", "amountRank", "merchant"]

x_train = payment_copy[features]
y_train = payment_copy["fraud"]
print(type(x_train))
# Standardized the dataset
scaler = StandardScaler()
# X = x_train.as_matrix().astype(np.float)
x_train = scaler.fit_transform(x_train)
print("after 1: ", type(x_train))
x_train = pd.DataFrame(x_train)
print("after 2: ", type(x_train))
#X_valid and y_valid are the validation sets
x_training, x_valid, y_training, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0) 
# fold has 5 pieces dataset, 1 piece for validation and 4 pieces for training
fold = KFold(x_training.shape[0], 5, shuffle = False)
# build a function to plot the confusion matrix
def plot_confusion_matrix(confusion_matrix):
    confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No Fraud', 'Fraud'), ('No Fraud', 'Fraud'))
    heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 10}, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 12)
#     plt.ylabel('True label', fontsize = 10)
#     plt.xlabel('Predicted label', fontsize = 10)
# build a function to print the scores after running servral rounds of cross validation
def print_scores_under_cv(model_name):
    recall_accs = []
    accuracy_accs = []
    f1_accs = []
    auc_accs = []
    for train,test in fold:
        # train is the indices of training dataset of x_training 
        model_name.fit(x_training.iloc[train,:], y_training.iloc[train])
        # predict values using the test indices in the training data
        y_pred = model_name.predict(x_training.iloc[test,:])      
        # calculate the recall and accuracy score and append it to a list 
        # for recall and accuracy scores representing the current C parameter
        recall_acc = recall_score(y_training.iloc[test], y_pred)
        accuracy_acc = accuracy_score(y_training.iloc[test], y_pred)
        f1_acc = f1_score(y_training.iloc[test], y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_training.iloc[test], y_pred)
        auc_acc = auc(fpr, tpr)
        recall_accs.append(recall_acc)
        accuracy_accs.append(accuracy_acc)
        f1_accs.append(f1_acc)
        auc_accs.append(auc_acc)
        
    recall_acc_mean = np.mean(recall_accs)
    accuracy_acc_mean = np.mean(accuracy_accs)
    f1_acc_mean = np.mean(f1_accs)
    auc_acc_mean = np.mean(auc_accs)
    print("")
    print("Mean of the recall score is: ", recall_acc_mean)
    print("Mean of the accuracy score is: ", accuracy_acc_mean)
    print("Mean of the f1 score is: ", f1_acc_mean)
    print("Mean of the auc score is: ", auc_acc_mean)
    print("")
# print the classifier report for each model
def classifier_report(model_name, x_train, y_train, x_test, y_test):
    model_name.fit(x_train, y_train)
    y_pred = model_name.predict(x_test)
    print(classification_report(y_test, y_pred))
# Check the Performances of different models under cross validation on imbalnaced dataset

lr = LogisticRegression()
dt = tree.DecisionTreeClassifier()
rf = RandomForestClassifier()
print("Logistic Regression Performance: ")
print_scores_under_cv(lr)
print("")
print("Decision Tree Performance: ")
print_scores_under_cv(dt)
print("")
print("Random Forest Performance: ")
print_scores_under_cv(rf)
print("")
    
# Check the classification report of different models
print("Logistic Regression: ")
classifier_report(lr, x_training, y_training, x_valid, y_valid)
print("Decision Tree: ")
classifier_report(dt, x_training, y_training, x_valid, y_valid)
print("Random Forest: ")
classifier_report(rf, x_training, y_training, x_valid, y_valid)
# Since we have more than 580000 of instances, we choose to under-sampling the dataset
 
data_majority = payment_copy[payment_copy["fraud"]==0]
data_minority = payment_copy[payment_copy["fraud"]==1]
# len(data_minority)
data_majority_undersampled = resample(data_majority,
                                      replace = True,
                                      n_samples = len(data_minority), # same number of samples as minority classe
                                      random_state = 1) # set the seed for random resampling

# Combine resampled results
data_undersampled = pd.concat([data_minority, data_majority_undersampled])
 
data_undersampled["fraud"].value_counts()
# Now that we have a 1:1 ratio for our classes, let’s train another logistic regression model
# Split the training and testing dataset
features = ["age", "gender", "category", "stepRank", "amountRank", "merchant"]

x_train_undersampled = data_undersampled[features]
y_train_undersampled = data_undersampled["fraud"]

# Standardized the dataset
scaler = StandardScaler()
# X = x_train.as_matrix().astype(np.float)
x_train_undersampled = scaler.fit_transform(x_train_undersampled)
x_train_undersampled = pd.DataFrame(x_train_undersampled)

#X_valid and y_valid are the validation sets
x_training_undersampled, x_valid_undersampled, y_training_undersampled, y_valid_undersampled = train_test_split(x_train_undersampled,
                                                                                                                y_train_undersampled, 
                                                                                                                test_size=0.2, 
                                                                                                                random_state=0) 
 
fold_undersampled = KFold(x_training_undersampled.shape[0], n_folds = 5, shuffle=False)
# build a function to print the scores after running servral rounds of cross validation based resampled dataset
def print_scores_under_cv_resample(model_name):
    recall_accs = []
    accuracy_accs = []
    f1_accs = []
    auc_accs = []
    for train,test in fold_undersampled:
        # train is the indices of training dataset of x_training 
        model_name.fit(x_training_undersampled.iloc[train,:], y_training_undersampled.iloc[train])
        # predict values using the test indices in the training data
        y_pred = model_name.predict(x_training_undersampled.iloc[test,:])      
        # calculate the recall and accuracy score and append it to a list 
        # for recall and accuracy scores representing the current C parameter
        recall_acc = recall_score(y_training_undersampled.iloc[test], y_pred)
        accuracy_acc = accuracy_score(y_training_undersampled.iloc[test], y_pred)
        f1_acc = f1_score(y_training_undersampled.iloc[test], y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_training_undersampled.iloc[test], y_pred)
        auc_acc = auc(fpr, tpr)
        recall_accs.append(recall_acc)
        accuracy_accs.append(accuracy_acc)
        f1_accs.append(f1_acc)
        auc_accs.append(auc_acc)
        
    recall_acc_mean = np.mean(recall_accs)
    accuracy_acc_mean = np.mean(accuracy_accs)
    f1_acc_mean = np.mean(f1_accs)
    auc_acc_mean = np.mean(auc_accs)
    print("")
    print("Mean of the recall score is: ", recall_acc_mean)
    print("Mean of the accuracy score is: ", accuracy_acc_mean)
    print("Mean of the f1 score is: ", f1_acc_mean)
    print("Mean of the auc score is: ", auc_acc_mean)
    print("")
# Check the Performances of different models under cross validation

lr_under = LogisticRegression()
dt_under = tree.DecisionTreeClassifier()
rf_under = RandomForestClassifier()
print("Logistic Regression Performance Based On Undersampled Dataset: ")
print_scores_under_cv_resample(lr_under)
print("")
print("Decision Tree Performance Based On Undersampled Dataset: ")
print_scores_under_cv_resample(dt_under)
print("")
print("Random Forest Performance Based On Undersampled Dataset: ")
print_scores_under_cv_resample(rf_under)
print("")
# check the classification reports
print("LR on undersampled test data: ")
classifier_report(lr_under, x_training_undersampled, y_training_undersampled, x_valid_undersampled, y_valid_undersampled)
print("LR on test data: ")
classifier_report(lr_under, x_training, y_training, x_valid, y_valid)
print("DT on undersampled test data: ")
classifier_report(dt_under, x_training_undersampled, y_training_undersampled, x_valid_undersampled, y_valid_undersampled)
print("DT on test data: ")
classifier_report(dt_under, x_training, y_training, x_valid, y_valid)
print("RF on undersampled test data: ")
classifier_report(rf_under, x_training_undersampled, y_training_undersampled, x_valid_undersampled, y_valid_undersampled)
print("RF on test data: ")
classifier_report(rf_under, x_training, y_training, x_valid, y_valid)
lr_penalized = LogisticRegression(penalty = "l1", C = 0.1, class_weight="balanced")
print("Penalized Logistic Regression Performance: ")
print_scores_under_cv(lr_penalized)
classifier_report(lr_penalized, x_training, y_training, x_valid, y_valid)
# build a function to find the best parameters for improving the performance of random forest
sample_leaves = [10, 20, 30, 40, 50, 60]
for sample_leaf in sample_leaves:
    RF = RandomForestClassifier(min_samples_leaf=sample_leaf, class_weight="balanced")
    print("min_sample_leaf = ", sample_leaf)
    print_scores_under_cv(RF)
    classifier_report(RF, x_training, y_training, x_valid, y_valid)
    print("")
plt.figure(figsize = (14,10))
plt.ylabel('True label', fontsize = 10)
plt.xlabel('Predicted label', fontsize = 10)
j = 1
for sample_leaf in sample_leaves:
    RF_pruned = RandomForestClassifier(min_samples_leaf=sample_leaf, class_weight="balanced")
    RF_pruned.fit(x_training, y_training)
    y_pred_RF_pruned = RF_pruned.predict(x_valid)
    plt.subplot(3,2,j)
    plt.title("min_samples_leaf = "+ str(sample_leaf))
    j = j + 1
    confusion_matrix_RF_pruned = confusion_matrix(y_valid, y_pred_RF_pruned)
    plot_confusion_matrix(confusion_matrix_RF_pruned)
# the best model we choose 
RF_Model = RandomForestClassifier(min_samples_leaf=60, class_weight="balanced")









