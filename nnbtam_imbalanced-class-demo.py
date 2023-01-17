# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
from sklearn.preprocessing import StandardScaler

# create StandardScaler and assign to std_scaler
std_scaler = StandardScaler()

# fit_transform "Amount", "Time" columns and replace old values with transformed values
df['Amount'] = std_scaler.fit_transform(df[['Amount']])
df['Time'] = std_scaler.fit_transform(df[['Time']])
# Check ratio between classes
percentage_fraud = (df['Class'] == 1).sum() / df.shape[0] * 100
percentage_no_fraud = (df['Class'] == 0).sum() / df.shape[0] * 100

print ('Percentage Fraud transactions: ', percentage_fraud)
print ('Percentage No-fraud transactions: ', percentage_no_fraud)
neg, pos = np.bincount(df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
fig = plt.figure(figsize=(7,7)) # Set figsize

sns.countplot(data=df, x='Class')

plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn import metrics
# Original data
X = df.drop(columns='Class')
y = df['Class']

print ('X shape:', X.shape)
print ('y shape:', y.shape)
from sklearn.model_selection import train_test_split
# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print("Number transactions training dataset: ", len(X_train))
print("Number transactions testing dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))
training_data = pd.concat ([X_train,y_train],axis = 1)
training_data['Class'].value_counts()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))
def plot_roc(name, model, y_test, X_test, **kwargs):
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.title(name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positives rate')
    plt.legend(loc=4)
    plt.show()
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
 
# Predict on training set
lr_pred = lr.predict(X_test)

predictions = pd.DataFrame(lr_pred)
print('Number of predicted labels:\n', predictions[0].value_counts())
print('Accuracy score - Logistic: ', accuracy_score(y_test, lr_pred))
print('Recall - Logistic: ', recall_score(y_test, lr_pred))
plot_cm(y_test, lr_pred)
# separate minority and majority classes
not_fraud = training_data[training_data.Class==0]
fraud = training_data[training_data.Class==1]
from sklearn.utils.class_weight import compute_class_weight
X_weighted = df.drop(columns='Class')
y_weighted = df['Class']
weights = compute_class_weight('balanced', [0,1], y_train)
print("Ratio label 0:1", weights)
lr_weighted = LogisticRegression(solver='liblinear', class_weight="balanced").fit(X_train, y_train)
 
# Predict on training set
lr_weighted_pred = lr_weighted.predict(X_test)

predictions = pd.DataFrame(lr_weighted_pred)
print('Number of predicted labels:\n', predictions[0].value_counts())
print('Accuracy score - Logistic Weighted: ', accuracy_score(y_test, lr_pred))
print('Recall - Logistic Weighted: ', recall_score(y_test, lr_pred))
from sklearn.utils import resample
# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
upsampled.Class.value_counts()
y_train_upsampled = upsampled.Class
X_train_upsampled = upsampled.drop('Class', axis=1)
upsampled_lr = LogisticRegression(solver='liblinear').fit(X_train_upsampled, y_train_upsampled)

upsampled_lr_pred = upsampled_lr.predict(X_test)

predictions = pd.DataFrame(upsampled_lr_pred)
print('Number of predicted labels:\n', predictions[0].value_counts())
print('Accuracy score - Logistic upsampled: ', accuracy_score(y_test, upsampled_lr_pred))
print('Recall - Logistic upsampled: ', recall_score(y_test, upsampled_lr_pred))
plot_cm(y_test, upsampled_lr_pred)
from imblearn.over_sampling import SMOTE
# upsample minority by SMOTE
sm = SMOTE(random_state=5)
X_smote, y_smote = sm.fit_resample(X_train, y_train)

print("Before SMOTE, counts of label '1': {}".format(sum(y==1)))
print("Before SMOTE, counts of label '0': {} \n".format(sum(y==0)))

print("After SMOTE, counts of label '1': {}".format(sum(y_smote==1)))
print("After SMOTE, counts of label '0': {}".format(sum(y_smote==0)))
smote_lr = LogisticRegression(solver='liblinear').fit(X_smote, y_smote)

smote_lr_pred = smote_lr.predict(X_test)

predictions = pd.DataFrame(smote_lr_pred)
print('Number of predicted labels:\n', predictions[0].value_counts())
print('Accuracy score - Logistic upsampled SMOTE: ', accuracy_score(y_test, smote_lr_pred))
print('Recall - Logistic upsampled SMOTE: ', recall_score(y_test, smote_lr_pred))
# plot_cm(y_test,smote_lr_pred)
# Random removal
# undersample minority
not_fraud_downsampled = resample(not_fraud,
                          replace=False, # sample without replacement
                          n_samples=len(fraud), # match number in minority class
                          random_state=5) # reproducible results

# combine minority and undersampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])

# check new class counts
downsampled.Class.value_counts()
y_train_downsampled = downsampled.Class
X_train_downsampled = downsampled.drop('Class', axis=1)
downsampled_lr = LogisticRegression(solver='liblinear').fit(X_train_downsampled, y_train_downsampled)

downsampled_lr_pred = downsampled_lr.predict(X_test)

predictions = pd.DataFrame(downsampled_lr_pred)
print('Number of predicted labels:\n', predictions[0].value_counts())
print('Accuracy score - Logistic downsampled: ', accuracy_score(y_test, downsampled_lr_pred))
print('Recall - Logistic downsampled: ', recall_score(y_test, downsampled_lr_pred))
plot_cm(y_test, downsampled_lr_pred)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

lr = LogisticRegression()
dtc = DecisionTreeClassifier()
xgb = XGBClassifier()

models = [lr, dtc, xgb]
models_name = ["Logistic Regression", "Decision Tree", "XGBoost"]
# Import confusion_matrix, classification_report
# Your code here
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score

# We create an utils function, that take a trained model as argument and print out confusion matrix
# classification report base on X and y
def evaluate_model(estimator, X, y, description):
    # Note: We should test on the original test set
    prediction = estimator.predict(X)
#     print('Confusion matrix:\n', confusion_matrix(y, prediction))
#     print('Classification report:\n', classification_report(y, prediction))
#     print('Testing set information:\n', "Your code here")

    # Set print options
    np.set_printoptions(precision=2)
    model_name = type(estimator).__name__
    return {'name': model_name, 
            'recall': recall_score(y, prediction),
            'precision': precision_score(y, prediction),
           'description': description}
# Now we will test on origin dataset (X_train, y_train)
# We loop for models
# For each model, we train with train_sub dataset
# and use evaluate_model function to test with test set
X_train = training_data.drop(columns='Class')
y_train = training_data['Class']
scores_origin = []
for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    # Your code here
    model.fit(X_train, y_train)
    scores_origin.append(evaluate_model(model, X_test, y_test, 'origin'))
    
    print("=======================================")
lr_weighted = LogisticRegression(class_weight="balanced")
dtc_weighted = DecisionTreeClassifier(class_weight="balanced")
xgb_weighted = XGBClassifier(scale_pos_weight=len(not_fraud) / len(fraud))

models_weighted = [lr_weighted, dtc_weighted, xgb_weighted]
models_weighted_name = ["Weighted Logistic Regression", "Weighted Decision Tree", "Weighted XGBoost"]
# Now we will test on origin dataset (X_train, y_train)
# We loop for models
# For each model, we train with train_sub dataset
# and use evaluate_model function to test with test set
X_train = training_data.drop(columns='Class')
y_train = training_data['Class']
scores_weighted = []
for idx, model in enumerate(models_weighted):
    print("Model: {}".format(models_weighted_name[idx]))
    # Your code here
    model.fit(X_train, y_train)
    scores_weighted.append(evaluate_model(model, X_test, y_test, 'weighted'))
    
    print("=======================================")
# Now we will test on Undersampled dataset (X_train_undersample, y_train_undersample)
# We loop for models
# For each model, we train with train_undersample dataset
# and use evaluate_model function to test with test set
scores_under = []
for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    # Your code here
    model.fit(X_train_downsampled, y_train_downsampled)
    scores_under.append(evaluate_model(model, X_test, y_test, 'under'))
    print("=======================================")
# Now we will test on Oversampled dataset (X_train_oversample, y_train_oversample)
# We loop for models
# For each model, we train with train_oversample dataset
# and use evaluate_model function to test with test set
scores_over = []
for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_train_upsampled, y_train_upsampled)
    scores_over.append(evaluate_model(model, X_test, y_test, 'oversample'))
    # Your code here
    print("=======================================")
scores = []

for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_smote, y_smote)
    scores.append(evaluate_model(model, X_test, y_test, 'smote'))
    # Your code here
    print("=======================================")  
df_imb = pd.DataFrame(scores)
df_under = pd.DataFrame(scores_under)
df_over  = pd.DataFrame(scores_over)
df_weighted = pd.DataFrame(scores_weighted)
df_origin = pd.DataFrame(scores_origin)

df_all = pd.concat([df_imb, df_under, df_over, df_weighted, df_origin])
df_all.sort_values(['recall'], inplace=True)
for label, df in df_all.groupby('description'):
    df.plot(x='name', kind='barh', title=label, figsize=(8, 4), xlim=(0, 1))
df_all.sort_values('recall', inplace=True)
for label, df in df_all.groupby('name'):
    df.plot(x='description', kind='barh', title=label, figsize=(8, 4), xlim=(0,1))
plot_roc("XGBoost", xgb, y_test, X_test, color=colors[0])
plot_roc("Cost-sensitive XGBoost", xgb_weighted, y_test, X_test, color=colors[0])
plot_roc("Random upsampled XGBoost", xgb.fit(X_train_upsampled, y_train_upsampled), y_test, X_test, color=colors[0])
plot_roc("SMOTE upsampled XGBoost", xgb.fit(X_smote, y_smote), y_test, X_test, color=colors[0])
plot_roc("Random Downsampled XGBoost", xgb.fit(X_train_downsampled, y_train_downsampled), y_test, X_test, color=colors[0])