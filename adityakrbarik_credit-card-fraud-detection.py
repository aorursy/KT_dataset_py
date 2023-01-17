# Importing Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Importing the Data
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(10)
# Name of the Columns of the Dataset
df.columns
# Basic Information about the Data
df.info()
# Print the Number of Duplicates in the Data
print(f'{df.duplicated().sum()} no. of duplicate records are removed among {df.shape[0]} records and the no. of remaining records is {df.drop_duplicates().shape[0]}.')
df = df.drop_duplicates()
# Pie Diagram showing the percentage of Fraudulent and Non-Fraudulent Transactions
fig, ax = plt.subplots()
ax.pie(df.Class.value_counts(), 
       labels = ['Non-Fraudulent Transaction', 'Fraudulent Transaction'], 
       autopct='%1.1f%%')

# Draw Circle
centre_circle = plt.Circle((0,0),0.75,fc = 'white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal Aspect Ratio ensures that Pie is drawn as a Circle
ax.axis('equal')  
plt.tight_layout()
plt.show()
print('Summary of the Amount Variable\n')
print(df.Amount.describe())
print(f'{df[df.Amount == 0].shape[0]} records found having Transaction Amount 0 have been dropped from the Data out of {df.shape[0]} records.')
df = df[df.Amount != 0]
# Histogram showing the Distribution of Amount
plt.figure(figsize= [20,10])
sns.distplot(df.Amount)
plt.title('Distribution of Amount')
plt.show()
# Boxplot of Amount grouped by Class
plt.figure(figsize = [16,8])
sns.boxplot(data = df, x = 'Class', y = 'Amount')
plt.title('Boxplot of Amount grouped by Class')
plt.show()
# Conversion of Time from Second to Hour
df['Hour'] = round(df['Time'] / (60 * 60)).astype(int)
# Histogram showing Frequency Distribution of Hour
plt.figure(figsize = [16, 8])
sns.distplot(df.Hour)
plt.title('Frequency Distribution of Hour')
plt.ylabel('Relative Frequency')
plt.show()
# Line Diagram showing the Relative Frequency Distribution of Time for Fraudulent and Non-Fraudulent Transaction
plt.figure(figsize = [16, 8])
sns.distplot(df[df.Class == 0].Hour, hist = False, color = 'blue', label = "Non-Fraudulent Transaction")
sns.distplot(df[df.Class == 1].Hour, hist = False, color = 'red', label = "Fraudulent Transaction")
plt.ylabel('Relative Frequency')
plt.title('Distribution of Time for Fraudulent and Non-Fraudulent Transaction')
plt.show()
# Heatmap showing the Correlations between each Pair of Variables
plt.figure(figsize = [10, 8])
sns.heatmap(df.corr())
plt.show()
# Few Scatter Plotsto check the Relation between a few Pair of Variables
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [16,16])

ax[0,0].scatter(x = df.V20, y = df.Amount)
ax[0,0].set_xlabel('V20')
ax[0,0].set_ylabel('Amount')
ax[0,0].set_title('Relation between V20 and Amount')

ax[0,1].scatter(x = df.V7, y = df.Amount)
ax[0,1].set_xlabel('V7')
ax[0,1].set_ylabel('Amount')
ax[0,1].set_title('Relation between V7 and Amount')

ax[1,0].scatter(x = df.V2, y = df.Amount)
ax[1,0].set_xlabel('V2')
ax[1,0].set_ylabel('Amount')
ax[1,0].set_title('Relation between V2 and Amount')

ax[1,1].scatter(x = df.V3, y = df.Hour)
ax[1,1].set_xlabel('V3')
ax[1,1].set_ylabel('Hour')
ax[1,1].set_title('Relation between V3 and Hour')

fig.suptitle('Few Scatter Plots')
plt.show()
# Boxplot of Logrithm of Amount grouped by Class
df['Log_Amount'] = np.log(df.Amount)
plt.figure(figsize = [16,8])
sns.boxplot(data = df, x = 'Class', y = 'Log_Amount')
plt.title('Boxplot of Logrithm of Amount grouped by Class')
plt.show()
# Function to return Indices of Outliers 
def indicies_of_outliers(x): 
    Q1, Q3 = x.quantile([0.25, 0.75]) 
    IQR = Q3 - Q1
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    return np.where((x > upper_limit) | (x < lower_limit))[0] 
outliers_indices = list(indicies_of_outliers(df[df['Class'] == 0]['Log_Amount']))
print(f'len(outliers_indices) records have been detected as outliers and removed from the data from {df.shape[0]} records.')
df = df[[not i for i in df.index.isin(outliers_indices)]]
plt.figure(figsize = [16,8])
sns.boxplot(data = df, x = 'Class', y = 'Log_Amount')
plt.title('Boxplot of Log_Amount grouped by Class')
plt.show()
Y = df.Class
X = df.drop(['Time', 'Amount', 'Class'], axis = 1)
print('For the Original Data \n'
f'No. of Record for Fraudulent Transactions : {Counter(Y)[1]} \n'
f'No. of Record for Non-Fraudulent Transactions : {Counter(Y)[0]}')
Random_Under_Sampling = RandomUnderSampler()
X_RUS, Y_RUS = Random_Under_Sampling.fit_resample(X, Y)
print('After applying Random Under Sampling on the Original Data \n'
f'No. of Record for Fraudulent Transactions : {Counter(Y_RUS)[1]} \n'
f'No. of Record for Non-Fraudulent Transactions : {Counter(Y_RUS)[0]}')
Random_Over_Sampling = RandomOverSampler()
X_ROS, Y_ROS = Random_Over_Sampling.fit_resample(X, Y)
print('After applying Random Over Sampling on the Original Data \n'
f'No. of Record for Fraudulent Transactions : {Counter(Y_ROS)[1]} \n'
f'No. of Record for Non-Fraudulent Transactions : {Counter(Y_ROS)[0]}')
SMOTE_ = SMOTE()
X_SMOTE, Y_SMOTE = SMOTE_.fit_resample(X, Y)
print('After applying Synthetic Minority Over-Sampling Technique on the Original Data \n'
f'No. of Record for Fraudulent Transactions : {Counter(Y_SMOTE)[1]} \n'
f'No. of Record for Non-Fraudulent Transactions : {Counter(Y_SMOTE)[0]}')
ADASYN_ = ADASYN()
X_ADASYN, Y_ADASYN = ADASYN_.fit_resample(X, Y)
print('After applying Adaptive Synthetic Sampling on the Original Data \n'
f'No. of Record for Fraudulent Transactions : {Counter(Y_ADASYN)[1]} \n'
f'No. of Record for Non-Fraudulent Transactions : {Counter(Y_ADASYN)[0]}')
def Model_Fit(Predictor, Response, Model, Test_Percentage = None,
              Imbalanced_Class = False, Imbalanced_Classification_Sampling_Technique = None):
    
    X, Y = Predictor, Response
    
    if Test_Percentage == None:
        while(True):
            test_per = input('Please give the percentage of Test Data \n (Value must be between 0 and 1) : ') 
            if float(test_per) > 0 and float(test_per) < 1:
                Test_Percentage = test_per
                break
            else:
                continue
    
    Models = ['LogisticRegression', 'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier']
    Imbalanced_Classification_Sampling_Techniques = ['RandomUnderSampler', 'RandomOverSampler', 'SMOTE', 'ADASYN']
    if Model in Models:
        Model = eval(Model + '()')
        if Imbalanced_Class:
            if Imbalanced_Classification_Sampling_Technique in Imbalanced_Classification_Sampling_Techniques:
                imb_ = eval(Imbalanced_Classification_Sampling_Technique + '()')
                
                X_New, Y_New = imb_.fit_resample(X, Y)
                X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_New, Y_New, test_size = Test_Percentage)
                
                model = Model
                model.fit(X_Train, Y_Train)
                Y_Pred = model.predict(X_Test)
                
                plt.figure(figsize = [8, 6])
                sns.heatmap(pd.DataFrame(confusion_matrix(Y_Test , Y_Pred)), annot = True, fmt = 'd', cmap = 'Blues')
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
                plt.title('Confusion Matrix')
                plt.show()
                
                Area_Under_ROC_Curve = roc_auc_score(Y_Test , Y_Pred)
                
                fpr, tpr, _ = roc_curve(Y_Test , Y_Pred)
                plt.figure(figsize = [8, 6])
                plt.plot(fpr, tpr, 'k-', label = f'AUC : {roc_auc_score(Y_Test , Y_Pred)}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc = 'lower right')
                plt.show()
                
                return(model, imb_, Area_Under_ROC_Curve)
        else:
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = Test_Percentage)
                
            model = Model
            model.fit(X_Train, Y_Train)
            Y_Pred = model.predict(X_Test)

            plt.figure(figsize = [8, 6])
            sns.heatmap(pd.DataFrame(confusion_matrix(Y_Test , Y_Pred)), annot = True, fmt = 'd', cmap = 'Blues')
            plt.xlabel('Predicted Class')
            plt.ylabel('Actual Class')
            plt.title('Confusion Matrix')
            plt.show()

            Area_Under_ROC_Curve = roc_auc_score(Y_Test , Y_Pred)

            fpr, tpr, _ = roc_curve(Y_Test , Y_Pred)
            plt.figure(figsize = [8, 6])
            plt.plot(fpr, tpr, 'k-', label = f'AUC : {roc_auc_score(Y_Test , Y_Pred)}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc = 'lower right')
            plt.show()

            return(model, Area_Under_ROC_Curve)
    else:
        print('Please Enter a Valid Model Name.')
Models = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier'] # SVC
IMB_Sampling_Techniques = ['RandomUnderSampler', 'RandomOverSampler', 'SMOTE', 'ADASYN']
Final_Model, Final_IMB, Final_AUC = None, None, 0
for Model in Models:
    for IMB_Sampling_Technique in IMB_Sampling_Techniques:
        print(f'Model : {Model} \nImbalanced_Classification_Sampling_Technique : {IMB_Sampling_Technique} \n\n')
        Model_, IMB_, AUC_ = Model_Fit(X, Y, Test_Percentage = 0.25, Model = Model, Imbalanced_Class = True, 
                                       Imbalanced_Classification_Sampling_Technique = IMB_Sampling_Technique)
        if AUC_ > Final_AUC: Final_Model, Final_IMB, Final_AUC = Model_, IMB_, AUC_
X = df.drop(['Time', 'Amount', 'Class'], axis = 1)
Y = df.Class
X_Final, Y_Final = Final_IMB.fit_resample(X, Y)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Final, Y_Final, test_size = 0.25)
Final_Model.fit(X_Train, Y_Train)
Y_Pred = Final_Model.predict(X_Test)
fpr, tpr, _ = roc_curve(Y_Test , Y_Pred)
plt.figure(figsize = [8, 6])
plt.plot(fpr, tpr, 'k-', label = f'AUC : {roc_auc_score(Y_Test , Y_Pred)}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', size = 14)
plt.ylabel('True Positive Rate', size = 14)
plt.title('Receiver Operating Characteristic (ROC) Curve', size = 15)
plt.legend(loc = 'lower right')
plt.show()
plt.figure(figsize = [8, 6])
sns.heatmap(pd.DataFrame(confusion_matrix(Y_Test , Y_Pred)), annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted Class', size = 12)
plt.ylabel('Actual Class', size = 12)
plt.title('Confusion Matrix', size = 15)
plt.show()
print(
'For the Final Model, \n\n'
f' Accuracy   :  {accuracy_score(Y_Test , Y_Pred)}\n'
f' Precision  :  {precision_score(Y_Test , Y_Pred)}\n'
f' Recall     :  {recall_score(Y_Test , Y_Pred)}\n'
f' F1 Score   :  {f1_score(Y_Test , Y_Pred)}\n\n'
'and a detail Classification Report is given below: \n\n'
f'{classification_report(Y_Test, Y_Pred, target_names = ["Non-Fraudulent Transactions (0)", "Fraudulent Transactions (1)"], digits = 8)}'
)