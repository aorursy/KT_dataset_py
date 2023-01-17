#Loading Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix,roc_curve,auc

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")
# loading data
df = pd.read_csv('../input/creditcard.csv')

## Analyse the Data
#  Dispcriptive statistics
df.shape       # shape - gives the total number of rows and columns
                      # it has 284807 rows and 31 columns
df.head()            # head () function - gives the top 5 rows
                     # it has 'Time' 'Amount' 'Class' and 28Variables(for security reasons actuall names are hidden and represented as V1,V2..etc.,)
                     # from the Table data identify 'Features(input)' and 'Labels(output)'
                     # As per the Data we Decide 'Class' is Our Label/output
                         # Class = 0 --No Fraud
                         # Class = 1 -- Fraud
                     # remaining all Columns we are taking as our 'Features(inputs)'
                     # check for CATOGORICAL Values , if there are any Catogorical Values convert it into "numerical format"
                     # as ML understands only Numerical format data
# checking datatypes   
df.info()           # all the features are of float datatype except the 'Class' which is of int type
# check for missing values in each feature and label
df.isnull().sum()       # missing values represented by 1 or more than 1
                           # no missing values represented by  0
                           # here there are no missing values
# statistical summary of the data 
# mean,standard deviation,count etc.,
df.describe()
# Check the Class distribution of 'Output Class'
# to identify whether our data is 'Balanced' or 'Imbalanced'

print(df['Class'].value_counts() )      # 0 - NonFraud Class
                                        # 1 - Fraud Class

# to get in percentage use 'normalize = True'
print('\nNoFrauds = 0 | Frauds = 1\n')
print(df['Class'].value_counts(normalize = True)*100)
# visualizing throug bar graph
df['Class'].value_counts().plot(kind = 'bar', title = 'Class Distribution\nNoFrauds = 0 | Frauds = 1'); # semicolon(;) to avoid '<matplotlib.axes._subplots.AxesSubplot at 0xe81c4b518>' in output
# No Missing Data, No Duplicates
# No Feature Selection as the feature names are hidden for security reasons
# 
# As the Data is PCA transformed we assume that the variables v1 - v28 are scaled 
# we scale leftout 'Time' and 'Amount' features


#visualizing through density plots using seaborn
import seaborn as sns
fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, figsize=(20, 5))

ax1.set_title(' Variable V1-V28\nAssuming as Scaled')  # plotting only few variables
sns.kdeplot(df['V1'], ax=ax1)                          # kde - kernel density estimate
sns.kdeplot(df['V2'], ax=ax1)
sns.kdeplot(df['V3'], ax=ax1)
sns.kdeplot(df['V25'], ax=ax1)
sns.kdeplot(df['V28'], ax=ax1)

ax2.set_title('Time Before Scaling')
sns.kdeplot(df['Time'], ax=ax2)

ax3.set_title('Amount Before Scaling')            
sns.kdeplot(df['Amount'], ax=ax3)

plt.show()
#Scaling data using RobustScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
rb = RobustScaler()
df['Time'] = rb.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount'] = rb.fit_transform(df['Amount'].values.reshape(-1,1))
df.head()
# lets Analyse why the accuracy is misleading(high) 

x = df.drop('Class',axis = 1)
y = df['Class']

#train and test split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size = 0.3,random_state = 42)

# spot check algorithms
classifiers = {"Logistic Regression":LogisticRegression(),
               "DecisionTree":DecisionTreeClassifier(),
               "LDA":LinearDiscriminantAnalysis()}        
# as the dataset is too big computation time will be high
# bcoz of which iam using only 3 classifiers

for name,clf in classifiers.items():
    accuracy      = cross_val_score(clf,xTrain,yTrain,scoring='accuracy',cv = 5)
    accuracyTest  = cross_val_score(clf,xTest,yTest,scoring='accuracy',cv = 5)
    
    precision     = cross_val_score(clf,xTrain,yTrain,scoring='precision',cv = 5)
    precisionTest = cross_val_score(clf,xTest,yTest,scoring='precision',cv = 5)
    
    recall        = cross_val_score(clf,xTrain,yTrain,scoring='recall',cv= 5)
    recallTest    = cross_val_score(clf,xTest,yTest,scoring='recall',cv = 5)
    
    print(name,'---','Train-Accuracy :%0.2f%%'%(accuracy.mean()*100),
                     'Train-Precision: %0.2f%%'%(precision.mean()*100),
                     'Train-Recall   : %0.2f%%'%(recall.mean()*100))
    
    print(name,'---','Test-Accuracy :%0.2f%%'%(accuracyTest.mean()*100),
                     'Test-Precision: %0.2f%%'%(precisionTest.mean()*100),
                     'Test-Recall   : %0.2f%%'%(recallTest.mean()*100),'\n')

# 1. split the 'Original Train data ' into train & test
# 2. Oversample or UnderSample the splitted train data
# 3. fit the model with Oversample or Undersampled train data
# 4. perform 'prediction' on Oversample or Undersampled train data
# 5. Finally perform 'prediction' on Original TEST Data

#step 1
xTrain_rus,xTest_rus,yTrain_rus,yTest_rus = train_test_split(xTrain,yTrain,test_size = 0.2,random_state = 42)

#step 2
rus = RandomUnderSampler()
x_rus,y_rus = rus.fit_sample(xTrain_rus,yTrain_rus)

#converting it to DataFrame to Visualize in pandas
df_x_rus = pd.DataFrame(x_rus)
df_x_rus['target'] = y_rus
print(df_x_rus['target'].value_counts())
print(df_x_rus['target'].value_counts().plot(kind = 'bar',title = 'RandomUnderSampling\nFrauds = 1 | NoFrauds = 0'))


#step 3
lr = LogisticRegression()
lr.fit(x_rus,y_rus)

#step 4
yPred_rus = lr.predict(xTest_rus)

rus_accuracy = accuracy_score(yTest_rus,yPred_rus)
rus_classReport = classification_report(yTest_rus,yPred_rus)
#print('\nTrain-Accuracy %0.2f%%'%(rus_accuracy*100),
#      '\nTrain-ClassificationReport:\n',rus_classReport,'\n')

#step 5
yPred_actual = lr.predict(xTest)
test_accuracy = accuracy_score(yTest,yPred_actual)
test_classReport = classification_report(yTest,yPred_actual)
print('\nTest-Accuracy %0.2f%%'%(test_accuracy*100),
      '\n\nTest-ClassificationReport:\n',test_classReport)

#step 1
xTrain_ros,xTest_ros,yTrain_ros,yTest_ros = train_test_split(xTrain,yTrain,test_size=0.2,random_state=42)

#step 2
ros = RandomOverSampler()
x_ros,y_ros = ros.fit_sample(xTrain_ros,yTrain_ros)

#Converting it to dataframe to visualize in pandas
df_x_ros = pd.DataFrame(x_ros)
df_x_ros['target'] = y_ros
print(df_x_ros['target'].value_counts())
print(df_x_ros['target'].value_counts().plot(kind = 'bar',title = 'RandomOverSampling\nFrauds = 0 | NoFrauds = 1'))

#step 3
lr = LogisticRegression()
lr.fit(x_ros,y_ros)

#step 4
yPred_ros = lr.predict(xTest_ros)

ros_accuracy = accuracy_score(yTest_ros,yPred_ros)
ros_classReport = classification_report(yTest_ros,yPred_ros)
print('\nTrain-Accuracy %0.2f%%'%(rus_accuracy*100),
      '\nTrain-ClassificationReport:\n',rus_classReport,'\n')

#step 5
yPred_actual = lr.predict(xTest)
test_accuracy = accuracy_score(yTest,yPred_actual)
test_classReport = classification_report(yTest,yPred_actual)
print('\nTest-Accuracy %0.2f%%'%(test_accuracy*100),
      '\n\nTest-ClassificationReport:\n',test_classReport)
#step 1
xTrain_smote,xTest_smote,yTrain_smote,yTest_smote = train_test_split(xTrain,yTrain,test_size = 0.2,random_state = 42 )

#step2
smote = SMOTE()
x_smote,y_smote = smote.fit_sample(xTrain_smote,yTrain_smote)
#Converting it to dataframe to visualize in pandas
df_x_smote = pd.DataFrame(x_smote)
df_x_smote['target'] = y_smote
print(df_x_smote['target'].value_counts())
print(df_x_smote['target'].value_counts().plot(kind = 'bar',title = 'SMOTE\nFrauds = 0 | NoFrauds = 1'))


rfc = RandomForestClassifier(random_state = 42)
rfc.fit(x_smote,y_smote)
ypred_smote = rfc.predict(xTest_smote)

rfc_prediction=rfc.predict(xTest)
print('RFC-Accuracy',accuracy_score(yTest,rfc_prediction),'\n')
print('Confusion_Matrix:\n',confusion_matrix(yTest,rfc_prediction),'\n')
print('Classification Report',classification_report(yTest,rfc_prediction))
#auc score
rfc_fpr,rfc_tpr,_ = roc_curve(yTest,rfc_prediction)
rfc_auc = auc(rfc_fpr,rfc_tpr)
print('RandomForestClassifier-auc : %0.2f%%'%(rfc_auc * 100))

#roc curve
plt.figure()
plt.plot(rfc_fpr,rfc_tpr,label ='RFC(auc = %0.2f%%)'%(rfc_auc *100))
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title('Smote with RandomForestClassifier\nROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
