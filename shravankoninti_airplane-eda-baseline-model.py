import pandas as pd                                                  # to import csv and for data manipulation

import matplotlib.pyplot as plt                                      # to plot graph

import seaborn as sns                                                # for intractve graphs

import numpy as np                                                   # for linear algebra

import datetime                                                      # to deal with date and time

%matplotlib inline

from sklearn.preprocessing import StandardScaler                     # for preprocessing the data

from sklearn.ensemble import RandomForestClassifier                  # Random forest classifier

from sklearn.tree import DecisionTreeClassifier                      # for Decision Tree classifier

from sklearn.svm import SVC                                          # for SVM classification

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.model_selection import GridSearchCV                     # for tunnig hyper parameter it will use all combination of given parameters

from sklearn.model_selection import RandomizedSearchCV               # same for tunning hyper parameter but will use random combinations of parameters

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')

from sklearn.utils import resample

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')

test_df.head()
sub_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/sample_submission.csv')

sub_df.head()
print(train_df['Severity'].nunique())

print(train_df['Severity'].unique())
print("The total number of Rows in Train dataset is : ", train_df.shape[0])

print("The total number of Rows in Test dataset is : ", test_df.shape[0])

print("The total number of Rows in both Train and Test dataset is : ", train_df.shape[0]+test_df.shape[0])
train_df.keys()
train_df.columns
test_df.columns
train_df.dtypes
train_df['Severity'].value_counts()
# Normalise can be set to true to print the proportions instead of Numbers.

train_df['Severity'].value_counts(normalize=True)
train_df['Severity'].value_counts().plot.bar(figsize=(4,4),title='Severity - Split for Train Dataset')

plt.xlabel('Severity')

plt.ylabel('Count')
train_df.columns
plt.figure(1)

plt.subplot(121)

train_df['Accident_Type_Code'].value_counts(normalize=True).plot.bar(figsize=(24,6), fontsize = 15.0)

plt.title('Accident_Type_Code', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)



plt.subplot(122)

train_df['Violations'].value_counts(normalize=True).plot.bar(figsize=(24,6), fontsize = 15.0)

plt.title('Violations', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)
cols = ['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints', 'Control_Metric']

for col in cols:    

    plt.figure(1)

    plt.subplot(121)

    sns.distplot(train_df[col])



    plt.subplot(122)

    train_df[col].plot.box(figsize=(16,5))



    plt.show()
cols = ['Turbulence_In_gforces',

       'Cabin_Temperature',  'Max_Elevation',

        'Adverse_Weather_Metric']

for col in cols:    

    plt.figure(1)

    plt.subplot(121)

    sns.distplot(train_df[col])



    plt.subplot(122)

    train_df[col].plot.box(figsize=(16,5))



    plt.show()
# Correlation between numerical variables

num_cols_data = (train_df[['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

                        'Control_Metric', 'Turbulence_In_gforces',

                       'Cabin_Temperature',  'Max_Elevation',

                        'Adverse_Weather_Metric'                       

                       ]])

matrix = num_cols_data.corr()

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
num_cols_data.describe()
# Check missing values

train_df.isnull().sum()
Accident_Type_Code=pd.crosstab(train_df['Accident_Type_Code'],train_df['Severity'])

Violations=pd.crosstab(train_df['Violations'],train_df['Severity'])





Accident_Type_Code.div(Accident_Type_Code.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6))

Violations.div(Violations.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6))

cols = ['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

                        'Control_Metric', 'Turbulence_In_gforces',

                       'Cabin_Temperature',  'Max_Elevation',

                        'Adverse_Weather_Metric']



train_df.groupby('Severity')['Safety_Score'].mean().plot.bar()



plt.ylabel('Mean_Safety_Score')
# cols = ['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

#                         'Control_Metric', 'Turbulence_In_gforces',

#                        'Cabin_Temperature',  'Max_Elevation',

#                         'Adverse_Weather_Metric']

plt.figure(1)

plt.subplot(121)

train_df.groupby('Severity')['Safety_Score'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Safety_Score', fontsize = 20.0)



plt.subplot(122)

train_df.groupby('Severity')['Days_Since_Inspection'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Since_Inspection', fontsize = 20.0)
plt.figure(1)

plt.subplot(121)

train_df.groupby('Severity')['Total_Safety_Complaints'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Total_Safety_Complaints', fontsize = 12.0)



plt.subplot(122)

train_df.groupby('Severity')['Control_Metric'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Control_Metric', fontsize = 12.0)
plt.figure(1)

plt.subplot(121)

train_df.groupby('Severity')['Turbulence_In_gforces'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Turbulence_In_gforces', fontsize = 12.0)



plt.subplot(122)

train_df.groupby('Severity')['Cabin_Temperature'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Cabin_Temperature', fontsize = 12.0)
# 'Max_Elevation',  'Adverse_Weather_Metric'

plt.figure(1)

plt.subplot(121)

train_df.groupby('Severity')['Max_Elevation'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Max_Elevation', fontsize = 12.0)



plt.subplot(122)

train_df.groupby('Severity')['Adverse_Weather_Metric'].mean().plot.bar(figsize=(18,6), fontsize = 15.0)

plt.title('Severity', fontweight="bold", fontsize = 22.0)

plt.ylabel('Mean_Days_Adverse_Weather_Metric', fontsize = 12.0)
train_df.columns
test_df.columns
features = ['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

       'Control_Metric', 'Turbulence_In_gforces', 'Cabin_Temperature',

       'Accident_Type_Code', 'Max_Elevation', 'Violations',

       'Adverse_Weather_Metric']

labels = train_df['Severity']
X = train_df.drop(['Accident_ID','Severity'],axis=1)

y = train_df['Severity']



test_X = test_df.drop(['Accident_ID'],axis=1)

# TODO: Shuffle and split the data into training and testing subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)



# Success

print ("Training and testing split was successful.")

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



model_log = LogisticRegression()

model_log.fit(X_train, y_train)

pred_cv = model_log.predict(X_valid)

accuracy_score(y_valid,pred_cv)
confusion_matrix = confusion_matrix( y_valid,pred_cv)

print("the recall for this model is :",confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0]))



fig= plt.figure(figsize=(6,3))# to plot the graph

print("TP",confusion_matrix[1,1,]) 

print("TN",confusion_matrix[0,0]) 

print("FP",confusion_matrix[0,1]) 

print("FN",confusion_matrix[1,0])

sns.heatmap(confusion_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)

plt.title("Confusion_matrix")

plt.xlabel("Predicted_class")

plt.ylabel("Real class")

plt.show()

print(confusion_matrix)

print("\n--------------------Classification Report------------------------------------")

print(classification_report(y_valid, pred_cv)) 
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

model_rf.fit(X_train, y_train)

pred_cv = model_rf.predict(X_valid)

accuracy_score(y_valid,pred_cv)
pred_test = model_rf.predict(test_X)

pred_test = pd.DataFrame(pred_test)

pred_test.columns = ['Severity']
pred_test.head()

print(len(pred_test))
importances=pd.Series(model_rf.feature_importances_, index=X.columns).sort_values()

importances.plot(kind='barh', figsize=(20,20))

plt.xlabel('Importance of Attributes - Score')

plt.ylabel('Attribute Name')

plt.title("Attribute Importance by RandomForest Application")
sub_df = test_df[['Accident_ID']]

# # Fill the target variable with the predictions

sub_df['Severity'] = pred_test['Severity']

# # # Converting the submission file to csv format

sub_df.to_csv('submission.csv', index=False)
sub_df.shape
sub_df.head()