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
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import shap
dataset= pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
dataset.head()
# Checking Missing values
dataset.isna().sum()
# Identifying datatypes and shape of training dataset
print(dataset.shape)
dataset.dtypes
# Drop ID
dataset.drop(labels='id',axis=1,inplace=True)
dataset.head()
# numeric features in dataset
dataset_num_features = ['Age', 'Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage']

#Catagorical features dataset
dataset_cat_features = ['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']
print(dataset_num_features)
dataset[dataset_num_features].describe()
for i in dataset_num_features:   
    plt.figure(figsize=(10,8),dpi=100)
    sns.violinplot(x="Response",y=i, data=dataset)
    plt.title(f"Response by {i}")
    plt.show()
 

# Age
Age_range = pd.Series(pd.cut(dataset.Age, bins = 6, precision = 0),name='Age_range')

# Region_Code
Region_Code_range = pd.Series(pd.cut(dataset.Region_Code, bins = 10, precision = 0),name='Region_Code_range')

#Annual_Primium
Annual_Primium_range = pd.Series(pd.cut(dataset.Annual_Premium, bins = 5, precision = 0),name='Annual_Primium_range')

#Policy_Sales_Channel
Policy_Sales_Channel_range = pd.Series(pd.cut(dataset.Policy_Sales_Channel,bins = 10, precision = 0),name='Policy_Sales_Channel_range')

#Vintage
Vintage_range =pd.Series(pd.cut(dataset.Vintage,bins = 5, precision = 0),name='Vintage_range')
# Modified Categorical Dataset:

mod_dataset = pd.concat([Age_range,
                Region_Code_range,
                Annual_Primium_range,
                Policy_Sales_Channel_range,
                Vintage_range], 
               axis=1)

print('Modified Dataset Shape :', mod_dataset.shape,'\n______________________________________\n')
print('New Columns:',mod_dataset.columns,'\n______________________________________\n' )
print(mod_dataset.dtypes,'\n______________________________________\n')

mod_dataset.head()
#Age_range
plt.figure(figsize=(10,7),dpi=300)
sns.countplot(x=Age_range, hue=dataset.Response)
plt.xticks()
plt.xlabel('Age_range',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.show()

#Region_Code_range
plt.figure(figsize=(10,7),dpi=300)
sns.countplot(x=Region_Code_range, hue=dataset.Response)
plt.xticks(fontsize=8)
plt.xlabel('Region_Code_range',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.show()

#Annual_Primium_range
plt.figure(figsize=(10,7),dpi=300)
sns.countplot(x=Annual_Primium_range, hue=dataset.Response)
plt.xticks(fontsize=8)
plt.xlabel('Annual_Primium_range',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.show()

#Policy_Sales_Channel_range
plt.figure(figsize=(10,7),dpi=300)
sns.countplot(x=Policy_Sales_Channel_range, hue=dataset.Response)
plt.xticks(fontsize=8)
plt.xlabel('Policy_Sales_Channel_range',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.show()

#Vintage_range
plt.figure(figsize=(10,7),dpi=300)
sns.countplot(x=Vintage_range, hue=dataset.Response)
plt.xticks(fontsize=8)
plt.xlabel('Vintage_range',fontsize=14)
plt.ylabel('Count',fontsize=14)

plt.show()

# plt.subplot(2,3,6)
# sns.countplot(x=Age_range)
# plt.xticks(fontsize=8)
# plt.xlabel('Age_range',fontsize=14)
# plt.ylabel('Count',fontsize=14)

dataset_cat_features
# getting Dummies for Binary Catagorical features in dataset_mod1:
dataset_mod1=pd.get_dummies(dataset, columns=['Gender','Driving_License','Previously_Insured','Vehicle_Damage'],
                            drop_first=True)
# getting Dummies for non-binary Catagorical features ataset_mod1:
dataset_mod1=pd.get_dummies(dataset_mod1, columns=['Vehicle_Age'])
print('\n columns :',dataset_mod1.columns)

print('\n________________________\n Shape:',dataset_mod1.shape)

dataset_mod1.head()
dataset_mod1_cat_features = ['Gender_Male',
                             'Driving_License_1',
                             'Previously_Insured_1',
                             'Vehicle_Age_1-2 Year',
                             'Vehicle_Age_< 1 Year',
                             'Vehicle_Age_> 2 Years',
                             'Vehicle_Damage_Yes']
# Data Counts and basic information in Catagorical features
for category in dataset_mod1_cat_features:
    print(dataset_mod1[category].value_counts(), '\n____________________________________\n')

# ploting counts 

plt.figure(figsize=(30,20),dpi=300)
#Gender
plt.subplot(2,3,1)
sns.countplot(x=dataset_mod1.Gender_Male)
plt.xlabel('Gender_Male',fontsize=14)
plt.ylabel('Count',fontsize=14)


# Driving_License
plt.subplot(2,3,2)
sns.countplot(x=dataset_mod1.Driving_License_1)
plt.xlabel('Driving_License_1',fontsize=14)
plt.ylabel('Count',fontsize=14)

# Previously_Insured
plt.subplot(2,3,3)
sns.countplot(x=dataset_mod1.Previously_Insured_1)
plt.xlabel('Previously_Insured_1',fontsize=14)
plt.ylabel('Count',fontsize=14)

# Vehicle_Age
plt.subplot(2,3,4)
sns.countplot(x=dataset_mod1['Vehicle_Age_> 2 Years'])
plt.xlabel('Vehicle_Age_> 2 Years',fontsize=14)
plt.ylabel('Count',fontsize=14)

# Vehicle_Damage
plt.subplot(2,3,5)
sns.countplot(x=dataset_mod1.Vehicle_Damage_Yes)
plt.xlabel('Vehicle_Damage_Yes',fontsize=14)
plt.ylabel('Count',fontsize=14)

#Response
plt.subplot(2,3,6)
sns.countplot(x=dataset.Response)
plt.xlabel('Response',fontsize=14)
plt.ylabel('Count',fontsize=14)

plt.show()

print(f"""Positive Response - {dataset.Response.value_counts()[1]/
(dataset.Response.value_counts()[1] + dataset.Response.value_counts()[0])*100}%""")


#data distribution of categorical features in both response group
for i in dataset_mod1_cat_features:   
    plt.figure(figsize=(10,8),dpi=100)
    sns.violinplot(x="Response",y=i, data=dataset_mod1)
    plt.title(f"Response by {i}")
    plt.show()
plt.figure(figsize=(10,7),dpi=100)
plt.title("Correlation plot")
sns.heatmap(dataset_mod1.corr(),linewidths=5, annot=True,annot_kws={'size': 8},cmap='coolwarm')
plt.figure(figsize=(10,7),dpi=100)
sns.scatterplot(x=dataset_mod1.Annual_Premium,y=dataset.Age, hue=dataset.Response)
plt.show()
plt.figure(figsize=(10,7),dpi=100)
sns.scatterplot(x=dataset_mod1.Annual_Premium,y=dataset.Vintage, hue=dataset.Response)
plt.show()
plt.figure(figsize=(10,7),dpi=100)
sns.scatterplot(x=dataset_mod1.Age,y=dataset.Annual_Premium, hue=dataset.Response)
plt.show()
# Colums in dataset_mod1
print(dataset_mod1.columns)
# Arranging columns in dataset_mod1:
dataset_mod1= dataset_mod1[['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel',
       'Vintage', 'Gender_Male', 'Driving_License_1',
       'Previously_Insured_1', 'Vehicle_Damage_Yes', 'Vehicle_Age_1-2 Year',
       'Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years','Response']]
# Renaming columns in dataset_mod1 to prevent future problems with XGBClassifier
dataset_mod1=dataset_mod1.rename(columns={'Vehicle_Age_1-2 Year':'Vehicle_Age_1_to_2 Year','Vehicle_Age_< 1 Year':'Vehicle_Age_lessthan_1_Year',
                             'Vehicle_Age_> 2 Years':'Vehicle_Age_morethan_2 Years'})
dataset_mod1
print('\n Shape:', dataset_mod1.shape,'\n__________________________\n')
print('Column:',dataset_mod1.columns)
# Previous Colum names:
print(dataset_mod1_cat_features)
# modified column names: 
dataset_mod1_cat_features = ['Gender_Male', 'Driving_License_1', 'Previously_Insured_1',
'Vehicle_Damage_Yes', 'Vehicle_Age_1_to_2 Year',
       'Vehicle_Age_lessthan_1_Year', 'Vehicle_Age_morethan_2 Years']
print(dataset_mod1_cat_features)
# Assigned independent and the target features.

X = dataset_mod1.iloc[:, 0:-1]
Y = dataset_mod1.iloc[:, -1]

print(X.shape)
print(Y.shape)
print(X)
print(Y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y= le.fit_transform(Y)
print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.01, random_state = 0)
print(X_train.shape)
print(X_train)
print(X_test.shape)
print(X_test)
print(Y_train.shape)
print(Y_train)
print(Y_test.shape)
print(Y_test)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(X_train.shape)
# print(X_train)
# print(X_test.shape)
# print(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier()
RF_classifier.fit(X_train, Y_train)
Y_pred = RF_classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print (classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print(f"Accuracy Score :{accuracy_score(Y_test, Y_pred)}")

plt.figure(figsize=(10,5),dpi=80)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2', cmap='Blues')
plt.xlabel('Predicted label',fontsize=14)
plt.ylabel('True label',fontsize=14)
plt.show()
from sklearn.model_selection import cross_val_score

CV_accuracies = cross_val_score(estimator = RF_classifier, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(CV_accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(CV_accuracies.std()*100))
feat_importances = pd.Series(RF_classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
Y_pred_proba = RF_classifier.predict_proba(X_test)
(fpr, tpr,_) = roc_curve(Y_test, Y_pred_proba[:,1])

plt.figure(figsize=(10,7),dpi=100)
plt.plot(fpr,tpr)
plt.title('Receiver operating characteristic Curve: HICSP')
plt.xlabel('False Positive Rate(FPR):Precision')
plt.ylabel('True Positive Rate (TPR): Recall')
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
from sklearn.model_selection import RandomizedSearchCV
# RF Params:
RF_parameters = {'n_estimators': [100],
                 'criterion': ['entropy', 'gini'],
                 'min_samples_split': [5, 7,10],
                 'min_samples_leaf': [4, 6, 8],
                 'max_depth': [2,3,4,5,6,7,10]
                }
RF_classifier_random = RandomizedSearchCV(estimator = RF_classifier, param_distributions = RF_parameters, n_iter = 10, 
                               cv = 10, verbose= 1, random_state= 404, n_jobs = -1)
RF_classifier_random.fit(X_train, Y_train)
Y_pred = RF_classifier_random.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print (classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print(f"Accuracy Score :{accuracy_score(Y_test, Y_pred)}")

plt.figure(figsize=(10,5),dpi=80)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2', cmap='Blues')
plt.xlabel('Predicted label',fontsize=14)
plt.ylabel('True label',fontsize=14)
plt.show()
Y_pred_proba = RF_classifier_random.predict_proba(X_test)
(fpr, tpr,_) = roc_curve(Y_test, Y_pred_proba[:,1])

plt.figure(figsize=(10,7),dpi=100)
plt.plot(fpr,tpr)
plt.title('Receiver operating characteristic Curve: HICSP')
plt.xlabel('False Positive Rate(FPR):Precision')
plt.ylabel('True Positive Rate (TPR): Recall')
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
from xgboost import XGBClassifier
XGB_classifier = XGBClassifier()
XGB_classifier.fit(X_train, Y_train)
Y_pred = XGB_classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print (classification_report(Y_test, Y_pred))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print(f"Accuracy Score :{accuracy_score(Y_test, Y_pred)}")

plt.figure(figsize=(10,5),dpi=80)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2', cmap='Blues')
plt.xlabel('Predicted label',fontsize=14)
plt.ylabel('True label',fontsize=14)
plt.show()
from sklearn.model_selection import cross_val_score

CVS_accuracies = cross_val_score(estimator = XGB_classifier, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(CVS_accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(CVS_accuracies.std()*100))
Y_pred_proba = XGB_classifier.predict_proba(X_test)
#Classifier_scores = classifier.predict_proba(X_test)[:,1]
(fpr, tpr,_) = roc_curve(Y_test, Y_pred_proba[:,1])

plt.figure(figsize=(10,7),dpi=100)
plt.plot(fpr,tpr)
plt.title('Receiver operating characteristic Curve: HICSP')
plt.xlabel('False Positive Rate(FPR):Precision')
plt.ylabel('True Positive Rate (TPR): Recall')
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
# explainer = shap.TreeExplainer(XGB_classifier)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# shap.summary_plot(shap_values, X_test, plot_size=(10,8))
feature_importance = XGB_classifier.get_booster().get_score(importance_type='total_gain')
keys = list(feature_importance.keys())
values = list(feature_importance.values())

feature_importance_data = pd.DataFrame(data=values, index=keys, 
                                       columns=["score"]).sort_values(by = "score", ascending=True)

feature_importance_data.plot(kind='barh',figsize=(10,7))
from sklearn.model_selection import RandomizedSearchCV
# A parameter grid for XGBClassifier
XGB_parameters = {'eta':[0.01,0.02,0.03,0.04,0.05],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [3, 5, 7, 9],
          'min_child_weight': [1, 5, 10],
          'subsample': [0.6, 0.8, 1.0],
          'colsample_bytree': [0.6, 0.8, 1.0]
         }
XGB_classifier_random = RandomizedSearchCV(estimator = XGB_classifier, param_distributions = XGB_parameters, n_iter = 10, 
                               cv = 10, verbose= 1, random_state= 42, n_jobs = -1)
XGB_classifier_random.fit(X_train, Y_train)
Y_pred = XGB_classifier_random.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print (classification_report(Y_test, Y_pred))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print(f"Accuracy Score :{accuracy_score(Y_test, Y_pred)}")

plt.figure(figsize=(10,5),dpi=80)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2', cmap='Blues')
plt.xlabel('Predicted label',fontsize=14)
plt.ylabel('True label',fontsize=14)
plt.show()
Y_pred_proba = XGB_classifier_random.predict_proba(X_test)
(fpr, tpr,_) = roc_curve(Y_test, Y_pred_proba[:,1])

plt.figure(figsize=(10,7),dpi=100)
plt.plot(fpr,tpr)
plt.title('Receiver operating characteristic Curve: HICSP')
plt.xlabel('False Positive Rate(FPR):Precision')
plt.ylabel('True Positive Rate (TPR): Recall')
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
from catboost import CatBoostClassifier, Pool
Cat_classifier = CatBoostClassifier()
Cat_classifier = Cat_classifier.fit(X_train,Y_train,
                                    eval_set = (X_test, Y_test), 
                                    cat_features = dataset_mod1_cat_features,
                                    use_best_model = True,
                                    plot = True,
                                    early_stopping_rounds = 10,
                                    )
Y_pred = Cat_classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print (classification_report(Y_test, Y_pred))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print(f"Accuracy Score :{accuracy_score(Y_test, Y_pred)}")

plt.figure(figsize=(10,5),dpi=80)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2', cmap='Blues')
plt.xlabel('Predicted label',fontsize=14)
plt.ylabel('True label',fontsize=14)
plt.show()
Y_pred_proba = Cat_classifier.predict_proba(X_test)
(fpr, tpr,_) = roc_curve(Y_test, Y_pred_proba[:,1])

plt.figure(figsize=(10,7),dpi=100)
plt.plot(fpr,tpr)
plt.title('Receiver operating characteristic Curve: HICSP')
plt.xlabel('False Positive Rate(FPR):Precision')
plt.ylabel('True Positive Rate (TPR): Recall')
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
explainer = shap.TreeExplainer(Cat_classifier)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test, plot_size=(10,8))
