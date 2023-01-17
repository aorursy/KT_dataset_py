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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score,accuracy_score

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
insurance_df = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
insurance_df.columns
insurance_df.dtypes
insurance_df.isnull().sum()
insurance_df.shape
categorical_columns=[]
continuous_columns=[]
for col in insurance_df.columns:
    if insurance_df[col].dtype!='object':
        continuous_columns.append(col)
    else:
        categorical_columns.append(col)
continuous_columns
plt.figure(figsize=(16,16))
for i, col in enumerate(['id','Age','Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage']):
    plt.subplot(4,4,i+1)
    sns.boxplot(insurance_df[col])
    plt.tight_layout()
insurance_df.loc[insurance_df.Annual_Premium> 400000,'Annual_Premium']=400000
insurance_df['Gender'].value_counts()
fig, ax =plt.subplots(1,2,figsize=(15,5))
sns.countplot(data=insurance_df,x='Gender',hue='Vehicle_Damage',ax=ax[0])
sns.countplot(data=insurance_df,x='Gender',hue='Previously_Insured',ax=ax[1])
fig.show()
fig, ax =plt.subplots(1,2,figsize=(15,5))
sns.countplot(data=insurance_df,x='Gender',hue='Vehicle_Age',ax=ax[0])
sns.countplot(data=insurance_df,x='Previously_Insured',hue='Vehicle_Damage',ax=ax[1])
fig.show()
fig, ax =plt.subplots(1,2,figsize=(15,5))
# fig, ax = plt.subplots() 
sns.countplot(data=insurance_df,x='Gender',hue='Previously_Insured',ax=ax[0])
sns.countplot(data=insurance_df,x='Gender',hue='Vehicle_Damage',ax=ax[1])
fig.show()
plt.figure(figsize=(20,9))
sns.FacetGrid(insurance_df, hue = 'Response',
             height = 6,xlim = (0,150)).map(sns.kdeplot, 'Age', shade = True,bw=2).add_legend()
plt.figure(figsize=(20,9))
sns.FacetGrid(insurance_df, hue = 'Gender',
             height = 6,xlim = (0,150)).map(sns.kdeplot, 'Age', shade = True,bw=2).add_legend()
plt.figure(figsize=(15,5))
sns.boxplot(y='Age', x ='Gender', hue="Previously_Insured", data=insurance_df)
plt.figure(figsize=(15,5))
sns.violinplot(y='Age', x ='Gender', hue="Response", data=insurance_df)
le = LabelEncoder()
insurance_df['Gender'] = le.fit_transform(insurance_df['Gender'])
insurance_df['Driving_License'] = le.fit_transform(insurance_df['Driving_License'])
insurance_df['Previously_Insured'] = le.fit_transform(insurance_df['Previously_Insured'])
insurance_df['Vehicle_Damage'] = le.fit_transform(insurance_df['Vehicle_Damage'])
insurance_df['Driving_License'] = le.fit_transform(insurance_df['Driving_License'])
insurance_df['Vehicle_Age'] = le.fit_transform(insurance_df['Vehicle_Age'])
insurance_df=insurance_df[['Gender', 'Age', 'Driving_License', 'Region_Code',
       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage', 'Response']]
plt.figure(figsize=(12,12))
sns.heatmap(insurance_df.corr())
def evaluation_stats(model,X_train, X_test, y_train, y_test,algo,is_feature=False):
    print('Train Accuracy')
    y_pred_train = model.predict(X_train)                           
    print(accuracy_score(y_train, y_pred_train))
    print('Validation Accuracy')
    y_pred_test = model.predict(X_test)                           
    print(accuracy_score(y_test, y_pred_test))
    print("\n")
    print("Train AUC Score")
    print(roc_auc_score(y_train, y_pred_train))
    print("Test AUC Score")
    print(roc_auc_score(y_test, y_pred_test))
    
    if is_feature:
        plot_feature_importance(rf_model.feature_importances_,X.columns,algo)

def training(model,X_train, y_train):
    return model.fit(X_train, y_train)

def plot_feature_importance(importance,names,model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
insurance_df.columns
insurance_df['Response'].value_counts()
X = insurance_df.drop(["Response"], axis=1)
y = insurance_df["Response"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)
rf_model = training(RandomForestClassifier(),X_train,y_train)
evaluation_stats(rf_model,X_train, X_test, y_train, y_test,'RANDOM FOREST')
xbg_model = training(XGBClassifier(),X_train,y_train)
evaluation_stats(xbg_model,X_train, X_test, y_train, y_test,'XGB')
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_resample(X_train, y_train)
rf_model = training(RandomForestClassifier(),X_res, y_res)
evaluation_stats(rf_model,X_res, X_test, y_res, y_test,'RANDOM FOREST')
xbg_model = training(XGBClassifier(),X_train,y_train)
evaluation_stats(xbg_model,X_res, X_test, y_res, y_test,'XGB')
rf_model = training(RandomForestClassifier(criterion='entropy',n_estimators=200,max_depth=3),X_res, y_res)
evaluation_stats(rf_model,X_res, X_test, y_res, y_test,'RANDOM FOREST')
xbg_model = training(XGBClassifier(n_estimators=1000,max_depth=10),X_res, y_res)
evaluation_stats(xbg_model,X_res, X_test, y_res, y_test,'XGB',is_feature=False)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(StandardScaler(), LogisticRegression())
clf.fit(X_res, y_res)
evaluation_stats(clf,X_train, X_test, y_train, y_test,'LR',is_feature=False)


