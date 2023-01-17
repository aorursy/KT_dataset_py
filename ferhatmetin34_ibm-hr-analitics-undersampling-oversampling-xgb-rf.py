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
#from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from scipy import stats

from scipy.stats import pearsonr

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows',None)

import itertools

from sklearn.model_selection import train_test_split

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve

from sklearn.model_selection import cross_val_score

from sklearn.tree import plot_tree

from xgboost import plot_tree,plot_importance

from sklearn import tree

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

!pip install dython

from dython import nominal
data=pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df=data.copy()

df.head()
df.shape
df.columns
df.describe().T
df.describe(include="object").T
df=df.drop(columns=["Over18"],axis=1)
df.isnull().sum().to_frame()
cat_columns=df.select_dtypes(include="object").columns

for i in cat_columns:

    plt.figure(figsize=(9,5));

    sns.countplot(df[i]);

    plt.xticks(rotation=90);

    plt.show();
for i in cat_columns[1:]:

    sns.catplot(x=i,y='MonthlyIncome',hue="Attrition",data=df,kind="bar",aspect=3);

    plt.xticks(rotation=90);
def diagnostic_plots(df, variable):

    

    plt.figure(figsize=(20, 9))



    plt.subplot(1, 3, 1)

    sns.distplot(df[variable], bins=30,kde_kws={'bw': 1.5})

    plt.title('Histogram')

    

    plt.subplot(1, 3, 2)

    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.ylabel('RM quantiles')



    plt.subplot(1, 3, 3)

    sns.boxplot(y=df[variable])

    plt.title('Boxplot')



    

    

    plt.show()
df.hist(edgecolor='black', linewidth=1.2, figsize=(22, 22));
for i in ["Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate"]:

       diagnostic_plots(df,i)
df.dtypes.to_frame()
df["EmployeeCount"].value_counts()
df["StandardHours"].value_counts()
sns.countplot(df["PerformanceRating"],palette="coolwarm");
df=df.drop(columns=["EmployeeCount","StandardHours"],axis=1)
num_columns=df.select_dtypes(exclude="object").columns
for i in num_columns:

    sns.boxplot(df[i],color="orangered");

    plt.show();
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(28,16))

corr=df.corr()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);
cat_columns=df.select_dtypes(include=["object"]).columns
nominal.associations(df[cat_columns],figsize=(20,10),mark_columns=True);
df["EmployeeNumber"].nunique()
df=df.drop("EmployeeNumber",axis=1)
df.head()
df_new=df.copy()

df_new=df_new.drop(columns=["Department","JobLevel","YearsInCurrentRole","TotalWorkingYears","PercentSalaryHike","YearsWithCurrManager"],axis=1)
df_new=pd.get_dummies(df_new,drop_first=True)
df_new.head()
X=df_new.drop("Attrition_Yes",axis=1)

y=df_new["Attrition_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.25, 

                                                    random_state=42,stratify=y)
xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy_score(y_test,y_pred)
cross_val_score(xgb_model,X,y).mean()
def conf_matrix(y_test,y_pred):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7,7))

    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'YlGnBu');

    plt.ylabel('Actual label');

    plt.xlabel('Predicted label');

    all_sample_title = 'Confusion Matrix - score:'+str(accuracy_score(y_test,y_pred))

    plt.title(all_sample_title, size = 15);

    plt.show()

    print(classification_report(y_test,y_pred))

conf_matrix(y_test,y_pred)
def plot_roc_curve(y_test,X_test,model):

    fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr_mlp, tpr_mlp)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC CURVE')

    plt.show()

plot_roc_curve(y_test,X_test,xgb_model)
plot_importance(xgb_model).figure.set_size_inches(10,8);
plot_tree(xgb_model,num_trees=2).figure.set_size_inches(200,200);

plt.show();


rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_resample(X, y)
X_resampled.shape,y_resampled.shape
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 

                                                    test_size=0.25, 

                                                    random_state=42,stratify=y_resampled)
xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy_score(y_test,y_pred)
cross_val_score(xgb_model,X_resampled,y_resampled).mean()
conf_matrix(y_test,y_pred)
plot_roc_curve(y_test,X_test,xgb_model)
roc_auc_score(y_test,y_pred)
plot_importance(xgb_model).figure.set_size_inches(10,8);
plot_tree(xgb_model,num_trees=2).figure.set_size_inches(200,200);

plt.show();


ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled.shape,y_resampled.shape
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 

                                                    test_size=0.25, 

                                                    random_state=42,stratify=y_resampled)
xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy_score(y_test,y_pred)
cross_val_score(xgb_model,X_resampled,y_resampled).mean()
conf_matrix(y_test,y_pred)
plot_roc_curve(y_test,X_test,xgb_model)
roc_auc_score(y_test,y_pred)
#%matplotlib inline

#%config InlineBackend.figure_format = 'retina'

plot_tree(xgb_model,num_trees=2).figure.set_size_inches(200,200);

plt.show();



plot_importance(xgb_model).figure.set_size_inches(10,8);
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.25, 

                                                    random_state=42,stratify=y)
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_score(rf_model,X,y).mean()
conf_matrix(y_test,y_pred)
plot_roc_curve(y_test,X_test,rf_model)
rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_resample(X, y)
X_resampled.shape,y_resampled.shape
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 

                                                    test_size=0.20, 

                                                    random_state=42)

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_score(rf_model,X_resampled,y_resampled).mean()
conf_matrix(y_test,y_pred)
plot_roc_curve(y_test,X_test,rf_model)
roc_auc_score(y_test,y_pred)
ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled.shape, y_resampled.shape
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 

                                                    test_size=0.20, 

                                                    random_state=42)
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_score(rf_model,X_resampled,y_resampled).mean()
conf_matrix(y_test,y_pred)
plot_roc_curve(y_test,X_test,rf_model)