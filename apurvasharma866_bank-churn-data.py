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
import pandas as pd
df = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
df.head()
df.shape
df.info()
len(df["Geography"].unique())
len(df["Surname"].unique())
len(df["Exited"].unique())
# Get unique count for each variable
df.nunique()
# Drop the columns as explained above
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
print("Min age in dataset", min(df["Age"].unique()))
print("Max age in dataset", max(df["Age"].unique()))
print("Min NumOfProducts in dataset", min(df["NumOfProducts"].unique()))
print("Max NumOfProducts in dataset", max(df["NumOfProducts"].unique()))
print("Min CreditScore in dataset", min(df["CreditScore"].unique()))
print("Max CreditScore in dataset", max(df["CreditScore"].unique()))
print("Min Tenure in dataset", min(df["Tenure"].unique()))
print("Max Tenure in dataset", max(df["Tenure"].unique()))
print("Min Tenure in dataset", min(df["Exited"].unique()))
print("Max Tenure in dataset", max(df["Exited"].unique()))
df.head()
print("Shape of the dataset ",df.shape)
df.describe()
df.isnull().sum()
df1 = df.copy()
import matplotlib.pyplot as plt
import seaborn as sns
corr = df.corr()
plt.figure(figsize = (12,12))
sns.heatmap(corr  , linewidths= 0.01 , linecolor= "white" , cmap= "coolwarm" , annot = True).set_title("Correlation".upper())
size  = df["IsActiveMember"].value_counts(sort =True)
colors = ["magenta","mediumslateblue"]
labels = ["Yes","No"]
explode = (0, 0.1)
plt.figure(figsize=(10 , 8))
plt.pie(size,colors=colors,autopct='%1.1f%%',shadow=True,startangle = 270 ,explode= explode, labels=labels)
plt.title("Active and Non active member")
plt.show()
labels = 'active', 'non active' 
colors = ["magenta","mediumslateblue"]

sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90 , colors = colors)
ax1.axis('equal')
plt.title("Proportion of customer active and non active", size = 20)
plt.show()
plt.figure(figsize=(8,4))
sns.countplot(x='IsActiveMember', data= df , palette= "magma",hue = "Gender")
plt.figure(figsize=(8,4))
sns.countplot(x='IsActiveMember', data= df , palette= "icefire",hue = "Geography")
plt.figure(figsize=(8,4))
sns.countplot(x='IsActiveMember', data= df , palette= "YlOrRd",hue = "Exited")
num_projects=df.groupby('Geography').count()
plt.bar(num_projects.index.values, num_projects['Gender'] , color = "slateblue" , edgecolor = "navy" ,linewidth = 5 )
plt.xlabel('Geography')
plt.ylabel('Gender')
plt.show()
num_projects=df.groupby('NumOfProducts').count()
plt.bar(num_projects.index.values, num_projects['Gender'] , color = "slateblue" , edgecolor = "navy" ,linewidth = 5 )
plt.xlabel('NumOfProducts')
plt.ylabel('Gender')
plt.show()
import numpy as np
bins = [ 0, 18, 24, 35, 60,92, 100]
labels = ["Unknown",'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df["age"] = df["Age"]
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
df["AgeGroup"].isnull().sum()

age_mapping = {"Unknown":0,'Teenager': 1, 'Student': 2, 'Young Adult': 3, 'Adult': 4, 'Senior': 5}
df["AgeGroup"].fillna(value = 'Unknown' ,inplace = True)


df['AgeGroup'] = df['AgeGroup'].map(age_mapping).astype("int")

print(df["AgeGroup"].unique())
label = {'France':1, 'Germany':2, 'Spain':3}
df.replace({'Geography':label}, inplace = True)
df.head()
label = {'Female':0, 'Male':1}
df.replace({'Gender':label}, inplace = True)
df.head()
df['BalanceSalaryRatio'] = df.Balance/df.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df , palette= "bone")
plt.ylim(-1, 5)
df['TenureByAge'] = df.Tenure/(df.age)
sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df , palette = "Blues" )
plt.ylim(-1, 1)
plt.show()

df['CreditScoreGivenAge'] = df.CreditScore/(df.age)
df.head()

continuous_vars = ['CreditScore',  'AgeGroup', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df = df[['Exited'] + continuous_vars + cat_vars]
df.head()

minVec = df[continuous_vars].min().copy()
maxVec = df[continuous_vars].max().copy()
df[continuous_vars] = (df[continuous_vars]-minVec)/(maxVec-minVec)
df.head()
def DfPrepPipeline(df_predict,df_Cols,minVec,maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_Cols]
    return df_predict

# Split Train, test data
df_train = df.sample(frac=0.7 ,random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))
df_test.head()
#Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='auto',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
print("Classification Report for Logistic Regression")
print(classification_report(df_test.Exited, log_primal.predict(df_test.loc[:, df_test.columns != 'Exited'])))

poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='auto', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,df_train.Exited)
print("Classification Report for logistic regression with pol 2 kernel")

print(classification_report(df_train.Exited,  log_pol2.predict(df_train_pol2)))

SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, 
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
print("Classification Report for SVM with RBF Kernel")

print(classification_report(df_train.Exited,  SVM_RBF.predict(df_train.loc[:, df_train.columns != 'Exited'])))

SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',  max_iter=-1,
              probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_POL.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
print("Classification Report for SVM with Pol Kernel")

print(classification_report(df_train.Exited,  SVM_POL.predict(df_train.loc[:, df_train.columns != 'Exited'])))

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
print("Classification Report for Random Forest classifier")

print(classification_report(df_train.Exited,  RF.predict(df_train.loc[:, df_train.columns != 'Exited'])))

XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, missing=None, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
XGB.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
print("Classification Report for Extreme Gradient Boost Classifier")

print(classification_report(df_train.Exited,  XGB.predict(df_train.loc[:, df_train.columns != 'Exited'])))
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('Logistic Regression', log_primal))
#models.append(('Logistic Regression for pol 2 kernel', poly2))
models.append(('SVM for RBF kernel', SVM_RBF))
models.append(('SVM for POL kernel', SVM_POL))
models.append(('Random Forest Classifier', RF))
models.append(('XGB', XGB))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, df_train.loc[:, df_train.columns != 'Exited'], df_train.Exited , cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure(figsize =(20,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
