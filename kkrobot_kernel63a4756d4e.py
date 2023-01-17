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


%matplotlib inline

import math, time, random, datetime

import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv
import time
def timer(self, threshold):
    if (time.time() - self.lastTime) > threshold:
        self.lastTime = time.time()
        return True
    else:
        return False
# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')
df_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_test=pd.read_csv("/kaggle/input/titanic/test.csv")
df_train=pd.read_csv("/kaggle/input/titanic/train.csv")
len(df_gender) #length of gender table

len(df_test)  #lenght of test table
len(df_train) #length of train table
df_train.head(11)
df_test.head(11)
df_gender.head(11)
df_gender.describe().columns

df_train.describe().columns

df_test.describe().columns


df_num = df_train[['Age','SibSp','Parch','Fare','Survived']]
df_cat = df_train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

print(df_num.corr())
sns.heatmap(df_num.corr())
for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()
for n in df_num.columns:
    sns.barplot(df_num[n].value_counts().index,df_num[n].value_counts()).set_title(n)
    plt.show()
missingno.matrix(df_train, figsize = (16,20))
df_train.isnull().sum()
df_bin=pd.DataFrame()
df_con=pd.DataFrame()
df_train.dtypes

df_train.Survived.isnull().sum()
df_train.Survived.value_counts()       # count or number of survived passanger

sns.set(style='dark')
ax=sns.countplot(y='Survived' ,palette='gist_rainbow_r', data=df_train);
sns.distplot(df_train.Survived , color = 'g')
df_bin['Survived'] = df_train['Survived']
df_con['Survived'] = df_train['Survived']   #add data to our subset dataframes
df_train.Pclass.isnull().sum()
sns.set(style='dark')
ax=sns.countplot(y='Pclass' ,palette='gist_rainbow', data=df_train);
sns.distplot(df_train.Pclass)
df_bin["Pclass"]=df_train["Pclass"]
df_con["Pclass"]=df_train["Pclass"]  # #add data to our subset dataframes
df_train.Name.isnull().sum()

df_train.Sex.isnull().sum()
plt.figure(figsize=(20,2))
sns.countplot(y='Sex' , data=df_train , palette='RdPu_r')
sns.set(style="darkgrid")
df_bin['Sex'] = df_train['Sex']
df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female

df_con['Sex'] = df_train['Sex']

df_train.Age.isnull
ax=df_train.Age.isnull().sum() # total of age null value
total=df_train.Age.count()  # ototal of age with null and filled value
filled=total-ax
filled                    #diference between total of age - null value
sns.distplot(df_train.Age)

df_train.SibSp.count()
df_train.SibSp.isnull().sum()
df_train.SibSp.value_counts()
plt.figure(figsize=(4,10))

plt.title("SibSp  , by Survived")

sns.barplot(x=df_train.SibSp , y=df_train.Survived)

plt.ylabel(" Average Children")
df_bin['SibSp'] = df_train['SibSp']
df_con['SibSp'] = df_train['SibSp'] #add data to our subset dataframes

def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
df_train.Parch.isnull().sum()
plot_count_dist(df_train, 
                bin_df=df_bin, 
                label_column='Survived', 
                target_column='SibSp', 
                figsize=(20, 10))
df_bin['Parch'] = df_train['Parch']
df_con['Parch'] = df_train['Parch']
df_train.Fare.isnull().sum()
plot_count_dist(df_train, 
                bin_df=df_bin, 
                label_column='Fare', 
                target_column='SibSp', 
                figsize=(10, 5))
df_bin['Fare'] = df_train['Fare']
df_con['Fare'] = df_train['Fare']
df_train.Embarked.isnull().sum()
df_bin['Embarked'] = df_train['Embarked']

df_con['Embarked'] = df_train['Embarked']
print(len(df_con))
df_con = df_con.dropna(subset=['Embarked'])
df_bin = df_bin.dropna(subset=['Embarked'])
print(len(df_con))
sns.countplot(y='Embarked', data=df_train);
df_bin.head()
encode = df_bin.columns.tolist()
encode.remove('Survived')
df_binencode = pd.get_dummies(df_bin, columns=encode)

df_binencode.head()
df_con.head()
df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], 
                                     prefix='embarked')

df_sex_one_hot = pd.get_dummies(df_con['Sex'], 
                                prefix='sex')

df_plcass_one_hot = pd.get_dummies(df_con['Pclass'], 
                                   prefix='pclass')
df_con_enc = pd.concat([df_con, 
                        df_embarked_one_hot, 
                        df_sex_one_hot, 
                        df_plcass_one_hot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df_con_enc.tail()
df_con_enc.describe()
df_con_enc.count()
df_con_enc
ML = df_con_enc

ML.shape
ML
ML.tail()
X_train=ML.drop("Survived" , axis=1)
y_train=ML.Survived
X_train.shape  ##ROW AND COLUMBS OF X_train
y_train.shape  ## ROWS OF y_train/?
def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv
start=time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(),X_train,y_train,10)
log=(time.time()- start)
print("Accuracy:%s "  %acc_log)
print("Accuracy CV 10-Fold :%s" %acc_cv_log)
print("Running time:%s" %datetime.timedelta(seconds=log))
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                y_train, 
                                                                10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
import time
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))
X_train.head()
X_train.tail()
y_train.describe()
features=np.where(X_train.dtypes  != np.float)[0]
features
pool_data=Pool(X_train ,  y_train , features)
Model=CatBoostClassifier(iterations=5000,custom_loss=['Accuracy'])
Model.fit(pool_data,plot=True)

metrics = ['Precision', 'Recall', 'F1', 'AUC']

eval_metrics = Model.eval_metrics(pool_data,metrics=metrics,plot=True)
for metric in metrics:
    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
X_train.head()
df_test.head()

df_test_embarked = pd.get_dummies(df_test['Embarked'], prefix='embarked')

df_test_sex = pd.get_dummies(df_test['Sex'], prefix='sex')

df_test_plcass = pd.get_dummies(df_test['Pclass'],prefix='pclass')
final= pd.concat([df_test, 
                  df_test_embarked, 
                  df_test_sex, 
                  df_test_plcass], axis=1)
final.head
final.tail()
final.describe()
wante = X_train.columns
wante
predict = Model.predict(final[wante])
predict[:2000]
submit = pd.DataFrame()
submit['PassengerId'] = df_test['PassengerId']
submit['Survived'] = predict # our model predictions on the test dataset

submit
submit.columns
submit['Survived'] = submit['Survived'].astype(int)  #convert our submission dataframe 'Survived' column to ints
print('DONE')
if len(submit) == len(df_test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submit)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")
#Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submit.to_csv('../catboost_submission.csv', index=False)
print('Submission is ready!')

# Check the submission csv to make sure it's in the right format
submissions_check = pd.read_csv("../catboost_submission.csv")
submissions_check.head()

final_submit=pd.read_csv("../catboost_submission.csv")
final_submit
final_submit.head


