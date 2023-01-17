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
df=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.info()
df.describe().style.background_gradient()
for i in df.columns:
    print("{} has ------------------------------------->{} unique values ".format(i,df[i].nunique()))
import matplotlib.pyplot as plt
import seaborn as sns
target="Outcome"
sns.countplot(df[target])
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
def plot_cont(df,con_ft,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[con_ft]),1):
        plt.subplot(len(list(con_ft)), 3, i)
#         sns.distplot(df[feature],kde=False)
        sns.boxplot(x=df[feature])
        plt.xlabel('{}'.format(feature))
        plt.title('skewness : {:.2f} kurtosis : {:.2f}'.format(df[feature].skew(),df[feature].kurtosis()),fontsize=12)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})
    plt.show()

cont_ft=[i for i in df.columns if i!=target]
plot_cont(df,cont_ft,(10,20))
def plot_cont_target(df,target,con_ft,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[con_ft]),1):
        plt.subplot(len(list(con_ft)), 3, i)
#         sns.distplot(df[feature],kde=False)
#         sns.violinplot(x=df[target],y=df[feature])
        sns.boxplot(x=df[target],y=df[feature])
        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})
    plt.show()

plot_cont_target(df,target,cont_ft,(10,20))
df[df["Insulin"]>800]
df.drop(df[df["Insulin"]>800].index,axis=0,inplace=True)
df.drop(df[df["Glucose"]==0].index,axis=0,inplace=True)
df.drop(df[df["SkinThickness"]>90].index,axis=0,inplace=True)

df
sns.pairplot(df,hue="Outcome")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
def make_model(df,target,model):
    X=df.drop(columns=[target])
    y=df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15615)
    model.fit(X_train,y_train)
    y_true=y_test
    y_pred=model.predict(X_test)
    print(classification_report(y_true, y_pred))
    plt.figure(figsize = (8,6))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(confusion_matrix(y_true,y_pred), cmap="Blues", annot=True)# font size
    plt.title(model)
    return model
baseline_logreg=LogisticRegression(max_iter=10000)
baseline_logreg=make_model(df,target,baseline_logreg)
from sklearn.utils import resample
df_log=np.log1p(df.drop("Outcome",axis=1))
df_log["Outcome"]=df["Outcome"]
df_minority = df_log[df_log["Outcome"]==1]
df_majority = df_log[df_log["Outcome"]==0]
df_upsampled = resample(df_minority, 
                 replace=True,     # sample with replacement
                 n_samples=500,    # to match majority class
                 random_state=123) # reproducible results 
df_train = pd.concat([df_majority, df_upsampled])
df_train
df_train.corr().style.background_gradient("Oranges")
sns.countplot(df_train[target])
baseline_logreg=make_model(df_train,target,baseline_logreg)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
baseline_dtree=make_model(df_train,target,DecisionTreeClassifier())
baseline_forest=make_model(df_train,target,RandomForestClassifier())
def roc_auc(y_val,y_pred):
    from sklearn.metrics import RocCurveDisplay
    from sklearn.metrics import roc_auc_score,roc_curve,auc
    from sklearn import metrics

    fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'red', label = 'ROC AUC score = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
def eval(model):
    X=df_log.drop(columns=[target])
    y=df_log[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    y_true=y_test
    y_pred=model.predict(X_test)
    print(classification_report(y_true, y_pred))
    plt.figure(figsize = (8,6))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(confusion_matrix(y_true,y_pred), cmap="Blues", annot=True)# font size
    plt.show()
    roc_auc(y_true,y_pred)
eval(baseline_forest)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3,4,5,6,7],
    'min_samples_leaf': [3, 4, 5,10],
    'min_samples_split': [8, 10, 12,20],
    'n_estimators': [100, 200, 300, 1000]
}

grid_rf=RandomForestClassifier()
grid_search = GridSearchCV(estimator = grid_rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_rf=make_model(df_train,target,grid_rf)
eval(grid_rf)
