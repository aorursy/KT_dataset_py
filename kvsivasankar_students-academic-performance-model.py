# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



from scipy import stats

from scipy.stats import norm, skew #for some statistics

from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder



from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
df = pd.read_csv('../input/xAPI-Edu-Data.csv')
df.head()
df.tail()
def DescriptiveStatistics(df):

    print("No of rwos and columns information:",df.shape)

    print("")

    print("---"*20)

    print("")

    print("Columns:")

    print("")

    print(df.columns.values)

    print("---"*20)

    print("")

    print(df.info())

    print("---"*20)

    print("")

    print(df.describe())
DescriptiveStatistics(df)
def CheckMissingInfo(df):

    print(df.isnull().sum())

    print("---"*20)

    print("")

    df_na = (df.isnull().sum() / len(df)) * 100

    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]

    missing_data = pd.DataFrame({'Missing Ratio' :df_na})

    print(missing_data)
CheckMissingInfo(df)
def GetColumnCount(df):

    int_columns = [col for col in df.columns if(df[col].dtype != "object")]

    print("No of integer type columns:",len(int_columns))

    print(int_columns)

    print("")

    obj_columns = [col for col in df.columns if(df[col].dtype == "object")]

    print("No of object type columns:",len(obj_columns))

    print(obj_columns)

    return int_columns,obj_columns
int_columns,obj_columns = GetColumnCount(df)
def GetCountPlots(df,obj_columns):

    for col in obj_columns:

        if(len(df[col].value_counts()) < 5):

            plt.figure(figsize=(5,5))

        else:

            plt.figure(figsize=(12,6))

        print(sns.countplot(x=col, data=df, palette="muted"))

        plt.show()
GetCountPlots(df,obj_columns)
def GetCardinality(df,obj_columns):

    for col in obj_columns:

        print("{0} :: {1}".format(col,len(df[col].value_counts())))

        

        print(df[col].value_counts())

        print("")
GetCardinality(df,obj_columns)
pd.crosstab(df['Class'],df['Topic'])
def GetCountPlots_with_hue(df,obj_columns,col_hue):

    for col in obj_columns:

        if(len(df[col].value_counts()) < 5):

            plt.figure(figsize=(5,5))

        else:

            plt.figure(figsize=(12,6))

        #print(sns.countplot(x=col, data=df, palette="muted"))

        sns.countplot(x=col,data = df, hue=col_hue,palette='bright')

        plt.show()
GetCountPlots_with_hue(df,obj_columns,'Class')
def GetBoxPlots(df,x_col):

    for col in int_columns:

        plt.figure(figsize=(5,5))

        boxplot1 = sns.boxplot(x=x_col, y=col, data=df)

        boxplot1 = sns.swarmplot(x=x_col, y=col, data=df, color=".15")

        plt.show()
GetBoxPlots(df,'Class')
df['Failed'] = np.where(df['Class']=='L',1,0)
df['AbsBoolean'] = df['StudentAbsenceDays']

df['AbsBoolean'] = np.where(df['AbsBoolean'] == 'Under-7',0,1)

df['AbsBoolean'].groupby(df['Topic']).mean()
df.head()
df.info()
def NumaricVariablesDistributions(df):

    int_columns=df.columns[df.dtypes==int]

    plt.figure(figsize=(10,7))

    for i, column in enumerate(int_columns):

        plt.subplot(3,2, i+1)

        sns.distplot(df[column], label=column, bins=10, fit=norm)

        plt.ylabel('Density');
NumaricVariablesDistributions(df)
def ApplyBoxcoxTransformation(df,columns):

    plt.figure(figsize=(10,7))

    for i, column in enumerate(columns):

        plt.subplot(2,2, i+1)

        df[column]=boxcox1p(df[column], 0.3)

        sns.distplot(df[column], label=column, bins=10, fit=norm)

        plt.ylabel('Density')
ApplyBoxcoxTransformation(df,['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'])
df['raisedhands_bin']=np.where(df.raisedhands>df.raisedhands.mean(),1,0)

df['VisITedResources_bin']=np.where(df.VisITedResources>df.VisITedResources.mean(),1,0)
GetBoxPlots(df,'Class')
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap='RdBu');
sns.pairplot(df);
print('Percent of students\' nationality - Kuwait or Jordan: {}'.format(

            round(100*df.NationalITy.isin(['KW','Jordan']).sum()/df.shape[0],2)))
target=df['Class']

df=df.drop('Class', axis=1)
df.head()
#Create new feature - type of topic (technical, language, other)

Topic_types={'Math':'technic', 'IT':'technic','Science':'technic','Biology':'technic',

 'Chemistry':'technic', 'Geology':'technic', 'Arabic':'language', 'English':'language',

 'Spanish':'language','French':'language', 'Quran':'other' ,'History':'other'}

df['Topic_type']=df.Topic.map(Topic_types)
df.head()
int_columns,obj_columns = GetColumnCount(df)
def ApplyScaling(df):

    for column in ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']:

        SS=StandardScaler().fit(df[[column]])

        df[[column]]=SS.transform(df[[column]])
ApplyScaling(df)
df.head()
int_columns,obj_columns = GetColumnCount(df)
def LabelEncoding(df):

    for column in obj_columns:

        #Binarize and LabelEncode

        if ((df[column].value_counts().shape[0]==2) | (column=='StageID') | (column=='GradeID')):

            le=LabelEncoder().fit(df[column])

            df[column]=le.transform(df[column])

    
LabelEncoding(df)
df.head()
#One-hot encoding

df=pd.get_dummies(df)
df.head()
df.columns
from sklearn.metrics import make_scorer, accuracy_score,roc_auc_score,confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.20, random_state=42)
#using cross_val_score

logis = LogisticRegression()

svm = SVC()

knn = KNeighborsClassifier()

dTmodel = DecisionTreeClassifier()

rForest = RandomForestClassifier()

grBoosting = GradientBoostingClassifier()

    

scores = cross_val_score(logis,x_train,y_train,cv=5)

print("Accuracy for logistic regresion: mean: {0:.2f} 2sd: {1:.2f}".format(scores.mean(),scores.std() * 2))

print("Scores::",scores)

print("\n")



scores2 = cross_val_score(svm,x_train,y_train,cv=5)

print("Accuracy for SVM: mean: {0:.2f} 2sd: {1:.2f}".format(scores2.mean(),scores2.std() * 2))

print("Scores::",scores)

print("\n")



scores3 = cross_val_score(knn,x_train,y_train,cv=5)

print("Accuracy for KNN: mean: {0:.2f} 2sd: {1:.2f}".format(scores3.mean(),scores3.std() * 2))

print("Scores::",scores)

print("\n")



scores4 = cross_val_score(dTmodel,x_train,y_train,cv=5)

print("Accuracy for Decision Tree: mean: {0:.2f} 2sd: {1:.2f}".format(scores4.mean(),scores4.std() * 2))

print("Scores::",scores4)

print("\n")



scores5 = cross_val_score(rForest,x_train,y_train,cv=5)

print("Accuracy for Random Forest: mean: {0:.2f} 2sd: {1:.2f}".format(scores5.mean(),scores5.std() * 2))

print("Scores::",scores5)

print("\n")



scores6 = cross_val_score(grBoosting,x_train,y_train,cv=5)

print("Accuracy for Gradient Boosting: mean: {0:.2f} 2sd: {1:.2f}".format(scores6.mean(),scores6.std() * 2))

print("Scores::",scores6)

print("\n")
from sklearn.metrics import roc_auc_score



def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):



  #creating a set of all the unique classes using the actual class list

  unique_class = set(actual_class)

  roc_auc_dict = {}

  for per_class in unique_class:

    #creating a list of all the classes except the current class 

    other_class = [x for x in unique_class if x != per_class]



    #marking the current class as 1 and all other classes as 0

    new_actual_class = [0 if x in other_class else 1 for x in actual_class]

    new_pred_class = [0 if x in other_class else 1 for x in pred_class]



    #using the sklearn metrics method to calculate the roc_auc_score

    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)

    roc_auc_dict[per_class] = roc_auc



  return roc_auc_dict
def modelling(model,model_name):

    print(model)

    print("\n")

    model.fit(x_train, y_train)

    preds=model.predict(x_test)

    preds_proba=model.predict_proba(x_test)

    print('Accuracy = {}'.format(100*round(accuracy_score(y_test,preds),2)))

    print(classification_report(y_test, preds))

    

    print("\n")

    print(model_name)

    lr_roc_auc_multiclass = roc_auc_score_multiclass(y_test, preds)

    print("AUC Score for each lable")

    print(lr_roc_auc_multiclass)

    print("\n")

    plt.figure(figsize=(7,5))

    sns.heatmap(confusion_matrix(y_test,preds), annot=True, vmax=50)

    plt.show()
modelling(LogisticRegression(),"Logistic Regression")
modelling(GradientBoostingClassifier(),"Gradient Boosting")
# Grid search cross validation



grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x_train,y_train)



print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("accuracy :",logreg_cv.best_score_)
modelling(LogisticRegression(C=1.0,penalty='l1'),"Logistic Regression tuned")