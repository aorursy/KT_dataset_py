# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# reference_idea_from https://www.kaggle.com/angps95/basic-classification-methods-for-titanic
%matplotlib inline 
import math
import numpy as np 
import scipy as sp 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
# Load SOLAR POWER GENERATION dataset
root_path = '/kaggle/input/titanic/'
df_train  = pd.read_csv(f"{root_path}/train.csv")
df_test   = pd.read_csv(f"{root_path}/test.csv") 
df_label = pd.read_csv(f"{root_path}/gender_submission.csv") 

#, parse_dates=['DATE_TIME'], infer_datetime_format=True)
df_train.head(5)
def survival_stacked_bar(variable, df=df_train):
    Died       = df[df["Survived"]==0][variable].value_counts()/len(df["Survived"]==0)
    Survived   = df[df["Survived"]==1][variable].value_counts()/len(df["Survived"]==1)
    data       = pd.DataFrame([Died,Survived])
    data.index = ["Death","Survived"]
    #
    data.plot(kind="bar",stacked=True,title="Percentage")
    return data.head()
# Compare death - survive case by Passenger class
survival_stacked_bar('Pclass')
# Compare death - survive case by Sex
survival_stacked_bar('Sex')
# Compare death - survive case by Port
survival_stacked_bar('Embarked')
# review missing data of training set
print(df_train.info())
print(df_test.info())
# Extract Title from Name
def extract_title(df):

    # Split title from passenger name
    df['title'] = df['Name'].map(lambda name: name.split(",")[1].split(".")[0].strip().lower())

    # Map title with feature labelling
    title_map={'mr':1, 'mrs':1, 'miss':1, 'mlle': 1, 'mme': 1, 'ms': 1,
               'master':2,
               'capt':3, 'col': 3, 'major': 3, 'dr': 3, 'rev': 3,
               'lady':4, 'sir': 4, 'the countess': 4, 'don': 4, 'dona': 4, 'jonkheer': 4 }

    df["title"] = df["title"].map(title_map)

    # review extract result
    print(df["title"].unique())

extract_title(df_train)
extract_title(df_test)
# Replace Embarked port
# Fill blank Embarked with 'S'
def extract_embark(df):
    df_train["Embarked"]    = df_train["Embarked"].fillna("S") # 'S' port is the most port passengers came from.
    # df_train['embarked_id'] = df_train['Embarked'].replace(['S','C','Q'],[0,1,2])
    print(df_train['Embarked'].unique())

extract_embark(df_train)
extract_embark(df_test)
# Replace Sex
def extract_sex(df):
    df['sex_id'] = df['Sex'].replace(['male','female'],[0,1])
    print(df['sex_id'].unique())

extract_sex(df_train)
extract_sex(df_test)
# Cleansing [Cabin]
def extract_cabin(df):
    df["Cabin"] = df["Cabin"].fillna("U")
    df["Cabin"] = df["Cabin"].map(lambda x: x[0])
    print(df['Cabin'].unique())

extract_cabin(df_train)
extract_cabin(df_test)
def extract_age(df):
    average_age = math.floor(df["Age"].mean())
    df["Age"] = df["Age"].fillna(average_age)

extract_age(df_train)
extract_age(df_test)
def extract_fare(df):
    average_fare = math.floor(df["Fare"].mean())
    df["Fare"] = df["Fare"].fillna(average_fare)

extract_fare(df_train)
extract_fare(df_test)
# Re-check Null data again <3
print(df_train.info())
print(df_test.info())
df_train.head(3)
import seaborn as sns
sns.set(style="whitegrid")

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df_train.corr(), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
def one_hot_cabin_features(dataset):
    dataset["Cabin A"]=np.where(dataset["Cabin"]=="A",1,0)
    dataset["Cabin B"]=np.where(dataset["Cabin"]=="B",1,0)
    dataset["Cabin C"]=np.where(dataset["Cabin"]=="C",1,0)
    dataset["Cabin D"]=np.where(dataset["Cabin"]=="D",1,0)
    dataset["Cabin E"]=np.where(dataset["Cabin"]=="E",1,0)
    dataset["Cabin F"]=np.where(dataset["Cabin"]=="F",1,0)
    dataset["Cabin G"]=np.where(dataset["Cabin"]=="G",1,0)
    dataset["Cabin T"]=np.where(dataset["Cabin"]=="T",1,0) 

one_hot_cabin_features(df_test)
one_hot_cabin_features(df_train)
def one_hot_embarked_features(dataset):
    dataset["Embarked S"]=np.where(dataset["Embarked"]=="S",1,0)
    dataset["Embarked C"]=np.where(dataset["Embarked"]=="C",1,0)
    dataset["Embarked Q"]=np.where(dataset["Embarked"]=="Q",1,0)
    
one_hot_embarked_features(df_test)
one_hot_embarked_features(df_train)
def drop_unused(dataset):
    dataset.drop(['Ticket','Name','Sex','Cabin','Embarked'],inplace=True,axis=1)
drop_unused(df_train)
drop_unused(df_test)
# Prepare train - test dataset
x_train = df_train.drop(["Survived"],axis=1)
y_train = df_train["Survived"]

x_test  = df_test
y_test  = df_label["Survived"]
print(x_train.shape , y_train.shape )
print(x_test.shape , y_test.shape )
x_train.head(3)
y_train
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import scikitplot as skplt
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
# KFold cross validation
split  = 5 
k_fold = KFold(n_splits=split, shuffle=True, random_state=0)
# Accuracy mean score
def acc_score(model):
    return np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy"))
# Define confision matrix
def confusion_matrix_model(model_used):
    cm  = confusion_matrix(y_test,model_used.predict(x_test))
    col = ["Predicted Dead","Predicted Survived"]
    cm  = pd.DataFrame(cm)
    cm.columns = ["Predicted Dead","Predicted Survived"]
    cm.index   = ["Actual Dead","Actual Survived"]
    cm[col]    = np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)
    return cm

def importance_of_features(model):
    features = pd.DataFrame()
    features['feature'] = x_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by = ['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(10,10))
# Model validation
def aucscore(model,has_proba=True):
    if has_proba:
        fpr,tpr,thresh = skplt.metrics.roc_curve(y_test,model.predict_proba(x_test)[:,1])
    else:
        fpr,tpr,thresh = skplt.metrics.roc_curve(y_test,model.decision_function(x_test))
    x   = fpr
    y   = tpr
    auc = skplt.metrics.auc(x,y)
    return auc

def plt_roc_curve(name,model,has_proba=True):
    if has_proba:
        fpr,tpr,thresh = skplt.metrics.roc_curve(y_test,model.predict_proba(x_test)[:,1])
    else:
        fpr,tpr,thresh = skplt.metrics.roc_curve(y_test,model.decision_function(x_test))
    x   = fpr
    y   = tpr
    auc = skplt.metrics.auc(x,y)
    plt.plot(x,y,label='ROC curve for %s (AUC = %0.2f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
Dec_tree=DecisionTreeClassifier(max_depth=4,random_state=5)
Dec_tree.fit(x_train,y_train)

print("Accuracy: " + str(acc_score(Dec_tree)))
confusion_matrix_model(Dec_tree)

#skplt.metrics.plot_confusion_matrix(y_test, Dec_tree.predict(x_test),normalize=True,figsize=(6,6),text_fontsize='small')
plt_roc_curve("Decision Tree",Dec_tree,has_proba=True)
ranfor = RandomForestClassifier(n_estimators=50, max_features='sqrt',max_depth=6,random_state=10)
ranfor = ranfor.fit(x_train,y_train)
print("Accuracy: " + str(acc_score(ranfor)))
confusion_matrix_model(ranfor)
plt_roc_curve("Random Forest",ranfor,has_proba=True)
naive_bayes = GaussianNB()
naive_bayes = naive_bayes.fit(x_train,y_train)
print("Accuracy: " + str(acc_score(naive_bayes)))
confusion_matrix_model(naive_bayes)
plt_roc_curve("Naive bayes",naive_bayes,has_proba=True)
multi_layer_perceptron = MLPClassifier(random_state=1, max_iter=300)
multi_layer_perceptron = multi_layer_perceptron.fit(x_train,y_train)
print("Accuracy: " + str(acc_score(multi_layer_perceptron)))
confusion_matrix_model(multi_layer_perceptron)
plt_roc_curve("Naive bayes",multi_layer_perceptron,has_proba=True)