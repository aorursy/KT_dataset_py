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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="darkgrid")
df = pd.read_csv("/kaggle/input/lead-scoring-x-online-education/Leads X Education.csv")

pd.set_option("display.max_columns",0)
df.head()
df.describe()
df.info()
df.isna().sum().sort_values(ascending=False).head(20)
categorical_col = df.select_dtypes(exclude =["number"]).drop("Prospect ID", axis=1).columns.values

numerical_col = df.select_dtypes(include =["number"]).drop("Lead Number", axis=1).columns.values
print("CATEGORICAL FEATURES : \n {} \n\n".format(categorical_col))

print("NUMERICAL FEATURES : \n {} ".format(numerical_col))
df[numerical_col].head()
df[categorical_col].head()
def Cat_info(df, categorical_column):

    df_result = pd.DataFrame(columns=["columns","values","unique_values","null_values","null_percent"])

    

    df_temp=pd.DataFrame()

    for value in categorical_column:

        df_temp["columns"] = [value]

        df_temp["values"] = [df[value].unique()]

        df_temp["unique_values"] = df[value].nunique()

        df_temp["null_values"] = df[value].isna().sum()

        df_temp["null_percent"] = (df[value].isna().sum()/len(df)*100).round(1)

        df_result = df_result.append(df_temp)

    

    df_result.sort_values("null_values", ascending =False, inplace=True)

    df_result.set_index("columns", inplace=True)

    return df_result
df_cat = Cat_info(df, categorical_col)

df_cat
def Num_info(df, numerical_column):

    df_result = pd.DataFrame(columns=["columns","unique_values","null_values","null_percent"])

    

    df_temp=pd.DataFrame()

    for value in numerical_column:

        df_temp["columns"] = [value]

        df_temp["unique_values"] = df[value].nunique()

        df_temp["null_values"] = df[value].isna().sum()

        df_temp["null_percent"] = (df[value].isna().sum()/len(df)*100).round(1)

        df_result = df_result.append(df_temp)



    df_result.sort_values("null_values", ascending =False, inplace=True)

    df_result.set_index("columns", inplace=True)

    return df_result
df_num = Num_info(df, numerical_col)

df_num
print("No of Converted clients out of 9240: \n{}\n".format(df["Converted"].value_counts()))

print("Percentage of Converted clients: \n{}".format((df["Converted"].value_counts()/9240*100).round(2)))
def Detect_outliers(df,col):

    df_outliers = pd.DataFrame(columns = ["columns", "outliers","lower_fence","higher_fence"])

    df_temp = pd.DataFrame()

    for column in col:

        q1 = df[column].quantile(0.25)

        q3 = df[column].quantile(0.75)

        iqr = q3-q1

        fence_low = q1 - (iqr*1.5)

        fence_high = q3 + (iqr*1.5)

        outlier = df[column][((df[column]>fence_high) | (df[column]<fence_low))].count() 

        df_temp["outliers"] = [outlier]

        df_temp["columns"] = column

        df_temp["lower_fence"] = fence_low

        df_temp["higher_fence"] = fence_high

        df_outliers = df_outliers.append(df_temp)

    df_outliers.set_index("columns", inplace=True)

    df_outliers.sort_values("outliers", ascending=False, inplace=True)

    return df_outliers
continuous_col = ['TotalVisits', 'Total Time Spent on Website',

                 'Page Views Per Visit', 'Asymmetrique Activity Score',

                 'Asymmetrique Profile Score']

df_out = Detect_outliers(df,continuous_col)

df_out 
data =df.copy()
data.head()
def fill_missing_values(df,col,n):

    for column in col:

        if n is "mean":

            df[column].fillna(df[column].mean(), inplace=True)

        elif n is "mode":

            df[column].fillna(df[column].mode(), inplace=True)

        elif n is "median":

            df[column].fillna(df[column].median(), inplace=True)

        elif n is "missing":

            df[column].fillna("---Missing---", inplace=True)

        else:

            print("Enter 'mean','median','mode' or 'missing'")

 
fill_missing_values(data,categorical_col,"missing")
def plot_categorical(df,col,num):

    for column in col:

        if (df[column].nunique()<num) and (df[column].nunique()!=1) :

            fig, (ax1, ax2)= plt.subplots(1,2, figsize=(15,5))

            sns.countplot(df[column],ax=ax1, palette="husl")

            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

            sns.countplot(df[column],hue=df["Converted"],ax=ax2)

            ax2.set_xticklabels(ax1.get_xticklabels(), rotation=90)

            ax2.set_ylabel("")

            ax2.legend(["NOT CONVERTED","CONVERTED"],loc="upper right")

    for column in col:

        if df[column].nunique()>=num:

            fig, (ax1, ax2)= plt.subplots(2,1, figsize=(15,8))

            sns.countplot(df[column],ax=ax1, palette="husl")

            ax1.set_xticklabels([])

            ax1.set_xlabel("")

            sns.countplot(df[column],hue=df["Converted"],ax=ax2)

            ax2.set_xticklabels(ax1.get_xticklabels(), rotation=90)

            ax2.legend(["NOT CONVERTED","CONVERTED"], loc="upper right")
sns.color_palette("husl")

sns.set_context('talk')

plot_categorical(data, categorical_col,10)
sns.jointplot('Total Time Spent on Website',

                 'TotalVisits',df, kind ="reg")
for columns in numerical_col:

    fig, (ax1, ax2)= plt.subplots(1,2, figsize=(15,5))

    #sns.hist(df["Converted"],ax=ax1, palette="husl")

    #ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    #sns.countplot(df["Total Visits"],hue=df["Converted"],ax=ax2)

    #ax2.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    sns.boxplot(data=df, x=columns, ax=ax1)

    sns.distplot(df[columns],ax=ax2)
sns.set_context("paper")

sns.pairplot(df[numerical_col],hue="Converted")
dataset = df.copy()
dataset.head()
dataset.drop(["Prospect ID","Lead Number"], axis=1, inplace=True)

dataset.head()
fill_missing_values(dataset,categorical_col,"missing") # Replaced with "Missing" as missing values are showing very different characterstics

fill_missing_values(dataset,numerical_col,"median") #Median as there are a lot of outliers

dataset["Country"][(dataset["Country"]!="India") & (dataset["Country"]!="Missing")]="Others" # People from abroad are almost negligible in number

dataset = pd.get_dummies(dataset,drop_first=True)

dataset
# Importing libraries

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix,roc_auc_score, classification_report, roc_curve, precision_recall_curve

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import recall_score, f1_score, precision_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from imblearn.over_sampling import SMOTE



!pip install imblearn
# Seprating dependant and independant data

X = dataset.iloc[:,1:]

y = dataset.iloc[:,0]



ss = StandardScaler()

ss.fit_transform(X)
# function to analyse all model without oversampling

def run_model(X, y, models):

    

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=20)

    

    for model in models:

        m = model()

        m.fit(X_train, y_train)

        y_pred = m.predict(X_test)

        fpr, tpr, threshold = roc_curve(y_test,y_pred)

        print(str(model)+ "\n")

        print("CONFUSION MATRIX: \n{}\n".format(confusion_matrix(y_test,y_pred)))

        print("CLASSIFICATION REPORT: \n{} ".format(classification_report(y_test,y_pred)))

        print("MODEL SCORE: {}".format(m.score(X_test,y_test)))

        print("ROC AUC SCORE: {}\n ".format(roc_auc_score(y_test,y_pred)))

        

        print("RECALL: \n{}\n ".format(cross_val_score(m,X,y,scoring="recall").mean()))

        plt.plot(fpr, tpr)

        plt.show()

        precision, recall, thresholds=precision_recall_curve(y_test,y_pred)

        print(precision)

        print(recall)

        print(thresholds)

        print("===="*20)
#models = {"RandomForestClassifier": RandomForestClassifier()}

models = [LogisticRegression,DecisionTreeClassifier,RandomForestClassifier, 

          AdaBoostClassifier, GradientBoostingClassifier,XGBClassifier]

run_model(X,y,models)
# function to analyse all model with oversampling

def run_model_balanced(X, y, models):

    

    

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=20)

    

    

    smote=SMOTE()

    b_X,b_y = smote.fit_sample(X_train,y_train)

    

    for model in models:

        m = model()

        m.fit(b_X, b_y)

        y_pred = m.predict(X_test)

        fpr, tpr, threshold = roc_curve(y_test,y_pred)

        print(str(model)+ "\n")

        print("CONFUSION MATRIX: \n{}\n".format(confusion_matrix(y_test,y_pred)))

        print("CLASSIFICATION REPORT: \n{} ".format(classification_report(y_test,y_pred)))

        print("MODEL SCORE: {}".format(m.score(X_test,y_test)))

        print("ROC AUC SCORE: {}\n ".format(roc_auc_score(y_test,y_pred)))

        print("RECALL: \n{}\n ".format(cross_val_score(m,X,y,scoring="recall").mean()))

        plt.plot(fpr, tpr)

        plt.show()

        print("===="*20)
models = [LogisticRegression,DecisionTreeClassifier,RandomForestClassifier, 

          AdaBoostClassifier, GradientBoostingClassifier,XGBClassifier]

run_model_balanced(X,y,models)
param={     "n_estimators":[int(x) for x in np.linspace(start = 10, stop = 190, num = 10) ],

            #"criterion":['gini','entropy'],

            "max_depth":[int(x) for x in np.linspace(50, 500, num = 6)],

            # "min_samples_split":[2, 5, 10],

            # "min_samples_leaf":[1, 2, 4],

            "max_features":['sqrt',"log2"],

            # "max_samples": [0.1,0.2,0.3],

            # "max_leaf_nodes": [25,50,75]     

      }



rf_classifier = RandomForestClassifier()

cv = RandomizedSearchCV(rf_classifier,param, verbose=2, scoring="recall", refit="recall")
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=20)
cv.fit(X_train, y_train)
cv.best_params_ #random
classifier_rf = RandomForestClassifier(n_estimators=170, max_depth= 230, max_features= "sqrt")

classifier_rf.fit(X_train,y_train)

y_pred=classifier_rf.predict(X_test)

fpr, tpr, threshold = roc_curve(y_test,y_pred)

print("CONFUSION MATRIX: \n{}\n".format(confusion_matrix(y_test,y_pred)))

print("CLASSIFICATION REPORT: \n{} ".format(classification_report(y_test,y_pred)))

print("MODEL SCORE: {}".format(classifier_rf.score(X_test,y_test)))

print("ROC AUC SCORE: {}\n ".format(roc_auc_score(y_test,y_pred)))

print("RECALL: \n{}\n ".format(recall_score(y_test,y_pred)))

plt.plot(fpr, tpr)

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.show()
pd.Series(classifier_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
## Hyper Parameter Optimization

params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]  

}
classifier=XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='recall', n_jobs=-1,cv=5,verbose=2)

random_search.fit(X_train,y_train)
random_search.best_estimator_
classifier_xgb = XGBClassifier(colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,min_child_weight=3)



classifier_xgb.fit(X_train,y_train)

y_pred=classifier_xgb.predict(X_test)

fpr, tpr, threshold = roc_curve(y_test,y_pred)

print("CONFUSION MATRIX: \n{}\n".format(confusion_matrix(y_test,y_pred)))

print("CLASSIFICATION REPORT: \n{} ".format(classification_report(y_test,y_pred)))

print("MODEL SCORE: {}".format(classifier_xgb.score(X_test,y_test)))

print("ROC AUC SCORE: {}\n ".format(roc_auc_score(y_test,y_pred)))

print("RECALL: \n{}\n ".format(recall_score(y_test,y_pred)))

plt.plot(fpr, tpr)

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.show()
pd.Series(classifier_xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
classifier_log = LogisticRegression()

classifier_log.fit(X_train, y_train)

y_pred_prob_log = classifier_log.predict_proba(X_train)

y_pred_prob_log = y_pred_prob_log[:,1]
def to_labels(pos_probs, threshold):

    return (pos_probs >= threshold).astype('int')
thresholds = np.arange(0, 1, 0.001)

scores_recall = [recall_score(y_train, to_labels(y_pred_prob_log, t)) for t in thresholds]

scores_precision = [precision_score(y_train, to_labels(y_pred_prob_log, t)) for t in thresholds]

data = {"thresholds": thresholds,"recall":scores_recall,"precision":scores_precision}

final_df = pd.DataFrame(data)

final_df = final_df[final_df.precision>=0.8]



final_df.head()
THRESHOLD=0.208

final_y_pred = classifier_log.predict_proba(X_test)

final_y_pred = np.where(final_y_pred[:,1]>THRESHOLD,1,0)



fpr, tpr, threshold = roc_curve(y_test,final_y_pred)

print("CONFUSION MATRIX: \n{}\n".format(confusion_matrix(y_test,final_y_pred)))

print("CLASSIFICATION REPORT: \n{} ".format(classification_report(y_test,final_y_pred)))

print("ROC AUC SCORE: {}\n ".format(roc_auc_score(y_test,final_y_pred)))

print("RECALL: \n{}\n ".format(recall_score(y_test,final_y_pred)))

print("PRECISION: \n{}\n ".format(precision_score(y_test,final_y_pred)))

plt.plot(fpr, tpr)

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.show()
def lead_score(X):

    score = pd.Series((classifier_log.predict_proba(X)[:,1]*100).round(), name="score")

    X = pd.concat([X,score], axis=1)

    return X
leadscore = lead_score(X)
leadscore
leadscore["score"]