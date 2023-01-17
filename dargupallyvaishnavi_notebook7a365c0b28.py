# The information are from here:
# https://www.kaggle.com/sk1812/titanic-ml-model

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# ML algorithms from scikit;
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Get train/test data
# Notice that train and test have same columns EXCEPT survial;
titanic = pd.read_csv('../input/titanic/train.csv')
titanic.head(10)
titanic_test = pd.read_csv('../input/titanic/test.csv')
titanic_test.head(10)
# TO DO
# Describe the data;
titanic.info()
titanic_test.info()
titanic.describe() 
titanic_test.describe() 
# Function to check the missing percent of a DatFrame;
def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

# TODO
# Check the data missing rate of titanic and titanic_test;
check_missing_data(titanic)

titanic.drop("PassengerId", axis=1, inplace=True)
titanic.head()
# TOTO
# Fill cabin with 0 if NaN; otherwise 1. Do this for both titanic and titanic_test;
titanic['Cabin'] = titanic['Cabin'].replace(np.nan, 0)
titanic.loc[titanic["Cabin"] != 0, "Cabin"] = 1

titanic_test['Cabin'] = titanic_test['Cabin'].replace(np.nan, 0)
titanic_test.loc[titanic_test["Cabin"] != 0, "Cabin"] = 1

titanic["Cabin"] = titanic["Cabin"].astype(np.int32)
titanic_test["Cabin"] = titanic_test["Cabin"].astype(np.int32)

titanic.describe() 
titanic["Age"].mean()
# TODO
# Fill 'Age' and 'Fare' missing data;
titanic["Age_aver"] = titanic["Age"].fillna(titanic["Age"].mean(),inplace = False)
titanic_test["Age_aver"] = titanic_test["Age"].fillna(titanic_test["Age"].mean(),inplace = False)

titanic["Fare_aver"] = titanic["Fare"].fillna(titanic["Fare"].mean(),inplace = False)
titanic_test["Fare_aver"] = titanic["Fare"].fillna(titanic_test["Fare"].mean(),inplace = False)
# TODO
# Chekck again if there is missing data;

check_missing_data(titanic)
# check_missing_data(titanic_test)
check_missing_data(titanic_test)
titanic["Embarked"].value_counts()

titanic["Embarked"].fillna("S",inplace = True)
check_missing_data(titanic)
# Function of drawing graph;
def draw(graph):
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
# def draw_data(data = titanic, con):
    
#     # Function of drawing graph;
#     def draw(graph):
#         for p in graph.patches:
#             height = p.get_height()
#             graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
            
#     for column in data.columns():
#         # Draw survided vs. non-survived of trainign data.
#         sns.set(style="darkgrid")
#         plt.figure(figsize = (8, 5))
#         graph= sns.countplot(x=column, hue="Survived", data=titanic)
#         draw(graph)
# Draw survided vs. non-survived of trainign data.
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph= sns.countplot(x='Survived', hue="Survived", data=titanic)
draw(graph)
 #Cabin and survived;
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Cabin", hue ="Survived", data = titanic)
draw(graph)

# TODO
# Do other plots, such as Sex vs Survived; Pclass vs Survived, and so on;
# Cabin and survived;
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Sex", hue ="Survived", data = titanic)
draw(graph)
# Cabin and survived;
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Pclass", hue ="Survived", data = titanic)
draw(graph)

# Cabin and survived;
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Cabin", hue ="Survived", data = titanic)
draw(graph)
# TODO
# Correlation of columns;
corr_matrix = titanic.corr()
sns.heatmap(corr_matrix, annot=True);

all_data = [titanic, titanic_test]
# Convert ‘Sex’ feature into numeric.
genders = {"male": 0, "female": 1}

for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(genders)
titanic['Sex'].value_counts()


# Use bin to group ages to bins;
for dataset in all_data:
    dataset['Age_aver'] = dataset['Age_aver'].astype(int)
    dataset.loc[ dataset['Age_aver'] <= 15, 'Age_aver'] = 0
    dataset.loc[(dataset['Age_aver'] > 15) & (dataset['Age_aver'] <= 20), 'Age_aver'] = 1
    dataset.loc[(dataset['Age_aver'] > 20) & (dataset['Age_aver'] <= 26), 'Age_aver'] = 2
    dataset.loc[(dataset['Age_aver'] > 26) & (dataset['Age_aver'] <= 28), 'Age_aver'] = 3
    dataset.loc[(dataset['Age_aver'] > 28) & (dataset['Age_aver'] <= 35), 'Age_aver'] = 4
    dataset.loc[(dataset['Age_aver'] > 35) & (dataset['Age_aver'] <= 45), 'Age_aver'] = 5
    dataset.loc[ dataset['Age_aver'] > 45, 'Age_aver'] = 6
titanic['Age'].value_counts()
# Combine SibSp and Parch as new feature; 
# Combne train test first;
all_data=[titanic,titanic_test]

for dataset in all_data:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create categorical of fare to plot fare vs Pclass first;
for dataset in all_data:
    dataset['Fare_cat'] = pd.cut(dataset['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','median_fare','Average_fare','high_fare'])
plt.figure(figsize = (8, 5))
ag = sns.countplot(x='Pclass', hue='Fare_cat', data=titanic)

# Re-organize the data; keep the columns with useful features;
input_cols = ['Pclass',"Sex","Age_aver","Cabin","Family"]
output_cols = ["Survived"]

# models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


# other utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np
import statsmodels.stats.proportion as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

def algorithm_show(data, input_cols, output_cols):
    """
    check the score of used algorithms
    
    @author: chenshuyi
    
    """

    temp = data.copy()
    temp.reset_index(drop=True, inplace = True)

    temp_X = temp[input_cols]
    temp_X = pd.get_dummies(temp_X)
    # print(temp_X.head(1))
    X_train, X_test, y_train, y_test = train_test_split(temp_X, temp[output_cols], test_size=0.25, random_state=20026, stratify=temp[output_cols])

    # modeling
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6,class_weight='balanced',random_state=20026)
    rf_model.fit(X_train,y_train.values.ravel())
    
    lr_model = LogisticRegression(class_weight='balanced',random_state=20026)
    lr_model.fit(X_train,y_train.values.ravel())
    
    knn_3 = KNeighborsClassifier(n_neighbors=3)
    knn_3.fit(X_train,y_train)
    knn_5 = KNeighborsClassifier(n_neighbors=5)
    knn_5.fit(X_train,y_train.values.ravel())
    
    clf = svm.SVC()
    clf.fit(X_train,y_train.values.ravel())
    
    clf_lin = svm.LinearSVC()
    clf_lin.fit(X_train,y_train.values.ravel())
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train.values.ravel())
    
    lgb_model = LGBMClassifier(max_depth=6,class_weight='balanced',random_state=20026)
    lgb_model.fit(X_train,y_train.values.ravel())
    
    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train,y_train.values.ravel())
    
    n=len(y_train[y_train[output_cols]==0])/len(y_train[y_train[output_cols]==1])
    xgb_model = XGBClassifier(max_depth=6,scale_pos_weight=n,random_state=20026,importance_type='weight')
    xgb_model.fit(X_train,y_train.values.ravel())

    def pred_result(model,x,y):
        ypred = model.predict(x)
        
        # SVM dont have prabability prediction, so no auc_score
        try:
            pscore = model.predict_proba(x)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y, pscore)
            auc_score = metrics.auc(fpr, tpr)
        except:
            auc_score = "NaN"
            
        accuracy = metrics.accuracy_score(y, ypred)
        precision = metrics.precision_score(y, ypred)
        precision_0 = metrics.precision_score(y, ypred,pos_label=0)
        recall = metrics.recall_score(y, ypred)
        recall_0 = metrics.recall_score(y, ypred,pos_label=0)
        f1score = metrics.f1_score(y, ypred)
        f1score_0 = metrics.f1_score(y, ypred,pos_label=0)
        accuracy_insample = model.score(X_train[input_cols],y_train)
        return [accuracy,auc_score,precision,recall,f1score,precision_0,recall_0,f1score_0,accuracy_insample]

    result = pd.DataFrame([],index=['accuracy','auc_score','precision','recall','f1score','precision_0','recall_0','f1score_0','accuracy_insample'])

    result['LogisticRegression'] = pred_result(lr_model,X_test,y_test)
    result['KNN_5'] = pred_result(knn_5,X_test,y_test)
    result['KNN_3'] = pred_result(knn_3,X_test,y_test)
    result['LSVM'] = pred_result(clf_lin,X_test,y_test)
    result['SVM'] = pred_result(clf,X_test,y_test)
    result['ngb'] = pred_result(gnb,X_test,y_test)
    result['LightGBM'] = pred_result(lgb_model,X_test,y_test)
    result['XGBoost'] = pred_result(xgb_model,X_test,y_test)
    result['decisiontree'] = pred_result(dt,X_test,y_test)
    result['RandomForest'] = pred_result(rf_model,X_test,y_test)

    return result
algorithm_show(titanic, input_cols, output_cols)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance
def change_name(dataframe):
    dataframe.loc[dataframe["Name"].str.contains("Miss.|Ms.|Ms|Countess.|Mlle."), "Name"] = "Miss"
    dataframe.loc[dataframe["Name"].str.contains("Mrs.|Mrs|Mme."), "Name"] = "Mrs"
    dataframe.loc[dataframe["Name"].str.contains("Master.|Master"), "Name"] = "Master"
    dataframe.loc[dataframe["Name"].str.contains("Mr.|Mr"), "Name"] = "Mr"
    dataframe.loc[dataframe["Name"].str.contains("Dr.|Dr|Don.|Rev.|Rev"), "Name"] = "Professional"
    dataframe.loc[dataframe["Name"].str.contains("Major.|Major|Capt.|Capt|Col| Col."), "Name"] = "Military"
    
    dataframe.loc[~dataframe["Name"]. \
           str.contains("Miss|Master|Mr|Professional|Military"), "Name"] = "NaN"
    
    dataframe.loc[(dataframe["Name"] == "NaN") &
           (dataframe["Sex"] == 1) &  (dataframe["Age"] <= 23), "Name"] = "Miss"
    dataframe.loc[(dataframe["Name"] == "NaN") &
           (dataframe["Sex"] == 1), "Name"] = "Mrs"
    dataframe.loc[(dataframe["Name"] == "NaN") &
           (dataframe["Sex"] == 0), "Name"] = "Mr"
    
    print(dataframe["Name"].value_counts())
def my_encoder(dataframe, column):
    n_encoder = OneHotEncoder(sparse=False)

    # fit encoder
    # If you run below line as "n_encoder.fit(train_X["Name"])"
    # with one square parantheses, you will get an error.
    n_encoder.fit(dataframe[[column]])

    # transform Name column of train_X
    name_one_hot_np = n_encoder.transform(dataframe[[column]])
    # convert datatype to int64
    name_one_hot_np = name_one_hot_np.astype(np.int32)
    
    alist = []
    for i in range(dataframe[column].nunique()):
        alist.append(column + "_" + str(i+1))

    # output of transform is a numpy array, convert it to Pandas dataframe
    name_one_hot_df = pd.DataFrame(name_one_hot_np,
                        columns=alist)
    # concatenate name_one_hot_df to train_X
    dataframe = pd.concat([dataframe, name_one_hot_df], axis=1)
    
    # drop categorical Name column
    # dataframe.drop(column, axis=1, inplace=True)
    dataframe.drop(column + "_" + str(i+1), axis=1, inplace=True)
    
    return dataframe

# titanic_test.drop("Fare_cat", axis=1, inplace=True)
# titanic.drop("Fare_cat", axis=1, inplace=True)

change_name(titanic)
change_name(titanic_test)

titanic=my_encoder(titanic, "Name")
titanic_test=my_encoder(titanic_test, "Name")

titanic=my_encoder(titanic, "Embarked")
titanic_test=my_encoder(titanic_test, "Embarked")
titanic.head(3)
 #Re-organize the data; keep the columns with useful features;
input_cols = ['Pclass',"Sex","Age","Cabin","Family","Fare","Name_1","Name_2","Name_3","Name_4","Embarked_1","Embarked_2"]
output_cols = ["Survived"]
def xgb_filler(train_X,test_X,column,input_cols,output_cols):
    
    reg_train = train_X[input_cols]
    reg_test = test_X[input_cols]

    # get only rows with non-null values
    reg_train_X = reg_train[reg_train[column].notnull()].copy()
    reg_train_Y = reg_train_X[[column]]
    reg_train_X.drop(column, axis=1, inplace=True)
    
    reg_test_X = reg_test[reg_test[column].notnull()].copy()
    reg_test_Y = reg_test_X[[column]]
    reg_test_X.drop(column, axis=1, inplace=True)    
    
    
    reg = xgb.XGBRegressor(colsample_bylevel=0.7,
                       colsample_bytree=0.5,
                       learning_rate=0.3,
                       max_depth=5,
                       min_child_weight=1.5,
                       n_estimators=18,
                       subsample=0.9)

    reg.fit(reg_train_X, reg_train_Y)
    
    all_predicted = reg.predict(train_X[input_cols].loc[:, (train_X[input_cols].columns != column)
                     ])
    
    train_X.loc[train_X[column].isnull(),
                    column] = all_predicted[train_X[column].isnull()]
    
    all_predicted_t = reg.predict(test_X[input_cols].loc[:, (test_X[input_cols].columns != column)
                     ])
    
    test_X.loc[test_X[column].isnull(),
                           column] = all_predicted_t[test_X[column].isnull()]

xgb_filler(titanic,titanic_test,'Age',input_cols,output_cols)
xgb_filler(titanic,titanic_test,'Fare',input_cols,output_cols)
titanic.info()

titanic_test.info()
# Get train/test data
# Notice that train and test have same columns EXCEPT survial;
titanic_cabin = pd.read_csv('../input/titanic/train.csv')["Cabin"]
titanic_test_cabin = pd.read_csv('../input/titanic/test.csv')["Cabin"]

titanic_cabin.value_counts()
147/len(titanic_cabin)
titanic['Cabin'] = titanic_cabin.str[:1]
titanic_test['Cabin'] = titanic_test_cabin.str[:1]
print(titanic["Cabin"].value_counts())

titanic['Cabin'] = titanic['Cabin'].replace(np.nan, "N")
titanic_test['Cabin'] = titanic_test['Cabin'].replace(np.nan, "N")

titanic=my_encoder(titanic, "Cabin")
titanic_test=my_encoder(titanic_test, "Cabin")

check_missing_data(titanic)
input_cols = ['Pclass',"Sex","Age","Parch","SibSp","Fare","Name_1","Name_2","Name_3","Name_4","Embarked_1","Embarked_2","Cabin_1","Cabin_2","Cabin_3","Cabin_4","Cabin_5","Cabin_6","Cabin_7","Cabin_8"]
output_cols = ["Survived"]

algorithm_show(titanic, input_cols, output_cols)


input_cols = ['Pclass',"Sex","Family","SibSp","Fare","Name_1","Name_2","Name_3","Name_4","Embarked_1","Embarked_2","Cabin_1","Cabin_2","Cabin_3","Cabin_4","Cabin_5","Cabin_6","Cabin_7","Cabin_8"]
output_cols = ["Survived"]
algorithm_show(titanic, input_cols, output_cols)
train_X = titanic[input_cols].copy()
train_Y = titanic[output_cols]

train_fold_X, valid_fold_X, train_fold_Y, valid_fold_Y = train_test_split(train_X, train_Y,
                                                            test_size=0.25, random_state=20026) 
cls = xgb.XGBClassifier()

parameters = {
    "colsample_bylevel": np.arange(0.4, 1.0, 0.1),
    "colsample_bytree": np.arange(0.7, 1.0, 0.1),
    "learning_rate": [0.4, 0.45, 0.5, 0.55],
    "max_depth": np.arange(4, 6, 1),
    "min_child_weight": np.arange(1, 2, 0.5),
    "n_estimators": [8, 10, 12],
    "subsample": np.arange(0.6, 1.0, 0.1)
}

skf_cv = StratifiedKFold(n_splits=5, random_state=20026, shuffle=False)

gscv = GridSearchCV(
    estimator=cls,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=-1,
    cv=skf_cv,
    verbose=True,
    refit=True,
)

gscv.fit(train_fold_X, train_fold_Y.values.ravel());

cls2 = xgb.XGBClassifier(**gscv.best_params_)

# Fit classifier to new split
cls2.fit(train_fold_X, train_fold_Y.values.ravel())

# Predict probabilities on valid_fold_X
split_pred_proba = cls2.predict_proba(valid_fold_X)

train_fold_Y.describe()

# Compute ROC AUC score
split_score = metrics.roc_auc_score(valid_fold_Y, split_pred_proba[:,1])
print("ROC AUC is {:.4f}".format(split_score))

# Draw ROC curve
fpr, tpr, thr = metrics.roc_curve(valid_fold_Y, split_pred_proba[:,1])
plt.plot(fpr, tpr, linestyle="-");
plt.title("ROC Curve");
plt.xlabel("False Positive Rate");
plt.ylabel("True Positive Rate");

# Compute optimal operating point and threshold
ind = np.argmin(list(map(lambda x, y: x**2 + (y - 1.0)**2, fpr,tpr)))

# Optimal threshold for our classifier cls2
opt_thr = thr[ind]

print("Optimal operating point on ROC curve (fpr,tpr) ({:.4f}, {:.4f})"
                                              "  with threshold {:.4f}".
                                    format(fpr[ind], tpr[ind], opt_thr))

y_pred = [1 if x > opt_thr else 0 for x in split_pred_proba[:,1]]
acc = metrics.accuracy_score(valid_fold_Y, y_pred)
print("Accuracy is {:.4f}".format(acc))
plot_importance(cls2);



input_cols = ['Pclass',"Sex","Family","SibSp","Fare","Name_1","Name_2","Name_3","Name_4","Embarked_1","Embarked_2","Cabin_1","Cabin_2","Cabin_4","Cabin_5","Cabin_6","Cabin_7","Cabin_8"]
output_cols = ["Survived"]
train_X = titanic[input_cols].copy()
train_Y = titanic[output_cols]

train_fold_X, valid_fold_X, train_fold_Y, valid_fold_Y = train_test_split(train_X, train_Y,
                                                            test_size=0.25, random_state=20026) 

cls3 = xgb.XGBClassifier(**gscv.best_params_)

# Fit classifier to new split
cls3.fit(train_fold_X, train_fold_Y.values.ravel())

# Predict probabilities on valid_fold_X
split_pred_proba = cls3.predict_proba(valid_fold_X)

# Compute ROC AUC score
split_score = metrics.roc_auc_score(valid_fold_Y, split_pred_proba[:,1])
print("ROC AUC is {:.4f}".format(split_score))

# Draw ROC curve
fpr, tpr, thr = metrics.roc_curve(valid_fold_Y, split_pred_proba[:,1])
plt.plot(fpr, tpr, linestyle="-");
plt.title("ROC Curve");
plt.xlabel("False Positive Rate");
plt.ylabel("True Positive Rate");

# Compute optimal operating point and threshold
ind = np.argmin(list(map(lambda x, y: x**2 + (y - 1.0)**2, fpr,tpr)))

# Optimal threshold for our classifier cls3
opt_thr = thr[ind]

print("Optimal operating point on ROC curve (fpr,tpr) ({:.4f}, {:.4f})"
                                              "  with threshold {:.4f}".
                                    format(fpr[ind], tpr[ind], opt_thr))

y_pred = [1 if x > opt_thr else 0 for x in split_pred_proba[:,1]]
acc = metrics.accuracy_score(valid_fold_Y, y_pred)
print("Accuracy is {:.4f}".format(acc))
plot_importance(cls3);

input_cols = ['Pclass',"Sex","Family","SibSp","Fare","Name_1","Name_2","Name_3","Name_4","Embarked_1","Embarked_2","Cabin_1","Cabin_2","Cabin_3","Cabin_4","Cabin_5","Cabin_6","Cabin_7","Cabin_8"]
output_cols = ["Survived"]

temp = titanic.copy()
temp.reset_index(drop=True, inplace = True)

temp_X = temp[input_cols]
temp_X = pd.get_dummies(temp_X)
# print(temp_X.head(1))
X_train, X_test, y_train, y_test = train_test_split(temp_X, temp[output_cols], test_size=0.25, random_state=20026, stratify=temp[output_cols])

cls5 = xgb.XGBClassifier(**gscv.best_params_)

# Fit classifier to new split
cls5.fit(X_train[input_cols], y_train.values.ravel())


# Predict probabilities on valid_fold_X
split_pred_proba = cls5.predict_proba(X_test[input_cols])

# Compute ROC AUC score
split_score = metrics.roc_auc_score(y_test, split_pred_proba[:,1])
print("ROC AUC is {:.4f}".format(split_score))

# Draw ROC curve
fpr, tpr, thr = metrics.roc_curve(y_test, split_pred_proba[:,1])
plt.plot(fpr, tpr, linestyle="-");
plt.title("ROC Curve");
plt.xlabel("False Positive Rate");
plt.ylabel("True Positive Rate");

# Compute optimal operating point and threshold
ind = np.argmin(list(map(lambda x, y: x**2 + (y - 1.0)**2, fpr,tpr)))

# Optimal threshold for our classifier cls5
opt_thr = thr[ind]

print("Optimal operating point on ROC curve (fpr,tpr) ({:.4f}, {:.4f})"
                                              " with threshold {:.4f}".
                                   format(fpr[ind], tpr[ind], opt_thr))

y_pred = [1 if x > opt_thr else 0 for x in split_pred_proba[:,1]]
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy is {:.4f}".format(acc))

plot_importance(cls5);

def pred_result_xgb(model,x,y,opt_thr):
    pscore = model.predict_proba(x)[:,1]
    ypred = [1 if x > opt_thr else 0 for x in pscore]
    fpr, tpr, thresholds = metrics.roc_curve(y, pscore)
    auc_score = metrics.auc(fpr, tpr)

    accuracy = metrics.accuracy_score(y, ypred)
    precision = metrics.precision_score(y, ypred)
    precision_0 = metrics.precision_score(y, ypred,pos_label=0)
    recall = metrics.recall_score(y, ypred)
    recall_0 = metrics.recall_score(y, ypred,pos_label=0)
    f1score = metrics.f1_score(y, ypred)
    f1score_0 = metrics.f1_score(y, ypred,pos_label=0)
    accuracy_insample = model.score(X_train[input_cols],y_train)
    return [accuracy,auc_score,precision,recall,f1score,precision_0,recall_0,f1score_0,accuracy_insample]

result = algorithm_show(titanic, input_cols, output_cols)
result['XGBoost_tune'] = pred_result_xgb(cls5,X_test[input_cols],y_test,opt_thr)
result
# ~~~~~~~~~~~~~~~~~ Example of Submission File ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
titanic_test["Cabin_8"] = 0 # in case of test file is too small, avoid error.
test_predict_proba = cls5.predict_proba(titanic_test[input_cols])

# Optimal threshold we found previously is opt_thr
y_pred_xgb = [1 if x > opt_thr else 0 for x in test_predict_proba[:,1]]

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": y_pred_xgb   # I have given prediction of random forest just change it to save prediction of other models here
    })
submission.to_csv('submission_xgb.csv', index=False)
submission = pd.read_csv('submission_xgb.csv')
submission.head(20)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6,class_weight='balanced',random_state=20026)
rf_model.fit(X_train,y_train.values.ravel())


y_pred_lr = rf_model.predict(titanic_test[input_cols])

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": y_pred_xgb   # I have given prediction of random forest just change it to save prediction of other models here
    })
submission.to_csv('submission_rf.csv', index=False)
submission = pd.read_csv('submission_rf.csv')
submission.head(20)
# Check if submit file exists
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))