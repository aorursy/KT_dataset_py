# List of Attributes :


# age            : continuous; age of the individual
# workclass      : Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
# fnlwgt         : continuous, financial weight of the individual
# education      : Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters,
#                  1st-4th, 10th, Doctorate, 5th-6th, Preschool
# education_num  : continuous; number of years of education of the individual
# marital_status : Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
# occupation     : Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners,
#                  Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv,
#                  Armed-Forces
# relationship   : Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
# race           : White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# sex            : Male, Female, Non-binary
# capital_gain   : continuous; capital gains of the individual
# capital_loss   : continuous; capital loss of the individual
# hours_per_week : continuous; working hours per week of the individual
# native_country : United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, 
#                  Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, 
#                  Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
#                  Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands

# income         : Target variable; categorical; income group of the individual (0:<=50K,1:>50K)
# Problem Statement : Build classification models for the binary clasification problem to predict whether an individual (with 
# given details) is part of one of the two income groups : (<=50K,>50K). 
# Evaluation Metric : ROC-AUC
import os 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import sklearn

import pickle
os.getcwd()
#Function for memory optimization
def reduce_mem_usage():
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """

    #Import the merged dataset (train + item_details + location_details + train_transactions)
    df=pd.read_csv('../input/income_classification_train.csv',na_values='?')
    print("Data Type of columns in the data frame before optimization")
    print(df.dtypes)
    print()    

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of the data frame before optimization is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of the data frame after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print()
    
    print("Data type of columns in the data frame after optimization")
    print(df.dtypes)
    print()

    return df


#Optimized train dataset
data=reduce_mem_usage()
data
#Function for memory optimization
def reduce_mem_usage():
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """

    #Import the merged dataset (train + item_details + location_details + train_transactions)
    df=pd.read_csv('../input/income_classification_test.csv',na_values='?')
    print("Data Type of columns in the data frame before optimization")
    print(df.dtypes)
    print()    

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of the data frame before optimization is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of the data frame after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print()
    
    print("Data type of columns in the data frame after optimization")
    print(df.dtypes)
    print()

    return df


#Optimized train dataset
test=reduce_mem_usage()
test
data
data.head()
data.head(n=19)
data.tail()
data.tail(n=23)
data.shape
data.describe()
data.describe().T
data.describe(include="all")
data.describe(include='category')
data.info()
data.dtypes
data.index.values
data.columns.values
data.iloc[:,:]
data.iloc[788,9]
data.iloc[:29,7:]
data.iloc[28674:,:14]
data.iloc[[563,837,9387,16362],[1,4,7,13]]
data.loc[:,:]
data.loc[:94,"marital_status"]
data.loc[26381:,['workclass','race','income']]
data.loc[[305,1829,24674,25767,],"sex":"native_country"]
f,ax=plt.subplots(1,2,figsize=(15,5))

ax[0] = data['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Income Share')


#f, ax = plt.subplots(figsize=(6, 8))
ax[1] = sns.countplot(x="income", data=data, palette="Set1")
ax[1].set_title("Frequency distribution of income variable")

plt.show()
f,ax = plt.subplots(figsize=(10,8))
ax = sns.countplot(x="income", hue="sex", data=data, palette="Set1")
ax.set_title("Frequency distribution of income variable wrt sex")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
ax = sns.countplot(x="income", hue="race", data=data, palette="Set1")
ax.set_title("Frequency distribution of income variable wrt race")
plt.show()
f,ax = plt.subplots(figsize=(15,8))
ax = sns.countplot(x="income", hue="occupation", data=data, palette="Set1")
ax.set_title("Frequency distribution of income variable wrt occupation")
plt.show()
f, ax = plt.subplots(figsize=(10,6))
ax = data.workclass.value_counts().plot(kind="bar", color="green")
ax.set_title("Frequency distribution of workclass variable")
ax.set_xticklabels(data.workclass.value_counts().index,rotation=30)
plt.show()
f,ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(x="workclass", hue="income", data=data, palette="Set1")
ax.set_title("Frequency distribution of workclass variable wrt income")
ax.legend(loc='upper right')
plt.show()
f,ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(x="workclass", hue="sex", data=data, palette="Set1")
ax.set_title("Frequency distribution of workclass variable wrt sex")
ax.legend(loc='upper right')
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['age']
ax = sns.distplot(x, bins=10, color='blue')
ax.set_title("Distribution of age variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['age']
x = pd.Series(x, name="Age variable")
ax = sns.kdeplot(x, shade=True, color='red')
ax.set_title("Distribution of age variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['fnlwgt']
ax = sns.distplot(x, bins=10, color='blue')
ax.set_title("Distribution of fnlwgt variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['fnlwgt']
x = pd.Series(x, name="fnlwgt variable")
ax = sns.kdeplot(x, shade=True, color='red')
ax.set_title("Distribution of fnlwgt variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['education_num']
ax = sns.distplot(x, bins=10, color='blue')
ax.set_title("Distribution of education_num variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['education_num']
x = pd.Series(x, name="education_num variable")
ax = sns.kdeplot(x, shade=True, color='red')
ax.set_title("Distribution of education_num variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
x = data['age']
ax = sns.boxplot(x)
ax.set_title("Visualize outliers in age variable")
plt.show()
f,ax = plt.subplots(figsize=(10,8))
ax = sns.boxplot(x="income", y="age", data=data)
ax.set_title("Visualize income wrt age variable")
plt.show()
plt.figure(figsize=(8,6))
ax = sns.catplot(x="income", y="age", col="sex", data=data, kind="box", height=8, aspect=1)
plt.show()
data.corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
sns.pairplot(data)
plt.show()
sns.pairplot(data,hue="income")
plt.show()
sns.pairplot(data,hue="sex")
plt.show()
# Distribution
fig,ax = plt.subplots(3,2, figsize=(25,20), facecolor="lavender")
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# age
sns.distplot(data.query("income=='<=50K'")["age"], ax=ax[0,0], label="<=50K", bins=30, kde=False)
sns.distplot(data.query("income=='>50K'")["age"], ax=ax[0,0], label=">50K", bins=30, kde=False)
ax[0,0].set_title("age distribution")
ax[0,0].set_ylabel("count")
ax[0,0].legend(facecolor="white")

# fnlwgt
sns.distplot(data.query("income=='<=50K'")["fnlwgt"], ax=ax[0,1], label="<=50K", bins=30, kde=False)
sns.distplot(data.query("income=='>50K'")["fnlwgt"], ax=ax[0,1], label=">50K", bins=30, kde=False)
ax[0,1].set_title("fnlwgt distribution")
ax[0,1].set_ylabel("count")
ax[0,1].legend(facecolor="white")

# capital-gain
sns.distplot(np.log(data.query("income=='<=50K'")["capital_gain"]+1), ax=ax[1,0], label="<=50K", bins=30, kde=False)
sns.distplot(np.log(data.query("income=='>50K'")["capital_gain"]+1), ax=ax[1,0], label=">50K", bins=30, kde=False)
ax[1,0].set_title("capital_gain distribution")
ax[1,0].set_xlabel("log(capital_gain)")
ax[1,0].set_ylabel("count")
ax[1,0].set_yscale("log")
ax[1,0].legend(facecolor="white")

# capital-loss
sns.distplot(np.log(data.query("income=='<=50K'")["capital_loss"]+1), ax=ax[1,1], label="<=50K", bins=30, kde=False)
sns.distplot(np.log(data.query("income=='>50K'")["capital_loss"]+1), ax=ax[1,1], label=">50K", bins=30, kde=False)
ax[1,1].set_title("capital_loss distribution")
ax[1,1].set_xlabel("log(capital_loss)")
ax[1,1].set_ylabel("count")
ax[1,1].set_yscale("log")
ax[1,1].legend(facecolor="white")

# education-num
sns.distplot(data.query("income=='<=50K'")["education_num"], ax=ax[2,0], label="<=50K", bins=30, kde=False)
sns.distplot(data.query("income=='>50K'")["education_num"], ax=ax[2,0], label=">50K", bins=30, kde=False)
ax[2,0].set_title("education-num distribution")
ax[2,0].set_ylabel("count")
ax[2,0].legend(facecolor="white")

# hours-per-week
sns.distplot(data.query("income=='<=50K'")["hours_per_week"], ax=ax[2,1], label="<=50K", bins=30, kde=False)
sns.distplot(data.query("income=='>50K'")["hours_per_week"], ax=ax[2,1], label=">50K", bins=30, kde=False)
ax[2,1].set_title("hours-per-week distribution")
ax[2,1].set_ylabel("count")
ax[2,1].legend(facecolor="white")
data.isnull().sum()
data.isnull().mean()*100
sns.heatmap(data.isnull(),yticklabels=False)
plt.show()
mode_columns={k:i for k,i in zip(["workclass","occupation","native_country"],data[["workclass","occupation","native_country"]].mode().values.reshape(3,))}
mode_columns
data[["workclass","occupation","native_country"]]=data[["workclass","occupation","native_country"]].fillna(mode_columns)
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False)
plt.show()
test.isnull().sum()
test.isnull().mean()*100
sns.heatmap(test.isnull(),yticklabels=False)
plt.show()
test[["workclass","occupation","native_country"]]=test[["workclass","occupation","native_country"]].fillna(mode_columns)
test.isnull().sum()
sns.heatmap(test.isnull(),yticklabels=False)
plt.show()
data_dummies=pd.get_dummies(data[['workclass','education','marital_status','occupation','relationship','race','sex','native_country']],drop_first=True)
data_dummies
dataDummies=pd.concat([data[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']],data_dummies,data['income']],axis=1)
dataDummies
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataDummies['income']= le.fit_transform(dataDummies['income'])
dataDummies
dataDummies.dtypes
dataDummies.columns.values
test_dummies=pd.get_dummies(test[['workclass','education','marital_status','occupation','relationship','race','sex','native_country']],drop_first=True)
test_dummies
testDummies=pd.concat([test[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']],test_dummies,test['income']],axis=1)
testDummies
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
testDummies['income']= le.fit_transform(testDummies['income'])
testDummies
testDummies.dtypes
testDummies.columns.values
X=dataDummies.drop("income",axis=1)
y=dataDummies["income"].to_frame()
X
y
X_test=testDummies.drop("income",axis=1)
y_test=testDummies["income"].to_frame()
X_test
y_test
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.20,random_state=0)
X_train
y_train
X_valid
y_valid
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']]=sc_X.fit_transform(X_train[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']])
X_valid[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']]=sc_X.transform(X_valid[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']])
X_train
X_valid
X_test[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']]=sc_X.transform(X_test[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']])
X_test
from sklearn.linear_model import LogisticRegression
logreg_classifier=LogisticRegression(random_state=0)
logreg_classifier.fit(X_train,np.ravel(y_train))
y_pred=logreg_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = logreg_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
cutoff=thresholds[np.argmax(tpr-fpr)]
cutoff
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2)
X_train_res,y_train_res = sm.fit_sample(X_train,np.ravel(y_train))

lr_smote_classifier = LogisticRegression()
lr_smote_classifier.fit(X_train_res, y_train_res)
y_pred=lr_smote_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = lr_smote_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
cutoff=thresholds[np.argmax(tpr-fpr)]
cutoff
# Try different numbers of C
C = [0.1,0.5,1,2,5,10,20,25,50]
scores_train = []
scores_valid = []
for n in C:
  lr_classifier=LogisticRegression(C=n,random_state=0,n_jobs=-1)
  lr_classifier.fit(X_train, y_train)
  y_pred_probs_train = lr_classifier.predict_proba(X_train)[:,1]
  scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
  y_pred_probs_valid= lr_classifier.predict_proba(X_valid)[:,1]
  scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.xlabel("C")
plt.ylabel("ROC-AUC")
plt.plot(C, scores_train, label="Train ROC-AUC")
plt.plot(C, scores_valid, label="Local Validation ROC-AUC")
plt.legend()
plt.show()
# Try different numbers of max_iter
iterations = [5,10,25,50,75,100,150]
scores_train = []
scores_valid = []
for n in iterations:
  lr_classifier=LogisticRegression(max_iter=n,random_state=0,n_jobs=-1)
  lr_classifier.fit(X_train, y_train)
  y_pred_probs_train = lr_classifier.predict_proba(X_train)[:,1]
  scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
  y_pred_probs_valid= lr_classifier.predict_proba(X_valid)[:,1]
  scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.xlabel("max_iter")
plt.ylabel("ROC-AUC")
plt.plot(iterations, scores_train, label="Train ROC-AUC")
plt.plot(iterations, scores_valid, label="Local Validation ROC-AUC")
plt.legend()
plt.show()
param_grid={'penalty': ['l1', 'l2'],
             'C': [0.1, 1, 10, 100, 1000,1100,1200],
             'class_weight': [{1: 0.4, 0: 0.6},
                              {1: 0.6, 0: 0.4},
                              {1: 0.7, 0: 0.3},
                              {1: 0.8, 0: 0.2},
                              {1: 0.9, 0: 0.1}],
             'solver': ['liblinear', 'saga']}
param_grid
from sklearn.model_selection import GridSearchCV

lr_gs=LogisticRegression(random_state=0,n_jobs=-1)
grid = GridSearchCV(estimator=lr_gs,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=10,
                    verbose=1,
                    n_jobs=-1)

grid_result = grid.fit(X_train, np.ravel(y_train))
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
logreg_classifier_final=LogisticRegression(C=1,class_weight={1:0.6,0:0.4},penalty='l2',solver='saga')
logreg_classifier_final.fit(X_train,np.ravel(y_train))
train_pred=logreg_classifier_final.predict(X_train)
train_pred
y_pred_probs = logreg_classifier_final.predict_proba(X_train)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_train, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_train, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
y_pred=logreg_classifier_final.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = logreg_classifier_final.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
cutoff=thresholds[np.argmax(tpr-fpr)]
cutoff
pickle.dump(logreg_classifier_final,open('LogisticRegressionModel.sav','wb'))
logreg_loaded_model = pickle.load(open('LogisticRegressionModel.sav','rb'))
y_pred_final=logreg_loaded_model.predict(X_test)
y_pred_final
y_pred_probs = logreg_loaded_model.predict_proba(X_test)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_test, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
cutoff=thresholds[np.argmax(tpr-fpr)]
cutoff
lr_submission=pd.concat([pd.DataFrame(pd.Series(np.arange(0,4884),name='index')),pd.DataFrame(pd.Series(y_pred_final,name='Churn'))],axis=1)
lr_submission
lr_submission.to_csv(r'lr_submission.csv',index=False)
from sklearn.naive_bayes import GaussianNB
nb_classifier=GaussianNB()
nb_classifier.fit(X_train,np.ravel(y_train))
train_pred=nb_classifier.predict(X_train)
train_pred
y_pred_probs = nb_classifier.predict_proba(X_train)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_train, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_train, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
y_pred=nb_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = nb_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
pickle.dump(nb_classifier,open('NaiveBayesClassificationModel.sav','wb'))
nb_loaded_model = pickle.load(open('NaiveBayesClassificationModel.sav','rb'))
y_pred_final=nb_loaded_model.predict(X_test)
y_pred_final
y_pred_probs = nb_loaded_model.predict_proba(X_test)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_test, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
nb_submission=pd.concat([pd.DataFrame(pd.Series(np.arange(0,4884),name='index')),pd.DataFrame(pd.Series(y_pred_final,name='Churn'))],axis=1)
nb_submission
nb_submission.to_csv(r'nb_submission.csv',index=False)
from sklearn.tree import DecisionTreeClassifier
dt_classifier=DecisionTreeClassifier(random_state=0)
dt_classifier.fit(X_train,np.ravel(y_train))
from sklearn.tree import export_graphviz
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import pydot

with open("tree1.dot", 'w') as dot:
    dot = export_graphviz(dt_classifier,
                          out_file=dot,
                          max_depth = 5,
                          impurity = True,
                          class_names = ['<=50K','>50K'],
                          feature_names = X_train.columns.values,
                          rounded = True,
                          filled= True )

    
# Annotating chart with PIL
(graph,) = pydot.graph_from_dot_file('tree1.dot')

graph.write_png('tree1.png')
PImage('tree1.png')
y_pred=dt_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = dt_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
feat_impt=pd.DataFrame(data=dt_classifier.feature_importances_).T
feat_impt.columns=X_train.columns.values
names = list(data.drop('income',axis=1).columns.values)
feature_importances=pd.DataFrame()
for column in names:
    value=feat_impt.filter(regex=column)
    value=value.mean(axis=1)
    feature_importances[column]=value

#feature_importances=pd.melt(feature_importances)
fig,ax=plt.subplots(figsize=(15,10))
p=sns.barplot(data=feature_importances,ax=ax)
p.set_xticklabels(p.get_xticklabels(),rotation=45)
feature_importances
# Try different numbers of max_depth
depth=[1,2,5,10,20,25,50,75,100,150,200]
scores_train = []
scores_valid = []
for n in depth:
    dt_classifier=DecisionTreeClassifier(max_depth=n,random_state=0)
    dt_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = dt_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= dt_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of max_depth")
plt.xlabel("max depth")
plt.ylabel("ROC AUC")
plt.plot(depth, scores_train, label="Train AUC")
plt.plot(depth, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of min_samples_leaf
leaf=[1,2,5,10,20,25,50,80]
scores_train = []
scores_valid = []
for n in leaf:
    dt_classifier=DecisionTreeClassifier(min_samples_leaf=n,random_state=0)
    dt_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = dt_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= dt_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of min_samples_leaf")
plt.xlabel("min samples leaf")
plt.ylabel("ROC AUC")
plt.plot(leaf, scores_train, label="Train AUC")
plt.plot(leaf, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of min_samples_split
samples=[2,5,10,20,25,50,80,100]
scores_train = []
scores_valid = []
for n in samples:
    dt_classifier=DecisionTreeClassifier(min_samples_split=n,random_state=0)
    dt_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = dt_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= dt_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of min_samples_split")
plt.xlabel("min samples split")
plt.ylabel("ROC AUC")
plt.plot(samples, scores_train, label="Train AUC")
plt.plot(samples, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
param_grid=params = {'max_depth':[10,100,150,250],
                     'max_features': ['auto','sqrt','log2'],
                     'min_samples_split': [2,100,200,500], 
                     'min_samples_leaf':[1,2,4,8,12]
                    }
param_grid
from sklearn.model_selection import GridSearchCV

dt_gs=DecisionTreeClassifier(random_state=0)
grid = GridSearchCV(estimator=dt_gs,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=10,
                    verbose=1,
                    n_jobs=-1)

grid_result = grid.fit(X_train,np.ravel(y_train))
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
dt_classifier_final=DecisionTreeClassifier(max_features='auto',max_depth=100,min_samples_leaf=4,min_samples_split=200,random_state=0)
dt_classifier_final.fit(X_train,np.ravel(y_train))
y_pred=dt_classifier_final.predict(X_train)
y_pred
y_pred_probs = dt_classifier_final.predict_proba(X_train)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_train, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_train, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
y_pred=dt_classifier_final.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = dt_classifier_final.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
pickle.dump(dt_classifier_final,open('DecisionTreeClassificationModel.sav','wb'))
dt_loaded_model = pickle.load(open('DecisionTreeClassificationModel.sav','rb'))
y_pred_final=dt_loaded_model.predict(X_test)
y_pred_final
y_pred_probs = dt_loaded_model.predict_proba(X_test)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_test, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
dt_submission=pd.concat([pd.DataFrame(pd.Series(np.arange(0,4884),name='index')),pd.DataFrame(pd.Series(y_pred_final,name='Churn'))],axis=1)
dt_submission
dt_submission.to_csv(r'dt_submission.csv',index=False)
from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(random_state=0,n_jobs=-1)
rf_classifier.fit(X_train,np.ravel(y_train))
y_pred=rf_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = rf_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
feat_impt=pd.DataFrame(data=rf_classifier.feature_importances_).T
feat_impt.columns=X_train.columns.values
names = list(data.drop('income',axis=1).columns.values)
feature_importances=pd.DataFrame()
for column in names:
    value=feat_impt.filter(regex=column)
    value=value.mean(axis=1)
    feature_importances[column]=value

#feature_importances=pd.melt(feature_importances)
fig,ax=plt.subplots(figsize=(15,10))
p=sns.barplot(data=feature_importances,ax=ax)
p.set_xticklabels(p.get_xticklabels(),rotation=45)
feature_importances
# Try different numbers of n_estimators
estimators = np.arange(10,500,20)
scores_train = []
scores_valid = []
for n in estimators:
    rf_classifier.set_params(n_estimators=n,random_state=0,n_jobs=-1)
    rf_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = rf_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid = rf_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("ROC AUC")
plt.plot(estimators, scores_train, label="Train AUC")
plt.plot(estimators, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of max_depth
depths = [1,5,10,20,25,50,75,100]
scores_train = []
scores_valid = []
for n in depths:
    rf_classifier.set_params(max_depth=n,random_state=0,n_jobs=-1)
    rf_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = rf_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid = rf_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of max_depth")
plt.xlabel("max depth")
plt.ylabel("ROC AUC")
plt.plot(depths, scores_train, label="Train AUC")
plt.plot(depths, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of min_samples_leaf
leaf = [1,2,5,10,15,20,25]
scores_train = []
scores_valid = []
for n in leaf:
    rf_classifier.set_params(min_samples_leaf=n,random_state=0,n_jobs=-1)
    rf_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = rf_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid = rf_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of min_samples_leaf")
plt.xlabel("min_samples_leaf")
plt.ylabel("ROC AUC")
plt.plot(leaf, scores_train, label="Train AUC")
plt.plot(leaf, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of min_samples_split
samples = [2,5,10,15,20,25]
scores_train = []
scores_valid = []
for n in samples:
    rf_classifier.set_params(min_samples_split=n,random_state=0,n_jobs=-1)
    rf_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = rf_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid = rf_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of min_samples_split")
plt.xlabel("min_samples_split")
plt.ylabel("ROC AUC")
plt.plot(samples, scores_train, label="Train AUC")
plt.plot(samples, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
param_grid = {'criterion':['gini','entropy'],
          'n_estimators':[10,120,150,200,250],
          'max_depth':[5,10,15,25],
          'min_samples_leaf':[1,2,5],
          'min_samples_split':[2,10,50,75,100]
         }
param_grid
from sklearn.model_selection import GridSearchCV

rf_gs=RandomForestClassifier(random_state=0,n_jobs=-1)
grid = GridSearchCV(estimator=rf_gs,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=10,
                    verbose=1,
                    n_jobs=-1)

grid_result = grid.fit(X_train,np.ravel(y_train))
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
param_grid = {'criterion':['entropy'],
          'n_estimators':[700,800],
          'max_depth':[35,50,70],
          'min_samples_leaf':[0.01,1],
          'min_samples_split':[50]
         }
param_grid
from sklearn.model_selection import GridSearchCV

rf_gs=RandomForestClassifier(random_state=0,n_jobs=-1)
grid = GridSearchCV(estimator=rf_gs,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=10,
                    verbose=1,
                    n_jobs=-1)

grid_result = grid.fit(X_train,np.ravel(y_train))
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
param_grid = {'criterion':['entropy'],
          'n_estimators':[800,1000,1200],
          'max_depth':[35],
          'min_samples_leaf':[0.1,1],
          'min_samples_split':[50]
         }
param_grid
from sklearn.model_selection import GridSearchCV

rf_gs=RandomForestClassifier(random_state=0,n_jobs=-1)
grid = GridSearchCV(estimator=rf_gs,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=10,
                    verbose=1,
                    n_jobs=-1)

grid_result = grid.fit(X_train,np.ravel(y_train))
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
from sklearn.ensemble import RandomForestClassifier
rf_classifier_final=RandomForestClassifier(criterion='entropy',max_depth=35,min_samples_leaf=1,min_samples_split=50,n_estimators=1000,random_state=0,n_jobs=-1)
rf_classifier_final.fit(X_train,np.ravel(y_train))
train_pred=rf_classifier_final.predict(X_train)
train_pred
y_pred_probs = rf_classifier_final.predict_proba(X_train)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_train, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_train, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
y_pred=rf_classifier_final.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = rf_classifier_final.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
pickle.dump(rf_classifier_final,open('RandomForestClassificationModel.sav','wb'))
rf_loaded_model = pickle.load(open('RandomForestClassificationModel.sav','rb'))
y_pred_final=rf_loaded_model.predict(X_test)
y_pred_final
y_pred_probs = rf_loaded_model.predict_proba(X_test)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_test, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
rf_submission=pd.concat([pd.DataFrame(pd.Series(np.arange(0,4884),name='index')),pd.DataFrame(pd.Series(y_pred_final,name='Churn'))],axis=1)
rf_submission
rf_submission.to_csv(r'rf_submission.csv',index=False)
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(random_state=0,n_jobs=-1)
xgb_classifier.fit(X_train, np.ravel(y_train))
y_pred=xgb_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = xgb_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
feat_impt=pd.DataFrame(data=xgb_classifier.feature_importances_).T
feat_impt.columns=X_train.columns.values
names = list(data.drop('income',axis=1).columns.values)
feature_importances=pd.DataFrame()
for column in names:
    value=feat_impt.filter(regex=column)
    value=value.mean(axis=1)
    feature_importances[column]=value

#feature_importances=pd.melt(feature_importances)
p=sns.barplot(data=feature_importances)
p.set_xticklabels(p.get_xticklabels(),rotation=45)
feature_importances
# Try different numbers of n_estimators
estimators = np.arange(60,700,20)
scores_train = []
scores_valid = []
for n in estimators:
    xgb_classifier.set_params(n_estimators=n,random_state=0,n_jobs=-1)
    xgb_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = xgb_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= xgb_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("ROC AUC")
plt.plot(estimators, scores_train, label="Train AUC")
plt.plot(estimators, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of learning_rate
rate=[0.005,0.01,0.05,0.1,0.5,1,5,8]
scores_train = []
scores_valid = []
for n in rate:
    xgb_classifier.set_params(learning_rate=n,random_state=0,n_jobs=-1)
    xgb_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = xgb_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= xgb_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of learning_rate")
plt.xlabel("learning rate")
plt.ylabel("ROC AUC")
plt.plot(rate, scores_train, label="Train AUC")
plt.plot(rate, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of max_depth
depth=[1,10,25,50,100]
scores_train = []
scores_valid = []
for n in depth:
    xgb_classifier.set_params(max_depth=n,random_state=0,n_jobs=-1)
    xgb_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = xgb_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= xgb_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of max_depth")
plt.xlabel("max depth")
plt.ylabel("ROC AUC")
plt.plot(depth, scores_train, label="Train AUC")
plt.plot(depth, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
# Try different numbers of gamma
gamma=[0,0.5,1,2,5,7,10,12,15]
scores_train = []
scores_valid = []
for n in gamma:
    xgb_classifier.set_params(gamma=n,random_state=0,n_jobs=-1)
    xgb_classifier.fit(X_train,np.ravel(y_train))
    y_pred_probs_train = xgb_classifier.predict_proba(X_train)[:,1]
    scores_train.append(metrics.roc_auc_score(y_train, y_pred_probs_train))
    y_pred_probs_valid= xgb_classifier.predict_proba(X_valid)[:,1]
    scores_valid.append(metrics.roc_auc_score(y_valid, y_pred_probs_valid))
plt.title("Effect of gamma")
plt.xlabel("gamma")
plt.ylabel("ROC AUC")
plt.plot(gamma, scores_train, label="Train AUC")
plt.plot(gamma, scores_valid, label="Local Validation AUC")
plt.legend()
plt.show()
param_grid = {"learning_rate"    : [0.01,0.5,0.1] ,
              "max_depth"        : [10,25,40],
              "gamma"            : [0,1,8,12]
            }
             
param_grid
from sklearn.model_selection import GridSearchCV

xgb_gs = XGBClassifier(random_state=0,n_jobs=-1)
grid = GridSearchCV(estimator=xgb_gs,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=5,
                    verbose=1,
                    n_jobs=-1)

grid_result = grid.fit(X_train,np.ravel(y_train))
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
xgb_classifier_final = XGBClassifier(gamma=8,learning_rate=0.1,max_depth=40,random_state=0,n_jobs=-1)
xgb_classifier_final.fit(X_train,np.ravel(y_train))
train_pred=xgb_classifier_final.predict(X_train)
train_pred
y_pred_probs = xgb_classifier_final.predict_proba(X_train)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_train, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_train, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
y_pred=xgb_classifier_final.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = xgb_classifier_final.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
pickle.dump(xgb_classifier_final,open('XGBoostClassificationModel.sav','wb'))
xgb_loaded_model = pickle.load(open('XGBoostClassificationModel.sav','rb'))
y_pred_final=xgb_loaded_model.predict(X_test)
y_pred_final
y_pred_probs = xgb_loaded_model.predict_proba(X_test)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_test, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
xgb_submission=pd.concat([pd.DataFrame(pd.Series(np.arange(0,4884),name='index')),pd.DataFrame(pd.Series(y_pred_final,name='Churn'))],axis=1)
xgb_submission
xgb_submission.to_csv(r'xgb_submission.csv',index=False)
logreg_loaded_model = pickle.load(open('LogisticRegressionModel.sav','rb'))
rf_loaded_model = pickle.load(open('RandomForestClassificationModel.sav','rb'))
xgb_loaded_model = pickle.load(open('XGBoostClassificationModel.sav','rb'))
logreg_loaded_model
rf_loaded_model
xgb_loaded_model
from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBClassifier
meta_classifier = StackingCVClassifier(classifiers=[logreg_loaded_model, rf_loaded_model, xgb_loaded_model],
                          shuffle=False,
                          use_probas=True,
                          cv=5,
                          meta_classifier=XGBClassifier())
meta_classifier.fit(X_train,np.ravel(y_train))
train_pred=meta_classifier.predict(X_train)
train_pred
y_pred_probs = meta_classifier.predict_proba(X_train)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_train, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_train, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
y_pred=meta_classifier.predict(X_valid)
y_pred
from sklearn import metrics
cm=metrics.confusion_matrix(y_true=y_valid,y_pred=y_pred)
cm
f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Confusion Matrix for Local Validation Set")
plt.xlabel("Predicted y values")
plt.ylabel("Real y values")
plt.show()
print(metrics.classification_report(y_valid,y_pred))
metrics.accuracy_score(y_valid,y_pred)
metrics.precision_score(y_valid,y_pred)
metrics.recall_score(y_valid,y_pred)
metrics.f1_score(y_valid,y_pred)
y_pred_probs = meta_classifier.predict_proba(X_valid)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_valid, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_valid, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
pickle.dump(meta_classifier,open('MetaClassificationModel.sav','wb'))
meta_loaded_model = pickle.load(open('MetaClassificationModel.sav','rb'))
y_pred_final=meta_loaded_model.predict(X_test)
y_pred_final
y_pred_probs = meta_loaded_model.predict_proba(X_test)[:,1]
print("Predicted Probabilities are :")
y_pred_probs
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_pred_probs)
fpr
tpr
thresholds
auc = metrics.roc_auc_score(y_test, y_pred_probs)
print("AUC =",auc)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
meta_submission=pd.concat([pd.DataFrame(pd.Series(np.arange(0,4884),name='index')),pd.DataFrame(pd.Series(y_pred_final,name='Churn'))],axis=1)
meta_submission
meta_submission.to_csv(r'meta_submission.csv',index=False)