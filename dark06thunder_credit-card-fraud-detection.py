import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

sns.set_style("whitegrid")



from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics



import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv("../input/credit-card-dataset/credit_dataset.csv",index_col=0)
dataset=data.copy()
# Reviewing the dataset using the head (first 5 rows of dataset) function 

dataset.head()
#shape of dataset

print("Rows    :",dataset.shape[0])

print("Columns :",dataset.shape[1])
#checking for NULL VALUES in dataset

dataset.isna().sum()
#datatypes of attributes present in dataset

dataset.dtypes
#checking for unique values in dataset

dataset.nunique()
# TARGET column

sns.countplot(dataset.TARGET).set_title("Class Distribution \n (0:Not Fraud || 1:Fraud)")

print(dataset.TARGET.value_counts(normalize=True))

plt.show()
labels = ['No','Yes']

data = dataset['TARGET'].value_counts(sort=True)

colours = ['Orange','black']

explode = (0.6,0)

plt.figure(figsize=(6,6))



plt.pie(data,colors = colours,shadow =True,startangle=180

        ,autopct="%1.1f%%",explode=explode,labels=labels)

plt.title("TARGET COLUMN",fontsize=20,color="b")

plt.tight_layout()



print('Not Frauds :', round(dataset['TARGET'].value_counts()[0]/len(dataset) * 100,2), '% of the dataset')

print('Frauds :', round(dataset['TARGET'].value_counts()[1]/len(dataset) * 100,2), '% of the dataset')
dataset.describe()
#Distribution of continuous 

dataset.hist(figsize=(20,12))

plt.show()
dataset.describe(include="O")
#all the categorical columns with their values -

cols=dataset.describe(include="O").columns

for i in cols:

    print("Distinct_values :\n 'column_name' =",i)

    print(dataset[i].unique())

    print("")
#dropping FLAG_MOBIL as all the values is this column is 1

plt.figure(figsize=(10,8))

sns.heatmap(dataset.drop("FLAG_MOBIL",axis=1).corr(),cmap="Blues",annot=True,fmt='.2f')

plt.show()
plt.figure(figsize=(16,4))

plt.subplot(121)

sns.countplot(dataset["GENDER"],hue="TARGET",data=dataset,palette="Blues")

plt.title("GENDER",fontsize=15,color="Red")

plt.subplot(122)

sns.boxplot(x="INCOME",y="GENDER",hue="TARGET",data=dataset,palette="Blues_r")

plt.title("GENDER vs INCOME",fontsize=15,color="Red")

plt.figure(figsize=(16,4))

sns.scatterplot(dataset["INCOME"],dataset["AGE"],hue=dataset["TARGET"])

plt.title("Distribution of Age Vs Income : On basis of TARGET",fontsize=15,color="Red")

plt.show()
plt.figure(figsize=(13,4))

sns.countplot(dataset["INCOME_TYPE"],hue="TARGET",data=dataset,palette="RdPu_r")

plt.show()
plt.figure(figsize=(13,4))

sns.countplot(dataset["EDUCATION_TYPE"],hue="TARGET",data=dataset,palette="Accent")

plt.show()
plt.figure(figsize=(13,4))

sns.countplot(dataset["FAMILY_TYPE"],hue="TARGET",data=dataset,palette="PuBu")

plt.show()
plt.figure(figsize=(13,5))

sns.boxplot(x="HOUSE_TYPE",y="INCOME",data=dataset,hue="TARGET",palette="Purples")

plt.show()
sns.boxplot(dataset["YEARS_EMPLOYED"])

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(dataset["YEARS_EMPLOYED"])

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(dataset["YEARS_EMPLOYED"],hue="TARGET",data=dataset)

plt.title("YEARS_EMPLOYED vs  TARGET",fontsize=20,color="BLUE")

plt.legend()

plt.show()
#checking the frequency of BEGIN_MONTH

plt.figure(figsize=(15,4))

sns.countplot(dataset["BEGIN_MONTH"])

plt.figure(figsize=(15,4))

sns.countplot(dataset["BEGIN_MONTH"],hue="TARGET",data=dataset)

plt.show()
plt.figure(figsize=(12,4))

plt.subplot(121)

sns.countplot(dataset["CAR"],palette="Blues")

plt.subplot(122)

sns.countplot(dataset.CAR,hue="TARGET",data=dataset,palette="Blues")

plt.show()
#these columns have continuous values other are having discrete entries

cont=dataset[["INCOME","BEGIN_MONTH","AGE","YEARS_EMPLOYED"]]
#Using Boxplot to detect the outliers-

plt.figure(figsize=(15,12))



for i ,col in enumerate(list(cont.columns)):

    plt.subplot(4,2,i+1)

    cont.boxplot(col)

    plt.grid()

    plt.tight_layout()
# Distribution of columns those have continuous -

#Histogram

plt.figure(figsize=(15,10))



for i ,cols in enumerate(list(cont.columns)):

    plt.subplot(3,2,i+1)

    sns.distplot(cont[cols])

    plt.xlabel(cols,fontsize=14,color="Red")

    plt.grid()

    plt.tight_layout()
plt.figure(figsize=(17,4))



plt.subplot(121)

res=stats.probplot(dataset.INCOME,plot=plt,dist="norm")

plt.xlabel("INCOME",fontsize=16,color="m")



plt.subplot(122)

res=stats.probplot(dataset.YEARS_EMPLOYED,plot=plt,dist="norm")

plt.xlabel("YEARS_EMPLOYED",fontsize=16,color="m")



plt.show()
#removing values those are greater then 600000

dataset=dataset[dataset['INCOME'] < 600000]
plt.figure(figsize=(17,4))



plt.subplot(121)

sns.boxplot(dataset.INCOME)



plt.subplot(122)

res=stats.probplot(dataset.INCOME,plot=plt,dist="norm")
#Dropping the values greater the 20

dataset=dataset[dataset['YEARS_EMPLOYED'] < 20]
plt.figure(figsize=(16,4))

plt.subplot(121)

sns.boxplot(dataset.YEARS_EMPLOYED)

plt.subplot(122)

res=stats.probplot(dataset.YEARS_EMPLOYED,plot=plt,dist="norm")
dataset.drop(columns={"ID"},inplace=True,axis=1)
#everyone has phone 

dataset.drop("FLAG_MOBIL",inplace=True,axis=1)
#converting float data types to INT64 datatype

floats=["INCOME","FAMILY SIZE","BEGIN_MONTH"]



for i in floats:

    dataset[i]=dataset[i].astype("int64")
from sklearn.preprocessing import LabelEncoder
labels=["GENDER","CAR","REALITY","INCOME_TYPE","EDUCATION_TYPE","HOUSE_TYPE","FAMILY_TYPE"]

label=LabelEncoder()



for i in labels:

    dataset[i]=label.fit_transform(dataset[i])
#dataset after using LabelEncoder

dataset.head()
#corr of the 

dataset.corr()
plt.figure(figsize=(18,10))

sns.heatmap(dataset.corr(),annot=True,fmt='.2',cmap="Greys")

plt.show()
#FAMILY_SIZE and NO_OF_CHILD are highly correlated to each other so we can drop one feature form the dataset

dataset.drop('NO_OF_CHILD',inplace=True,axis=1)
from sklearn.tree import ExtraTreeClassifier
X=dataset.drop("TARGET",axis=1)

y=dataset.TARGET
print("X :",X.shape)

print("y :",y.shape)
model=ExtraTreeClassifier()

model.fit(X,y)
print(model.feature_importances_)
feat=pd.Series(model.feature_importances_,index=X.columns)
feats=feat.to_frame().reset_index()

feats.columns=["Features","Scores"]

features=feats.sort_values(by="Scores",ascending=False)

top_features=features.nlargest(12,"Scores")

top_features
plt.figure(figsize=(14,6))

sns.barplot(y="Features",x="Scores",data=top_features)

plt.xticks(rotation=90)

plt.show()
new_data=dataset[["GENDER","BEGIN_MONTH","AGE","INCOME","YEARS_EMPLOYED","FAMILY SIZE","INCOME_TYPE","FAMILY_TYPE","WORK_PHONE"

                  ,"PHONE","HOUSE_TYPE","EDUCATION_TYPE","TARGET"]]
new_data.shape
plt.figure(figsize=(12,8))

sns.heatmap(new_data.corr(),fmt='.2',annot=True,cmap="Greys")

plt.show()
new_data.head()
dummy_data=pd.get_dummies(new_data,columns={"GENDER","FAMILY SIZE","INCOME_TYPE","FAMILY_TYPE","WORK_PHONE","PHONE",

                                            "HOUSE_TYPE","EDUCATION_TYPE"},drop_first=True)
#shape of the dataset after creating dummy variables

dummy_data.shape
#checking the correlation of the new dataset

plt.figure(figsize=(12,8))

sns.heatmap(dummy_data.corr(),fmt='.2',cbar=False,cmap="RdPu_r")

plt.show()
data=dummy_data.copy()
data_minority=data[dataset["TARGET"] == 1]

data_majority=data[dataset["TARGET"] == 0]
print("Data_majority :",data_majority.shape)

print("Data_minority :",data_minority.shape)
#library for perfroming SMOTE

from imblearn.over_sampling import SMOTE
data_oversampled=data.copy()
data_oversampled.TARGET.value_counts()
#setting TARGET variable SMOTE

X=data_oversampled.drop("TARGET",axis=1)

y=data_oversampled.TARGET
X_smote,y_smote=SMOTE().fit_sample(X,y)
#shape of dataset after using SMOTE

print("X",X_smote.shape)

print("y",y_smote.shape)
#head of data after using SMOTE

#X_smote.head()
sns.countplot(y_smote)

plt.show()
#considering the continuous variables -

X_smote_continuous=X_smote[["BEGIN_MONTH","AGE","INCOME","YEARS_EMPLOYED"]]
#distribution of the continuous variables after sampling the data using SMOTE -

plt.figure(figsize=(13,9))



for i,col in enumerate((X_smote_continuous.columns)):

    plt.subplot(3,2,i+1)

    sns.distplot(X_smote_continuous[col])

    plt.xlabel(col,fontsize=14,color="Magenta")

    plt.grid()

    plt.tight_layout()
plt.figure(figsize=(10,8))

matrix = np.triu(X_smote.corr())

sns.heatmap(X_smote.corr(),mask=matrix,cbar=False)

plt.show()
X_train_over,X_test_over,y_train_over,y_test_over=train_test_split(X_smote,y_smote,test_size=0.25,random_state=99)
#size of data after spliting

print("X_train_over : ",X_train_over.shape,"\ny_train_over : ",y_train_over.shape)

print("X_train_over : ",X_test_over.shape,"\ny_train_over : ",y_test_over.shape)
random_over=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=25, max_features='sqrt', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=2, min_samples_split=50,

            min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)
random_over.fit(X_train_over,y_train_over)
pred_over=random_over.predict(X_test_over)
print(metrics.classification_report(pred_over,y_test_over))
print(metrics.precision_score(pred_over,y_test_over))
score_over = {}

score_over["Random_Forest"] ={}

score_over["Random_Forest"]["Precision"]=metrics.precision_score(pred_over,y_test_over)*100

score_over["Random_Forest"]["Recall"]=metrics.recall_score(pred_over,y_test_over)*100

score_over["Random_Forest"]["Accuracy"]=metrics.accuracy_score(pred_over,y_test_over)*100
sns.heatmap(metrics.confusion_matrix(y_test_over,pred_over),annot=True,cbar=False

            ,cmap="Blues",fmt="1",linecolor="Black",linewidth=0.3)

plt.xlabel("PREDICTED_LABEL",fontsize=12,color='r')

plt.ylabel("ACTUAL_LBLE",fontsize=12,color='r')

plt.title("CONFUISON_MATRIX   :-",fontsize=14,color="b")

plt.show()
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()

tree.fit(X_train_over,y_train_over)
pred_tree=tree.predict(X_test_over)
print(metrics.classification_report(y_test_over,pred_tree))
score_over["Decision_Tree"] ={}

score_over["Decision_Tree"]["Precision"]=metrics.precision_score(pred_tree,y_test_over)*100

score_over["Decision_Tree"]["Recall"]=metrics.recall_score(pred_tree,y_test_over)*100

score_over["Decision_Tree"]["Accuracy"]=metrics.accuracy_score(pred_tree,y_test_over)*100
sns.heatmap(metrics.confusion_matrix(y_test_over,pred_tree),annot=True,cbar=False

            ,cmap="Blues",fmt="1",linecolor="Black",linewidth=0.3)

plt.xlabel("PREDICTED_LABEL",fontsize=12,color='r')

plt.ylabel("ACTUAL_LBLE",fontsize=12,color='r')

plt.title("CONFUISON_MATRIX   :-",fontsize=14,color="b")

plt.show()
score_oversampling=pd.DataFrame(score_over)
#library for selecting data at random

from sklearn.utils import resample
data_undersampled=data.copy()
#target class count bofer under_sampling

data_undersampled.TARGET.value_counts()
#splitting the whole data set into to parts for under_sampling method-

data_majority_undersampled=data_undersampled[data_undersampled["TARGET"] == 0]

data_minority_undersampled=data_undersampled[data_undersampled["TARGET"] == 1]
#here we will be keeing the majority class double the size of minority class to prevent the model from underfitting

data_under_sampled=resample(data_majority_undersampled,n_samples=844,replace=True,random_state=42)
#shape of the majority class after resampling :

print(data_under_sampled.shape)
data_under=pd.concat([data_under_sampled,data_minority])
#data afrer under sample

data_under.head()
#size of new data_ after under_sampling

data_under.shape
#data distribution after under sampling

data_under.hist(figsize=(20,20))

plt.show()
#target column of sampled data

print(data_under.TARGET.value_counts())
#Corr plot of the under_sampled data look same the normal data Major difference is not seen here in the plot.

#Removing from the correlation plot INCOME_TYPE_3 because it has all the values 1

plt.figure(figsize=(12,8))

sns.heatmap(data_under.drop("INCOME_TYPE_3",axis=1).corr(),fmt='.2',cbar=False)

plt.show()
#after under_sampling the data now target class have 1:2 distribution

sns.countplot(data_under.TARGET,palette="rocket")

plt.xlabel("Target_class Distribution",fontsize=15)

plt.show()
X_under=data_under.drop("TARGET",axis=1)

y_under=data_under.TARGET
#spliting the data for train and test:

X_train_under,X_test_under,y_train_under,y_test_under=train_test_split(X_under,y_under,test_size=0.25,random_state=71)
#size of data after spliting

print("X_train_under : ",X_train_under.shape,"\ny_train_under : ",y_train_under.shape)

print("X_train_under : ",X_test_under.shape,"\ny_train_under : ",y_test_under.shape)
random_under=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=25, max_features='sqrt', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=2, min_samples_split=50,

            min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)
random_under.fit(X_train_under,y_train_under)
pred_under=random_under.predict(X_test_under)

print(metrics.classification_report(pred_under,y_test_under))
precision_under=(metrics.precision_score(pred_under,y_test_under))

precision_under
score_under = {}

score_under["Random_Forest"] ={}

score_under["Random_Forest"]["Precision"]=metrics.precision_score(pred_under,y_test_under)*100

score_under["Random_Forest"]["Recall"]=metrics.recall_score(pred_under,y_test_under)*100

score_under["Random_Forest"]["Accuracy"]=metrics.accuracy_score(pred_under,y_test_under)*100
sns.heatmap(metrics.confusion_matrix(y_test_under,pred_under),annot=True,cbar=False

            ,cmap="Greys",fmt="1",linecolor="Black",linewidth=0.3)

plt.xlabel("PREDICTED_LABEL",fontsize=12,color='r')

plt.ylabel("ACTUAL_LBLE",fontsize=12,color='r')

plt.title("CONFUISON_MATRIX   :-",fontsize=14,color="b")

plt.show()
tree_under=DecisionTreeClassifier()

tree_under.fit(X_train_under,y_train_under)
pred_under_tree=tree_under.predict(X_test_under)
print(metrics.classification_report(pred_under_tree,y_test_under))
score_under["Decision_Tree"] ={}

score_under["Decision_Tree"]["Precision"]=metrics.precision_score(pred_under_tree,y_test_under)*100

score_under["Decision_Tree"]["Recall"]=metrics.recall_score(pred_under_tree,y_test_under)*100

score_under["Decision_Tree"]["Accuracy"]=metrics.accuracy_score(pred_under_tree,y_test_under)*100
sns.heatmap(metrics.confusion_matrix(y_test_under,pred_under_tree),annot=True,cbar=False

            ,cmap="Blues",fmt="1",linecolor="Black",linewidth=0.3)

plt.xlabel("PREDICTED_LABEL",fontsize=12,color='r')

plt.ylabel("ACTUAL_LBLE",fontsize=12,color='r')

plt.title("CONFUISON_MATRIX   :-",fontsize=14,color="b")

plt.show()
score_undersampling=pd.DataFrame(score_under)
score_oversampling.plot(kind="bar",figsize=(10,3))

plt.xticks(rotation=(0))

#plt.xlabel("Evaluation Metrics",fontsize=15)

plt.ylabel("%",fontsize=15)

plt.title("Scores Of UnderSampled Data: SMOTE",fontsize=15,color="Red")



score_undersampling.plot(kind="bar",figsize=(10,3))

plt.xticks(rotation=(0))

plt.xlabel("Evaluation Metrics",fontsize=15)

plt.ylabel("%",fontsize=15)

plt.title("Scores Of UnderSampled Data: Random Resampling",fontsize=15,color="Red")

plt.show()
from sklearn.metrics import roc_auc_score,roc_curve



ytest_pred=random_over.predict_proba(X_test_over)

print("Xtest : {} ".format(roc_auc_score(y_test_over,ytest_pred[:,-1])))

ytest_pred_ = ytest_pred[:,-1]



fpr,tpr,thresholds = roc_curve(y_test_over,ytest_pred_)



#plt.figure(figsize=(12,8))

plt.plot(fpr,tpr,label='roc_curve',color='MAGENTA')

plt.plot([0,1],[0,1],color='darkblue',linestyle='--')

plt.ylabel("True positive rate",fontsize=15)

plt.xlabel("False positive rate",fontsize=15)

plt.title("Receiver Operating Characteristic (Roc) curve -",fontsize=15)

plt.legend()

plt.show()