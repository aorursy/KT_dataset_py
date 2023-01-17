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
#imoporting libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
####import dataset set from local system
#import os
#DATA_PATH=r"C:\Users\Ravi\Downloads"
#csv_name="Placement_Data_Full_Class.csv"
#def load_data(data_path=DATA_PATH):
    #csv_path=os.path.join(data_path,csv_name)
    #return pd.read_csv(csv_path)
#loding dataset
#getting the first five entries of data
dataset=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
dataset.head()
#shape of dataset 215-rows and 15-columns
dataset.shape
#finding where are the null values in dataset
dataset.isna().sum()
#plotting bar graph of missing values using missingno library ~
import missingno as ms
ms.bar(dataset)
#using info function to get the datatypes or dtypes of dataset
dataset.info()
#checking how many unique values are present in each columns/attributes
dataset.nunique()
#using describe function to find the mean, median, and other parameters for numeric data in dataset
dataset.describe()
#desc. function fo categorical data in dataset
dataset.describe(include="O")
dataset['salary'].isna().sum()/len(dataset["salary"])*100
#filling the missing values with 0 using fillna function
dataset["salary"]=dataset['salary'].fillna(value=0)
dataset["salary"].isna().sum()
dataset["salary"]
col=dataset.describe(include="O").columns
#printing the distinct values of categorical data from the dataset
for i in col:
    print(i,"Distinct_values : ")
    print(dataset[i].unique())
    print("-"*30)
    print(dataset[i].value_counts())
    print("-"*30)
    print("")
sns.set_style("whitegrid")
def count(col): 
    print(dataset[col].value_counts(normalize=True)) 
    ax=plt.figure(figsize=(14,4)) 
    plt.subplot(121) 
    graph=sns.countplot(dataset[col],palette="winter_r")
    graph.set_xticklabels(graph.get_xticklabels())
    i=0
    for p in graph.patches:
        height=p.get_height()
        graph.text(p.get_x()+p.get_width()/2.,height + 1.5,
              dataset[col].value_counts()[i],ha="center")
        i+=1
    plt.title(col,fontsize=20)
    plt.subplot(122) 
    sns.countplot(dataset[col],hue="status",data=dataset,palette="Blues").set_title(col)
    plt.show()
count("gender")
count("ssc_b")
count("hsc_b")
count("hsc_s")
count("degree_t")
count("workex")
count("specialisation")
ax=plt.figure(figsize=(15,4))
plt.subplot(121)
sns.distplot(dataset["ssc_p"]).set_title("SSC_results")
plt.subplot(122)
sns.distplot(dataset["hsc_p"]).set_title("HSC_results")
plt.show()
ax=plt.figure(figsize=(15,4))
plt.subplot(121)
sns.distplot(dataset["degree_p"]).set_title("degree_results")
plt.subplot(122)
sns.distplot(dataset["etest_p"]).set_title("etest_results")
plt.show()
ax=plt.figure(figsize=(15,4))
plt.subplot(121)
sns.distplot(dataset["mba_p"]).set_title("MBA_results")
plt.subplot(122)
sns.distplot(dataset["salary"]).set_title("Salary")
plt.show()
outs=dataset[["ssc_p","hsc_p","degree_p","etest_p","mba_p"]]
ax=plt.figure(figsize=(9,5))
dat=pd.melt(outs,var_name="features",value_name="value")
sns.boxplot(x="features",y="value",data=dat)
plt.xticks(rotation=(90))
plt.show()
#in salary we have only one outlier 
plt.subplot(122)
sns.boxplot(dataset["salary"],orient="var")
plt.show()
sns.kdeplot(dataset.ssc_p[dataset.status=="Placed"])
sns.kdeplot(dataset.ssc_p[dataset.status=="Not Placed"])
plt.legend(["Placed","Not_placed"])
plt.xlabel("ssc_students")
plt.show()
sns.kdeplot(dataset.hsc_p[dataset.status=='Placed'])
sns.kdeplot(dataset.hsc_p[dataset.status=="Not Placed"])
plt.legend(["Placed","Not_Placed"])
plt.xlabel("HSC_scores")
plt.show()
sns.kdeplot(dataset["degree_p"][dataset["status"]=="Placed"])
sns.kdeplot(dataset["degree_p"][dataset["status"]=="Not Placed"])
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.kdeplot(dataset.mba_p[dataset.status=="Placed"])
sns.kdeplot(dataset.mba_p[dataset.status=="Not Placed"])
plt.legend(["Placed","Not_Placed"])
plt.xlabel("MBA_scores")
plt.subplot(122)
sns.kdeplot(dataset.etest_p[dataset.status=='Placed'])
sns.kdeplot(dataset.etest_p[dataset.status=='Not Placed'])
plt.legend(["Placed","Not_Placed"])
plt.xlabel("ETEST")
plt.show()
plt.figure(figsize=(12,6))
dataset.groupby(["ssc_b","hsc_b","hsc_s"])["status"].count().plot(kind="bar")
on_basis_of_lowerClass=dataset.groupby(["ssc_b","hsc_b","hsc_s"])["status"].count()
pd.DataFrame(on_basis_of_lowerClass).style.background_gradient(cmap="Blues")
plt.figure(figsize=(12,6))
dataset.groupby(["degree_t","workex","specialisation"])["status"].count().plot(kind="bar")
on_basis_of_higherEducation=dataset.groupby(["degree_t","workex","specialisation"])["status"].count()
pd.DataFrame(on_basis_of_higherEducation).style.background_gradient(cmap="Blues")
import plotly_express as px
gapminder=px.data.gapminder()
px.scatter(dataset,x="mba_p",y="etest_p",color="status",facet_col="workex")
px.scatter(dataset,x="mba_p",y="degree_p",color="status",facet_col="specialisation")
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(10,6))

sal = dataset[dataset.salary != 0] #taking psoitive values
sns.boxplot(sal["salary"], ax=ax_box)
sns.distplot(sal["salary"], ax=ax_hist,color="k")
 
# Remove x axis name for the boxplot
ax_box.set(xlabel="")
plt.show()
plt.figure(figsize=(15,4))
plt.subplot(121)
sns.boxplot(sal["gender"],sal["salary"],hue=sal["specialisation"])
plt.subplot(122)
sns.boxplot(sal["workex"],sal["salary"],hue=sal["specialisation"],palette="PuBu")
plt.show()
plt.figure(figsize=(15,4))
plt.subplot(121)
sns.boxplot(sal["hsc_b"],sal["salary"],hue=sal["specialisation"])
plt.subplot(122)
sns.boxplot(sal["degree_t"],sal["salary"],hue=sal["specialisation"],palette="Blues")
plt.show()
dataset.drop("sl_no",axis=1,inplace=True)
plt.figure(figsize=(9,6))
dataset.corr()
sns.heatmap(dataset.corr(),fmt=".2f",annot=True,cmap="Greys")
plt.show()
dataset.head()
from sklearn.preprocessing import LabelEncoder
lab=['gender','ssc_b','hsc_b',"hsc_s","degree_t","workex","specialisation",'status']
label=LabelEncoder()
for i in lab:
    dataset[i]=label.fit_transform(dataset[i])
dataset.head()
data=pd.get_dummies(columns=['hsc_s','degree_t',"workex","specialisation"],data=dataset)
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scaled=scale.fit_transform(data.drop("status",axis=1))
data_scaled=pd.DataFrame(scaled,columns=data.drop("status",axis=1).columns)
data_scaled["status"]=dataset["status"]
data_scaled.head()
X=data_scaled.drop("status",axis=1)
y=data_scaled.status
print("X :",X.shape)
print("y :",y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print("Train:",X_train.shape,y_train.shape)
print("Test:",X_test.shape,y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,classification_report
from sklearn.metrics import f1_score ,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
print("Accuracy_score :\n",accuracy_score(pred_rf,y_test))
print("Confusion_matrix :\n",confusion_matrix(pred_rf,y_test))
print("Classificaiton_report :\n",classification_report(pred_rf,y_test))
con=pd.DataFrame(confusion_matrix(pred_rf,y_test))
sns.heatmap(con,annot=True,cmap="Blues")
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
print("Accuracy_score :\n",accuracy_score(pred_lr,y_test))
print("Classification_report :\n",classification_report(pred_lr,y_test))
con_lr=pd.DataFrame(confusion_matrix(pred_lr,y_test))
sns.heatmap(con_lr,annot=True,cmap="PuBu")
plt.ylabel("Predicted_values")
plt.xlabel("Actual_values")
plt.show()
error_rate=[]

for i in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_kn=knn.predict(X_test)
    error_rate.append(np.mean(pred_kn != y_test))
plt.plot(range(1,30),error_rate,marker="o",label="k_value",linestyle="dashed")
plt.legend()
plt.show()
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred_kn=knn.predict(X_test)
print("Accuracy_score :\n",accuracy_score(pred_kn,y_test)*100)
print("Classification_report :\n",classification_report(pred_kn,y_test))
con_kn=pd.DataFrame(confusion_matrix(pred_kn,y_test))
sns.heatmap(con_kn,annot=True)
plt.xlabel("Actual_values")
plt.ylabel("Predicted_values")
plt.show()
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
y_test_pred=lr.predict_proba(X_test)
print("x_test {} ".format(roc_auc_score(y_test,y_test_pred[:,-1])))
y_test_pred_lr=lr.predict_proba(X_train)
print("x_train {} ".format(roc_auc_score(y_train,y_test_pred_lr[:,-1])))
ytest=y_test_pred[:,-1]
tpr , fpr ,threshold = roc_curve(y_test,ytest)
plt.plot(tpr,fpr,label="logistic_regression",marker='*')
plt.plot([0,1],[0,1],"r--")
plt.legend()
plt.title("ROC_CURVE")
plt.xlabel("True positive_rate")
plt.ylabel("False positive_rate")
plt.show()