import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pylab import rcParams 
dataset=pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/train_data.csv")
dataset.head(10)
dataset['Age'].value_counts()
dataset[dataset['Age']== "91-100"]
dataset[dataset['Age']== "0-10"]
dataset[dataset['Hospital_code']== 6]["Hospital_type_code"].nunique()
dataset.groupby(dataset["Department"]).nunique()
dataset[["Hospital_code","Hospital_type_code" ]].nunique()
dataset.groupby(dataset["Hospital_code"]).nunique()
dataset.groupby(dataset["patientid"] ).nunique()
dataset[['Department','Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness']].describe()
dataset.describe()
plt.boxplot(dataset["Admission_Deposit"])
dataset[dataset["Admission_Deposit"] > 10000] 
dataset.isnull().sum()
import matplotlib.pyplot as plt
#plt.pairplot(dataset)
import seaborn as sns
#sns.pairplot(dataset)
dataset.head()
dataset1=dataset.drop(["case_id","Hospital_code", "City_Code_Hospital","Hospital_type_code" ,"Hospital_region_code","Ward_Facility_Code","patientid","City_Code_Patient", "Visitors with Patient"], axis= 1)
dataset1
from sklearn.preprocessing import LabelEncoder
lr= LabelEncoder()
dataset1["Department"]=lr.fit_transform(dataset1["Department"])

dataset1["Ward_Type"]=lr.fit_transform(dataset1["Ward_Type"])

dataset1["Type of Admission"]=lr.fit_transform(dataset1["Type of Admission"])

dataset1["Severity of Illness"]=lr.fit_transform(dataset1["Severity of Illness"])

dataset1["Age"]=lr.fit_transform(dataset1["Age"])

dataset1["Stay"]=lr.fit_transform(dataset1["Stay"])
dataset1.describe()
dataset1.dropna(inplace= True)
dataset1.reset_index(drop=True)
dataset1.columns
dataset1['Stay'].value_counts().sum()
dataset1.describe()
dataset1.head()

from sklearn.preprocessing import StandardScaler
dataset1.columns
sd= StandardScaler()
X=sd.fit_transform(dataset1[['Available Extra Rooms in Hospital', 'Department', 'Ward_Type',
       'Bed Grade', 'Type of Admission', 'Severity of Illness', 'Age',
       'Admission_Deposit']])

X
X = pd.DataFrame(X,columns = ['Available Extra Rooms in Hospital', 'Department', 'Ward_Type',
       'Bed Grade', 'Type of Admission', 'Severity of Illness', 'Age',
       'Admission_Deposit'])
X.head()
X.describe()
X.isna().sum()
print(dataset1['Stay'].count())

y = dataset1['Stay']
print(y.count())
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier()
clf.fit(X,y)
from matplotlib.pylab import rcParams
rcParams["figure.figsize"]=10,20
from sklearn.tree import plot_tree
#plot_tree(clf)
print("Hello")
f_name = ["Available Extra Rooms in Hospital","Department","Ward_Type","Bed Grade","Type of Admission","Severity of Illness","Age","Admission_Deposit"]


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
with open("Tree.dot","w") as p:
    p = tree.export_graphviz(clf, feature_names=f_name, out_file=p)
