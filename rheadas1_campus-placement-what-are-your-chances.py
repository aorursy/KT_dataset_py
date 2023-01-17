# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas_profiling



#Plotly Libraris



# Run the below code if PLOTLY is not installed

#!pip install plotly



import plotly.express as px

import plotly.graph_objects as go

from plotly.colors import n_colors

from plotly.subplots import make_subplots

#Run the below code in anaconda promt for pandas-profiling to work

#conda install -c conda-forge pandas-profiling



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

data.drop("sl_no", axis=1, inplace=True)

print(data.shape)

data.head()
data.info()
data.describe()
data.describe(include="O") #O -> include categorical columns
data.nunique()
data.isna().sum()
# Salary has missing values. Lets replace that with '0' as the status of all Salary='Null' is 'Not Placed'

# We cannot drop the values as the dataset is small and also it might give valueable information as to why the student did not get place



print(round(data['salary'].isnull().sum()/len(data['salary'])*100,2),"% of data in Salary column is NULL!")
data['salary'] = data['salary'].fillna(value=0)
from pandas_profiling import ProfileReport



ProfileReport(data)
print("Salary Distribution as per gender")



plt.figure(figsize=(15,5))

sns.kdeplot(data.salary[ data.gender == "M"])

sns.kdeplot(data.salary[ data.gender == "F"])

plt.legend(["Male", "Female"])

plt.xlabel("Salary(100)")

plt.show()
print("Placement vs Marks at differnt education level")



f, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False, squeeze = True)

#sns.despine(left=True)



sns.kdeplot(data.ssc_p[ data.status== "Placed"], ax=axes[0, 0])

sns.kdeplot(data.ssc_p[ data.status== "Not Placed"], ax=axes[0, 0])

plt.xlabel("10th Marks")





sns.kdeplot(data.hsc_p[ data.status== "Placed"], ax=axes[0, 1])

sns.kdeplot(data.hsc_p[ data.status== "Not Placed"], ax=axes[0, 1])

plt.xlabel("12th Marks")



sns.kdeplot(data.degree_p[ data.status== "Placed"], ax=axes[1, 0])

sns.kdeplot(data.degree_p[ data.status== "Not Placed"], ax=axes[1, 0])

plt.xlabel("Degree_Marks")



sns.kdeplot(data.mba_p[ data.status== "Placed"], ax=axes[1, 1])

sns.kdeplot(data.mba_p[ data.status== "Not Placed"], ax=axes[1, 1])

plt.legend(["Placed", "Not Placed"])

plt.xlabel("MBA_Marks")



plt.setp(axes, yticks=[])

plt.tight_layout()
fig = px.bar(data, x="gender", y="salary",color="gender",facet_row="workex", facet_col="specialisation")

fig.update_layout(title_text='Facet view of Student Salary wrt Gender, Specialization in Higher education and previous work experience')

fig.show()
fig = px.scatter(data,x="mba_p",y="salary",color="specialisation", facet_row='gender', facet_col="workex")

fig.update_layout(title_text='Facet view of Student Salary wrt Gender, MBA%, HighEd Specialization and previous work experience')

fig.show()
status = data['status'].value_counts().to_frame().reset_index().rename(columns={'index':'Status','status':'Count'})

status
fig = go.Figure([go.Pie(labels = status['Status'], values = status['Count'], hole=0.6)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Placement staus",title_x=0.5)

fig.show()
sunburst = data[['gender','status','specialisation','salary','degree_t']].groupby(['gender','status','specialisation','salary','degree_t']).agg('max').reset_index()

fig = px.sunburst(sunburst, path=['gender','status','specialisation','degree_t'], values='salary')

fig.update_layout(title="Salary Distribution by Gender, Placement Status, HigherEd Specialization, Degree Subject",title_x=0.5)

fig.show()
print("Tabuler Format - Job offers as per SSC & HSC")

Table_10_12 = pd.DataFrame(data.groupby(["ssc_b", "hsc_b", "hsc_s"])["status"].count()).style.background_gradient(cmap="bone_r")

Table_10_12
print("Tabuler Format - Job offers as per Degree, Work Experience & MBA Specialsation")

Table_deg_workex_mba = pd.DataFrame(data.groupby(["degree_t", "workex", "specialisation"])["status"].count()).style.background_gradient(cmap="bone_r")

Table_deg_workex_mba
### Placed --> 1, Not Placed --> 0

## Splitting dataset into X dataset (Predictor variables) & Y dataset (Target variable):

data['status'] = np.where(data['status'] == "Placed", 1,0)

X = data.drop('status', axis=1)

Y = data['status']
# X dataset - creating dummies

X = pd.get_dummies(X)

print("The Dimension of X (Predictor Dataset):",X.shape)

print("The Dimension of Y (Target Dataset):",Y.shape)

X.head()
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()

scaled = scale.fit_transform(X)

X = pd.DataFrame(scaled,columns = X.columns)

X.head()
# Split x and y into training and testing set (70%-30% ratio and a random state of 200)



import sklearn.model_selection as ms

x_train, x_test, y_train, y_test= ms.train_test_split(X,Y, test_size=0.3, random_state=200)
print("X-Train :", x_train.shape)

print("Y-Train :", y_train.shape)  # Labels of training dataset

print("X-Test  :", x_test.shape)

print("Y-Test  :", y_test.shape)   # Labels of testing dataset
from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import f1_score ,confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve

import sklearn.metrics as metrics
#Using Random Forest Algorithm

RF = RandomForestClassifier(n_estimators=100)

RF.fit(x_train, y_train)

y_pred = RF.predict(x_test)
print("Random Forest Model Results:\n")

print("Accuracy Score:", round(accuracy_score(y_test, y_pred),2)*100,"%")

print("***************************************************\n")

print("Classification Report:\n", classification_report(y_test, y_pred))

print("***************************************************\n")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("***************************************************")
CM = pd.DataFrame(confusion_matrix(y_test, y_pred))



sns.heatmap(CM, annot=True, annot_kws={"size": 15}, cmap="cividis_r", linewidths=0.9)

plt.title('Confusion matrix for RF', y=1.1, fontdict = {'fontsize': 20})

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
## ROC curve for RF:

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)

auc = metrics.roc_auc_score(y_test, y_pred)



plt.figure(figsize=(10,5))

plt.style.use('seaborn')

plt.plot(fpr,tpr,label="AUC ="+str(auc))

plt.plot([0,1],[0,1],"r--")

plt.title("ROC for RF model", fontdict = {'fontsize': 20})

plt.xlabel("True positive_rate")

plt.ylabel("False positive_rate")

plt.legend(loc= 4, fontsize = "x-large")
# Feature Selection

print("**Dataframe showing Feature Importance in descending order**")

best_features = pd.DataFrame({'Features': x_train.columns, 'Importance':RF.feature_importances_})

best_features.sort_values('Importance', ascending=False)
from sklearn.neighbors import KNeighborsClassifier
## Getting values of k

error_rate=[]

for i in range(1,20):

    knn= KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train, y_train)

    y_pred_kn= knn.predict(x_test)

    error_rate.append(np.mean(y_pred_kn != y_test))  

    

## Plotting values of k



plt.figure(figsize=(15,5))

plt.style.use('seaborn')

plt.plot(range(1,20), error_rate, marker ='o', label= "k-value", linestyle="dashed" )

plt.title(label= "Error rate of all the values of K", fontdict = {'fontsize': 20})

plt.legend(fontsize = "xx-large")

plt.show()
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

y_pred_kn = knn.predict(x_test)
print("KNN Model Results:\n")

print("Accuracy Score:", round(accuracy_score(y_test, y_pred_kn),4)*100,"%")

print("***************************************************\n")

print("Classification Report:\n", classification_report(y_test, y_pred_kn))

print("***************************************************\n")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_kn))

print("***************************************************")
CM_knn = pd.DataFrame(confusion_matrix(y_test, y_pred_kn))



sns.heatmap(CM_knn, annot=True, annot_kws={"size": 15}, cmap="cividis_r", linewidths=0.9)

plt.title('Confusion matrix for KNN', y=1.1, fontdict = {'fontsize': 20})

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
## ROC curve for KNN:

fpr1, tpr1, _ = metrics.roc_curve(y_test, y_pred_kn)

auc1 = metrics.roc_auc_score(y_test, y_pred_kn)



plt.figure(figsize=(10,5))

plt.style.use('seaborn')

plt.plot(fpr1, tpr1, label="AUC ="+str(auc1))

plt.plot([0,1],[0,1],"r--")

plt.title("ROC for KNN model", fontdict = {'fontsize': 20})

plt.xlabel("True positive_rate")

plt.ylabel("False positive_rate")

plt.legend(loc= 4, fontsize = "x-large")