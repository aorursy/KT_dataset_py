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
data =  pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.shape
data.head()
data.isnull().sum()
data[data['salary'].isnull()]
data[data['status']=='Not Placed']
# Making null value as zero.
data.fillna(0,inplace=True)
data.isnull().sum()
data.head()
data.columns
# Making a dataset with all precentage columns- 
#['ssc_p', 'hsc_p','degree_p', 'etest_p',  'mba_p','status']
modal_data=data[['ssc_p', 'hsc_p','degree_p', 'etest_p',  'mba_p','status']]
modal_data.head()
# Extracting features and target

X=modal_data.iloc[:,:-1]
y=modal_data.iloc[:,-1]
# Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Spliting data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)
l_reg = LogisticRegression()
l_reg.fit(X_train,y_train)
y_pred=l_reg.predict(X_test)
metrics.accuracy_score(y_pred,y_test)
df = pd.DataFrame(columns=['Actual','Predicted'])
df['Actual']=y_test
df['Predicted']=y_pred
df['Actual']=y_test
df['Predicted']=y_pred
df
# Second Predicting salary
modal_data.head()
modal2_data=modal_data[modal_data['status']=='Placed'].iloc[:,:-1]
modal2_data['salary']=data[data['salary']>0]['salary']
modal2_data.head()
X=modal2_data.iloc[:,:-1]
y=modal2_data.iloc[:,-1]
# Spliting data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
r2_score(y_test,y_pred)
df1 = pd.DataFrame(columns=['Actual','Predicted'])
df1['Actual']=y_test
df1['Predicted']=y_pred
df1
import matplotlib.pyplot as plt
plt.scatter(modal2_data['salary'],modal2_data['ssc_p'])
plt.scatter(modal2_data['hsc_p'],modal2_data['salary'])
plt.scatter(modal2_data['degree_p'],modal2_data['salary'])
plt.scatter(modal2_data['etest_p'],modal2_data['salary'])
plt.scatter(modal2_data['mba_p'],modal2_data['salary'])
data.head()
# Seaborn Library.
import seaborn as sns
import matplotlib.pyplot as plt
data =  pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.gender.value_counts()
sns.kdeplot(data.salary[ data.gender=="M"],)
sns.kdeplot(data.salary[ data.gender=="F"])
plt.legend(["Male", "Female"])
plt.xlabel("Salary")
plt.show()
plt.figure(figsize =(14,6))
sns.boxplot("salary", "gender", data=data)
plt.show()
sns.countplot("ssc_b", hue="status", data=data)
plt.show()
sns.scatterplot(x='ssc_p',y='salary',hue='ssc_b',data=data)
sns.countplot("hsc_b", hue="status", data=data)
plt.show()
sns.scatterplot(x='hsc_p',y='salary',hue='hsc_b',data=data)

sns.countplot("hsc_s", hue="status", data=data)
plt.show()
sns.scatterplot(x='hsc_p',y='salary',hue='hsc_s',data=data)
sns.countplot("degree_t", hue="status", data=data)
plt.show()
sns.scatterplot(x='degree_p',y='salary',hue='degree_t',data=data)
## Work Experience

sns.countplot("workex", hue="status", data=data)
plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "workex", data=data)
plt.show()
## Employment test
sns.lineplot("etest_p", "salary", data=data)
plt.show()
sns.scatterplot("etest_p", "salary", data=data)
# Density plot
sns.kdeplot(data.etest_p[ data.status=="Placed"])
sns.kdeplot(data.etest_p[ data.status=="Not Placed"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Employability-test %")
plt.show()
## POST GRAD SPECIALISATION
sns.countplot("specialisation", hue="status", data=data)
plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "specialisation", data=data)
plt.show()
# Post grad percentage
sns.lineplot("mba_p", "salary", data=data)
plt.show()
sns.scatterplot("mba_p", "salary", data=data)
plt.show()
# We have to drop ssc_b and hsc_b, sl.no
data.drop(['ssc_b','hsc_b','sl_no'], axis=1, inplace=True)
data.head()
data.dtypes
# Encoding categorical columns
data["gender"] = data.gender.map({"M":0,"F":1})
data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["status"] = data.status.map({"Not Placed":0, "Placed":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
from sklearn.preprocessing import StandardScaler# for scaling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data.isnull().sum()
#dropping NaNs (in Salary)
data.dropna(inplace=True)
#dropping Status = "Placed" column
data.drop("status", axis=1, inplace=True)
data.head()
#Seperating Depencent and Independent Vaiiables
X = data.iloc[:,:-1]
y = data.iloc[:,-1] #Dependent Variable
X
#Scalizing (Normalization)
X_scaled = StandardScaler().fit_transform(X)
X_scaled.shape
lr=LinearRegression(fit_intercept=True,normalize=True)
lr.fit(X_scaled,y)
y_pred=lr.predict(X_scaled)
print(f"R2 Score: {r2_score(y, y_pred)}")
print(f"MAE: {mean_absolute_error(y, y_pred)}")
from sklearn.model_selection import GridSearchCV
model=LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(model,parameters)
grid.fit(X_scaled, y)
print("r2 / variance : ", grid.best_score_)
pred=pd.DataFrame(columns=['salary'])
pred['salary']=y_pred
sns.kdeplot(y)
sns.kdeplot(pred['salary'])
plt.legend(["Actual", "Predicted"])
plt.xlabel("Salary")
plt.show()
# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
X_poly=poly.fit_transform(X_scaled)
lr_poly=LinearRegression()
lr_poly.fit(X_poly,y)
y_pred=lr_poly.predict(X_poly)
type(X_scaled)
X_poly.shape
r2_score(y,y_pred)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 19)
type(X_test)
r2_train=[]
r2_test=[]
for i in range(1,10):
    poly=PolynomialFeatures(degree=i)
    X_poly=poly.fit_transform(X_train)
    lr_poly=LinearRegression().fit(X_poly,y_train)
    y_pred_train=lr_poly.predict(X_poly)
    
    X_test_poly=poly.fit_transform(X_test)
    y_pred=lr_poly.predict(X_test_poly)
    
    r2_train.append(r2_score(y_train,y_pred_train))
    r2_test.append(r2_score(y_test,y_pred))
x_ax=np.arange(10)+1
r2_train
r2_test