import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.isnull().sum()
df.info()
df.describe()
plt.style.use('dark_background') 
plt.figure(figsize=(10,5))
sns.countplot('gender',data = df,palette = 'inferno')
plt.title("Distribution of Males and Females in our Data",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('ssc_b',data = df,palette = 'inferno')
plt.title("Distribution of the Boards the Students belong to in 10th",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('hsc_b',data = df,palette = 'inferno')
plt.title("Distribution of the Boards the Students belong to in 10th",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('hsc_s',data = df,palette = 'inferno')
plt.title("Distribution of the Streams that students chose in High school",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('degree_t',data = df,palette = 'inferno')
plt.title("Distribution of the Type of Degrees",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('workex',data = df,palette = 'inferno')
plt.title("Distribution of how many students have prior work experience",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('specialisation',data = df,palette = 'inferno')
plt.title("Distribution of the Types of Specialisation",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('status',data = df,palette = 'inferno')
plt.title("Distribution of the Placements",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('gender',data = df,palette = 'inferno',hue = 'status')
plt.title("Distribution of Placements in Males and Females",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('workex',data = df,palette = 'inferno',hue = 'status')
plt.title("Distribution of Placements in Males and Females",fontsize = 15)
plt.figure(figsize=(10,5))
sns.boxplot('ssc_b','ssc_p',data = df,palette = 'inferno')
plt.title("Relation between the Student's Boards and their score during their Secondary education",fontsize = 15)
plt.figure(figsize=(10,5))
sns.boxplot('status','ssc_p',data = df,palette = 'inferno')
plt.title("Relation between the Students that were placed and their score during their Secondary education",fontsize = 15)
plt.figure(figsize=(10,5))
sns.boxplot('hsc_b','hsc_p',data = df,palette = 'inferno')
plt.title("Relation between the Student's Boards and their score in High school",fontsize = 15)
plt.figure(figsize=(10,5))
sns.boxplot('status','hsc_p',data = df,palette = 'inferno')
plt.title("Relation between the Students that were placed and their score in High school",fontsize = 15)
plt.figure(figsize=(10,5))
sns.boxplot('status','mba_p',data = df,palette = 'inferno')
plt.title("Relation between the Students that were placed and their score during MBA",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('ssc_b',data = df,palette = 'inferno',hue = 'status')
plt.title("Relation between the Students that were placed and the boards that they were in during Secondary Education",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('hsc_b',data = df,palette = 'inferno',hue = 'status')
plt.title("Relation between the Students that were placed and the boards that they were in High School",fontsize = 15)
plt.figure(figsize=(10,5))
sns.boxplot('status','degree_p',data = df,palette = 'inferno')
plt.title("Relation between the Students that were placed and their degree percentage",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('hsc_s',data = df,palette = 'inferno',hue = 'status')
plt.title("Relation between the streams that students chose in highschool and their placement",fontsize = 15)
plt.figure(figsize=(10,5))
sns.countplot('degree_t',data = df,palette = 'inferno',hue = 'status')
plt.title("Relation between the degree types that students chose and their placement",fontsize = 15)
#df.head()
df1 = df.copy()
df1.drop(['sl_no','salary'],axis = 1,inplace = True)
df1.head()
df1['status']= df1['status'].map({'Placed':1,'Not Placed':0})
df1['workex']= df1['workex'].map({'Yes':1,'No':0})
df1['gender']= df1['gender'].map({'M':1,'F':0})
df1['hsc_b']= df1['hsc_b'].map({'Central':1,'Others':0})
df1['ssc_b']= df1['ssc_b'].map({'Central':1,'Others':0})
df1['degree_t']= df1['degree_t'].map({'Sci&Tech':0,'Comm&Mgmt':1,'Others':2})
df['specialisation']= df1['specialisation'].map({'Mkt&HR':1,'Mkt&Fin':0})
df1['hsc_s']= df1['hsc_s'].map({'Commerce':0,'Science':1,'Arts':2})
df1.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score,recall_score,precision_score
X = df1[['ssc_p','hsc_p','degree_p','workex','mba_p','etest_p','gender','degree_t','specialisation']]
y = df1['status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print("Accuracy:",accuracy_score(y_test, pred)*100)
print("Precision:",precision_score(y_test, pred)*100)
print("Recall:",recall_score(y_test, pred)*100)
user_input1 = [[67.00,91.00,58.00,0,58.80,55.00,1,0,1]]
user_pred1 = lr.predict(user_input1)
if user_pred1 == 1:
    print("The Candidate will be Placed!")
else:
    print("The Candidate won't be Placed :(")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print("Accuracy:",accuracy_score(y_test, prediction)*100)
print("Precision:",precision_score(y_test, prediction)*100)
print("Recall:",recall_score(y_test, prediction)*100)
user_input2 = [[67.00,91.00,58.00,0,58.80,55.00,1,0,1]]
user_pred2 = knn.predict(user_input2)
if user_pred1 == 1:
    print("The Candidate will be Placed!")
else:
    print("The Candidate won't be Placed :(")
error = []
for i in  range(1,100):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize = (10,6))
plt.plot(range(1,100),error)
plt.title('K-values')
plt.xlabel('K')
plt.ylabel('Error')
knn1 = KNeighborsClassifier(n_neighbors=20)
knn1.fit(X_train,y_train)
prediction1 = knn1.predict(X_test)
print("Accuracy:",accuracy_score(y_test, prediction1)*100)
print("Precision:",precision_score(y_test, prediction1)*100)
print("Recall:",recall_score(y_test, prediction1)*100)
user_input3 = [[67.00,91.00,58.00,0,58.80,55.00,1,0,1]]
user_pred3 = knn1.predict(user_input1)
if user_pred3 == 1:
    print("The Candidate will be Placed!")
else:
    print("The Candidate won't be Placed :(")
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
print("Accuracy:",accuracy_score(y_test, svc_pred)*100)
print("Precision:",precision_score(y_test, svc_pred)*100)
print("Recall:",recall_score(y_test, svc_pred)*100)
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose = 3)
grid.fit(X_train,y_train)
grid.best_estimator_
grid.best_params_
grid_pred = grid.predict(X_test)
print("Accuracy:",accuracy_score(y_test, grid_pred)*100)
print("Precision:",precision_score(y_test, grid_pred)*100)
print("Recall:",recall_score(y_test, grid_pred)*100)
user_input4 = [[56.00,52.00,52.00,0,59.43,66.00,1,0,1]]
user_pred4 = grid.predict(user_input4)
if user_pred4 == 1:
    print("The Candidate will be Placed!")
else:
    print("The Candidate won't be Placed :(")
