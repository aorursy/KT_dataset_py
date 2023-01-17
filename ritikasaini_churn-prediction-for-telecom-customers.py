# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import seaborn as sns #for visualisation
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv("../input/telecom-churn/Churn.csv")
df.head()
print ("\nFeatures : \n" ,df.columns.tolist())  #listing all the features
df.describe() 
Null_val = [(c, df[c].isna().mean()*100) for c in df]
Null_val = pd.DataFrame(Null_val, columns=["column_name", "percentage"])
Null_val
df.info()   
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')   #changing to numeric
print ("\nUnique values for each column:\n",df.nunique())
df.drop(["customerID"],axis=1,inplace = True) #dropping CustomerID column because it has got nothing to do with analysis of Churn
df.head()
df.gender = [1 if each == "Male" else 0 for each in df.gender] #mapping male to 1 and female to 0
df.head()
df.gender[df.gender == 'male'] = 1
df.gender[df.gender == 'female'] = 0

#mapping male to 1 and female to 0
change_to_num = ['Partner', 
                      'Dependents', 
                      'PhoneService', 
                      'MultipleLines',
                      'OnlineSecurity',
                      'OnlineBackup',
                      'DeviceProtection',
                      'TechSupport',
                      'StreamingTV',
                      'StreamingMovies',
                      'PaperlessBilling', 
                      'Churn']

for x in change_to_num:
    df[x] = [1 if each == "Yes" else 0 if each == "No" else -1 for each in df[x]]
    
df.head()

sns.countplot(x="Churn",data=df) #Visualising the distribution of Churn values

sns.pairplot(df,vars = ['tenure','MonthlyCharges','TotalCharges'], hue="Churn") 
#plotting the three numeric features with hue as "Churn" 
#Hue is a Variable in data to map plot aspects to different colors.

v=sns.catplot(x="Contract", y="Churn", data=df,kind="bar")
v.set_ylabels("Probability of Churn to be 1")
# All types of contract vs Churning probability
u=sns.catplot(x="InternetService", y="Churn", data=df,kind="bar")
u.set_ylabels("Probability of churn to be 1")
#All types of IS vs CHurn probability
u=sns.catplot(x="TechSupport", y="Churn", data=df,kind="bar")
u.set_ylabels("Probability of churn")
u=sns.catplot(x="gender", y="Churn", data=df,kind="bar")
u.set_ylabels("Probabilty for churn to be 1")
u=sns.catplot(x="SeniorCitizen", y="Churn", data=df,kind="bar")
u.set_ylabels("Churn Probability")
u=sns.catplot(x="OnlineSecurity", y="Churn", data=df,kind="bar")
u.set_ylabels("Churn probability")
u=sns.catplot(x="DeviceProtection", y="Churn", data=df,kind="bar")
u.set_ylabels("Churning Probability")
u=sns.catplot(x="PaperlessBilling", y="Churn", data=df,kind="bar")
u.set_ylabels("Churn Probability")
#Now we will map the remaining columns (InternetService, Contract, PaymentMethod)
df = pd.get_dummies(data=df)
df.head()

p=df.corr() #Finding the correlation between the columns so that I can remove one of two highly correlated column 
#Usually the values ranging from +/-0.5 to +/-1 are said to be highly correlated, so we will look for it
p
p['Churn'].sort_values() 
#No value is highly correlated, so we are good to go
df = df.reset_index()
y=df.Churn.values #storing Churn(which is to be predicted) in variable Y
df1=df.drop(["Churn"],axis=1) #dropping Churn column, so that we can be left with rest of the features
x = (df1-np.min(df1))/(np.max(df1)-np.min(df1)).values
pd.isnull(x).sum() > 0  #finding the column where lies any null value
x=x.fillna(x.mean()) #replacing the null value with mean
np.any(np.isnan(x)) #checking whethetr a null value still exists in the dataframe
x = x[np.isfinite(x).all(1)]  #Only keeping finite values
np.all(np.isfinite(x)) #Checking whether all values are finite
print(x.astype(np.float32)) #to avoid any dtype error, finding the value that exceeds the bounds of a float 32 dtype
X = np.nan_to_num(x.astype(np.float32)) #bringing value in the bound of float 32 dtype
print(X)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1) 
#I've split the data in the ratio 80:20
from sklearn.tree import DecisionTreeClassifier         #Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
accuracy_dt = dt_model.score(x_test,y_test)
print("Decision Tree's accuracy:",accuracy_dt)

from sklearn.svm import SVC                             #SVM
svc_model = SVC(random_state = 1)
svc_model.fit(x_train,y_train)
accuracy_svc = svc_model.score(x_test,y_test)
print("Accuracy using SVM :",accuracy_svc)
from sklearn.naive_bayes import GaussianNB              #Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
accuracy_nb = nb_model.score(x_test,y_test)
print("Accuracy using Naive Bayes :",accuracy_nb)
from sklearn.linear_model import LogisticRegression    #Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
accuracy_lr = lr_model.score(x_test,y_test)
print(" Accuracy using Logistic Regression is:",accuracy_lr)
from sklearn.neighbors import KNeighborsClassifier    #K-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors = 3) #set K neighbor as 3
knn.fit(x_train,y_train)
predicted_y = knn.predict(x_test)
print("KNN accuracy when k=3:",knn.score(x_test,y_test))
arr1 = []
for each in range(1,25):
    knn_loop = KNeighborsClassifier(n_neighbors = each) #set K neighbor as 3
    knn_loop.fit(x_train,y_train)
    arr1.append(knn_loop.score(x_test,y_test))
    
plt.plot(range(1,25),arr1)
plt.xlabel("Range")
plt.ylabel("Score")
plt.show()

#KNN gives highest accuracy at k=16
knn_model = KNeighborsClassifier(n_neighbors = 16) #at k=16
knn_model.fit(x_train,y_train)
predicted_y = knn_model.predict(x_test)
accuracy_knn = knn_model.score(x_test,y_test)
print("KNN accuracy when K=16:",accuracy_knn)
from sklearn.ensemble import RandomForestClassifier     #Random Forest
rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
rf_model_initial.fit(x_train,y_train)
print("Random Forest accuracy for 7 trees is:",rf_model_initial.score(x_test,y_test))

arr = []   #plotting a graph to find best value of K that would give us maximum accuracy
for each in range(1,50):
    rf_loop = RandomForestClassifier(n_estimators = each, random_state = 1)
    rf_loop.fit(x_train,y_train)
    arr.append(rf_loop.score(x_test,y_test))
    
plt.plot(range(1,50),arr)
plt.xlabel("Range")
plt.ylabel("Score")
plt.show()
#Accuracy of RF is highest at 35 and 42
rf_model = RandomForestClassifier(n_estimators = 35, random_state = 1) 
rf_model.fit(x_train,y_train)
accuracy_rf = rf_model.score(x_test,y_test)
print("Random Forest accuracy for 35 trees is:",accuracy_rf)

from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
cm_lr = confusion_matrix(y_test,lr_model.predict(x_test)) #for Logistic Regression
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "blue", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of LR")
plt.show()

def print_scores(headline, y_true, y_pred):
    print(headline)
    acc_score = accuracy_score(y_true, y_pred)
    print("accuracy: ",acc_score)
    pre_score = precision_score(y_true, y_pred)
    print("precision: ",pre_score)
    rec_score = recall_score(y_true, y_pred)                            
    print("recall: ",rec_score)
    f_score = f1_score(y_true, y_pred, average='weighted')
    print("f1_score: ",f_score)

print_scores("Logistic Regression;",y_test, lr_model.predict(x_test))
print_scores("SVC;",y_test, svc_model.predict(x_test))
print_scores("KNN;",y_test, knn_model.predict(x_test))
print_scores("Naive Bayes;",y_test, nb_model.predict(x_test))
print_scores("Decision Tree;",y_test, dt_model.predict(x_test))
print_scores("Random Forest;",y_test, rf_model.predict(x_test))
report = classification_report(y_test, lr_model.predict(x_test))  #Report of best performing LR model
print(report)
