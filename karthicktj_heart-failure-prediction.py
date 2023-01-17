# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import data 
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
df.info()
print("shape:{} and size:{} of the data".format(df.shape , df.size))
df.describe()
df.columns
print("If any Missing Value is present in the data")
df.isnull().sum()
df.DEATH_EVENT.describe()
fig,ax = plt.subplots(1,2,figsize=(10,5))
sns.countplot(data = df , x= "DEATH_EVENT" ,palette = "Set3" ,ax=ax[0])
plt.title("Death_event")
df.DEATH_EVENT.value_counts().plot.pie(explode =[0.1,0] ,autopct = "%0.2f%%" ,shadow = True ,ax = ax[1])
plt.show()
def univarient(data , feature):
    plt.figure(figsize = (10,10))
    sns.distplot(data[feature])
    plt.show()
feature = ["age" , "creatinine_phosphokinase" ,"ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]
for var in feature:
    univarient(df,var)
def outlier(data ,feature):
    plt.figure(figsize = (10,10))
    sns.boxplot(data[feature])
    plt.show()
    
feature = ["age" , "creatinine_phosphokinase" ,"ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]
for var in feature:
    outlier(df,var)
def univarient_cat(data ,feature):
    plt.figure(figsize = (10,10))
    sns.countplot(x = feature ,hue ="DEATH_EVENT" , data = data , palette = "rainbow" )
    plt.show()
feature_b = ["anaemia"  ,"diabetes","high_blood_pressure","sex","smoking"]
for var in feature_b:
    univarient_cat(df ,var)
def bivarient(data ,feature):
    plt.figure(figsize = (10,10))
    sns.boxplot(y = feature ,x ="DEATH_EVENT" , data = data , palette = "rainbow" )
    plt.show()
feature = ["age" , "creatinine_phosphokinase" ,"ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]
for var in feature:
    bivarient(df,var)
def bivarient_conti(data , feature):
    plt.figure(figsize =(10,10))
    sns.lineplot(x = "age" , y = feature , hue = "DEATH_EVENT" , data = data)
    plt.title("Relationship between Age and Hue =DEATH_EVENT")
    plt.show()
feature_c = [ "creatinine_phosphokinase" ,"ejection_fraction","platelets","serum_creatinine","serum_sodium"]
for var in feature_c:
    bivarient_conti(df,var)
plt.figure(figsize = (10,10))
plt.title("Relation between death based on Sex and Diabetes ")
sns.catplot(kind = "count",x = "sex" ,hue = "diabetes" ,col = "DEATH_EVENT" ,data = df,palette = "rainbow")

plt.figure(figsize = (10,10))
plt.title("Relation between death based on Sex and High_blood_pressure ")
x = sns.catplot(kind = "count",x = "sex" ,hue = "high_blood_pressure" ,col = "DEATH_EVENT" ,data = df,palette = "rainbow")
x
plt.figure(figsize=(10,15))
sns.heatmap(df.corr() , annot = True ,cmap = "Blues" )
plt.show()
sns.pairplot(df)
plt.show()
#outlier calculation for Extreme and Nominal
def IQR_CAL(data , feature):
    IQR = data[feature].quantile(0.75) - data[feature].quantile(0.25)
    E_upper = data[feature].quantile(0.75) + (3 * IQR)
    E_lower = data[feature].quantile (0.25)- (3 * IQR)
    N_upper = data[feature].quantile(0.75) + (1.5 * IQR) # apply Nominal outlier
    N_lower = data[feature].quantile(0.25) + (1.5 * IQR)
    print("Inter Quantile Range  {}:{}".format(feature,IQR))
    print("Extreme outlier for Upper Boundary  {}:{}".format(feature,E_upper))
    print("Extreme outlier for Lower Boundary  {}:{}".format(feature,E_lower))
    print("Nominal outlier for Upper Boundary  {}:{}".format(feature,N_upper))
    print("Nominal outlier for Lower Boundary  {}:{}".format(feature,N_lower))
    
    
outlier_f = [ "creatinine_phosphokinase" ,"ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]
for var in outlier_f:
    IQR_CAL(df,var)
#outlier removal
df.loc[df["serum_sodium"] < 125 ,  "serum_sodium" ] = 125.0
df.loc[df["serum_creatinine"] > 2.14 ,"serum_creatinine" ] = 2.14

df.loc[df["platelets"] > 440000 , "platelets"] =440000  #nominal outlier
df.loc[df["ejection_fraction"] > 67.5 , "ejection_fraction"] =  67.5   #nominal outlier
df.loc[df["creatinine_phosphokinase"] > 1280.25 , "creatinine_phosphokinase"] = 1280.25 #nominal outlier

def outlier_removal_f(data,var):
    plt.figure(figsize = (10,10))
    sns.distplot(data[var],color="y")
    plt.title(var)
    plt.show()
outlier_f = [ "creatinine_phosphokinase" ,"ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]
for var in outlier_f:
    outlier_removal_f(df,var)
df.head()
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

y.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report, precision_score,accuracy_score

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
print("shape of x_train:{} and x_test:{}".format(x_train.shape,x_test.shape))
print("shape of y_train:{} and y_test:{}".format(y_train.shape,y_test.shape))
x_train_std = StandardScaler().fit_transform(x_train)
x_test_std = StandardScaler().fit_transform(x_test)

lr = LogisticRegression(penalty = "l2" , fit_intercept=True,verbose = 2 ,n_jobs = -1)
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
con = confusion_matrix(y_test,pred)
print(con)
plt.title("confusion_matrix for Logistic Regression")
sns.heatmap(con , annot = True ,cmap = "Blues")
print("accuracy Score for Logistic Regression Before appling Standardisation:{}".format(accuracy_score(y_test,pred) * 100))
lr = LogisticRegression(penalty = "l2" , fit_intercept=True)
lr.fit(x_train_std,y_train)
pred = lr.predict(x_test_std)
con = confusion_matrix(y_test,pred)
print(con)
plt.title("confusion_matrix for Logistic Regression")
sns.heatmap(con , annot = True ,cmap = "Blues")
print("accuracy Score for Logistic Regression after appling Standardisation:{}".format(accuracy_score(y_test,pred) * 100))
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
con = confusion_matrix(y_test,pred)
print(con)
plt.title("confusion_matrix for RandomForestClassifier")
sns.heatmap(con , annot = True ,cmap = "Blues")
print("accuracy Score for RandomClassifier:{}".format(accuracy_score(y_test,pred) * 100 ,"%"))
print("Classification_report for Random_forest")
print(classification_report(y_test,pred))
gb = GradientBoostingClassifier(n_estimators = 50 ,max_depth = 10 ,criterion = "mse",random_state = 3)
gb.fit(x_train,y_train)
pred = rf.predict(x_test)
con = confusion_matrix(y_test,pred)
print(con)
plt.title("confusion_matrix for GradientBoostinClassifier")
sns.heatmap(con , annot = True ,cmap = "bone")
print("accuracy Score for GradientBoosting:{}".format(accuracy_score(y_test,pred) * 100 ,"%"))
print("Classification_report for GradientBoosting")
print(classification_report(y_test,pred))
Knn = KNeighborsClassifier(n_neighbors=100, weights='uniform')
Knn.fit(x_train,y_train)
knnpred = Knn.predict(x_test)
con = confusion_matrix(y_test,knnpred)
print(con)
plt.title("confusion_matrix for KneighborsClassifier")
sns.heatmap(con , annot = True ,cmap = "YlGnBu")
print("accuracy Score for KneighborsClassifier:{}".format(accuracy_score(y_test,knnpred) * 100 ))
print("Classification_report for KneighborsClassifier")
print(classification_report(y_test,knnpred))

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
svcpred = svc.predict(x_test)
con = confusion_matrix(y_test,svcpred)
print(con)
plt.title("confusion_matrix for Support Vector Classifier")
sns.heatmap(con , annot = True ,cmap = "Pastel2")
print("accuracy Score for Support Vector Classifier:{}".format(accuracy_score(y_test,svcpred) * 100 ))
print("Classification_report for Support Vector Classifier")
print(classification_report(y_test,svcpred))


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train_std,y_train)
svcpred = svc.predict(x_test_std)
con = confusion_matrix(y_test,svcpred)
print(con)
plt.title("confusion_matrix for Support Vector Classifier")
sns.heatmap(con , annot = True ,cmap = "YlGnBu")
print("accuracy Score for Support Vector Classifier Standardized:{}".format(accuracy_score(y_test,svcpred) * 100 ))
print("Classification_report for Support Vector Classifier")
print(classification_report(y_test,svcpred))


dt = DecisionTreeClassifier(max_depth = 10 ,criterion = "entropy",splitter = "best")
dt.fit(x_train,y_train)
dtpred = dt.predict(x_test)
con = confusion_matrix(y_test,dtpred)
print(con)
plt.title("confusion_matrix for Decision Tree Classifier")
sns.heatmap(con , annot = True ,cmap = "YlGnBu")
print("accuracy Score for Decision Tree Classifier:{}".format(accuracy_score(y_test,dtpred) * 100 ))
print("Classification_report for Decision Tree Classifier")
print(classification_report(y_test,dtpred))


rf = RandomForestClassifier()
cr = cross_val_score(rf,x,y,cv = 10)
print("Cross Value Score Random Forest:{}".format(cr.mean()))
rf=RandomForestClassifier(n_estimators=300,criterion='entropy',
                             max_features='sqrt',min_samples_leaf=10,random_state=100)
rf.fit(x_train,y_train)
rfpred = rf.predict(x_test)
con = confusion_matrix(y_test,rfpred)
print(con)
plt.title("confusion_matrix for RandomForest Classifier")
sns.heatmap(con , annot = True ,cmap = "Pastel1")
print("accuracy Score for RandomForest Classifier:{}".format(accuracy_score(y_test,rfpred) * 100 ))
print("Classification_report for RandomForest Classifier")
print(classification_report(y_test,rfpred))
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 2, stop = 2000, num = 1000)]
max_features = ['auto', 'sqrt','log2']
max_depth = [int(x) for x in np.linspace(1, 1000,500)]
min_samples_split = [2, 5, 10,14]
min_samples_leaf = [1, 2, 4,6,8]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)
rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
rf_randomcv.fit(x_train,y_train)
rf_randomcv.best_params_
rbest = rf_randomcv.best_estimator_
rbest
rfpred = rbest.predict(x_test)
confusion = confusion_matrix(y_test , rfpred)
print(confusion)
print("Accuray Score HyperTuning Random forest:{}".format(accuracy_score(y_test,rfpred) * 100))
sns.heatmap(confusion ,annot = True , cmap = "rainbow")
plt.title("Confusion_matrix for HyperTuning Random forest")
print("Classification Report HyperTuning Random forest:{}".format(classification_report(y_test , rfpred)))
