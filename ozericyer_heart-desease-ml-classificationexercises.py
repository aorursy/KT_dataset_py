# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score,mean_squared_error,roc_curve,roc_auc_score,classification_report,r2_score,confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor


# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()
# Plotly for interactive graphics 
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df=data.copy()
df.head()
data.sample(5)  #chose randon sample from row
df.info()
df.target.unique()
df.isnull().sum()  
df["target"].value_counts()
df.describe()
df.corr()
sns.countplot(df.target, palette=['green', 'red'])
plt.title("[0] == Not Disease, [1] == Disease");
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()
f, ax = plt.subplots(figsize=(10,6)) #DISTRUBUTION OF AGE WITH DISTPLOT
x = df['age']
ax = sns.distplot(x, bins=10)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))   #DISTRUBUTION OF AGE WITH BOXPLOT
sns.boxplot(x=df["age"])
plt.show()
young_ages=df[(df.age>=29)&(df.age<40)] 
middle_ages=df[(df.age>=40)&(df.age<55)]
elderly_ages=df[(df.age>55)]
print('Young Ages :',len(young_ages))
print('Middle Ages :',len(middle_ages))
print('Elderly Ages :',len(elderly_ages))
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()
colors = ['blue','green','yellow']  #we can see in pie.
explode = [0,0,0.1]
plt.figure(figsize = (10,10))
#plt.pie([target_0_agerang_0,target_1_agerang_0], explode=explode, labels=['Target 0 Age Range 0','Target 1 Age Range 0'], colors=colors, autopct='%1.1f%%')
plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)],labels=['young ages','middle ages','elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.title('Age States',color = 'blue',fontsize = 15)
plt.show()
plt.figure(figsize=(15,7))
sns.violinplot(x=df.age,y=df.target)
plt.xticks(rotation=90)
plt.legend()
plt.title("Age & Target System")
plt.show()
df.columns
plt.figure(figsize=(10,7))
sns.barplot(x="sex",y = 'ca',hue = 'target',data=df);
plt.figure(figsize=(10,7))
sns.barplot(x="sex",y = 'oldpeak',hue = 'restecg',data=df);
sns.countplot(df.target,hue=df.sex)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target & Sex Counter 1 & 0')
plt.show()
plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()#Number of people who have heart disease according to age 
# Let's make our correlation matrix a little prettier
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
df.groupby('cp',as_index=False)['target'].mean()
df.groupby('slope',as_index=False)['target'].mean()
df.groupby('target').mean()
num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]
sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')
plt.show()
num_var = ['cp', 'slope', 'exang', 'thalach', 'oldpeak','ca','thal', 'target' ]
sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')
plt.show()
df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target")
f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))# with jitter
sns.stripplot(x="target", y="thalach", data=df, jitter = 0.01)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))  #with boxplot
sns.boxplot(x="target", y="thalach", data=df)
plt.show()
y = df.target.values
x_dat = df.drop(['target'], axis = 1)
x=(x_dat-np.min(x_dat))/(np.max(x_dat)-np.min(x_dat)).values
y=df.target.values
x_dat=df.drop(["target"],axis=1)
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,GridSearchCV
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = 'liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
lr #We can see what there is in lr(icinde hangi secenekler vargormek icin) 

#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   #intercept_scaling=1, l1_ratio=None, max_iter=100,
                  # multi_class='auto', n_jobs=None, penalty='l2',
                  # random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                  # warm_start=False)
lr.intercept_  #sabit katsayi
lr.coef_   #degisken katsayilari
l_score=accuracy_score(y_test,y_pred)
l_score
#The y predicted by the y in the test are compared(test deki y ile tahmin edilen yler karsilastiriliyor.Dogru tahmin etme yuzdesi bulunuyor)
c_l=confusion_matrix(y_test,y_pred)# We found the numbers of guessing with confusion matrix, 31 for 1 correct guess, 0 for 35 correct guess
c_l                               #The top was imported.
#confusion matrixle tahmin etme sayilarini bulduk,1 icin 31 i dogru tahmin,0 icin 35 i dogru tahmin
#En ustte import edildi.
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=lr.predict(x_test)
cmlr=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmlr, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print(classification_report(y_test,y_pred)) #yukarda import edildi
lr.predict(x_test)[0:10] #ilk 10 datatest deki tahminlerimiz
lr.predict_proba(x_test)[0:10] #1.si 0 olma 2.si 1 olma olasiligi oranlari
y_probs = lr.predict_proba(x_test)[:,1]
y_pred = [1 if i>0.52 else 0 for i in y_probs]
y_pred[-10:]
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
y = df.target
x = df.drop('target',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.25,
                                                random_state = 42)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred=nb.predict(x_test)
y_pred[:10]
nb  #we can look at which option is there in GaussionNB
n_score=accuracy_score(y_test,y_pred)
n_score
c_nb=confusion_matrix(y_test,y_pred)
c_nb
#confusion matrixle tahmin etme sayilarini bulduk,1 icin 32 i dogru tahmin,0 icin 30 i dogru tahmin
#En ustte import edildi.
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=nb.predict(x_test)
cmnb=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmnb, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print(classification_report(y_test,y_pred)) #yukarda import edildi
nb.predict(x_test)[0:10] #ilk 10 datatest deki tahminlerimiz
nb.predict_proba(x_test)[0:10] #1.si 0 olma 2.si 1 olma olasiligi oranlari
y_probs = nb.predict_proba(x_test)[:,1]
y_pred = [1 if i>0.45 else 0 for i in y_probs]
y_pred[0:10]
nb_tuned_bestscore=accuracy_score(y_test,y_pred)
nb_tuned_bestscore

cmnb_best=confusion_matrix(y_test,y_pred) 
cmnb_best
from sklearn.neighbors import KNeighborsClassifier
y=df.target
x=df.drop("target",axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.25,
                                                random_state = 42)
knn = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
y_pred
knn  ##we can look at which option is there in KNeighborsClassifier
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     #metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     #weights='uniform')
knn_score=accuracy_score(y_test,y_pred)
knn_score
c_knn=confusion_matrix(y_test,y_pred)
c_knn
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=knn.predict(x_test)
cmknn=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmknn, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print(classification_report(y_test,y_pred)) #yukarda import edildi
knn.predict(x_test)[0:10] #ilk 10 datatest deki tahminlerimiz
knn.predict_proba(x_test)[0:10] #1.si 0 olma 2.si 1 olma olasiligi oranlari
RMSE = []   # ERROR ON TRAIN DATA

for k in range(30):
    k = k+1
    knn = KNeighborsRegressor(n_neighbors = k).fit(x_train, y_train)
    y_pred = knn.predict(x_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    RMSE.append(rmse) 
    print("k =" , k , "için RMSE değeri: ", rmse)
from sklearn.model_selection import GridSearchCV  
#We use Grid for tuning
knn_params = {'n_neighbors': np.arange(1,30,1)} #we obta
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv = 10) #cross validation yontemi kullaniliyor.nesnesi tanimlandi
knn_cv_model.fit(x_train, y_train)
print("Best Score:"+str(knn_cv_model.best_score_))
print("Best Parameters:"+str(knn_cv_model.best_params_))
knn_tuned =KNeighborsClassifier(n_neighbors = 21)
knn_tuned = knn_tuned.fit(x_train,y_train)
y_pred = knn_tuned.predict(x_test)
knn_tuned_score=accuracy_score(y_test,y_pred)
knn_tuned_score
np.sqrt(mean_squared_error(y_test, knn_tuned.predict(x_test)))
knn_tune2 =KNeighborsClassifier(n_neighbors = 21,metric='hamming')
knn_tune2.fit(x_train,y_train)
y_pred = knn_tune2.predict(x_test)
knn_tuned_bestscore=accuracy_score(y_test,y_pred)
knn_tuned_bestscore
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=knn_tune2.predict(x_test)
cmknn_best=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmknn_best, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
from sklearn.svm import SVC
y=df.target
x=df.drop("target",axis=1)
y = df.target
x = df.drop('target',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.25,
                                                random_state = 42)
svm = SVC(C=5,degree=9,kernel = 'poly')
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
svm
#SVC(C=5, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    #decision_function_shape='ovr', degree=9, gamma='scale', kernel='poly',
    #max_iter=-1, probability=False, random_state=None, shrinking=True,
    #tol=0.001, verbose=False)
y_pred
svm_score1 = accuracy_score(y_test,y_pred)
svm_score1
c_svm=confusion_matrix(y_test,y_pred)
c_svm
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=svm.predict(x_test)
cmsvm=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmsvm, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print(classification_report(y_test,y_pred)) #yukarda import edildi
#EN UYGUN C VE GAMMA DEGERI BULMA
svc_params ={"C":[0.00001,0.001,0.01,5,10,50,100],
            "gamma":[0.0001,0.001,0.01,1,5,10,50,100]}
svc =SVC()
svc_cv_model = GridSearchCV(svc,svc_params,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2)
svc_cv_model.fit(x_train,y_train)
print("Best Parameters:"+str(svc_cv_model.best_params_))
# svm_tune1= SVC(C=100,gamma= 0.0001,degree=9,kernel = 'poly')
# svm_tune1.fit(x_train,y_train)
# y_pred = svm.predict(x_test)  # cok uzun suruyor
svm_score2 = accuracy_score(y_test,y_pred)
svm_score2
#we changed the kernel,We can use linear,poly,rbf...
svm_tune2 = SVC(C=100,degree=9,kernel = 'linear')
svm_tune2.fit(x_train,y_train)
y_pred = svm_tune2.predict(x_test)
accuracy_score(y_test,y_pred)
#we changed the kernel,We can use linear,poly,rbf...
svm_tune3 = SVC(C=100,degree=9,kernel = 'rbf')
svm_tune3.fit(x_train,y_train)
y_pred = svm_tune3.predict(x_test)
accuracy_score(y_test,y_pred)
# svc_tuned=SVC(C=100,gamma=0.0001,kernel = 'linear')
# svc_tuned.fit(x_train,y_train)
# y_pred = svc_tuned.predict(x_test)
# accuracy_score(y_test,y_pred)    #uzun suruyor
from sklearn.ensemble import RandomForestClassifier
y=df.target
x=df.drop("target",axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.25,
                                                random_state = 42)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
y_pred
rf
#RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       #criterion='gini', max_depth=None, max_features='auto',
                       #max_leaf_nodes=None, max_samples=None,
                      # min_impurity_decrease=0.0, min_impurity_split=None,
                      # min_samples_leaf=1, min_samples_split=2,
                      # min_weight_fraction_leaf=0.0, n_estimators=100,
                      # n_jobs=None, oob_score=False, random_state=None,
                      # verbose=0, warm_start=False)
rf_score=accuracy_score(y_test,y_pred)
rf_score
c_rf=confusion_matrix(y_test,y_pred)
c_rf
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=rf.predict(x_test)
cmlr=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmlr, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print(classification_report(y_test,y_pred)) #yukarda import edildi
rf.predict(x_test)[0:10] #ilk 10 datatest deki tahminlerimiz
rf.predict_proba(x_test)[0:10] #1.si 0 olma 2.si 1 olma olasiligi oranlari
from sklearn.ensemble import RandomForestClassifier  #n_estimotors=11 is best
score_list=[]
for each in range(1,75):
    rf2=RandomForestClassifier(n_estimators=each, random_state=42)
    rf2.fit(x_train, y_train)
    score_list.append(100*rf2.score(x_test, y_test))
    print("n_estimators=", each, "--> Accuracy:", 100*rf2.score(x_test, y_test), "%")

plt.plot([*range(1,75)], score_list)
plt.xlabel("n_estimators Value")
plt.ylabel("Accuracy %")
plt.show()
Importance = pd.DataFrame({"Importance": rf.feature_importances_*100},
                         index = x_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "g")

plt.xlabel("Variable Severity Levels");
y=df.target
x=df[['ca','oldpeak','thal','cp','thalach','age']]

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
rf_2 = RandomForestClassifier().fit(x_train, y_train)
y_pred = rf_2.predict(x_test)
rf_2_score=accuracy_score(y_test, y_pred)
rf_2_score
c_rf2=confusion_matrix(y_test,y_pred)
c_rf2
rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}
rf_model1 = RandomForestClassifier()

rf_cv_model1 = GridSearchCV(rf_model1, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2)
#rf_cv_model1.fit(x_train, y_train)    #uzun suruyor
#print("Best Parameters: " + str(rf_cv_model1.best_params_))  #uzun suruyor
rf_tuned1 = RandomForestClassifier(max_depth = 2, 
                                  max_features = 2, 
                                  min_samples_split = 2,
                                  n_estimators = 500)

rf_tuned1.fit(x_train, y_train)
y_pred = rf_tuned1.predict(x_test)
rf_tuned_score=accuracy_score(y_test, y_pred)
rf_tuned_score
from sklearn.tree import DecisionTreeClassifier
y=df.target
x=df.drop("target",axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.25,
                                                random_state = 42)
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
y_pred
dtc_score=accuracy_score(y_test,y_pred)
dtc_score
c_dtc=confusion_matrix(y_test,y_pred)
c_dtc
from sklearn.metrics import confusion_matrix   #Hepsi icin yapilabilir
y_true=y_test
y_pred=dtc.predict(x_test)
cmdtc=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(cmdtc, annot=True)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
print(classification_report(y_test,y_pred))
tree_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }
tree1 = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree1, tree_grid, cv = 10, n_jobs = -1, verbose = 2)
tree_cv_model = tree_cv.fit(x_train, y_train)
print("Best Parameters: " + str(tree_cv_model.best_params_))
tree1 = DecisionTreeClassifier(max_depth = 3, min_samples_split = 2)
tree_tuned1 = tree1.fit(x_train, y_train)
y_pred = tree_tuned1.predict(x_test)
dtc_tuned_bestscore=accuracy_score(y_test, y_pred)
dtc_tuned_bestscore
Importance = pd.DataFrame({"Importance": dtc.feature_importances_*100},
                         index = x_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "g")

plt.xlabel("Variable Severity Levels");
y=df.target
x=df[['ca','oldpeak','thal','cp','thalach','age']]

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
dtc2 = RandomForestClassifier().fit(x_train, y_train)
y_pred = dtc2.predict(x_test)
dtc2_score=accuracy_score(y_test, y_pred)
dtc2_score
c_dtc2=confusion_matrix(y_test,y_pred)
c_dtc2
dtc_tuned1 = RandomForestClassifier(max_depth = 2, 
                                  max_features = 2, 
                                  min_samples_split = 2,
                                  n_estimators = 500)
dtc_tuned1.fit(x_train, y_train)
y_pred = dtc_tuned1.predict(x_test)
dtc_tuned_bestscore=accuracy_score(y_test, y_pred)
dtc_tuned_bestscore
c_bestdtc=confusion_matrix(y_test,y_pred)
c_bestdtc


indexx = ["Log","KNN","SVM","NB","RF","DT"]
regressions = [l_score,knn_tuned_bestscore,svm_score1,nb_tuned_bestscore,rf_2_score,dtc_tuned_bestscore]

plt.figure(figsize=(8,6))
sns.barplot(x=indexx,y=regressions)
plt.xticks()
plt.title('Model Comparision',color = 'orange',fontsize=20);
plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(c_l,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cmknn_best,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(c_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cmnb_best,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(c_bestdtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(c_rf2,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()
