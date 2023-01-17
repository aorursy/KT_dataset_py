import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score,mean_squared_error,roc_curve,roc_auc_score,classification_report,r2_score,confusion_matrix

# Visualization Seaborn & Matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly for interactive graphics 
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
#Disabling the warnings
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/diabetes/diabetes.csv')
df.head()
df.info()
cdf = df.copy()
cdf.Pregnancies.value_counts()
plt.figure(figsize = (12,6)) 
sns.heatmap(cdf.corr(),robust=True,fmt='.1g',linewidths=1.3,linecolor = 'gold', annot=True,);
cdf.nunique()
cdf.describe().T
print("satir ve sutun=",cdf.shape)
print("boyut sayisi = ",cdf.ndim)
print("boyut sayisi = ",cdf.ndim)
sns.countplot(cdf.Outcome);
import missingno as msno
msno.matrix(cdf);
x = cdf.drop(["Outcome"],axis = 1) #independent value
y = cdf.Outcome
x = (x-np.min(x))/(np.max(x)-np.min(x)).values
x.head()
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,GridSearchCV
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression().fit(x_train,y_train)
log_reg
log_reg.intercept_
log_reg.coef_
y_pred = log_reg.predict(x_test)
y_pred[0:10]
y_probs = log_reg.predict_proba(x_test)[:,1]
y_pred = [1 if i >0.45 else 0 for i in y_probs]
y_pred[:10]
log_score = accuracy_score(y_test,y_pred)
print ("log score=",log_score)
from sklearn.metrics import accuracy_score

log_score = accuracy_score(y_test,y_pred)
print ("log score=",log_score)
confusion_matrix(y_test,y_pred)
y_pred = [1 if i >0.52 else 0 for i in y_probs]
y_pred[:10]

from sklearn.ensemble import RandomForestClassifier
r_for = RandomForestClassifier().fit(x_train,y_train)
r_for
y_pred = r_for.predict(x_test)
rf_score = accuracy_score(y_test,y_pred)
rf_score
Importance = pd.DataFrame({"Importance": r_for.feature_importances_*100},
                         index = x_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "g")

plt.xlabel("Değişken Önem Düzeyleri")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
knn
y_pred = knn.predict(x_test)
knn_score = accuracy_score(y_test,y_pred)
knn_score
confusion_matrix(y_test,y_pred)
knn_params = {"n_neighbors":np.arange(1,50)}
knn =KNeighborsClassifier()
knn_cv = GridSearchCV(knn,knn_params,cv = 10)
knn_cv = knn_cv.fit(x_train,y_train)
print("Best Score:"+str(knn_cv.best_score_))
print("Best Parameters:"+str(knn_cv.best_params_))
knn_final = KNeighborsClassifier(n_neighbors=1)
knn_final.fit(x_train,y_train)
y_pred = knn_final.predict(x_test)
knn_fscore = accuracy_score(y_test,y_pred)
knn_fscore
confusion_matrix(y_test,y_pred)
from sklearn.svm import SVC
svm_model = SVC(C=5,degree=9,kernel = 'poly').fit(x_train,y_train)
svm_model
y_pred = svm_model.predict(x_test)
svm_score = accuracy_score(y_test,y_pred)
svm_score
confusion_matrix(y_test,y_pred)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(x_train, y_train)
nb_model
nb_model.predict(x_test)[:10]
nb_model.predict_proba(x_test)[0:10]  

y_pred = nb_model.predict(x_test)
nb_score = accuracy_score(y_test, y_pred)
print("NB_class_SCORE = ", nb_score)  
cross_val_score(nb_model, x_test, y_test, cv = 20).mean()  
indexx = ["Log","RF","KNN","SVM","NB"]
regressions = [log_score,rf_score,knn_fscore,svm_score,nb_score]

plt.figure(figsize=(8,6))
sns.barplot(x=indexx,y=regressions)
plt.xticks()
plt.title('Model Comparision',color = 'orange',fontsize=20);

indexx = ["Log","RF","KNN","SVM","NB"]
regressions = [log_score,rf_score,knn_fscore,svm_score,nb_score]

import plotly.express as px

fig = px.pie(df, values=regressions, names=indexx, title='Regression Score Results')
fig.show()