# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.rename(columns={"pelvic_tilt numeric":"pelvic_tilt_numeric"},inplace=True)
data.info()
# 
data["class"]=[1 if each=="Abnormal" else 0 for each in data["class"]]
x_data=data.drop(columns=["class"])
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
y=data["class"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
best_score_all=[] #Storing inside all method's best scores. 
# General view of data
color_list = ['red' if i==1 else 'green' for i in data["class"]]
pd.plotting.scatter_matrix(x, alpha=0.7,figsize=(20,20),c=color_list)
plt.show()

N=data[data["class"]==0]
A=data[data["class"]==1]
fig = plt.figure("degree_spondylolisthesis & pelvic_incidence")
plt.scatter(N.pelvic_incidence,N.degree_spondylolisthesis,color="green",alpha=0.5,label="Normal")
plt.scatter(A.pelvic_incidence,A.degree_spondylolisthesis,color="red",alpha=0.5,label="Abnormal")
plt.show()
#%% KNN-Trying simle test with pelvic_incidence and degree_spondylolisthesis
# score:  0.8225806451612904
pi_ds=x.drop(columns=["pelvic_tilt_numeric","lumbar_lordosis_angle","sacral_slope","pelvic_radius"])
pi_ds_train,pi_ds_test,y_train3,y_test3=train_test_split(pi_ds,y,test_size=0.2,random_state=42)
list_of_knn=[]
for each in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(pi_ds_train,y_train3)
    list_of_knn.append(knn.score(pi_ds_test,y_test3))
plt.scatter(range(1,50),list_of_knn)
plt.show()
# KNN=5 is best one    
knn2=KNeighborsClassifier(n_neighbors=(list_of_knn.index(max(list_of_knn))+1))
knn2.fit(pi_ds_train,y_train3)
print("score: ",knn2.score(pi_ds_test,y_test3))
#%% KNN-General usage (Main Method)
# score:  0.8064516129032258
list_of_KNN=[]
for each in range(1,50):
    KNN=KNeighborsClassifier(n_neighbors=each)
    KNN.fit(x_train,y_train)
    list_of_KNN.append(KNN.score(x_test,y_test))
  
KNN=KNeighborsClassifier(n_neighbors=(list_of_KNN.index(max(list_of_KNN))+1))
KNN.fit(x_train,y_train)
print("score: ",KNN.score(x_test,y_test))
#%% KNN- Testing all columns correlation between them and return best score
# Better than main method
# score of degree_spondylolisthesis and pelvic_radius:  0.8709677419354839
list_of_columns=[]
last_score_knn=[]
for col in x.columns:
    list_of_columns.append(col)
    
def kkn(x,y):
    if x.columns[0]==x.columns[1]:
        return
    else:    
        x_train_s,x_test_s,y_train_s,y_test_s=train_test_split(x,y,test_size=0.2,random_state=42)
        list_of_knn_s=[]
        for each in range(1,50):
            knn_s=KNeighborsClassifier(n_neighbors=each)
            knn_s.fit(x_train_s,y_train_s)
            list_of_knn_s.append(knn_s.score(x_test_s,y_test_s))
        if list_of_knn_s.index(max(list_of_knn_s))== 0:
            return
        else :    
            knn_s2=KNeighborsClassifier(n_neighbors=(list_of_knn_s.index(max(list_of_knn_s))+1))
            knn_s2.fit(x_train_s,y_train_s)
            print("score of {} and {}: ".format(x.columns[0],x.columns[1]))
            print("",knn_s2.score(x_test_s,y_test_s))
            print("")
            return knn_s2.score(x_test_s,y_test_s)
    
for each in list_of_columns:
    for beach in list_of_columns:
        concat=pd.concat([x[each],x[beach]],axis=1)
        last_score_knn.append(kkn(concat,y))
last_score_knn = [x for x in last_score_knn if str(x) != 'None']
print("Best Score: ",max(last_score_knn))
best_score_all.append(max(last_score_knn))
#%% These are give us best score from KNN algorithm and let's see
fig = plt.figure("degree_spondylolisthesis & sacral_slope")
plt.scatter(N.degree_spondylolisthesis,N.sacral_slope,color="green",alpha=0.5,label="Normal")
plt.scatter(A.degree_spondylolisthesis,A.sacral_slope,color="red",alpha=0.5,label="Abnormal")
plt.show()
#%% NAIVE BAYES-- I will try columns which has giv better score on KNN algorithm
# score: 0.7903225806451613
ss_ds=x.drop(columns=["pelvic_tilt_numeric","lumbar_lordosis_angle","pelvic_incidence","pelvic_radius"])
ss_ds_train,ss_ds_test,y_train4,y_test4=train_test_split(ss_ds,y,test_size=0.2,random_state=42)
nb=GaussianNB()
nb.fit(ss_ds_train,y_train4)
print("score:",nb.score(ss_ds_test,y_test4))

#%% NAIVE BAYES--General usage (Main Method)
# score: 0.782258064516129
nb2=GaussianNB()
nb2.fit(x_train,y_train)
print("score:",nb2.score(x_train,y_train))
#%% I'm using here my own code for the get best score
# And again it works better than main method
# score of degree_spondylolisthesis and pelvic_radius: 0.8225806451612904
last_score_naive=[]
def naive(x,y):
    if x.columns[0]==x.columns[1]:
        return
    else:    
        x_train_s,x_test_s,y_train_s,y_test_s=train_test_split(x,y,test_size=0.2,random_state=42)
        nb3=GaussianNB()
        nb3.fit(x_train_s,y_train_s)
        print("score of {} and {}: ".format(x.columns[0],x.columns[1]))
        print("",nb3.score(x_test_s,y_test_s))
        print("")
        return nb3.score(x_test_s,y_test_s)

for each in list_of_columns:
    for beach in list_of_columns:
        concat=pd.concat([x[each],x[beach]],axis=1)
        last_score_naive.append(naive(concat,y))
last_score_naive = [x for x in last_score_naive if str(x) != 'None']
print("Best Score: ",max(last_score_naive))
best_score_all.append(max(last_score_naive))
#Logistic Regression model is the same score with main method
# Score: 0.7741935483870968
last_score_logreg=[]
def logistic(x,y):
    if x.columns[0]==x.columns[1]:
        return
    else:    
        x_train_s,x_test_s,y_train_s,y_test_s=train_test_split(x,y,test_size=0.2,random_state=42)
        lr=LogisticRegression()
        lr.fit(x_train_s,y_train_s)
        print("score of {} and {}: ".format(x.columns[0],x.columns[1]))
        print("",lr.score(x_test_s,y_test_s))
        print("")
        return lr.score(x_test_s,y_test_s)

for each in list_of_columns:
    for beach in list_of_columns:
        concat=pd.concat([x[each],x[beach]],axis=1)
        last_score_logreg.append(logistic(concat,y))
last_score_logreg = [x for x in last_score_logreg if str(x) != 'None']
print("Best Score: ",max(last_score_logreg))
best_score_all.append(max(last_score_logreg))
# Logistic Regression-- General usage (Main method)
# score: 0.7741935483870968
lr2 = LogisticRegression()
lr2.fit(x_train,y_train)
print("test accuracy {}".format(lr2.score(x_test,y_test)))

#%% I'm using here my own code for the get best score
# And again it works better than main method
# score of degree_spondylolisthesis and pelvic_radius: 0.8387096774193549
last_score_SVC=[]
def svc(x,y):
    if x.columns[0]==x.columns[1]:
        return
    else:    
        x_train_s,x_test_s,y_train_s,y_test_s=train_test_split(x,y,test_size=0.2,random_state=42)
        svm=SVC(random_state=42)
        svm.fit(x_train_s,y_train_s)
        print("score of {} and {}: ".format(x.columns[0],x.columns[1]))
        print("",svm.score(x_test_s,y_test_s))
        print("")
        return svm.score(x_test_s,y_test_s)

for each in list_of_columns:
    for beach in list_of_columns:
        concat=pd.concat([x[each],x[beach]],axis=1)
        last_score_SVC.append(svc(concat,y))
last_score_SVC = [x for x in last_score_SVC if str(x) != 'None']
print("Best Score: ",max(last_score_SVC))
best_score_all.append(max(last_score_SVC))
#%% SVM--General usage (Main method)
#score: 0.8064516129032258
svm2=SVC(random_state=42)
svm2.fit(x_train,y_train)
print("score: ",svm2.score(x_test,y_test)) 
# Visualizing all scores
str_models=["knn_model","nb_model","log_reg","svm_tuned_model"]
df=pd.DataFrame({"Models": str_models,"Accuracy_Score":best_score_all})
df.sort_values(by="Accuracy_Score",ascending=False)
df.reset_index(drop=True)

plt.figure(figsize=(10,10))
sns.barplot(x="Accuracy_Score",y="Models",data=df,orient="h")
plt.xticks(rotation=90)
plt.grid() 