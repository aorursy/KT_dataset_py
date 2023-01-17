import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import os

train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Income"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

test_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
fold_count=10
train_data.shape
train_data.head()
test_data.shape
test_data.head()
n_train_data=train_data.dropna()
n_train_data.shape
num_train_data=n_train_data.iloc[1:,:]
num_train_data=num_train_data.apply(preprocessing.LabelEncoder().fit_transform)
num_train_data.head()
num_train_data.shape
train_X = num_train_data[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]] 
train_Y = num_train_data.Income

train_Y
train_X.shape
train_Y.shape
cross_scores=[]
k_list=[]

#testaremos K de 1 à 100
for i in range(1,100):
    
    k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    cross_scores.append(np.mean(cross_val_score(knn, train_X, train_Y,cv=fold_count)))
    #print(cross_scores)
plt.plot(k_list,cross_scores)
    
max_k_num=cross_scores.index(max(cross_scores))
max_k_num
cross_scores[max_k_num]
cat_train_X = num_train_data[["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
cat_train_Y = num_train_data.Income
print(cat_train_X.shape)
cat_train_X.head()
print(cat_train_Y.shape)
cat_train_Y.head()
cat_test_X.shape
cat_cross_scores=[]
cat_k_list=[]

#testaremos K de 1 à 100
for i in range(1,100):
    
    cat_k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    cat_cross_scores.append(np.mean(cross_val_score(knn, cat_train_X, cat_train_Y,cv=fold_count)))
    #print(cat_cross_scores)
    
plt.plot(cat_k_list,cat_cross_scores)
max_k_cat_num=cat_cross_scores.index(max(cat_cross_scores))
max_k_cat_num
cat_cross_scores[max_k_cat_num]
null_data=pd.isnull(test_data["Workclass"])

test_data[null_data]
null_data=pd.isnull(test_data["Country"])

test_data[null_data]
null_data=pd.isnull(test_data["Occupation"])

test_data[null_data]
faltantes_train_X = num_train_data[["Age","Workclass","Education-Num","Occupation","Capital Gain", "Capital Loss", "Hours per week", "Country"]] 
faltantes_train_Y = num_train_data.Income

faltantes_cross_scores=[]
faltantes_k_list=[]

k_init=20

#testaremos K de 20 a 40
for i in range(k_init,40):
    
    faltantes_k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    faltantes_cross_scores.append(np.mean(cross_val_score(knn, faltantes_train_X, faltantes_train_Y,cv=fold_count)))
    #print(cross_scores)
plt.plot(faltantes_k_list,faltantes_cross_scores)
max_k_faltantes_num=faltantes_cross_scores.index(max(faltantes_cross_scores))
max_k_faltantes_num+k_init
faltantes_cross_scores[max_k_faltantes_num]
faltantes_train_X = num_train_data[["Age","Education-Num","Sex","Capital Gain", "Capital Loss", "Hours per week"]] 
faltantes_train_Y = num_train_data.Income

faltantes_cross_scores=[]
faltantes_k_list=[]

k_init=20

#testaremos K de 20 a 40
for i in range(k_init,40):
    
    faltantes_k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    faltantes_cross_scores.append(np.mean(cross_val_score(knn, faltantes_train_X, faltantes_train_Y,cv=fold_count)))
    #print(cross_scores)
plt.plot(faltantes_k_list,faltantes_cross_scores)
max_k_faltantes_num=faltantes_cross_scores.index(max(faltantes_cross_scores))
max_k_faltantes_num+k_init
faltantes_cross_scores[max_k_faltantes_num]
faltantes_train_X = num_train_data[["Age","Education-Num","Race", "Capital Gain", "Capital Loss", "Hours per week"]] 
faltantes_train_Y = num_train_data.Income

faltantes_cross_scores=[]
faltantes_k_list=[]

k_init=20

#testaremos K de 20 a 40
for i in range(k_init,40):
    
    faltantes_k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    faltantes_cross_scores.append(np.mean(cross_val_score(knn, faltantes_train_X, faltantes_train_Y,cv=fold_count)))
    #print(cross_scores)
plt.plot(faltantes_k_list,faltantes_cross_scores)
max_k_faltantes_num=faltantes_cross_scores.index(max(faltantes_cross_scores))
max_k_faltantes_num+k_init
faltantes_cross_scores[max_k_faltantes_num]
faltantes_train_X = num_train_data[["Age","Education-Num","Relationship", "Capital Gain", "Capital Loss", "Hours per week"]] 
faltantes_train_Y = num_train_data.Income

faltantes_cross_scores=[]
faltantes_k_list=[]

k_init=20

#testaremos K de 20 a 40
for i in range(k_init,40):
    
    faltantes_k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    faltantes_cross_scores.append(np.mean(cross_val_score(knn, faltantes_train_X, faltantes_train_Y,cv=fold_count)))
    #print(cross_scores)
plt.plot(faltantes_k_list,faltantes_cross_scores)
max_k_faltantes_num=faltantes_cross_scores.index(max(faltantes_cross_scores))
max_k_faltantes_num+k_init
faltantes_cross_scores[max_k_faltantes_num]
f_train_X = num_train_data[["Age","Education-Num","Relationship","Sex","Capital Gain", "Capital Loss", "Hours per week"]] 
f_train_Y = num_train_data.Income

f_cross_scores=[]
f_k_list=[]

k_init=20

#testaremos K de 20 a 40
for i in range(k_init,40):
    
    f_k_list.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    
    f_cross_scores.append(np.mean(cross_val_score(knn, f_train_X, f_train_Y,cv=fold_count)))
    #print(cross_scores)
plt.plot(f_k_list,f_cross_scores)
max_k_f_num=f_cross_scores.index(max(f_cross_scores))
max_k_f_num+k_init
f_cross_scores[max_k_f_num]
knn=KNeighborsClassifier(n_neighbors=26)
knn.fit(f_train_X,f_train_Y)
test_data=test_data.iloc[1:,:]
test_X = test_data[["Age","Education-Num","Relationship","Sex","Capital Gain", "Capital Loss", "Hours per week"]] 
test_X=test_X.apply(preprocessing.LabelEncoder().fit_transform)
pred_Y= knn.predict(test_X)
pred_Y
final_Y=[]
for i in range(len(pred_Y)):
    if pred_Y[i]==1:
        final_Y.append(">50K")
    else:
        final_Y.append("<=50K")
final_Y
result=np.array(final_Y)
result=result.transpose()
result.shape
submission=pd.DataFrame(result)
submission.columns=["income"]
submission.to_csv("submission.csv", index = True, index_label = 'Id')
submission
