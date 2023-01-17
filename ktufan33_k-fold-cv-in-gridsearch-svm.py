
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# düz okuması için yapılan 

data=pd.read_csv("../input/pd_speech_features.csv",header=1) 
data.head(10)

data.tail(10)
# 
data.info()
# data.value_counts()
data["class"].value_counts()

print(data.describe())
sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
# 1) önclikle 3 satır bir satır olsun ()
# 2) sadece 1. satırlar
# 3) sadece 2. satırlar
# 4) sadece 3. satırlar
# 5) her iç satırın ortalaması
# 6) devam edecek

# for i in range(0, 12, 3):
#     print(i)
row1=data.iloc[0,:]
# print(row1)
# df_3in1row=[] #create an empty datafame
df_3in1row=data.iloc[0:3,:]
#print(df_3in1row.head())

df_3in1row.index = df_3in1row.index + 1
df_3in1row = df_3in1row.stack()
# df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
df_3in1row.index = df_3in1row.index.map('{0[1]}'.format)
df_3in1row=df_3in1row.to_frame().T
#df_3in1row.head()
for i in range(3, 756, 3):
    """
    1) drop [id,class] columns for row2, [id] cloumns for row3 and class cloumn from row1
    2) concate 3 rows into a single row
    """
    # GET THE NEXT 3 ROWS AND MAKE ONE RAW DATAFRAME
    df_test3=data.iloc[i:i+3,:]
    #   print(df_test3.head())

    df_test3.index = df_test3.index + 1
    df_out2 = df_test3.stack()
    # df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
    df_out2.index = df_out2.index.map('{0[1]}'.format)
    df_out2=df_out2.to_frame().T
    #     print(df_out2.head())
    # CONCATANATE 
    df_3in1row = df_3in1row.append(df_out2)
df_3in1row.head()
#df_3in1row.info()
# burada silinecek sütunlar myid ve myclass olarak kaydediliyor
testcol=df_3in1row[['id', 'class']]
testcol.columns = ['a', 'b','c','d','e','f']
myid=testcol[['a']]
myclass=testcol[['d']]
myclass.head()

# myindexes=[]
# a=df_3in1row.columns.get_loc("id")
# for i in range(len(a)):
#     if a[i]==True:
#         myindexes.append(i)
        
# print(myindexes)
        
#silme (id ve class sütunları 3 defa tekrarlanıyor. birr indeksleri verilince tamamı siliniyor. bunun yerinde isimleriyle silinirse daha iyi olur)
cols = [0,754]
print(df_3in1row.info())
print("*"*30)
df_3in1row.drop(df_3in1row.columns[cols],axis=1,inplace=True)
print(df_3in1row.head())
print(df_3in1row.info())
#concat myclass to df_3in1row
# result2 = pd.concat([myid,df_3in1row, myclass],axis=1) myid ye gerek yok
data_result = pd.concat([df_3in1row, myclass],axis=1)
data_result.head()
print(data_result.head())
# data_result.shape
data_result.head()
# burada istenilen column indexleri nerde o bulunacak
"""
sonuçlar:
tqwt_kurtosisValue_dec_36 --> [752, 1505, 2258]
tqwt_kurtosisValue_dec_1  --> [717, 1470, 2223]

"""
myindex=[]
a=data_result.columns.get_loc("tqwt_kurtosisValue_dec_36")
# print(len(a))

# a=pd.DataFrame(a)
for i in range(len(a)):
    if a[i]==True:
        myindex.append(i)
    
# myindex
print(myindex)

"""
k-fold CV in hyperparameter optimization with linearSVM
"""
# test01_data=data_result.iloc[:,1350:1510]
# test01_data=data_result.iloc[:,250:1505]
# class_weight = {0: 1.,
#                 1: 3.04}

# """


# test01_data=data_result.iloc[:,250:1505]
# test01_data.head()

# test01_data = pd.concat([test01_data, myclass],axis=1)
# # test01_data.head()
# df=test01_data
# y=df.iloc[:,-1]
# X=df.iloc[:, :-1]
# from sklearn import preprocessing
# scaler=preprocessing.MinMaxScaler()
# X=scaler.fit_transform(X)
# X=pd.DataFrame(X)
# X.columns =X.columns
# class_weight = {0: 1.,
#                 1: 3.04}
# # test
# from sklearn.svm import SVC
# from sklearn import metrics
# from sklearn.model_selection import cross_val_score
# C_range=list(np.arange(0.2,1.9,0.1))
# acc_scoreLSVM=[]
# for c in C_range:
#     print(c)
#     svc = SVC(kernel='linear', C=c,max_iter=500000,class_weight=class_weight)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     #     print(scores)
#     acc_scoreLSVM.append(scores.mean())
# print(acc_scoreLSVM)

"""
Bu kısımda RBF ve POLY için SVM sınıflama yapılıyor
Poly için sadece 1 degree değeri test edilebiliyor. eğer değişik degree değerleri denenmek istenirse
bir for döngüsü de degree için yapılması lazım.
yine degree değeri sonuçlar üzerinde yazdılıcak ise o kısım da düzenlenmelidir.

"""

# burada istersek sigmoid de var. o işlenmedi
test01_data=data_result.iloc[:,715:1500] # sınıflama kullanılacak olan sınıflar...[0--2258] arası...
test01_data.head()

test01_data = pd.concat([test01_data, myclass],axis=1) # burada class column ile birleştiriliyor
# print(test01_data.head(20))
df=test01_data
y=myclass
# y=df.iloc[:,-1]
X=df.iloc[:, :-1]
# print(y.head(20))
# print(X.shape)

from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler() # SVM için scaling önemli
X=scaler.fit_transform(X)
X=pd.DataFrame(X)

# print(len(df.columns))
# X.columns =df.columns[:len(df.columns)-1]
# y.columns=["Class"]
# print(y.head(20))
# print(X.head(20))
class_weight = {0: 1.,1: 3.04}
# degree=[2,3,4,5,6]
# test
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# C_range=list(np.arange(0.2,1.9,0.1))
# gamma_range=[0.0001,0.001,0.01,0.1,1,3,7,10,20,35,60,100]
# degree=[2,3,4,5,6]
C_range=[1.5]
gamma_range=[5]
degree=3 #poly için gerekli

acc_score=[]
mydummy=np.zeros(14)
columns = ['C','Gamma','Fold01', 'Fold02','Fold03', 'Fold04','Fold05', 'Fold06','Fold07', 'Fold08','Fold09', 'Fold10',"Average", "Median"]
myCVDF = pd.DataFrame(index=range(1000),columns=columns) # sonuçları tutacak dataframe
i=0
for g in gamma_range:
    for c in C_range:
        print(i)
        # svc = SVC(kernel='sigmoid',C=c, gamma=g)
#         svc = SVC(kernel='rbf',C=c, gamma=g,max_iter=50,class_weight=class_weight)
        svc = SVC(kernel='poly',C=c, degree=5, gamma=g,class_weight=class_weight,max_iter=50000000)
        scores = cross_val_score(svc, X, y.values.ravel(), cv=10, scoring='accuracy')
        print("Score: ",scores)
        acc_score.append(scores.mean())
        # test to take all scores
        # all_accuracies = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    #     print("gamma = ",g )
    #     print(scores)  
    #     print("="*30)
        """
        here to fill C and gamma, fold values, mean and median values into dataframe - myCVDF
        """
        mydummy[0]=c
        mydummy[1]=g 
        mydummy[2:12]=scores[:]
#         print("mydummy is: ",mydummy)
        s1=pd.DataFrame(mydummy)
#       print("S1 is: ",s1)
        for x in range(12):
            myCVDF.iloc[i,x]=s1[0][x]
        myCVDF.iloc[i,12]=np.mean(scores)
        myCVDF.iloc[i,13]=np.median(scores) 
        i +=1
        # myCVDF.append(s1,ignore_index=True)
        # myCVDF[i,:]=mydummy
# print("+"*50)
# print("Accuracy: ",acc_score)
# print("+"*50)
# print(myCVDF.head(50))
#  # s1.shape
#  y.head()
myCVDF.head(200)
# s1.shape


# # test
# C_range=list(np.arange(0.1,2,0.4))
# acc_score=[]
# for c in C_range:
#     print(c)
#     svc = SVC(kernel='linear', C=c)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(acc_score) 

# df=test01_data
# y=df.iloc[:,-1]
# #df.corr()
# df.isnull().sum()
# df.shape
# df[df.label=="d"].shape[0]
# df[df.iloc[:,36]==0].count()
# df.describe()
# X=df.iloc[:, :-1]
# X.head()
# y=df.iloc[:,-1]
# y.head()


# # Scale the data to be between -1 and 1 YOL 2
# # from sklearn import preprocessing
# from sklearn import preprocessing
# scaler=preprocessing.MinMaxScaler()
# X=scaler.fit_transform(X)
# X=pd.DataFrame(X)
# X.columns =X.columns
# # X

#testxx=pd.DataFrame(minmaxX)
# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# X_train.shape 
# X_train.head()
# from sklearn.svm import SVC
# from sklearn import metrics
# svc=SVC(gamma="scale")
# svc.fit(X_train,y_train)
# y_pred=svc.predict(X_test)
# print("Accuracy Score: ",metrics.accuracy_score(y_test,y_pred))
# svc=SVC(kernel='linear')
# svc.fit(X_train,y_train)
# y_pred=svc.predict(X_test)
# print('Accuracy Score:')
# print(metrics.accuracy_score(y_test,y_pred))
# svc=SVC(kernel='rbf')
# svc.fit(X_train,y_train)
# y_pred=svc.predict(X_test)
# print('Accuracy Score:')
# print(metrics.accuracy_score(y_test,y_pred))
# svc=SVC(kernel='poly')
# svc.fit(X_train,y_train)
# y_pred=svc.predict(X_test)
# print('Accuracy Score:')
# print(metrics.accuracy_score(y_test,y_pred))
# from sklearn.model_selection import cross_val_score
# svc=SVC(kernel='linear')
# scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
# print(scores)
# np.median(scores)

# from sklearn.model_selection import cross_val_score
# svc=SVC(kernel='rbf',gamma="scale")
# scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
# print(scores)
# from sklearn.model_selection import cross_val_score
# svc=SVC(kernel='poly',gamma="scale")
# scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
# print(scores)
# C_range=list(range(1,26))
# acc_score=[]
# for c in C_range:
#     svc = SVC(kernel='linear', C=c,gamma="scale")
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(sum(acc_score)/len(acc_score))    
    
# C_range=list(np.linspace(0.025,5,40))
# acc_score=[]
# for c in C_range:
#     svc = SVC(kernel='linear', C=c)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# #print(acc_score)    
# print(sum(acc_score)/len(acc_score))   
# print(max(acc_score))

# import matplotlib.pyplot as plt
# %matplotlib inline


# #C_values=list(range(1,26))
# C_range=list(np.linspace(0.25,5,40))
# # plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
# # plt.plot(C_values,acc_score)
# plt.plot(C_range,acc_score)
# # plt.xticks(np.arange(0,27,2))
# plt.xlabel('Value of C for SVC')
# plt.ylabel('Cross-Validated Accuracy')
# C_range=list(np.arange(0.1,6,0.1))
# acc_score=[]
# for c in C_range:
#     svc = SVC(kernel='linear', C=c)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(acc_score) 
# import matplotlib.pyplot as plt
# %matplotlib inline

# C_values=list(np.arange(0.1,6,0.1))
# # plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(C_values,acc_score)
# plt.xticks(np.arange(0.0,6,0.3))
# plt.xlabel('Value of C for SVC ')
# plt.ylabel('Cross-Validated Accuracy')
# gamma_range=[0.0001,0.001,0.01,0.1,1,3,7,10,20,35,60,100]
# acc_score=[]
# for g in gamma_range:
#     svc = SVC(kernel='rbf', gamma=g)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(acc_score)    
    
# import matplotlib.pyplot as plt
# %matplotlib inline

# gamma_range=[0.0001,0.001,0.01,0.1,1,3,7,10,20,35,60,100]
# # plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(gamma_range,acc_score)
# from sklearn.model_selection import cross_val_score  
# gamma_range=[0.0001,0.001,0.01,0.1,1,3,7,10,20,35,60,100]
# acc_score=[]
# mydummy=np.zeros(13)
# columns = ['Gamma','Fold01', 'Fold02','Fold03', 'Fold04','Fold05', 'Fold06','Fold07', 'Fold08','Fold09', 'Fold10',"Average", "Median"]
# myCVDF = pd.DataFrame(index=range(100),columns=columns) # sonuçları tutacak dataframe
# i=0
# for g in gamma_range:
#     svc = SVC(kernel='rbf', gamma=g)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
#     # test to take all scores
#     # all_accuracies = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
# #     print("gamma = ",g )
# #     print(scores)  
# #     print("="*30)
#     """
#     here to fill gamma, fold values, mean and median values into dataframe - myCVDF
#     """
#     mydummy[0]=g 
#     mydummy[1:11]=scores[:]
#     # print(mydummy)
#     s1=pd.DataFrame(mydummy)
#     for x in range(11):
#         myCVDF.iloc[i,x]=s1[0][x]
#     myCVDF.iloc[i,11]=np.mean(scores)
#     myCVDF.iloc[i,12]=np.median(scores) 
#     # myCVDF.append(s1,ignore_index=True)
#     # myCVDF[i,:]=mydummy
    
#     # myCVDF.append(pd.Series(mydummy),ignore_index=True)
#     # s1.append(s2)
#     i +=1
# # print ("average accuracy: ")
# # print(acc_score)  
# # myCVDF.head()
    

# myCVDF.head()
# myCVDF["Median"].max()
# C= np.arange(0.1,1,0.1)
# C[0]
# # burada istersek sigmoid de var. 
# from sklearn.model_selection import cross_val_score  
# gamma_range=[0.0001,0.001,0.01,0.1,1,3,7,10,20,35,60,100]
# C= np.arange(0.1,1,0.1)
# acc_score=[]
# mydummy=np.zeros(14)
# columns = ['C','Gamma','Fold01', 'Fold02','Fold03', 'Fold04','Fold05', 'Fold06','Fold07', 'Fold08','Fold09', 'Fold10',"Average", "Median"]
# myCVDF = pd.DataFrame(index=range(1000),columns=columns) # sonuçları tutacak dataframe
# i=0
# for g in gamma_range:
#     for c in C:
#         # svc = SVC(kernel='sigmoid',C=c, gamma=g)
#         svc = SVC(kernel='rbf',C=c, gamma=g)
#         scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#         acc_score.append(scores.mean())
#         # test to take all scores
#         # all_accuracies = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     #     print("gamma = ",g )
#     #     print(scores)  
#     #     print("="*30)
#         """
#         here to fill gamma, fold values, mean and median values into dataframe - myCVDF
#         """
#         mydummy[0]=c
#         mydummy[1]=g 
#         mydummy[2:12]=scores[:]
#         # print(mydummy)
#         s1=pd.DataFrame(mydummy)
#         for x in range(11):
#             myCVDF.iloc[i,x]=s1[0][x]
#         myCVDF.iloc[i,12]=np.mean(scores)
#         myCVDF.iloc[i,13]=np.median(scores) 
#         # myCVDF.append(s1,ignore_index=True)
#         # myCVDF[i,:]=mydummy

#         # myCVDF.append(pd.Series(mydummy),ignore_index=True)
#         # s1.append(s2)
#         i +=1
# # print ("average accuracy: ")
# # print(acc_score)  
# # myCVDF.head()
    
# myCVDF.head()
# myCVDF["Median"].max()
# myCVDF["Median"].max()
# # TO GET THE İNDEX OF THE MAXIMUM MEDIAN VALUE
# index_max_median=np.argmax(myCVDF["Median"])[0]
# #index_max_median
# myCVDF.iloc[index_max_median,:]
# myCVDF["Median"].max()
# # myCVDF.index(myCVDF["Median"].max())
# # myCVDF['Median'].idxmax()
# # myCVDF.dtypes
# # myCVDF2=pd.DataFrame(myCVDF)
# #myCVDF['Median'].idxmax()
# from sklearn.model_selection import cross_val_score  
# C_range=np.arange(0.05,3,0.05)
# acc_score=[]
# mydummy=np.zeros(13)
# columns = ['C','Fold01', 'Fold02','Fold03', 'Fold04','Fold05', 'Fold06','Fold07', 'Fold08','Fold09', 'Fold10',"Average", "Median"]
# myCVDF = pd.DataFrame(index=range(100),columns=columns) # sonuçları tutacak dataframe
# i=0
# for c in C_range:
#     svc = SVC(kernel='linear', C=c)
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
#     # test to take all scores
#     # all_accuracies = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
# #     print("gamma = ",g )
# #     print(scores)  
# #     print("="*30)
#     """
#     here to fill gamma, fold values, mean and median values into dataframe - myCVDF
#     """
#     mydummy[0]=c 
#     mydummy[1:11]=scores[:]
#     # print(mydummy)
#     s1=pd.DataFrame(mydummy)
#     for x in range(11):
#         myCVDF.iloc[i,x]=s1[0][x]
#     myCVDF.iloc[i,11]=np.mean(scores)
#     myCVDF.iloc[i,12]=np.median(scores) 
#     # myCVDF.append(s1,ignore_index=True)
#     # myCVDF[i,:]=mydummy
    
#     # myCVDF.append(pd.Series(mydummy),ignore_index=True)
#     # s1.append(s2)
#     i +=1
# # print ("average accuracy: ")
# # print(acc_score)  
# # myCVDF.head()
# myCVDF.head()
# myCVDF["Median"].max()
# from sklearn.model_selection import cross_val_score  
# gamma_range=[0.001,0.01,0.1,1,3,7,10]
# C= np.arange(0.1,1,0.3)
# degree=[2,3,4,5,6]
# acc_score=[]
# mydummy=np.zeros(15)
# columns = ['C','Gamma','Degree','Fold01', 'Fold02','Fold03', 'Fold04','Fold05', 'Fold06','Fold07', 'Fold08','Fold09', 'Fold10',"Average", "Median"]
# myCVDF = pd.DataFrame(index=range(1000),columns=columns) # sonuçları tutacak dataframe
# i=0
# for d in degree:
#     for g in gamma_range:
#         for c in C:
#             svc = SVC(kernel='poly',C=c, gamma=g,degree=d)
#             scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#             acc_score.append(scores.mean())
#             # test to take all scores
#             # all_accuracies = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#         #     print("gamma = ",g )
#         #     print(scores)  
#         #     print("="*30)
#             """
#             here to fill gamma, fold values, mean and median values into dataframe - myCVDF
#             """
#             mydummy[0]=c
#             mydummy[1]=g
#             mydummy[2]=d 
#             mydummy[3:13]=scores[:]
#             # print(mydummy)
#             s1=pd.DataFrame(mydummy)
#             for x in range(11):
#                 myCVDF.iloc[i,x]=s1[0][x]
#             myCVDF.iloc[i,13]=np.mean(scores)
#             myCVDF.iloc[i,14]=np.median(scores) 
#             # myCVDF.append(s1,ignore_index=True)
#             # myCVDF[i,:]=mydummy

#             # myCVDF.append(pd.Series(mydummy),ignore_index=True)
#             # s1.append(s2)
#             i +=1
# # print ("average accuracy: ")
# # print(acc_score)  
# # myCVDF.head()
# myCVDF.head(5)
# myCVDF["Median"].median()
# myCVDF["Median"].max()



# degree=[2,3,4,5,6]
# acc_score=[]
# for d in degree:
#     svc = SVC(kernel='poly', degree=d,gamma="scale")
#     scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
#     acc_score.append(scores.mean())
# print(acc_score) 
# import matplotlib.pyplot as plt
# %matplotlib inline

# degree=[2,3,4,5,6]

# # plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(degree,acc_score,color='r')
# plt.xlabel('degrees for SVC ')
# plt.ylabel('Cross-Validated Accuracy')
# from sklearn.svm import SVC
# svm_model= SVC()
# tuned_parameters = {
#  'C': (np.arange(0.1,1,0.1)) , 'kernel': ['linear'],
#  'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel': ['rbf'],
#  'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05], 'C':(np.arange(0.1,1,0.1)) , 'kernel':['poly']
#                    }
# from sklearn.model_selection import GridSearchCV

# model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')
# model_svm.fit(X_train, y_train)
# print(model_svm.best_score_)
# print(model_svm.best_params_)
# y_pred= model_svm.predict(X_test)
# print(metrics.accuracy_score(y_pred,y_test))
# my=metrics.accuracy_score(y_pred,y_test)
# my


