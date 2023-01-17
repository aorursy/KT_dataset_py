import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.cluster import KMeans



from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_fscore_support as score



from numpy.linalg import matrix_rank
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
features=df.columns[0:-1]

X_o=df[features]

y=df.target
# 'cp', 'thal' and 'slope' 

df['cp'] = df['cp'].astype('category')

df['thal'] = df['thal'].astype('category')

df['slope'] = df['slope'].astype('category')



a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")



frm = [X_o,a, b, c]

X_wCatVar = pd.concat(frm, axis = 1)



X = X_wCatVar.drop(columns = ['cp', 'thal', 'slope'])



print ('Number of Features in original data : {} , rank of data matrix  : {}'.format(X_o.shape[1],matrix_rank(X_o)))

print ('Number of Features added with categorical data : {} , rank of data matrix  : {}'.format(X_wCatVar.shape[1],matrix_rank(X_wCatVar)))

print ('Number of Features after original categorical features removed : {} , rank of data matrix  : {}'.format(X.shape[1],matrix_rank(X)))

abc=pd.concat([a, b, c], axis = 1)

print("Dimension of dummy variables",abc.shape[1],"rank of dummy variables", matrix_rank(abc))
# Split the data into a training set and a test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
acc_lin_1=[]

acc_q_1=[]

TP_lin_1=[]

TP_q_1=[]



for i in range(1,X.shape[1]+1):

    

    pca = PCA(n_components=i)

    X_train_pc = pca.fit_transform(X_train)

    X_test_pc=pca.transform(X_test)

    

    lda= LinearDiscriminantAnalysis(solver="svd", store_covariance=True)

    y_lin_pred = lda.fit(X_train_pc, y_train).predict(X_test_pc)

    acc_lin_1.append(accuracy_score(y_test, y_lin_pred))

    precision,recall,fscore,support=score(y_test, y_lin_pred)

    TP_lin_1.append(recall[1]) 



    qda = QuadraticDiscriminantAnalysis(store_covariance=True)

    y_q_pred = qda.fit(X_train_pc, y_train).predict(X_test_pc)

    acc_q_1.append(accuracy_score(y_test, y_q_pred))

    precision,recall,fscore,support=score(y_test, y_q_pred)

    TP_q_1.append(recall[1])
fig, ax = plt.subplots(figsize=(8, 5))

ii=np.arange(1,X.shape[1]+1)

var_exp_1=(pca.explained_variance_ratio_*100)

ax.plot(ii, var_exp_1,'rx-',label='Explained Variace Ratio')

ax.set_xticks(np.arange(0, 21, step=1))

ax.set_xlabel("Number of Principal Components")

ax.set_ylabel("Variance Ratio")

ax.set_title("Scree Plot - without standardization of the data",fontsize=16)

#ax.legend()

#plt.savefig("Scree Plot before std.png")

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(14, 5))

ii=np.arange(1,X.shape[1]+1)

ax1.plot(ii, acc_lin_1,'rx-',label='Accuracy-LDA')

ax1.plot(ii, acc_q_1, 'go-',label='Accuracy-QDA')

ax1.set_xlabel("Number of Principal Components")

ax1.set_ylabel("Accuracy")

ax1.set_xticks(np.arange(0, 21, step=1))

ax1.set_title("Accuracy")

ax1.legend()



ax2.plot(ii, TP_lin_1,'rx-',label='Recall-TP-LDA')

ax2.plot(ii, TP_q_1, 'go-',label='Recall-TP-QDA')

ax2.set_xlabel("Number of Principal Components")

ax2.set_ylabel("Recall-TP")

ax2.set_title("Recall-TP")

ax2.set_xticks(np.arange(0, 21, step=1))

ax2.legend()

f.suptitle('LDA-QDA :Performance Based on Number of Prin Comps Before Standardization', fontsize=16)

#plt.savefig("LDA QDA  Performanve before std.png")

plt.show()
scaler = StandardScaler()

X_std=scaler.fit_transform(X)

X_std_train, X_std_test, y_train, y_test = train_test_split(X_std, y,random_state=0)
acc_lin_2=[]

acc_q_2=[]

TP_lin_2=[]

TP_q_2=[]



for i in range(1,X.shape[1]+1):

    

    pca_std = PCA(n_components=i)

    X_std_train_pc = pca_std.fit_transform(X_std_train)

    X_std_test_pc=pca_std.transform(X_std_test)

    

    lda= LinearDiscriminantAnalysis(solver="svd", store_covariance=True)

    y_lin_pred = lda.fit(X_std_train_pc, y_train).predict(X_std_test_pc)

    acc_lin_2.append(accuracy_score(y_test, y_lin_pred))

    precision,recall,fscore,support=score(y_test, y_lin_pred)

    TP_lin_2.append(recall[1]) 



    qda = QuadraticDiscriminantAnalysis(store_covariance=True)

    y_q_pred = qda.fit(X_std_train_pc, y_train).predict(X_std_test_pc)

    acc_q_2.append(accuracy_score(y_test, y_q_pred))

    precision,recall,fscore,support=score(y_test, y_q_pred)

    TP_q_2.append(recall[1])
fig, ax = plt.subplots(figsize=(8, 5))

ii=np.arange(1,X.shape[1]+1)

var_exp_2=(pca_std.explained_variance_ratio_*100)

ax.plot(ii, var_exp_2,'rx-',label='Explained Variace Ratio')

ax.set_xticks(np.arange(0, 21, step=1))

ax.set_xlabel("Number of Principal Components")

ax.set_ylabel("Variance Ratio")

ax.set_title("Scree Plot - without standardization of the data",fontsize=16)

#ax.legend()

#plt.savefig("Scree Plot before std.png")

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(14, 5))

ii=np.arange(1,X.shape[1]+1)

ax1.plot(ii, acc_lin_2,'rx-',label='Accuracy-LDA')

ax1.plot(ii, acc_q_2, 'go-',label='Accuracy-QDA')

ax1.set_xlabel("Number of Principal Components")

ax1.set_ylabel("Accuracy")

ax1.set_xticks(np.arange(0, 21, step=1))

ax1.set_title("Accuracy")

ax1.legend()



ax2.plot(ii, TP_lin_2,'rx-',label='Recall-TP-LDA')

ax2.plot(ii, TP_q_2, 'go-',label='Recall-TP-QDA')

ax2.set_xlabel("Number of Principal Components")

ax2.set_ylabel("Recall-TP")

ax2.set_title("Recall-TP")

ax2.set_xticks(np.arange(0, 21, step=1))

ax2.legend()

f.suptitle('LDA-QDA :Performance Based on Number of Prin Comps Before Standardization', fontsize=16)

#plt.savefig("LDA QDA  Performanve before std.png")

plt.show()
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ii, var_exp_1,'bo-',label='Without standardization')

ax.plot(ii, var_exp_2,'rx-',label='After standardization')

ax.set_xticks(np.arange(0, 21, step=1))

ax.set_xlabel("Number of Principal Components")

ax.set_ylabel("Variance Ratio")

ax.set_title("Scree Plot",fontsize=16)

ax.legend()

#plt.savefig("Scree Plot before and after.png")

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharey=False,figsize=(18, 18))

ii=np.arange(1,X.shape[1]+1)



ax1.plot(ii, acc_lin_1,'cx--',label='Without Standardization - Accuracy-LDA')

ax1.plot(ii, acc_q_1, 'yo--',label='Without Standardization - Accuracy-QDA')



ax1.plot(ii, acc_lin_2,'rx-',label='After Standardization - Accuracy-LDA')

ax1.plot(ii, acc_q_2, 'go-',label='After Standardization - Accuracy-QDA')



ax1.set_xlabel("Number of Principal Components",fontsize=16)

ax1.set_ylabel("Accuracy",fontsize=16)

ax1.set_xticks(np.arange(0, 21, step=1))

ax1.legend(loc='lower center')

#ax1.legend()



ax2.plot(ii, TP_lin_1,'cx--',label='Without Standardization -Recall-TP-LDA')

ax2.plot(ii, TP_q_1, 'yo--',label='Without Standardization - Recall-TP-QDA')



ax2.plot(ii, TP_lin_2,'rx-',label='After Standardization - Recall-TP-LDA')

ax2.plot(ii, TP_q_2, 'go-',label='After Standardization - Recall-TP-QDA')



ax2.set_xlabel("Number of Principal Components",fontsize=16)

ax2.set_ylabel("Recall-TP",fontsize=16)

ax2.legend()

#f.suptitle('Performance Based on Number of Principal Components After Standardization', fontsize=16)

ax1.set_title('Accuracy Based on Number of Principal Components', fontsize=20)

ax2.set_title('TP Based on Number of Principal Components', fontsize=20)

ax2.set_xticks(np.arange(0, 21, step=1))

#plt.savefig("Performance Plot before and after.png")

plt.show()
#def kmeansf_1 ():

acc_3=[]

TP_3=[]

for i in range(1,X_std.shape[1]+1):

    pca = PCA(n_components=i)

    X_train_pc = pca.fit_transform(X_train)

    X_test_pc=pca.transform(X_test)



    kmeans = KMeans(n_clusters=2)

    y_k_pred=kmeans.fit(X_train_pc).predict(X_test_pc)



    acc_3.append(accuracy_score(y_test, y_k_pred))

    cm_3 = confusion_matrix(y_test, y_k_pred) 

    precision,recall,fscore,support=score(y_test, y_k_pred)

    TP_3.append(recall[1])

    ii=np.arange(1,X.shape[1]+1)               



    if len(ii)==len(acc_3):

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(14, 5))



        ax1.plot(ii, acc_3,'ro-',label='Accuracy')

        ax1.set_xticks(np.arange(0, 21, step=1))



        ax1.set_xlabel("Number of Principal Components")

        ax1.set_ylabel("Accuracy")

        ax1.set_title("Accuracy")

        ax1.legend()



        ax2.plot(ii, TP_3, 'go-',label='Recall-TP')

        ax2.set_xlabel("Number of Principal Components")

        ax2.set_ylabel("Recall-TP")

        ax2.set_title("Recall-TP")

        ax2.legend()

        ax2.set_xticks(np.arange(0, 21, step=1))

        f.suptitle('KMeans : Performance Based on Number of Principal Components Before Standardization', fontsize=16)

        #plt.savefig("K Means:Performance before standardization.png")

        plt.show()

#def kmeansf_2 ():

acc_4=[]

TP_4=[]

for i in range(1,X_std.shape[1]+1):

    pca_std = PCA(n_components=i)

    X_std_train_pc = pca_std.fit_transform(X_std_train)

    X_std_test_pc=pca_std.transform(X_std_test)



    kmeans = KMeans(n_clusters=2)

    y_k_pred=kmeans.fit(X_std_train_pc).predict(X_std_test_pc)



    acc_4.append(accuracy_score(y_test, y_k_pred))

    cm_4 = confusion_matrix(y_test, y_k_pred) 

    precision,recall,fscore,support=score(y_test, y_k_pred)

    TP_4.append(recall[1])

    ii=np.arange(1,X.shape[1]+1)               



    if len(ii)==len(acc_4):

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(14, 5))



        ax1.plot(ii, acc_4,'ro-',label='Accuracy')

        ax1.set_xticks(np.arange(0, 21, step=1))



        ax1.set_xlabel("Number of Principal Components")

        ax1.set_ylabel("Accuracy")

        ax1.set_title("Accuracy")

        ax1.legend()



        ax2.plot(ii, TP_4, 'go-',label='Recall-TP')

        ax2.set_xlabel("Number of Principal Components")

        ax2.set_ylabel("Recall-TP")

        ax2.set_title("Recall-TP")

        ax2.legend()

        ax2.set_xticks(np.arange(0, 21, step=1))

        f.suptitle('KMeans : Performance Based on Number of Principal Components After Standardization', fontsize=16)

        #plt.savefig("K Means:Performance after standardization.png")

        plt.show()



        print("Max Accuracy  :")

        print("___________________________________")

        print('LDA-wo standardization- {:.3f} with {} components'.format(max(acc_lin_1),(acc_lin_1.index(max(acc_lin_1))+1)))

        print('LDA-with standardization- {:.3f} with {} components'.format(max(acc_lin_2),(acc_lin_2.index(max(acc_lin_2))+1)))

        print('QDA-wo standardization- {:.3f} with {} components'.format(max(acc_q_1),(acc_q_1.index(max(acc_q_1))+1)))

        print('QDA-with standardization- {:.3f} with {} components'.format(max(acc_q_2),(acc_q_2.index(max(acc_q_2))+1)))

        print('KMeans-wo standardization- {:.3f} with {} components'.format(max(acc_3),(acc_3.index(max(acc_3))+1)))

        print('KMeans-with standardization- {:.3f} with {} components'.format(max(acc_4),(acc_4.index(max(acc_4))+1)))



        print()

        print("Max Recall-TP :")

        print("___________________________________")

        print('LDA-wo standardization- {:.3f} with {} components'.format(max(TP_lin_1),(TP_lin_1.index(max(TP_lin_1))+1)))

        print('LDA-with standardization- {:.3f} with {} components'.format(max(TP_lin_2),(TP_lin_2.index(max(TP_lin_2))+1)))

        print('QDA-wo standardization- {:.3f} with {} components'.format(max(TP_q_1),(TP_q_1.index(max(TP_q_1))+1)))

        print('QDA-with standardization- {:.3f} with {} components'.format(max(TP_q_2),(TP_q_2.index(max(TP_q_2))+1)))

        print('KMeans-wo standardization- {:.3f} with {} components'.format(max(TP_3),(TP_3.index(max(TP_3))+1)))

        print('KMeans-with standardization- {:.3f} with {} components'.format(max(TP_4),(TP_4.index(max(TP_4))+1)))



f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(14, 5))



ax1.plot(ii, acc_3,'cx--',label='Without Standardization - Accuracy')

ax1.plot(ii, acc_4,'yo-',label='After Standardization - Accuracy')

ax1.set_xticks(np.arange(0, 21, step=1))



ax1.set_xlabel("Number of Principal Components")

ax1.set_ylabel("Accuracy")

ax1.set_title("Accuracy")

ax1.legend(loc='center', bbox_to_anchor=(0.60, 0.3))



ax2.plot(ii, TP_3, 'rx--',label='Without Standardization - Recall-TP')

ax2.plot(ii, TP_4, 'go-',label='After Standardization - Recall-TP')



ax2.set_xlabel("Number of Principal Components")

ax2.set_ylabel("Recall-TP")

ax2.set_title("Recall-TP")

ax2.legend(loc='center', bbox_to_anchor=(0.60, 0.3))

ax2.set_xticks(np.arange(0, 21, step=1))

f.suptitle('KMeans : Performance Based on Number of Principal Components', fontsize=16)

#plt.savefig("K Means:Combined Performance after standardization.png")

plt.show()