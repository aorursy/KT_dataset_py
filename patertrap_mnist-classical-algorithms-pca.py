# load the modules



import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style("dark")

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix,classification_report, f1_score

from sklearn.svm import SVC

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from sklearn.neural_network import MLPClassifier

def classif_summary(y_T,y_pred):

    a=classification_report(y_T,y_pred)

    return a
def plot_conf_matrix(y_T,y_pred):

    matrix=confusion_matrix(y_T,y_pred)

    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(matrix, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)
# read in the data



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

y = train["label"]

train=train.drop(["label"],axis=1)

train=train/255 #Pixel intensities 0-255, normalization is done here.

train.shape
# plot some of the numbers



#plt.figure(figsize(5,5))

for digit_num in range(0,64):

    plt.subplot(8,8,digit_num+1)

    grid_data = train.iloc[digit_num].values.reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")

    plt.xticks([]) #Remove the axes/ticks, they are not wanted.

    plt.yticks([])
pca=PCA(n_components=2) #2 for visualizing. I'm not crazy.



pca.fit(train)

transf_2=pca.transform(train)
plt.figure(figsize=(10,8))

plt.scatter(transf_2[:,0],transf_2[:,1], c=target, cmap="jet")

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.colorbar()

max_comp=([1,2,5,10,50,100,200,500,700]) #More than this may be excessive. Can be refined if this list is too coarse.

#THIS IS A LIST. IF I DON'T INCLUDE THE (), THE LOOP WILL FIT PCA WITH THE INDICES, MAX_COMP[1], MAX_COMP[2]...MAX_COMP[100]

#WHICH DOESN'T EXIST. I WANT TO PASS A LIST OF VALUES, NOT A LIST OF INDICES

j=0

variance_exp=np.zeros(len(max_comp)) #Zeros because I am doing a cumulative sum. It is done manually.

#I store the values at the i-th element. Then I sum all the values until that element.



for i in max_comp:

    pca=PCA(n_components=i)

    pca.fit(train)

    variance_exp[j]=sum(pca.explained_variance_ratio_)

    j=j+1

    
plt.figure(figsize=(10,8))

plt.plot(max_comp,variance_exp)

plt.xlabel("N principal components")

plt.ylabel("Explained variance")

plt.grid()



pca=PCA(n_components=200) #200 from previous analysis, 100 for speed.



pca.fit(train)

train_2=pca.transform(train)
y_plot=y.value_counts()

fig, ax = plt.subplots(figsize=(8,5))

g = sns.countplot(y)

X_1,X_test,y_1,y_test=train_test_split(train_2,y,test_size=0.2,random_state=13,stratify=y)

X_train,X_CV,y_train,y_CV=train_test_split(X_1,y_1,test_size=0.2,random_state=13,stratify=y_1)

#X_train.shape

RF_class=RandomForestClassifier()



param_alpha={"n_estimators":[500],

            "max_depth":[10,None], #More than 3 may overfit unnecessarily. Small gamma is also tricky

            "max_features":[10, "sqrt"]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_RF=GridSearchCV(estimator=RF_class,param_grid=param_alpha,scoring="f1_micro",cv=3,n_jobs=-1)

CV_RF=CV_RF.fit(X_train,y_train)

#I can directly predict with the best param!!

pred_RF=CV_RF.predict(X_CV)



print("The best parameter alpha for the RF classifier is:",CV_RF.best_params_)

print("The best score (f1_micro) for the RF classifier is:",CV_RF.best_score_)

y_RF=CV_RF.predict(X_test)



summary_RF=classif_summary(y_test,y_RF)

print(summary_RF)

plot_conf_matrix(y_test,y_RF)

LR_class=LogisticRegression(solver="sag",penalty="l2",multi_class="multinomial",max_iter=1000,warm_start=True)



param_alpha={"C":[1e-5,3e-5,9e-5,3e-4,9e-4]} #Based on a coarser log check I updated the values



CV_LR=GridSearchCV(estimator=LR_class,param_grid=param_alpha,scoring="f1_micro",cv=3,n_jobs=-1)

CV_LR=CV_LR.fit(X_train,y_train)

#I can directly predict with the best param!!

pred_LR=CV_LR.predict(X_CV)



print("The best parameter alpha for the LR classifier is:",CV_LR.best_params_)

print("The best score (f1_micro) for the LR classifier is:",CV_LR.best_score_)
y_LR=CV_LR.predict(X_test)



summary_LR=classif_summary(y_test,y_LR)

print(summary_LR)

plot_conf_matrix(y_test,y_LR)
KNN_class=KNeighborsClassifier()

f1_micro=[]

for i in range(1,20):

    knn=KNeighborsClassifier(n_neighbors=i,n_jobs=-1);

    knn.fit(X_train,y_train)

    pred_knn=knn.predict(X_CV)

    f1_micro.append(f1_score(y_CV,pred_knn, average="micro"))



plt.figure(figsize=(10,8))

plt.plot(range(1,20),f1_micro)

plt.title("Accuracy vs. KNeighbors")

plt.ylabel("Accuracy")

plt.xlabel("K")

plt.grid()

knn=KNeighborsClassifier(n_neighbors=1,n_jobs=-1);

knn.fit(X_train,y_train)

y_knn=knn.predict(X_test)
summary_KNN=classif_summary(y_test,y_knn)

print(summary_KNN)

plot_conf_matrix(y_test,y_knn)
print("This is very slow, as expected, uncomment if you are bored")

# SVC_class=SVC()



# param_alpha={"C":[1],

#             "kernel":["linear"]} #I think this values are enough



# CV_SVC=GridSearchCV(estimator=SVC_class,param_grid=param_alpha,scoring="f1_micro",cv=3,n_jobs=-1)

# CV_SVC=CV_SVC.fit(X_train,y_train)

# #I can directly predict with the best param!!

# pred_SVC=CV_SVC.predict(X_CV)



# print("The best parameter alpha for the SVC classifier is:",CV_SVC.best_params_)

# print("The best score (f1_micro) for the SVC classifier is:",CV_SVC.best_score_)

print("Uncomment if you uncomment the previous cell")

# y_SVC=SVC.predict(X_test)

# summary_SVC=classif_summary(y_test,y_SVC)

# print(summary_SVC)

# plot_conf_matrix(y_test,y_SVC)
#Don't forget n_init. It's important in clustering

KMeans_class=KMeans(n_clusters=10,n_init=100, n_jobs=-1, max_iter=1000)

KMeans_class.fit(X_train)

y_KMeans=KMeans_class.predict(X_test)

summary_KMeans=classif_summary(y_test,y_KMeans)

print(summary_KMeans)

plot_conf_matrix(y_test,y_KMeans)

GMM_class=GaussianMixture(n_components=10,n_init=1, max_iter=1000)

GMM_class.fit(X_train)

y_GMM=GMM_class.predict(X_test)

summary_GMM=classif_summary(y_test,y_KMeans)

print(summary_GMM)

plot_conf_matrix(y_test,y_GMM)
MLP_class=MLPClassifier(max_iter=1000)



param_alpha={"hidden_layer_sizes":[(200,5),(200,4),(300,4),(300,5)],#Don't get too crazy with this

            "activation":["relu"]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_MLP=GridSearchCV(estimator=MLP_class,param_grid=param_alpha,scoring="f1_micro",cv=3,n_jobs=-1,refit=True)



CV_MLP=CV_MLP.fit(X_train,y_train)

#I can directly predict with the best param!!

pred_MLP=CV_MLP.predict(X_CV)



print("The best parameter alpha for the LR classifier is:",CV_MLP.best_params_)

print("The best score (f1_micro) for the LR classifier is:",CV_MLP.best_score_)
y_MLP=CV_MLP.predict(X_test)



summary_MLP=classif_summary(y_test,y_MLP)

print(summary_MLP)

plot_conf_matrix(y_test,y_MLP)