import pandas as pd
import numpy as np
import seaborn as sns
bcan=pd.read_csv(r"../input/data.csv")
bcan.head()
slc=bcan.iloc[:,2:32]
print(type(slc))
slc.head()
bcr=bcan.diagnosis
bcr.head()
bcd=pd.get_dummies(bcr)
bcd.head()
bcm=bcd.M
bcm.head()
#identifying collinearity
import pandas as pd
import numpy as np
bccor=np.corrcoef(slc.T)
type(bccor)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(8,6))
sns.heatmap(bccor, vmin=0.85, vmax=1,\
            cmap=plt.cm.Spectral_r)
#PCA for dimensional reduction
#PCA on standard/scaled data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
slc_scaled=scaler.fit_transform(slc)
pca=PCA(n_components=.95)
x_pca=pca.fit_transform(slc_scaled)
print(x_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
loadings=pca.components_
#PCA after removing collinear variables
slc_scaled=pd.DataFrame(slc_scaled)
xx=slc_scaled.drop(slc_scaled.columns[[2,3,22,12,13]],axis=1)
pca=PCA(n_components=.95)
x_pca=pca.fit_transform(xx)
print(x_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
loadings=pca.components_
#PCA on raw data
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
x_pca=pca.fit_transform(slc)
print(x_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
loadings=pca.components_
#PCA Variance plots
import matplotlib.pyplot as plt
n=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
index=np.arange(10)

var_1=np.array([9.82044672e-01, 1.61764899e-02, 1.55751075e-03, 1.20931964e-04,
 8.82724536e-05, 6.64883951e-06, 4.01713682e-06, 8.22017197e-07,
 3.44135279e-07 ,1.86018721e-07])
var_2=np.array([0.44272026,0.18971182, 0.09393163,0.06602135, 0.05495768, 0.04024522
 ,0.02250734 ,0.01588724 ,0.01389649 ,0.01168978])
var_3=np.array([0.42452421 ,0.17289338 ,0.09930622, 0.07549947,0.06283005 ,0.04819281
 ,0.02115356 ,0.01866259 ,0.01450551 ,0.01184149])


var=np.vstack([var_1,var_2,var_3]).T
df=pd.DataFrame(var,index=n,columns=['raw','scaled','scale_non_col'])
df
#plotting variance
%matplotlib inline
fig, ax=plt.subplots(1,3, sharey=True, figsize=(18,4))
r1=ax[0].bar(index,df['raw'],width=0.5, align='center')
ax[0].set_title('PCA on raw data',fontsize=18)

r1=ax[1].bar(index,df['scaled'],width=0.5, align='center')
ax[1].set_title('PCA on scaled data',fontsize=18)


r1=ax[2].bar(index,df['scale_non_col'],width=0.5, align='center')
ax[2].set_title('PCA on non collinear scaled data',fontsize=18)
#deciding on the best model kNN, logistics,support vector classifier(svc)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
bcan=pd.read_csv(r"../input/data.csv")
bcan.head()
pcan=bcan.iloc[:,2:32]
bcan.iloc[:,2:32].describe()
bd=pd.get_dummies(bcan.diagnosis)
ym=bd.M
ym.head()
#standardizing and PCA
scal=StandardScaler()
pcan_scaled=scal.fit_transform(pcan)
pcan_scaled=pd.DataFrame(pcan_scaled)
ppcan=pcan_scaled.drop(pcan_scaled.columns[[2,3,22,23,12,13]], axis=1)
pca=PCA(n_components=0.95)
pcan_pca=pca.fit_transform(ppcan)
print(pcan_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
n=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','diagnosis']
bcd=bcan.iloc[:,1:2]
xbcan=pd.DataFrame(np.hstack([pcan_pca,bcd.values]),columns=n)
sns.lmplot("PC1","PC2",hue="diagnosis",data=xbcan,fit_reg=False,markers=["o","*"],palette="Set1")
plt.show()

#training and test
from sklearn.model_selection import train_test_split
X=(xbcan.iloc[:,0:11]).values
X_train,X_test,y_train,y_test=train_test_split(X,ym,test_size=0.25,random_state=0)

#kNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn=KNeighborsClassifier()
k_range=list(range(1,50))
k_scores=[]
for k in k_range:
        knn=KNeighborsClassifier(n_neighbors=k)
        scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='recall')
        k_scores.append(scores.mean())
print(np.round(k_scores,3))
from matplotlib import pyplot as plt
plt.plot(k_range,k_scores,color='blue')
plt.xlabel('k values')
plt.ylabel('Recall')
plt.show()
#use grid search for accuracy, tuning hyper parameters for accuracy

# fitting the optimal model (i.e. knn with k=5 based upon recall score) onto the training data
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# for display purposes, we fit the model on the first two components i.e. PC1, and PC2
knn.fit(X_train[:,0:2], y_train)
# Plotting the decision boundary for all data (both train and test)
# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#AAFFAA','#FFAAAA'])
cmap_bold = ListedColormap(['#0000FF','#FF0000'])
# creating a meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h=0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
xy_mesh=np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(xy_mesh)
%matplotlib inline
Z = Z.reshape(xx.shape)
#print(Z)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
ax=plt.scatter(X[:, 0], X[:, 1], c=ym, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max());plt.ylim(yy.min(), yy.max())
plt.xlabel('PC1');plt.ylabel('PC2')
plt.title('KNN')
plt.show()
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# KNN
# fitting the knn model on the training data
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn =knn.predict(X_test)
# computing and plotting confusion matrix
c_m = confusion_matrix(y_test, y_pred_knn)
print('KNN:\n confusion matrix\n', c_m,'\n\n')
ax=plt.matshow(c_m,cmap=plt.cm.Reds)
print('Confusion matrix plot of KNN classifier')
plt.colorbar(ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# classification report
print('\n Classification report \n',classification_report(y_test, y_pred_knn))
print ('#############################################################################')
