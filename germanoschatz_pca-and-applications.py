import numpy as np
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine

wine = load_wine()
#dict_keys(['data', 'target', 'feature_names', 'DESCR'])
dataset = pd.DataFrame(wine.data, columns=wine.feature_names)
dataset['Type'] = wine.target
print(dataset.head())
#Design the target and feature variables
X = dataset.drop(['Type'], axis=1)
Y = dataset['Type']

#split the dataset 90% and 10%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
print("X_train_dimensions:",X_train.shape)
print("X_test_dimensions:",X_test.shape)
print("Y_train_dimensions:",Y_train.shape)
print("Y_test_dimensions:",Y_test.shape)
#SUPPORT VECTOR CLASSIFIER
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# model evaluation for testing set
y_test_predict = model.predict(X_test)

cm=confusion_matrix(Y_test, y_test_predict)
accuracy = np.trace(cm)/np.sum(cm)
print("Accuracy:",accuracy)
#Z SCORE SCALE
# WE CALCULATE MEAN AND STANDAR DEVIATION 
#OF TRAIN SET AND APPLY THE Z SCORE SCALE TO TEST SET
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#building the model using sklearn
pca = PCA(n_components=13)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#---suport vector clasifier with PCA---------
model = SVC(kernel='linear')
model.fit(X_train[:,0:6], Y_train)

# model evaluation for testing set
y_test_predict = model.predict(X_test[:,0:6])

cm=confusion_matrix(Y_test, y_test_predict)
accuracy = np.trace(cm)/np.sum(cm)
print("Accuracy_SVC_WIH_PCA:",accuracy)
#--------------explained variance------------------
explained_variance = pca.explained_variance_ratio_
plt.plot(np.cumsum(explained_variance))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig("D:\website\PCA\Variance.png", dpi=150)
#-----Boundaries PLOT with 2 pca without train test split------
from mlxtend.plotting import plot_decision_regions
sc2 = StandardScaler()
X = sc2.fit_transform(X)
pca = PCA(n_components=13)
X = pd.DataFrame(pca.fit_transform(X))
model = SVC(kernel='linear').fit(X.iloc[:,[0,2]],Y)
fig=plt.figure()
plot_decision_regions(np.array(X.iloc[:,[0,2]]),np.array(Y), clf=model,legend=2)

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('SVM on Wine')
plt.show()
#fig.savefig("\PCA\SVC_IRIS.png",dpi=120)
# ---------------PLOT PCA obsjected oriented & using seaborn-------------
principalDf = pd.DataFrame(data = X_train[:,0:2]
             , columns = ['PCA1', 'PCA2'])
Df_WITH_TARGET = pd.concat([principalDf, Y_train.reset_index()], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = Df_WITH_TARGET['Type'] == target
    ax.scatter(Df_WITH_TARGET.loc[indicesToKeep, 'PCA1']
               , Df_WITH_TARGET.loc[indicesToKeep, 'PCA2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#---------------------------USING SEABORN---------------------------------
plot_sc = sns.scatterplot(x="PCA1", y="PCA2", hue="Type",
                     data=Df_WITH_TARGET,palette='deep')
#plt.savefig("\Wine_2pca.png", dpi=150)