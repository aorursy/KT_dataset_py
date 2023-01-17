import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#For ignoring warnings
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/Dataset_spine.csv")
data.sample(5)
data["Class_att"].unique()
data.dropna(axis = 1, inplace = True)
data.sample()
#correlation map
import seaborn as sns
f, ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()
data.Col2.plot(label = "Col2")
data.Col1.plot(label = "Col1")
plt.xlabel("index", color = "red")
plt.ylabel("values", color = "red")
plt.legend()
plt.title("Col1 and Col2")
plt.show()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'Class_att']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'Class_att'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()
sns.countplot(data = data, x = "Class_att")
plt.show()
data.loc[:,"Class_att"].value_counts()
#x, y Split and Normalization
x_data = data.iloc[:, 0:12].values
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))#Normalization
y = data.iloc[:, 12]

#Train, Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)

#Grid Search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv = 10)#cv = How many data split do we want
knn_cv.fit(x_train, y_train)
print("Best number of neighbors is {}".format(knn_cv.best_params_["n_neighbors"]))
print("Best score is {}".format(round(knn_cv.best_score_,2)))

#Grid Search Visualization
score = []
for i in range(1,50):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(x_train, y_train)
    score.append(knn2.score(x_test, y_test))
plt.plot(np.arange(1,50), score)
plt.xlabel("Number of neighbors", color = "red", fontsize = 14)
plt.ylabel("Score", color = "red", fontsize = 14)
plt.show()
#Clone our data
data_pca = data.copy()
data_pca["Class_att"] = [1 if i == "Abnormal" else 0 for i in data_pca["Class_att"]]
#Then put it in PCA
from sklearn.decomposition import PCA
pca_model = PCA(n_components = 6)
pca_model.fit(data_pca)
data_pca = pca_model.transform(data_pca)
# PCA Variance
plt.bar(range(pca_model.n_components_), pca_model.explained_variance_ratio_*100 )
plt.xlabel('PCA n_components',size=12,color='red')
plt.ylabel('Variance Ratio(%)',size=12,color='red')
plt.show()
#In this scenario, I want my data in 2 dimension.
pca_model = PCA(n_components = 2)
pca_model.fit(data_pca)
data_pca = pca_model.transform(data_pca)
print("My old shape:", data.shape)
print("My new shape:", data_pca.shape)
x_pca = data_pca[:,0]
y_pca = data_pca[:,1]
plt.scatter(x_pca, y_pca, c = ["red","green"])
plt.show()

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_normal = KNeighborsClassifier()
knn_normal.fit(x_train, y_train)
print("KNN without PCA score :", knn_normal.score(x_test, y_test))


## KNN With PCA
#Train, Test Split
x_pca = x_pca.reshape(-1,1)
y_pca = y_pca.reshape(-1,1)
y_pca_edit = [round(float(i),0) for i in y_pca]
y_pca_edit = ["Abnormal" if i>0 else "Normal" for i in y_pca_edit]
y_pca_edit = np.array(y_pca_edit)
from sklearn.model_selection import train_test_split
x_pca_train, x_pca_test, y_pca_train, y_pca_test = train_test_split(x_pca, y_pca_edit, test_size = 0.1, random_state = 1)


from sklearn.neighbors import KNeighborsClassifier
knn_pca = KNeighborsClassifier()
knn_pca.fit(x_pca_train, y_pca_train)
print("KNN with PCA score    :", knn_pca.score(x_pca_test, y_pca_test))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)# 13 ----> 2
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn_lda = KNeighborsClassifier()
knn_lda.fit(x_train_lda, y_train)

print("KNN score :", knn_normal.score(x_test, y_test))
print("KNN with PCA score    :", knn_pca.score(x_pca_test, y_pca_test))
print("KNN with LDA score    :", knn_lda.score(x_test_lda, y_test))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 100)#max_iter is for forward and backward propogation
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 1)
dtc.fit(x_train, y_train)

#K-fold Cross Validation
from sklearn.model_selection import cross_val_score
cvs_scores = cross_val_score(knn_normal, x_pca_test, y_pca_test, cv=5) #cv=5 means, we will split our data into 5 pieces
print("Cross Validation score is", cvs_scores.mean())

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, dtc.predict(x_test))
print("Confusion Matrix \n",cm)
#x, y Split and Normalization
x_data = data.iloc[:, 0:12].values
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))#Normalization
y = data.iloc[:, 12]

#Train, Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)

y_train=y_train.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)

#Naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train)

print("Navie Bayes score is", gnb.score(x_test, y_test))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=100)
#max_depth = The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure
#n_estimators = The number of trees in the forest.
rfc.fit(x_train, y_train)

print("Random Forest score is", rfc.score(x_test, y_test))
from sklearn import svm
model = svm.SVC() 
model.fit(x_train, y_train)
print("Support Vector Machine score is", model.score(x_test, y_test))