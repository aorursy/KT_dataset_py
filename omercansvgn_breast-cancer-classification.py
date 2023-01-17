import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis,LocalOutlierFactor

from sklearn.decomposition import PCA

from warnings import filterwarnings

filterwarnings('ignore')
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

data.head()
data = data.rename(columns= {'diagnosis':'target'})#I want to change the diagnosis variable name to target.

data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
data.head()
data['target'] = [1 if i.strip() == 'M'else 0 for i in data.target] # I need to convert the M and B in the target variable to 0 and 1.

# .strip () removes spaces in string expressions.
# How many M and how many B's are we examining them.

palette=["#FBC00E","#29B3FF"]

sns.countplot(data['target'],palette=palette);

print(data.target.value_counts())
# 1 malignant so M

# 0 benign so B

data.head()
data.info()
data.describe()

# strictly standardization process is required for this data
# Since all variables we have are numeric, we look at corr ().

ax = plt.figure(figsize=(15,8))

ax = sns.heatmap(data.corr());
# Since this is so confusing, I'll set a Threshold and cover those above it.

cor_mat = data.corr()

threshold = 0.75

filters = np.abs(cor_mat['target']) > threshold

corr_features = cor_mat.columns[filters].tolist()

ax = plt.figure(figsize=(15,8))

ax = sns.heatmap(data[corr_features].corr(),annot=True,linewidths=.3)

plt.title('Correlation Between Features w Corr Threshold 0.75');
data_melted = pd.melt(data,id_vars='target',var_name='features',value_name='value')

ax = plt.figure(figsize=(15,8))

ax = sns.boxplot(x='features',y='value',hue='target',data=data_melted,palette=palette)

plt.xticks(rotation=90);

# Because the data is not standardized here, it becomes a strange table, we will use it later.
# Pair plot is one of the most effective methods used in numerical data.

# This will not look nice either, because the data needs to be standardized.

sns.pairplot(data[corr_features],diag_kind='kde',markers='+',hue='target',palette=palette);

# But I just used corr_features for images.
y = data['target']

x = data.drop(['target'],axis=1)

columns = x.columns.tolist()
clf = LocalOutlierFactor()

y_pred = clf.fit_predict(x)

x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()

outlier_score['score'] = x_score

outlier_score.head()
plt.figure(figsize=(8,5))

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='#29B3FF',s=3,label='Data Points')

plt.legend();
radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
plt.figure(figsize=(8,5))

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='#FBC00E',s=3,label='Data Points')

plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors='#29B3FF',facecolors='none',label='Outlier Scores')

plt.legend()

plt.show()
# We are looking at contradictory observations.

threshold = -2.5

filtre = outlier_score['score'] < threshold

outlier_index = outlier_score[filtre].index.tolist()

plt.figure(figsize=(8,5))

plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color='#D55250',s=50,label='Outlier')

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='#FBC00E',s=3,label='Data Points')

plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors='#29B3FF',facecolors='none',label='Outlier Scores')

plt.legend()

plt.show()
# Drop outliers

x = x.drop(outlier_index)

y = y.drop(outlier_index).values

# All of these, there are a lot of other variables waiting for outlier observations for columns 0 and 1.
# Train Test seperation

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# Standart

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
# Let's visualize boxplot that was not visualized before.

x_train_df = pd.DataFrame(x_train,columns=columns)

x_train_df['target'] = y_train

data_melted = pd.melt(x_train_df,id_vars='target',var_name='features',value_name='value')

plt.figure(figsize=(15,8))

sns.boxplot(x='features',y='value',hue='target',data=data_melted,palette=palette)

plt.xticks(rotation=90)

plt.show()
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

score = knn.score(x_test,y_test)

print('Score:',score)

print('Confusion Matrix:',cm)

print('Basic Accuracy Score:',acc)
def knn_best_params(x_train,x_test,y_train,y_test):

    k_range = list(range(1,31))

    weight_options = ['uniform','distance']

    print()

    param_grid = dict(n_neighbors = k_range,weights=weight_options)

    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')

    grid.fit(x_train,y_train)

    print('Best Training Score {} with parameters: {}'.format(grid.best_score_,grid.best_params_))

    print()

    

    knn = KNeighborsClassifier(**grid.best_params_)

    knn.fit(x_train,y_train)

    

    y_pred_test = knn.predict(x_test)

    y_pred_train = knn.predict(x_train)

    cm_test = confusion_matrix(y_test,y_pred_test)

    cm_train = confusion_matrix(y_train,y_pred_train)

    acc_test = accuracy_score(y_test,y_pred_test)

    acc_train = accuracy_score(y_train,y_pred_train)

    print('Test Score: {},Train Score: {}'.format(acc_test,acc_train))

    print()

    print('Confusion Matrix Test: {}'.format(cm_test))

    print('Confusion Matrix Train: {}'.format(cm_train))

    

    return grid
grid = knn_best_params(x_train,x_test,y_train,y_test)
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)

pca.fit(x_scaled)

x_reduced_pca = pca.transform(x_scaled)

pca_data = pd.DataFrame(x_reduced_pca,columns=['p1','p2'])

pca_data['target'] = y

plt.figure(figsize=(8,5))

sns.scatterplot(x='p1',y='p2',hue='target',data=pca_data,palette=palette)

plt.title('PCA: p1 vs p2')

plt.show()

# We reduced 30 dimensional data to 2 dimensions with PCA.
# Now we will do a chnn using 2 dimensional data.

x_train_pca,x_test_pca,y_train_pca,y_test_pca = train_test_split(x_reduced_pca,y,test_size=0.3,random_state=42)

grid_pca = knn_best_params(x_train_pca,x_test_pca,y_train_pca,y_test_pca)
# We use a visualization to see how the split is decided.

cmap_light = ListedColormap(['#FBC00E','#29B3FF'])

cmap_bold = ListedColormap(['darkorange','darkblue'])

h = .05

X = x_reduced_pca

x_min,x_max = X[:,0].min() -1,X[:,0].max() + 1

y_min,y_max = X[:,1].min() -1 ,X[:,1].max() + 1

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),

                   np.arange(y_min,y_max,h))

Z = grid_pca.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,8))

plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,

           edgecolors='k',s=20)

plt.xlim(xx.min(),xx.max())

plt.ylim(yy.min(),yy.max())

plt.title("%i-Class Classification (k= %i,weights = '%s')"%(len(np.unique(y)),grid_pca.best_estimator_.n_neighbors,grid_pca.best_estimator_.weights))
nca = NeighborhoodComponentsAnalysis(n_components=2,random_state=42)

nca.fit(x_scaled,y)

x_reduced_nca = nca.transform(x_scaled)

nca_data = pd.DataFrame(x_reduced_nca,columns=['p1','p2'])

nca_data['target'] = y

plt.figure(figsize=(10,8))

sns.scatterplot(x='p1',y='p2',hue='target',data=nca_data,palette=palette)

plt.title('NCA : p1 vs p2')

plt.show()
x_train_nca,x_test_nca,y_train_nca,y_test_nca = train_test_split(x_reduced_nca,y,test_size=0.3,random_state=42)

grid_nca = knn_best_params(x_train_nca,x_test_nca,y_train_nca,y_test_nca)