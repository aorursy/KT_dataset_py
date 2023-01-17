# Load Libraries:

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import plotly.offline as pyo 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from matplotlib.colors import ListedColormap

#

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

#

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score

from sklearn import metrics

#

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor

from sklearn.decomposition import PCA

#

import warnings

warnings.filterwarnings("ignore")

#

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

data.head()
# Drop Unnecessary columns

data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

data.head()
# data shape:

row, columns = data.shape

print("Data Row:", row)

print("Data Columns:", columns)

# column names:

data.columns

# descriptions 

display(data.describe().T)

# class distribution 

print("Data is  balanced:",data.groupby('diagnosis').size())
# correlation:

corr_matrix = data.corr()

sns.clustermap(corr_matrix,annot=True,fmt=".2f",figsize=(20,14))

plt.title("Correlation Between Features")
data.info()
data_m = data[data.diagnosis == "M"]

data_b = data[data.diagnosis == "B"]
trace = [go.Bar(x=data.diagnosis.unique(), y=(len(data_m),len(data_b)),

               marker=dict(color=["blue","brown"]))]

               

layout = go.Layout(title="Count of M = malignant, B = benign ")# üst üste gelecek şekilde..

fig = go.Figure(data=trace,layout=layout)   

pyo.iplot(fig)
labels = ["M","B"]

values = [len(data_m),len(data_b)]

trace = [go.Pie(labels=labels, values=values,

               marker=dict(colors=["blue","brown"]))]

layout = go.Layout(title="Percentage of M = malignant, B = benign ")

fig = go.Figure(data=trace,layout=layout)

pyo.iplot(fig)
def dist_plot(data_feature): 

    hist_data = [data_m[data_feature], data_b[data_feature]]

    

    group_labels = ['malignant', 'benign']

    colors=["blue","brown"]

    

    fig = ff.create_distplot(hist_data, group_labels, colors = colors)

    fig['layout'].update(title = data_feature)

    return pyo.iplot(fig)
dist_plot('radius_mean')

dist_plot('texture_mean')
# Change object to integer:

data["diagnosis"] = [1 if item == "M" else 0  for item in data["diagnosis"]]
y = data["diagnosis"]

x = data.drop(["diagnosis"],axis=1)
columns = x.columns.tolist()
clf = LocalOutlierFactor()

y_pred = clf.fit_predict(x)
y_pred[:10]
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()

outlier_score["score"] = X_score
outlier_score.head()
# So make threshold: we decide about max and min of "outlier_score"

threshold = -2

filtre = outlier_score["score"] < threshold

outlier_index = outlier_score[filtre].index.tolist()
# Radius for our outliers

radius = (X_score.max()-X_score)/(X_score.max()-X_score.min())
trace0 = go.Scatter(x=x.iloc[outlier_index,0], y=x.iloc[outlier_index,1],

                   mode="markers",

                   marker=dict(size=10,color="brown"),

                   name="outliers"

                   )



trace1 = go.Scatter(x=x.iloc[:,0], y=x.iloc[:,1],

                   mode="markers",

                   marker=dict(size=50*radius,color="gold"),

                   name="real points"

                   )

 

layout = go.Layout(title="Outliers (Depends on Threshold Value)",hovermode="closest")

fig = go.Figure(data=[trace0,trace1],layout=layout)

pyo.iplot(fig)
x = x.drop(outlier_index)

y = y.drop(outlier_index)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test  = sc.transform(x_test) 
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(y_test,y_pred)

knn_acc = metrics.accuracy_score(y_test, y_pred)

print(knn_cm)

print(knn_acc)
# Tuning Decision Tree Model

n_neighbors = [5,7,9,11,13,15,17,19,21]

weights = ["uniform","distance"]

metric = ["euclidean","manhattan","minkowski"]

param_grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
knn = KNeighborsClassifier()

gs = GridSearchCV(estimator=knn,param_grid=param_grid,scoring="accuracy", cv=10)

grid_search = gs.fit(x_train,y_train)

best_score = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Score:",best_score)

print("Best Parameters:",best_parameters)
knn = KNeighborsClassifier(metric='manhattan',n_neighbors=9,weights='distance')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
knn_cm = confusion_matrix(y_test,y_pred)

knn_acc = metrics.accuracy_score(y_test, y_pred)

print(knn_cm)

print(knn_acc)
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

# Drop Unnecessary columns

data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

# Change object to integer:

data["diagnosis"] = [1 if item == "M" else 0  for item in data["diagnosis"]]

y = data["diagnosis"]

x = data.drop(["diagnosis"],axis=1)
# PCA needs scaled data

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
# Build PCA

pca = PCA(n_components = 2)

pca.fit(x_scaled)

X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2"])

pca_data["diagnosis"] = y
hue =pca_data["diagnosis"]

data = [go.Scatter(x = pca_data.p1,

                   y = pca_data.p2,

                   mode = 'markers',

                   marker=dict(

                           size=12,

                           color=hue,

                           symbol="pentagon",

                           line=dict(width=2) #çevre çizgileri

                           ))]  

                            

layout = go.Layout(title="PCA",

                   xaxis=dict(title="p1"),

                   yaxis=dict(title="p2"),

                   hovermode="closest")

fig = go.Figure(data=data,layout=layout)   

pyo.iplot(fig)                
pca_data.head()
y_pca = pca_data.diagnosis

x_pca = pca_data.drop(["diagnosis"],axis=1)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, y_pca, test_size=0.33, random_state=42)
knn_pca = KNeighborsClassifier()

knn_pca.fit(x_train_pca, y_train_pca)

y_pred_pca = knn_pca.predict(x_test_pca)
knn_cm_pca = confusion_matrix(y_test_pca,y_pred_pca)

knn_acc_pca = metrics.accuracy_score(y_test_pca, y_pred_pca)

print(knn_cm_pca)

print(knn_acc_pca)
# visualize 

cmap_light = ListedColormap(['orange',  'cornflowerblue'])

cmap_bold = ListedColormap(['darkorange', 'darkblue'])



h = .05 # step size in the mesh

X = x_pca

x_min, x_max = (X.iloc[:, 0].min() - 1), (X.iloc[:, 0].max() + 1)

y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(figsize=(20, 10), dpi=80)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



# Plot also the training points

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold,

            edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)

nca.fit(x_scaled, y)

x_nca = nca.transform(x_scaled)

nca_data = pd.DataFrame(x_nca, columns = ["p1","p2"])

nca_data["diagnosis"] = y
hue =nca_data["diagnosis"]

data_nca = [go.Scatter(x = nca_data.p1,

                   y = nca_data.p2,

                   mode = 'markers',

                   marker=dict(

                           size=7,

                           color=hue,

                           symbol="circle",

                           line=dict(width=2) 

                           ))]  

                            

layout = go.Layout(title="NCA",

                   xaxis=dict(title="p1"),

                   yaxis=dict(title="p2"),

                   hovermode="closest")

fig = go.Figure(data=data_nca,layout=layout)   

pyo.iplot(fig) 
y_nca = nca_data.diagnosis

x_nca = nca_data.drop(["diagnosis"],axis=1)
x_train_nca, x_test_nca, y_train_nca, y_test_nca = train_test_split(x_nca, y_nca, test_size=0.33, random_state=42)
knn_nca = KNeighborsClassifier()

knn_nca.fit(x_train_nca, y_train_nca)

y_pred_nca = knn_nca.predict(x_test_nca)
knn_cm_nca = confusion_matrix(y_test_nca,y_pred_nca)

knn_acc_nca = metrics.accuracy_score(y_test_nca, y_pred_nca)

print(knn_cm_nca)

print(knn_acc_nca)
# visualize 

cmap_light = ListedColormap(['orange',  'cornflowerblue'])

cmap_bold = ListedColormap(['darkorange', 'darkblue'])



h = .3 # step size in the mesh

X = x_nca

x_min, x_max = (X.iloc[:, 0].min() - 1), (X.iloc[:, 0].max() + 1)

y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



Z = knn_nca.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(figsize=(20, 10), dpi=80)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



# Plot also the training points

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold,

            edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())
models = ["Default","PCA","NCA"]

values = [0.946,0.952,0.984]
# Compare Model's Acc

f,ax = plt.subplots(figsize = (10,7))

sns.barplot(x=models, y=values,palette="viridis");

plt.title("Compare Accuracies",fontsize = 20,color='blue')

plt.xlabel('Analysis',fontsize = 15,color='blue')

plt.ylabel('Accuracies',fontsize = 15,color='blue')