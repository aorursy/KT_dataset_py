# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

!pip install chart_studio
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objects as go

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.describe()
data.columns
data.info()
plt.figure(figsize = (10,6))
var = data['class']
var_values = var.value_counts()
plt.bar(var_values.index, var_values)
plt.show()
def plot_hist(variable):
    
    
    var = data[variable]
    
    #visualiez
    
    plt.figure(figsize = (10,3))
    plt.hist(var,bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} Distrubiton with Histogram".format(variable))
    plt.show()
numerical_variables = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle','sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']

for each in numerical_variables:
    plot_hist(each)
data.head()
data[["pelvic_incidence","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'pelvic_incidence', ascending = False)
data[["pelvic_tilt numeric","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'pelvic_tilt numeric', ascending = False)
data[["lumbar_lordosis_angle","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'lumbar_lordosis_angle', ascending = False)
data[["sacral_slope","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'sacral_slope', ascending = False)
data[["pelvic_radius","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'pelvic_radius', ascending = False)
data[["degree_spondylolisthesis","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'degree_spondylolisthesis', ascending = False)
from collections import Counter
def detect_outliers(data,features):
    outlier_indices = []
    
    for i in features:
        
        Q1 = np.percentile(data[i],25)
        Q3 = np.percentile(data[i],75)
        
        IQR = Q3-Q1
        
        outlier_step = IQR*1.5
        
        outlier_list_cols = data[(data[i] < Q1-outlier_step) | (data[i]>Q3+outlier_step)].index
        
        outlier_indices.extend(outlier_list_cols)
        
        
    outlier_indices = Counter(outlier_indices)
    
    multiple_outliers = list(c for c,k in outlier_indices.items() if k>2)
    
    return multiple_outliers
data.loc[detect_outliers(data,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis'])]
data = data.drop(detect_outliers(data,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis']), axis=0).reset_index(drop = True)
data.loc[detect_outliers(data,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis'])]
data.columns[data.isnull().any()]
data.corr()
data = data.replace({'Abnormal':0 , 'Normal':1 })
data.head()
f,ax = plt.subplots(figsize = (13,13))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()
g = sns.FacetGrid(data, col = "class",size=5)
g.map(sns.distplot, "pelvic_radius", bins = 20)
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
sns.swarmplot(x="class", y="pelvic_incidence", data=data, ax=ax,size=9)
plt.show()
g = sns.FacetGrid(data, col = "class",size=5)
g.map(sns.distplot, "pelvic_incidence", bins = 20)
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
sns.swarmplot(x="class", y="pelvic_tilt numeric", data=data, ax=ax,size=9)
plt.show()
g = sns.FacetGrid(data, col = "class",size=5)
g.map(sns.distplot, "pelvic_tilt numeric", bins = 20)
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
sns.swarmplot(x="class", y="lumbar_lordosis_angle", data=data, ax=ax,size=9)
plt.show()
g = sns.FacetGrid(data, col = "class",size=5)
g.map(sns.distplot, "lumbar_lordosis_angle", bins = 20)
plt.show()
g = sns.lmplot(x="sacral_slope", y="class", data=data, y_jitter=.02, logistic=True, truncate=False,size=8)
g.set(xlim=(9, 82), ylim=(-0.1, 1.1))
plt.show()
g = sns.FacetGrid(data, col = "class",size=5)
g.map(sns.distplot, "sacral_slope", bins = 20)
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
sns.swarmplot(x="class", y="sacral_slope", data=data, ax=ax,size=9)
plt.show()
g = sns.lmplot(x="degree_spondylolisthesis", y="class", data=data, y_jitter=.02, logistic=True, truncate=False,size=8)
g.set(xlim=(-20, 150), ylim=(-0.1, 1.1))
plt.show()
g = sns.FacetGrid(data, col = "class",size=5)
g.map(sns.distplot, "degree_spondylolisthesis", bins = 20)
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
sns.swarmplot(x="class", y="degree_spondylolisthesis", data=data, ax=ax,size=9)
plt.show()
data1 = data[data["class"] == 1] #Creating data contains only normal people
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1) #pelvic_incidence is our feature
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)  #Sacral slope is our target

#Visualize
plt.figure(figsize=(10,10))
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
predict_space = np.linspace(min(x), max(x),num=208).reshape(-1,1)
#Fit
lr.fit(x,y)
#Predict
predicted = lr.predict(predict_space)
#Visualize
plt.subplots(figsize=(9,7))
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
#R^2 Score
print('R^2 score: ',lr.score(x, y))

results = {}
data.head()
y = data['class'].values
x_data = data.drop(["class"],axis=1)
#Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# KNN MODEL
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
results["KNN: "] = knn.score(x_test, y_test)
score_list = []
for each in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test, y_test))

#plotting    
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1,30),y=score_list))
fig.update_layout(title = "KNN Classification Scores by Number of Neighbours", xaxis_title='Number of Neighbours',yaxis_title='Score')
fig.show()
    
    
print("Accuracy of KNN Algorithm: {:.2f} % ".format(100*knn.score(x_test,y_test)))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Test Accuracy {:.2f} %".format(100*lr.score(x_test,y_test)))

results["LogisticRegression: "] = lr.score(x_test,y_test)

from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train, y_train)
results["SuperVectorMachine:"] = svm.score(x_test,y_test)
print("Accuracy of SVM Algorithm: {:.2f} %".format(100*svm.score(x_test,y_test)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

results["NaiveBayes: "] = nb.score(x_test,y_test)
print("Accuracy of Naive Bayes: {:.2f} %".format(100*nb.score(x_test,y_test)))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)

results["DecissionTree: "] = dtc.score(x_test,y_test)
print("Accuracy of Decision Tree: {:.2f} %".format(100*dtc.score(x_test,y_test)))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 17, random_state = 1)

rfc.fit(x_train,y_train)

results["RandomForest: "] = rfc.score(x_test,y_test)
print("Accuracy of Random Forest: {:.2f} %".format(100*rfc.score(x_test,y_test)))
score_list = []
for each in np.arange(1,60):
    rfc2 = RandomForestClassifier(n_estimators = each, random_state = 1)

    rfc2.fit(x_train,y_train)
    score_list.append(rfc2.score(x_test, y_test))

#plotting    
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1,60),y=score_list))
fig.update_layout(title = "Random Forest Classification Scores by Number of Estimators", xaxis_title='Number of Estimators',yaxis_title='Score')
fig.show()
    
for a,b in results.items():
    print("Score of {}: {:.4f} % ".format(a,100*b))
    
print("*"*50)
print("Best Score => {} : {} ".format(max(results, key=results.get),max(results.values())))
print("*"*50)
y_pred = knn.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)
#Visualize of Confussion Matrix
f,ax = plt.subplots(figsize=(12,10))
sns.set(font_scale=1.5)
ax =sns.heatmap(cm,annot=True,fmt='.0f')
ax.tick_params(labelsize=25)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
f,ax = plt.subplots(figsize =(9,4))
x=data["pelvic_radius"]
y=data["degree_spondylolisthesis"]
plt.scatter(x,y)
plt.show()
data2 = data.loc[:,["pelvic_radius","degree_spondylolisthesis"]]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)

cdict = {0:'green',1:'purple'}
arrayim = []
for i in labels:
    arrayim.append(cdict[i])

f,ax = plt.subplots(figsize =(9,4))
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],color=arrayim)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
inertia_list = np.empty(8)

for i in range(1,8):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
    
    
plt.plot(range(0,8),inertia_list, '-o')
plt.xlabel("Number of Cluster")
plt.ylabel("Inertia")
plt.show()



