# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/data.csv")
data
data.tail(30)
del data["diagnostics_Versions_PyRadiomics"]
del data["diagnostics_Versions_SimpleITK"]
del data["diagnostics_Configuration_Settings"]
del data["diagnostics_Configuration_EnabledImageTypes"]
del data["diagnostics_Image-original_Hash"]
del data["diagnostics_Image-original_Dimensionality"]
del data["diagnostics_Versions_Numpy"]
del data["diagnostics_Versions_PyWavelet"]
del data["diagnostics_Versions_Python"]
del data["Unnamed: 0"]
del data["diagnostics_Image-original_Spacing"]
del data["diagnostics_Image-original_Size"]
del data["diagnostics_Mask-original_Hash"]
del data["diagnostics_Mask-original_Spacing"]
del data["diagnostics_Mask-original_Size"]
del data["diagnostics_Mask-original_BoundingBox"]
del data["diagnostics_Mask-corrected_Size"]
del data["diagnostics_Mask-corrected_Spacing"]
del data["diagnostics_Mask-original_CenterOfMassIndex"]
del data["diagnostics_Mask-original_CenterOfMass"]
del data["diagnostics_Mask-corrected_BoundingBox"]
del data["diagnostics_Mask-corrected_CenterOfMassIndex"]

del data["diagnostics_Mask-corrected_CenterOfMass"]


data=data.drop([32],axis=0)
data.tail(30)
data.columns[data.isnull().any()]
data.isnull().sum()
sns.countplot(x="CovidORnot", data=data)
data.loc[:,'CovidORnot'].value_counts()
data.columns
data.original_shape_MinorAxisLength
data.original_shape_SurfaceArea
# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['CovidORnot'] == 1]
x = np.array(data1.loc[:,'original_shape_SurfaceArea']).reshape(-1,1)
y = np.array(data1.loc[:,'original_shape_MinorAxisLength']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('original_shape_SurfaceArea')
plt.ylabel('original_shape_MinorAxisLength')
plt.show()
# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# Predict space
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
# Fit
reg.fit(x,y)
# Predict
predicted = reg.predict(predict_space)
# R^2 
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('original_shape_SurfaceArea')
plt.ylabel('original_shape_MinorAxisLength')
plt.show()
x_data = data.drop(["CovidORnot"],axis=1)
y = data.CovidORnot
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'CovidORnot'], data.loc[:,'CovidORnot']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 2)
x,y = data.loc[:,data.columns != 'CovidORnot'], data.loc[:,'CovidORnot']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=2) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))
# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,20:30]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,30:40]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,40:50]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,50:60]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,60:70]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,70:80]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,80:90]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,90:100]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,100:110]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,110:118]],axis=1)
data = pd.melt(data,id_vars="CovidORnot",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="CovidORnot", data=data)

plt.xticks(rotation=90)
data
