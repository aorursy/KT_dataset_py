# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #Linear Regression 
from sklearn.neighbors import KNeighborsClassifier #KNN Algorithm
from sklearn.model_selection import train_test_split #Trian and Test Split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe1 = pd.read_csv('../input/column_2C_weka.csv') #read to file
print(dataframe1.columns) 
print(dataframe1.info())
newBioDataFrame = dataframe1.loc[:,["sacral_slope","pelvic_radius"]]
import missingno as msno
msno.matrix(newBioDataFrame)
plt.show()
msno.bar(newBioDataFrame)
plt.show()
dataframe1.head()
dataframe1.tail()
dataframe1.describe()
trace1 = go.Box(
    y = dataframe1.sacral_slope,
    name = "sacral_slope",
    marker = dict(color = "red")
)
trace2 = go.Box(
    y = dataframe1.pelvic_radius,
    name = "pelvic_radius",
    marker = dict(color = "blue")
)
concatTrace = [trace1,trace2]
iplot(concatTrace)
dataFilter = dataframe1[dataframe1['class'] == 'Abnormal']
linear_regression = LinearRegression()
x = dataFilter.pelvic_incidence.values.reshape(-1,1)
y = dataFilter.sacral_slope.values.reshape(-1,1)
linear_regression.fit(x,y)

y_head = linear_regression.predict(x)

plt.figure(figsize=[15,15])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')

plt.plot(x,y_head,color="green",linewidth=2)
plt.show()

from sklearn.metrics import r2_score
print('R^2 score: ',r2_score(y,y_head))
Abnormal = dataframe1[dataframe1["class"] == "Abnormal"]
Normal = dataframe1[dataframe1["class"] == "Normal"]

plt.figure(figsize=(15,15))
plt.scatter(Abnormal.pelvic_radius,Abnormal.lumbar_lordosis_angle,color="blue",label="pelvic_radius")
plt.scatter(Normal.pelvic_radius,Normal.lumbar_lordosis_angle,color="lime",label="lumbar_lordosis_angle")
plt.legend()
plt.xlabel("pelvic_radius")
plt.ylabel("lumbar_lordosis_angle")
plt.show()
dataframe1["class"] = [1 if(each == "Abnormal") else 0 for each in dataframe1["class"]]
y = dataframe1["class"].values
x_data = dataframe1.drop(["class"],axis=1)

#Normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#Train and Test values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
score_list = []
for eachs in range(1,15):
    knnAlgorithm1 = KNeighborsClassifier(n_neighbors = eachs)
    knnAlgorithm1.fit(x_train,y_train) #Modeli eğitiyorum
    score_list.append(knnAlgorithm1.score(x_test,y_test))
plt.figure(figsize=(15,15))
plt.plot(range(1,15),score_list)
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.show()
import warnings
warnings.filterwarnings("ignore")

section = {"n_neighbors":np.arange(1,50)}
knnAlgorithm2 = KNeighborsClassifier()
knnAlgorithm_cv = GridSearchCV(knnAlgorithm2,section,cv = 10)
knnAlgorithm_cv.fit(x_train,y_train)
print("Best K value: ", knnAlgorithm_cv.best_params_)
print("And the best guess score: ",knnAlgorithm_cv.best_score_)
knnAlgorithm = KNeighborsClassifier(n_neighbors=13)
knnAlgorithm.fit(x_train,y_train)
predict = knnAlgorithm.predict(x_test)
print("{} nn Score {}: ".format(13,knnAlgorithm.score(x_test,y_test)))
truePredict = 0
falsePredict = 0
for p in range(len(predict)):
    for y in range(p,len(y_test)):
        if (predict[p] == y_test[y]):
            truePredict = truePredict +1
            break
        else:
            falsePredict = falsePredict +1
            break
print("True Predict: ",truePredict)
print("False Predict",falsePredict)
print("-------------------------------------------------------------------------------------")
print("Predict: ",predict)
print("-------------------------------------------------------------------------------------")
print("y_test: ",y_test)
x_Axis = ["True","False"]
y_Axis = [truePredict,falsePredict]

plt.figure(figsize=(15,15))
sns.barplot(x=x_Axis,y=y_Axis,palette = sns.cubehelix_palette(len(x_Axis)))
plt.xlabel("Disease Class")
plt.ylabel("Frequency")
plt.title("Abnormal and normal type diseases")
plt.show()
conf_matrix = confusion_matrix(y_test,predict)
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(conf_matrix,annot=True,linewidths=0.5,linecolor="white",fmt=".0f",ax=ax)
plt.xlabel("y_test")
plt.ylabel("predict")
plt.show()