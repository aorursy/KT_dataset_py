import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_2C = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

df_3C = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')

compareScore3 =[]

compareScore2 =[]
print("column_2C_weka")

color_list = ['red' if i == 'Abnormal' else 'green' for i in df_2C.loc[:,'class']]

pd.plotting.scatter_matrix(df_2C.loc[:,df_2C.columns!='class'],

                                        c = color_list,

                                        figsize=[15,15],

                                        diagonal = 'hist',

                                        alpha = 0.5,

                                        s = 200,

                                        marker = '*',

                                        edgecolor = 'black')



plt.show()



print("column_3c")

color_list2 = ['red' if i != 'Normal' else 'green' for i in df_3C.loc[:,'class']]

pd.plotting.scatter_matrix(df_3C.loc[:,df_3C.columns!='class'],

                                      c = color_list,

                                      figsize = [15,15],

                                      diagonal = 'hist',

                                      alpha = 0.5,

                                      s = 200,

                                      marker = '*',

                                      edgecolor = 'black')





plt.show()
sns.countplot(x = "class",data = df_2C)

df_2C.loc[:,'class'].value_counts()
sns.countplot(x = "class",data = df_3C)

df_3C.loc[:,'class'].value_counts()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x_2C,y_2C = df_2C.loc[:,df_2C.columns != 'class'], df_2C.loc[:,df_2C.columns == 'class']

knn.fit(x_2C,y_2C)

prediction = knn.predict(x_2C)

print("KNN score for 1. Dataset: ",knn.score(x_2C,y_2C))



# confusion matrix

y_pred = knn.predict(x_2C)

y_true = y_2C



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_true,y_pred)

print("Confusion matrix for 1. dataset")

import matplotlib.pyplot as plt

import seaborn as sns

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn2 = KNeighborsClassifier(n_neighbors = 3)

x_3C, y_3C = df_3C.loc[:,df_3C.columns != 'class'], df_3C.loc[:,df_3C.columns == 'class']

knn2.fit(x_3C,y_3C)

prediction = knn2.predict(x_3C)

print("KNN score for 2. dataset: ",knn2.score(x_3C,y_3C))







y_true = y_3C

y_pred = knn2.predict(x_3C)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



print("Confusion matrix for 2. dataset")

import seaborn as sns

import matplotlib.pyplot as plt

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_true")

plt.ylabel("y_predict")

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

x2_train, x2_test, y2_train, y2_test = train_test_split(x_2C,y_2C,test_size=0.3,random_state=42)



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x2_train,y2_train)

print("{} nn score {}: ".format(3,knn.score(x2_test,y2_test)))

knnScore2 = knn.score(x2_test, y2_test) * 100

compareScore2.append(knnScore2)



# confusion-matrix

y_true = y2_test

y_pred = knn.predict(x2_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt





f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_true")

plt.ylabel("y_pred")

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

x3_train, x3_test, y3_train, y3_test = train_test_split(x_3C,y_3C,test_size = 0.3, random_state = 42)

knn2 = KNeighborsClassifier(n_neighbors = 3)

knn2.fit(x3_train,y3_train)

print("{} nn score: {}".format(3,knn2.score(x3_test,y3_test)))

knnScore3 = knn2.score(x3_test, y3_test) * 100

compareScore3.append(knnScore3)

# confusion-matrix



y_true = y3_test

y_pred = knn2.predict(x3_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel('y_true')

plt.ylabel('y_pred')

plt.show()
# evelation regresion model performance with R-square

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')



data = df[df['class'] == 'Normal']

x = np.array(data.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data.loc[:,'sacral_slope']).reshape(-1,1)



# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



predict_space = np.linspace(min(x),max(x)).reshape(-1,1)

lr.fit(x,y)

predicted = lr.predict(predict_space)



print('R^2 score: ',lr.score(x, y))



# Plot regression line and scatter

plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# SVM(Support Vector Machine) for 1. dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')



df_abnormal = df[df['class'] == 'Abnormal']

df_normal = df[df['class'] == 'Normal']



sns.countplot(x = 'class', data = df)

df.loc[:,'class'].value_counts()
plt.scatter(df_abnormal.sacral_slope,df_abnormal.pelvic_radius,label = 'abnormal',color="red")

plt.scatter(df_normal.sacral_slope,df_normal.pelvic_radius,label = 'normal',color="green")

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')

plt.show()
y2 = df['class'].values.reshape(-1,1)

x2_data = df.drop('class',axis = 1)

x2 = ((x2_data - np.min(x2_data)) / (np.max(x2_data) - np.min(x2_data))).values

from sklearn.model_selection import train_test_split

x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,test_size=0.3,random_state=42)



from sklearn.svm import SVC



svm = SVC(random_state=1)

svm.fit(x2_train,y2_train)



print("print accuracy of svm algo for 1. dataset: ",svm.score(x2_test,y2_test))

svmScore2 = svm.score(x2_test, y2_test) * 100

compareScore2.append(svmScore2)

#confusion_matrix



y_true = y2_test

y_pred = svm.predict(x2_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



print("Confusion matrix for 1. dataset")

      

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel('y_true')

plt.ylabel('y_pred')

plt.show()
# SVM(Support Vector Machine) for 2. dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')



df_abnormal = df[df['class'] == 'Abnormal']

df_normal = df[df['class'] == 'Normal']



sns.countplot(x = 'class', data = df)

df.loc[:,'class'].value_counts()
plt.scatter(df_abnormal.sacral_slope,df_abnormal.pelvic_radius,label = 'abnormal',color="red")

plt.scatter(df_normal.sacral_slope,df_normal.pelvic_radius,label = 'normal',color="green")

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')

plt.show()
y3 = df['class'].values.reshape(-1,1)

x3_data = df.drop('class',axis = 1)

x3 = ((x3_data - np.min(x3_data)) / (np.max(x3_data) - np.min(x3_data))).values

from sklearn.model_selection import train_test_split

x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3,test_size=0.3,random_state=42)



from sklearn.svm import SVC



svm2 = SVC(random_state=1)

svm2.fit(x3_train,y3_train)



print("print accuracy of svm algo for 2. dataset: ",svm2.score(x3_test,y3_test))

svmScore3 = svm2.score(x3_test, y3_test) * 100

compareScore3.append(svmScore3)

#confusion_matrix



y_true = y3_test

y_pred = svm2.predict(x3_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



print("Confusion matrix for 2. dataset")

      

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel('y_true')

plt.ylabel('y_pred')

plt.show()
# Naive Bayes for 1. dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')



df_Abnormal = df[df['class'] == 'Abnormal']

df_Normal = df[df['class'] == 'Normal']



plt.scatter(df_Abnormal.sacral_slope,df_Abnormal.pelvic_radius,label="Abnormal",color = "red")

plt.scatter(df_Normal.sacral_slope,df_Normal.pelvic_radius,label="Normal",color = "green")

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')

plt.show()
y2 = df['class'].values

x2_data = df.drop('class',axis = 1)



x2 = ((x2_data - np.min(x2_data)) / (np.max(x2_data) - np.min(x2_data))).values



from sklearn.model_selection import train_test_split



x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,test_size=0.3,random_state=42)



from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x2_train,y2_train)



print("accuracy on naive bayes: ",nb.score(x2_test, y2_test))

nbScore2 = nb.score(x2_test, y2_test) * 100

compareScore2.append(nbScore2)

#confusion_matrix



y_true = y2_test

y_pred = nb.predict(x2_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



print("Confusion matrix for 1. dataset")

      

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel('y_true')

plt.ylabel('y_pred')

plt.show()
# Naive Bayes for 2. dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')



df_Abnormal = df[df['class'] == 'Abnormal']

df_Normal = df[df['class'] == 'Normal']



plt.scatter(df_Abnormal.sacral_slope,df_Abnormal.pelvic_radius,label="Abnormal",color = "red")

plt.scatter(df_Normal.sacral_slope,df_Normal.pelvic_radius,label="Normal",color = "green")

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')

plt.show()
y3 = df['class'].values

x3_data = df.drop('class',axis = 1)



x3 = ((x3_data - np.min(x3_data)) / (np.max(x3_data) - np.min(x3_data))).values



from sklearn.model_selection import train_test_split



x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3,test_size=0.3,random_state=42)



from sklearn.naive_bayes import GaussianNB



nb2 = GaussianNB()

nb2.fit(x3_train,y3_train)



print("accuracy on naive bayes: ",nb2.score(x3_test, y3_test))

nbScore3 = nb2.score(x3_test, y3_test) * 100

compareScore3.append(nbScore3)

#confusion_matrix



y_true = y3_test

y_pred = nb2.predict(x3_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



print("Confusion matrix for 2. dataset")

      

import seaborn as sns

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel('y_true')

plt.ylabel('y_pred')

plt.show()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



algoList = ["KNN", "SVM", "NaiveBayes"]

comparison2 = {"Models" : algoList, "Accuracy" : compareScore2}

dfComparison = pd.DataFrame(comparison2)



newIndex = (dfComparison.Accuracy.sort_values(ascending = False)).index.values

sorted_dfComparison = dfComparison.reindex(newIndex)





data = [go.Bar(

               x = sorted_dfComparison.Models,

               y = sorted_dfComparison.Accuracy,

               name = "Scores of Models",

               marker = dict(color = "rgba(116,173,209,0.8)",

                             line=dict(color='rgb(0,0,0)',width=1.0)))]



layout = go.Layout(xaxis= dict(title= 'Models',ticklen= 5,zeroline= False))



fig = go.Figure(data = data, layout = layout)

print("Results for 1. dataset")

iplot(fig)
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



algoList = ["KNN", "SVM", "NaiveBayes"]

comparison3 = {"Models" : algoList, "Accuracy" : compareScore3}

dfComparison = pd.DataFrame(comparison3)



newIndex = (dfComparison.Accuracy.sort_values(ascending = False)).index.values

sorted_dfComparison = dfComparison.reindex(newIndex)





data = [go.Bar(

               x = sorted_dfComparison.Models,

               y = sorted_dfComparison.Accuracy,

               name = "Scores of Models",

               marker = dict(color = "rgba(116,173,209,0.8)",

                             line=dict(color='rgb(0,0,0)',width=1.0)))]



layout = go.Layout(xaxis= dict(title= 'Models',ticklen= 5,zeroline= False))



fig = go.Figure(data = data, layout = layout)

print("Results for 2. dataset")

iplot(fig)