import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot, plot

import seaborn as sns

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier









from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn import metrics





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv',encoding='ISO-8859-1')

data.info()
data.describe()
eksik_degerler = data.isnull().sum()

eksik_degerler_yüzde = 100*eksik_degerler/len(data)



eksik_deger_tablosu = pd.DataFrame({"Eksik Değer Sayısı" : eksik_degerler , "Eksik Değerlerin Yüzdesi" : eksik_degerler_yüzde})



eksik_deger_tablosu
x = data["class"].unique()

y = data["class"].value_counts()



normallik = pd.DataFrame({"Normallik Durumu" : x ,"Kişi Sayısı" : y})





plt.Figure(figsize=(80,45))

sns.barplot(x = normallik["Normallik Durumu"], y =normallik["Kişi Sayısı"] ,color= "red")

plt.show()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()

x = data.pelvic_incidence.values.reshape(-1,1)

y = data.sacral_slope.values.reshape(-1,1)



linear_reg = LinearRegression()

linear_reg2 = LinearRegression()
score_list = []

for each in range(1,9):

    x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y,test_size=(each/10),random_state=42)

    linear_reg2.fit(x_train1,y_train1)



    y_head2 = linear_reg2.predict(x_test1)

    score_list.append(r2_score(y_test1,y_head2))

    

plt.plot(range(1,9) , score_list)

plt.xlabel("Test Size")

plt.ylabel("Accuracy")

plt.show()

    

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)



linear_reg.fit(x_train,y_train)



y_head = linear_reg.predict(x_test)



print("R^2 Skoru:" , r2_score(y_test,y_head))
array = np.array([20,30,40,50,60,70,80,90,100,120]).reshape(-1,1)

plt.scatter(x,y)

y_head2 = linear_reg.predict(array)

plt.plot(array, y_head2, color = "black" , linewidth = 3)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
data.head()
data["class"] = [0 if each == "Abnormal" else 1 for each in data["class"]]



y = data["class"].values

x_data = data.drop("class" , axis =1)



x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

score_list = []



for each in range(1,150):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,150),score_list)

plt.xlabel("K Values")

plt.ylabel("Accuracy")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 20)

knn.fit(x_train , y_train)

pred = knn.predict(x_test)



print("{} nn skoru: {}".format(5,knn.score(x_test,y_test)))