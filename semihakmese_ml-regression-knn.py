# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



from sklearn.linear_model import LinearRegression #Linear Regresyon Lib

from sklearn.preprocessing import PolynomialFeatures #Polynomial Linear Reg Lib

from sklearn.tree import DecisionTreeRegressor #Decision Tree Reg Lib

from sklearn.ensemble import RandomForestRegressor #Random Forest Reg Lib

from sklearn.metrics import r2_score #Performance Analysis with R_Square



data3c = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv") 

data2c = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv") 
data3c.head(10)
data2c.head(10)
data2c.columns
data2c["class"].value_counts()
data3c["class"].value_counts()
data2c.info()

data3c.info()
#Visualize Categorical Variable - Burada sadece Class bu tanıma uyuyor 

plt.figure(figsize = (9,5))

var = data3c["class"]

var_values = var.value_counts()

plt.bar(var_values.index, var_values)

plt.show()
data3c.info()
#Visualize Numerical Variable 

def histogram_plot(variable):

    var= data3c[variable]

    

    plt.figure(figsize =(7,3))

    plt.hist(var,bins=50,color ="grey")

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distrubiton with Histogram".format(variable))



numericVar = ["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope",

             "pelvic_radius","degree_spondylolisthesis"]    

for each in numericVar:

    histogram_plot(each)





#Pelvic Incıdence vs Class

data3c[["pelvic_incidence","class"]].groupby(["class"], as_index = False).mean().sort_values(by="pelvic_incidence",ascending = False)
#pelvic_tilt vs Class

data3c[["pelvic_tilt","class"]].groupby(["class"], as_index = False).mean().sort_values(by = "pelvic_tilt", ascending = False)
#lumbar lordosis angle vs class

data3c[["lumbar_lordosis_angle","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'lumbar_lordosis_angle', ascending = False)
#sacral_slope vs class

data3c[["sacral_slope","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'sacral_slope', ascending = False)
#Pelvic Radius vs Class

data3c[["pelvic_radius","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'pelvic_radius', ascending = False)
#degree spondyylolisthesis vs class

data3c[["degree_spondylolisthesis","class"]].groupby(["class"], as_index = False).mean().sort_values(by = 'degree_spondylolisthesis', ascending = False)
def detect_outliers(df,features):

    outlier_indices=[] 

    

    for c in features:

        Q1= np.percentile(df[c],25)

        Q2= np.percentile(df[c],50)

        Q3= np.percentile(df[c],75)

        

        IQR = Q3 - Q1

        outlier_step = IQR * 1.5

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        

        outlier_indices.extend(outlier_list_col)

   

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i,v in outlier_indices.items() if v>2)

    

    return multiple_outliers 



        

                                  
data3c.loc[detect_outliers(data3c,["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"])]
data3c.columns[data3c.isnull().any()] #Herhangi null değeri barındıran bir column yok
#Oncelikle HeatMap ile Korelasyon Analizi Yapalım 

list1 = data3c.iloc[:,:-1] #class hariç tüm sütunları aldık Çünkü class object olacagundan heatmap de gözükmez

sns.heatmap(list1.corr(), annot = True, fmt = ".3f")

#Class ı da görmek istersek 

#data3c = data3c.replace({'Spondylolisthesis':0 , 'Normal':1 , 'Hernia':2}) #şeklide de yapabilirdik
g = sns.factorplot(x="lumbar_lordosis_angle",y="class", kind = "bar", data=data3c, size =6)

g.set_ylabels("Class Type",size = 14)

plt.show()  

print("Ortalama Lumbar Lordoz Açısı : ",data3c.lumbar_lordosis_angle.mean()) #Ortalama lumbar lordosis angle'da yazdırıldı

#Eger sekildeki gibi Object i Y'ye yazarsak yan bar plot cıkar
#Pelvic Radius ve Class ı karsılastıralım 

g = sns.factorplot(x="class",y="pelvic_radius", kind= "bar",data=data3c)
#Facetgrid Pelvic Tilt ile Class ı görelim 

g= sns.FacetGrid(data3c, col="class",size =4)

g.map(sns.distplot, "pelvic_tilt" , bins=30, color = "red")

plt.show()
#Birde swarm plot ile sacral_slope ve Class karsılastırması yapalım 

fig, ax = plt.subplots(figsize=(9,5))

sns.swarmplot(x="class",y="sacral_slope", data=data3c, ax=ax, size=8)

plt.show()
#Şimdi de inset plot ile Pelvic Tilt- Class ve Pelvic Radius - Class ilişkilerini görelim 

trace1 = go.Scatter(

    y=data3c["class"],

    x=data3c["pelvic_tilt"],

    name ="Pelvic Tilt - Class",

    marker=dict(color="rgba(120,158,245,0.75)")

)

trace2 = go.Scatter(

    y=data3c["class"],

    x=data3c["pelvic_radius"],

    xaxis ="x2",

    yaxis ="y2",

    name ="Pelvic Radius - Class",

    marker =dict(color="rgba(0,55,100,0.65)")

)

data = [trace1,trace2]

layout = go.Layout(xaxis = dict(domain=[0.6,0.95],anchor="x2"),

                   yaxis = dict(domain=[0.6,0.95],anchor="y2")

)

fig= go.Figure(data=data, layout=layout)

iplot(fig)
data3c.columns
dataspondy = data3c[data3c["class"]== "Spondylolisthesis"]

datanorm = data3c[data3c["class"]== "Normal"]

datahernia = data3c[data3c["class"]== "Hernia"] 
#Burada Spondylolisthesis sınıfını seçme sebebim Outlier değerlerine sahip olması outlierları atarak ve atmadan sonucları karsılastırma fırsatımız olacak



#Bu kısımda pelvic_incidence ve sacral_slope arasında linear reg yapalım çünkü bu verilerin ortalama değerleri birbirine yakın ve nokta dağılımı belli bir nokta etrafında dağılmıs



# x= dataspondy["pelvic_incidence"].values.reshape(-1,1) #Burada .values metodu ile np array' a cevirdik 

# y= np.array(dataspondy.loc[:,"degree_spondylolisthesis"]).reshape(-1,1) #Bu sekilde .values yerine np.array metodu ile cevirdik

x=np.array(dataspondy.loc[:,"pelvic_incidence"]).reshape(-1,1)

y=np.array(dataspondy.loc[:,"sacral_slope"]).reshape(-1,1)



#Şimdi bu modelde çizdirme işlemini yapalım 

plt.scatter(x,y,color="red")

plt.show()



#Linear Regression >>>>> y = b0 + b1*x

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()



#Bir de fit edelim 

linear_reg.fit(x,y) 



#Şimdi b0 ı bulalım 

#b0 = linear_reg.predict([[0]]) şeklinde de yapabilrdik çünkü y eksenini kestiği noktada x=0 dolayısıyla b1 = 0 ve y= b0 sonucuna ulasırdık



b0 = linear_reg.intercept_  

b1 = linear_reg.coef_ 

print("b0 = {}, b1 = {} \n".format(b0,b1)) #Artık b0 ve b1 değerlerini biliyoruz 

x_ = np.linspace(min(x), max(x)).reshape(-1,1)

y_head = linear_reg.predict(x_)

#Görsellestirme

plt.figure(figsize =(10,8))

plt.scatter(x,y,color ="red")

plt.plot(x_,y_head,color ="blue")

plt.show()
#Bir de performans analizi için R_Square kullanalım

from sklearn.metrics import r2_score

print("R^2 Score : ",linear_reg.score(x,y)) 

#Buradaki hata sample sayısının eşit olmaması dolayısıyla eşitleyince de sonuc negatif cıkıyor
#Polynomial Linear Regression  >>>>> y = b0 + b1*x + b2*x^2 + .. + bn*x^n 



from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree = 2)

#Bir de fit edelim 

x_polynomial = polynomial_reg.fit_transform(x)

linear_reg2 =LinearRegression()

linear_reg2.fit(x_polynomial,y)



#Görsellestirme 

y_head2 = linear_reg2.predict(x_polynomial)

plt.scatter(x,y)

plt.plot(x,y_head2,color ="blue")

plt.show()



from sklearn.metrics import r2_score

print(linear_reg2.score(y,y_head2))

#R Square hesabında hata olustu 
#Decision Tree 

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)

xt_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

yt_ = tree_reg.predict(xt_)



#Visualize 

plt.scatter(x,y,color="red")

plt.plot(xt_,yt_,color="blue")

plt.show()



from sklearn.metrics import r2_score

print(linear_reg2.score(y,yt_))
data= pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data["class"].value_counts()
data.info()
data["class"] = [1 if each == "Abnormal" else 0 for each  in data["class"]]

y = data["class"].values

x_data = data.iloc[:,:-1]
#Normalizasyon

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 13) 

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {}".format(13,knn.score(x_test, y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test, y_test))



plt.figure(figsize=(8,5))

plt.scatter(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()