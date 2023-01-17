# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Veri tanımlama

import numpy as np 

import pandas as pd 



# Veri Görselleştirme

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



#Sistem kütüphaneleri

import os

import warnings



# Çıktılarda karmaşıklığa sebep olduğu için uyarılırı iptal ediyoruz

warnings.filterwarnings("ignore")

print("Warnings Ignore")



print(os.listdir("../"))

print(os.listdir("../input/red-wine-quality-cortez-et-al-2009"))
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data.info()
print(data.columns)
data.shape
data.head()
data.describe()
data.quality.unique()
data.quality.value_counts()
data.groupby(["quality"], as_index = True).mean()
data.corr()


data["status"] = data.quality.apply(lambda x: "Good" if x > 5 else "Bad")

data.head()
plt.figure(figsize = (8,8))

labels = data.status.value_counts().index

plt.pie(data.status.value_counts(), autopct='%1.1f%%', pctdistance=0.8, textprops={'size':"30", 'color':"w"},

        shadow=True, startangle=360)

plt.legend(labels, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=16)

plt.axis('equal')

plt.title('QUALİTY', fontsize=32)

plt.show()
x = list(data.columns)

for i in x:

    data.plot(kind='scatter', x=i, y='quality',alpha = 0.5,color ="red" )   

    plt.title('Scatter Plot')          

    plt.show()
#correlation matrix

plt.figure(figsize=(10,5))

heatmap = sns.heatmap(data.corr(), annot=True, fmt=".1f")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)

plt.show()
cols_sns = ['residual sugar', 'chlorides', 'density', 'pH', 'alcohol', 'quality']

sns.set(style="ticks")

sns.pairplot(data[cols_sns], hue='quality')

plt.show()
sns.countplot(x='quality', data=data)

plt.show()
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
x = data.drop(['quality','status'], axis =1)

y = data['quality'] 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  



x = scaler.fit_transform(x)
x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train,y_train)

tree_pred = decision_tree.predict(x_test)

print('Decision Tree:', accuracy_score(y_test, tree_pred)*100,'%')
rf = RandomForestClassifier()

rf.fit(x_train,y_train)

rf_pred = rf.predict(x_test)

print('Random Forest:', accuracy_score(y_test, rf_pred)*100,'%')
KN = KNeighborsClassifier()

KN.fit(x_train,y_train)

KN_pred = KN.predict(x_test)

print('KNeighbors:',accuracy_score(y_test, KN_pred)*100,'%')
Gaussian = GaussianNB()

Gaussian.fit(x_train,y_train)

Gaussian_pred = Gaussian.predict(x_test)

print('GaussianNB:',accuracy_score(y_test, Gaussian_pred)*100,'%')
svc = SVC()

svc.fit(x_train,y_train)

svc_pred = svc.predict(x_test)

print('SVC:',accuracy_score(y_test, svc_pred)*100,'%')
k = []

l = []

for i in range(1,250):

    rf_tune = RandomForestClassifier(n_estimators=i)

    rf_tune.fit(x_train,y_train)

    y_pred = rf_tune.predict(x_test)

    k.append(float(accuracy_score(y_test, y_pred)*100))

    l.append(i)

    

k = pd.DataFrame(k , columns=['Accuracy']) 

l = pd.DataFrame(l , columns=['n_estimator'])

df = pd.concat([k, l], axis = 1)
df.sort_values(by='Accuracy', ascending=False)
from sklearn.decomposition import PCA

pca = PCA(n_components = 3, whiten= True )  # whitten = normalize

pca.fit(x)



x_pca = pca.transform(x)



print("variance ratio: ", pca.explained_variance_ratio_)



print("sum: ",sum(pca.explained_variance_ratio_))



#%%
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3)
k = []

l = []

for i in range(1,250):

    rf_tune = RandomForestClassifier(n_estimators=i)

    rf_tune.fit(x_train,y_train)

    y_pred = rf_tune.predict(x_test)

    k.append(float(accuracy_score(y_test, y_pred)*100))

    l.append(i)

    

k = pd.DataFrame(k , columns=['Accuracy']) 

l = pd.DataFrame(l , columns=['n_estimator'])

df = pd.concat([k, l], axis = 1)
df.sort_values(by='Accuracy', ascending=False)
data.corr()
x = data.drop(['fixed acidity','citric acid','free sulfur dioxide','total sulfur dioxide','pH','quality', 'status'], axis =1)

y = data['quality'] 
x.corr()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
k = []

l = []

for i in range(1,250):

    rf_tune = RandomForestClassifier(n_estimators=i)

    rf_tune.fit(x_train,y_train)

    y_pred = rf_tune.predict(x_test)

    k.append(float(accuracy_score(y_test, y_pred)*100))

    l.append(i)

    

k = pd.DataFrame(k , columns=['Accuracy']) 

l = pd.DataFrame(l , columns=['n_estimator'])

df = pd.concat([k, l], axis = 1)
df.sort_values(by='Accuracy', ascending=False)
data["status"] = data.quality.apply(lambda x: 1 if x > 5 else 0)

data.head()
x = data.drop(['quality','status' ,'fixed acidity','residual sugar','free sulfur dioxide','pH'], axis =1)

y = data['status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("test accuracy {}".format(lr.score(x_test,y_test)))
#%%

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



y_tahmin = lr.predict(x_test)

results = confusion_matrix(y_tahmin, y_test)

print("Confusion Matrix : ")

print(results)
data_cls = data.drop(['quality','status'], axis =1)
from sklearn.cluster import KMeans

wcss = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data_cls)

    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,5))

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value" , fontsize=26)

plt.ylabel("Within Cluster Sum of Squares" , fontsize=26)

plt.show()
kmeans2 = KMeans(n_clusters=3)

clusters= kmeans2.fit_predict(data_cls)

data_cls["label"] = clusters
data_cls
from sklearn.decomposition import PCA

pca = PCA(n_components =2 , whiten= True )  # whitten = normalize

pca.fit(data_cls)



scatter =pd.DataFrame(pca.transform(data_cls))
scatter["label"] = clusters



plt.scatter(scatter[0][scatter.label == 0 ],scatter[1][scatter.label == 0],color = "red")

plt.scatter(scatter[0][scatter.label == 1 ],scatter[1][scatter.label == 1],color = "green")

plt.scatter(scatter[0][scatter.label == 2 ],scatter[1][scatter.label == 2],color = "blue")

plt.show()