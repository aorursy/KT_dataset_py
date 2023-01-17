#1. kutuphaneler

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris

#2. Veri Onisleme

#2.1. Veri Yukleme

veriler = pd.read_csv('../input/iris/Iris.csv')

print(veriler.head())





x = veriler.iloc[:,1:5].values #bağımsız değişkenler

#y = veriler.iloc[:,5:].values #bağımlı değişken

y = veriler.Species.values

print(x[:5])

print(y[:5])



#verilerin egitim ve test icin bolunmesi

from sklearn.cross_validation import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)



#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.transform(x_test)
# 1. Logistic Regression

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train,y_train) #egitim



y_pred = logr.predict(X_test) #tahmin

#print(y_pred)

#print(y_test)



#karmasiklik matrisi

cm = confusion_matrix(y_test,y_pred)

#print('LR')

# %% cm visualization

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train,y_train)



y_pred = knn.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('KNN')

#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
# 3. SVC (SVM classifier)

from sklearn.svm import SVC

svc = SVC(kernel='poly')

svc.fit(X_train,y_train)



y_pred = svc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('SVC')

#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
# 4. Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)



y_pred = gnb.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('GNB')

#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
# 5. Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')



dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('DTC')

#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
# 6. Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')

rfc.fit(X_train,y_train)



y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('RFC')

#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
# 7. ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test)

#print(y_test)

#print(y_proba[:,0])



from sklearn import metrics

fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')

print(fpr)

print(tpr)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



veriler = pd.read_csv('../input/clustering/musteriler.csv')

print(veriler.head())

X = veriler.iloc[:,3:].values

#print(X)

x1 = X[:,0]

y1 = X[:,1]

plt.scatter(x1,y1)

plt.show()
from sklearn.cluster import KMeans

sonuclar = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init='k-means++', random_state= 123)

    kmeans.fit(X)

    sonuclar.append(kmeans.inertia_)



plt.plot(range(1,11),sonuclar)
kmeans = KMeans( n_clusters = 3, init = 'k-means++')

kmeans.fit(X)

print(kmeans.cluster_centers_)



clusters = kmeans.fit_predict(X)

veriler["label"] = clusters

#print(clusters)



plt.scatter(veriler.Hacim[veriler.label == 0 ],veriler.Maas[veriler.label == 0],color = "red")

plt.scatter(veriler.Hacim[veriler.label == 1 ],veriler.Maas[veriler.label == 1],color = "green")

plt.scatter(veriler.Hacim[veriler.label == 2 ], veriler.Maas[veriler.label == 2],color = "blue")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = "yellow")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(X,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

Y_tahmin = ac.fit_predict(X)

#print(Y_tahmin)



plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')

plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')

plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')

plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')

plt.title('HC')

plt.show()
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import r2_score

import statsmodels.api as sm



veriler = pd.read_csv('../input/comparing-regression2/maaslar2.csv')

print(veriler.head())

x = veriler.iloc[:,1:2]

y = veriler.iloc[:,2:]

X = x.values

Y = y.values

veriler.head()
veriler.corr()
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X,Y)

model = sm.OLS(lin_reg.predict(X),X)

print(model.fit().summary())

print("Linear R2 degeri:")

print(r2_score(Y, lin_reg.predict((X))))



plt.scatter(X,Y,color='red')

plt.plot(x,lin_reg.predict(X), color = 'blue')

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(X)

#print(x_poly)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,y)

model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)



print("Polynomial R2 degeri:")

print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)) ))



plt.scatter(X,Y,color = 'red')

plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.show()
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()

y_olcekli = sc2.fit_transform(Y)



from sklearn.svm import SVR



svr_reg = SVR(kernel = 'rbf')

svr_reg.fit(x_olcekli,y_olcekli)



model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)

print(model3.fit().summary())

print("SVR R2 degeri:")

print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )
plt.scatter(x_olcekli,y_olcekli,color='red')

plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

plt.show()
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X,Y)

print('dt ols')

model4 = sm.OLS(r_dt.predict(X),X)

print(model4.fit().summary())

print("Decision Tree R2 degeri:")

print(r2_score(Y, r_dt.predict(X)) )
Z = X + 0.5

K = X - 0.4



plt.scatter(X,Y, color='red')

plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')

plt.plot(x,r_dt.predict(K), color = 'yellow')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)

rf_reg.fit(X,Y)

print('dt ols')

model5 = sm.OLS(rf_reg.predict(X),X)

print(model5.fit().summary())

print("Random Forest R2 degeri:")

print(r2_score(Y, rf_reg.predict(X)) )
plt.scatter(X,Y, color='red')

plt.plot(x,rf_reg.predict(X), color = 'blue')

plt.plot(x,rf_reg.predict(Z), color = 'green')

plt.show()
print('----------------')

print("Linear R2 value:")

print(r2_score(Y, lin_reg.predict((X))))



print("Polynomial R2 value:")

print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)) ))



print("SVR R2 value:")

print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )



print("Decision Tree R2 value:")

print(r2_score(Y, r_dt.predict(X)) )



print("Random Forest R2 value:")

print(r2_score(Y, rf_reg.predict(X)) )