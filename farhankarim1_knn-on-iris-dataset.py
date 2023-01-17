import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df1 = pd.read_csv("../input/Iris.csv", index_col=0)
df1.loc[(df1['Species']=='Iris-setosa'),'Species']=0
df1.loc[(df1['Species']=='Iris-versicolor'),'Species']=1
df1.loc[(df1['Species']=='Iris-virginica'),'Species']=2
df1.head(150)
df = df1.drop('Species',axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_features = scaler.transform(df)
df_scaled = pd.DataFrame(scaled_features,columns = df.columns)
df_scaled.head()
from sklearn.cross_validation import train_test_split
df.columns
X=df_scaled
y=df1['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []

for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(10,6))
plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=12)
plt.title('Error rate per k value')
plt.xlabel('K-Value')
plt.ylabel('Error Rate')


error_rate[6]
for i in range (1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

