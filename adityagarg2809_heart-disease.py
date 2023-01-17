import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.isnull().sum()
df.describe().transpose()
import matplotlib.pyplot as plt 

import seaborn as sns 



l = df.columns
l
df['sex'].value_counts().plot(kind='bar')
plt.scatter(df['chol'],df['trestbps'])
sns.heatmap(df.corr())
plt.scatter(df['thal'],df['oldpeak'])
sns.scatterplot(df['thalach'], df['slope'], hue=df['target'])
X = df[l[:-1]]



y = df[l[-1]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.33)



scaled_train = scaler.fit_transform(X_train)



scaled_test = scaler.transform(X_test)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression()



lr_model.fit(scaled_train,y_train)



preds = lr_model.predict(scaled_test)
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
from sklearn.decomposition import PCA
pca = PCA(3)



reduced_train = pca.fit_transform(scaled_train)



reduced_test = pca.transform(scaled_test)
reduced_train.shape
plt.scatter(reduced_train[:,0],reduced_train[:,1])
plt.scatter(reduced_train[:,1],reduced_train[:,2])
plt.scatter(reduced_train[:,0],reduced_train[:,2])
lr_model = LogisticRegression()



lr_model.fit(reduced_train,y_train)



preds = lr_model.predict(reduced_test)
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
from sklearn.metrics import accuracy_score

iv = []

av = []

for i in range(1,13):

    pca = PCA(i)



    reduced_train = pca.fit_transform(scaled_train)



    reduced_test = pca.transform(scaled_test)

    

    lr_model = LogisticRegression()



    lr_model.fit(reduced_train,y_train)



    preds = lr_model.predict(reduced_test)

    

    iv.append(i)

    av.append(accuracy_score(y_test,preds))
plt.plot(iv,av)
from sklearn.svm import SVC
iv = []

av = []

for i in range(1,13):

    pca = PCA(i)



    reduced_train = pca.fit_transform(scaled_train)



    reduced_test = pca.transform(scaled_test)

    

    svc = SVC()



    svc.fit(reduced_train,y_train)



    preds = svc.predict(reduced_test)

    

    iv.append(i)

    av.append(accuracy_score(y_test,preds))
plt.plot(iv,av)
pca = PCA(4)



reduced_train = pca.fit_transform(scaled_train)



reduced_test = pca.transform(scaled_test)

    

svc = SVC()



svc.fit(reduced_train,y_train)



preds = svc.predict(reduced_test)
print(classification_report(y_test,preds))
svc = SVC()



svc.fit(scaled_train,y_train)



preds = svc.predict(scaled_test)
print(classification_report(y_test,preds))
from sklearn.preprocessing import PolynomialFeatures





poly = PolynomialFeatures(2)



x_poly_train = poly.fit_transform(scaled_train)



x_poly_test = poly.transform(scaled_test)
svc = SVC()



svc.fit(x_poly_train,y_train)



preds = svc.predict(x_poly_test)
print(classification_report(y_test,preds))
iv = []

av = []

for i in range(1,106):

    pca = PCA(i)



    reduced_train = pca.fit_transform(x_poly_train)



    reduced_test = pca.transform(x_poly_test)

    

    svc = SVC()



    svc.fit(reduced_train,y_train)



    preds = svc.predict(reduced_test)

    

    iv.append(i)

    av.append(accuracy_score(y_test,preds))
plt.plot(iv,av)
iv = np.array(iv)

av = np.array(av)



iv[av.argmax()]
pca = PCA(8)



reduced_train = pca.fit_transform(x_poly_train)



reduced_test = pca.transform(x_poly_test)

    

svc = SVC()



svc.fit(reduced_train,y_train)



preds = svc.predict(reduced_test)
print(classification_report(y_test,preds))