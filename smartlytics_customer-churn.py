# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# The dataset contains the information of 7042 Customers and their churn value.

data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data = data.drop(['customerID'], axis=1)

data
# list(data)

categorical = [#'customerID',

 'gender',

 'SeniorCitizen',

 'Partner',

 'Dependents',

 #'tenure',

 'PhoneService',

 'MultipleLines',

 'InternetService',

 'OnlineSecurity',

 'OnlineBackup',

 'DeviceProtection',

 'TechSupport',

 'StreamingTV',

 'StreamingMovies',

 'Contract',

 'PaperlessBilling',

 'PaymentMethod',

 #'MonthlyCharges',

 #'TotalCharges',

 #'Churn'

]
# Churn summary

print(data.groupby('Churn').Churn.count())
data.SeniorCitizen = ['Yes' if sc == 1 else 'No' for sc in data.SeniorCitizen]

data.Churn = [1 if c == 'Yes' else 0 for c in data.Churn]
def perc(x):

    return str(round(100*x,1))+'%'

fig=plt.figure(figsize=(16,15))

fig.suptitle('Churn rate by category',fontsize='x-large')

for i in range(0,len(categorical)):

    category = categorical[i]

    ax = fig.add_subplot(4,4,i+1)

    group_churn = data.groupby(category).Churn.mean()

    k = group_churn.keys()

    v = group_churn.values

    v2 = v/np.sum(v)

    left = np.cumsum(v2)

    plt.barh([1],v2[0],height=0.2,label=k[0]+': '+perc(v[0]))

    for j in range(1,len(v)):

        plt.barh([1],v2[j],left=left[j-1],height=0.2,label=k[j]+': '+perc(v[j]))

    plt.ylim([0.4,1.6])

    plt.xlim(-0.1,1.1)

    plt.axis('off')

    plt.legend()

    plt.title(category)

fig=plt.figure(figsize=(16,15))

fig.suptitle('Disparities in Churn rates for each category',fontsize='x-large')

plt.show()
# Preprocessing and Dummy variables



data.TotalCharges = [0 if tc==' ' else float(tc) for tc in data.TotalCharges]

data = pd.get_dummies(data,columns=categorical, drop_first=True)



# Dependent and Independent variables

X = data.values

y = data.Churn.values



# Feature Scaling

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X = scale.fit_transform(X)
from sklearn.decomposition import KernelPCA

kpca=KernelPCA(n_components = 2,kernel = 'rbf')

Xpca = kpca.fit_transform(X)



from sklearn.cluster import KMeans

clusters = 2

kmeans = KMeans(n_clusters = clusters,init = 'k-means++',max_iter=300,n_init=10,random_state=0)

y_kmeans = kmeans.fit_predict(X)

data['Clusters'] = y_kmeans



mean_values = data.groupby('Clusters')['Churn'].mean().values

rate = [str(round(100*mean_values.min(),1))+'%',str(round(100*mean_values.max(),1))+'%']

colors = ['skyblue','tomato']

if mean_values[0]>mean_values[1]:

    colors = colors[::-1]

    rate = rate[::1]

    

plt.rcParams['axes.facecolor'] = 'whitesmoke'

plt.figure(figsize=(12,6))



plt.subplot(1,2,1)

plt.scatter(Xpca[data.Clusters==0][:,0],Xpca[data.Clusters==0][:,1],marker='.',color=colors[0],label='Churn rate = '+rate[0])

plt.scatter(Xpca[data.Clusters==1][:,0],Xpca[data.Clusters==1][:,1],marker='.',color=colors[1],label='Churn rate = '+rate[1])

plt.legend()

plt.title('Churn by Cluster')

plt.xlabel('PCA component 1')

plt.ylabel('PCA component 2')

plt.axis('equal')



plt.subplot(1,2,2)

plt.title('Cluster Size')

plt.pie(data.groupby('Clusters')['Clusters'].count().values,colors = colors,autopct='%1.1f%%')

plt.axis('equal')

plt.show()



plt.show()
clusters = 5

kmeans = KMeans(n_clusters = clusters,init = 'k-means++',max_iter=300,n_init=10,random_state=0)

y_kmeans = kmeans.fit_predict(X)

data['Clusters'] = y_kmeans



X = scale.inverse_transform(X)
plt.rcParams['axes.facecolor'] = 'silver'

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)



colors = ['beige','palegoldenrod','orange','orangered','firebrick']



mean_values = data.groupby('Clusters')['Churn'].mean().values

risk_sort = np.argsort(data.groupby('Clusters')['Churn'].mean().values)



plt.bar(x=np.arange(0,clusters),height = (mean_values)[risk_sort],color= colors)

plt.plot(np.arange(-1,clusters+1),np.full(clusters+2,data.Churn.mean()),'k--',label='Average Churn')

plt.xlim([-.75,clusters-0.25])

plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.xlabel('Customer Clusters')

plt.ylabel('Churn rate')

plt.title('Churn by Cluster')

plt.legend()



plt.subplot(1,2,2)

plt.title('Cluster Size')

plt.pie(data.groupby('Clusters')['Clusters'].count().values[risk_sort],autopct='%1.1f%%',colors = colors)

plt.axis('equal')

plt.show()
print('Risk Cluster size:\t\t',str(round(100*data.groupby('Clusters')['Churn'].count().values.max()/len(data),1))+'%')

print('Risk Cluster churn rate:\t',str(round(100*mean_values.max(),1))+'%')
# Training set and Test set

from sklearn.model_selection import train_test_split

X = data.drop(columns=['Churn','Clusters']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



# RF Classifier

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200)

rf_model.fit(X_train,y_train)



# Prediciton

y_pred = rf_model.predict(X_test)



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



print('Model:\t\t\tRandom Forest Classification')

print('Test Set Accuracy:\t' ,str(round(100*np.trace(cm)/np.sum(cm),1))+'%')
# XGBoost Classifier

from xgboost import XGBClassifier

xgb_model = XGBClassifier()

xgb_model.fit(X_train,y_train)



# Prediciton

y_pred = xgb_model.predict(X_test)



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



print('Model:\t\t\tXGBoost Classification')

print('Test Set Accuracy:\t' ,str(round(100*np.trace(cm)/np.sum(cm),1))+'%')
# Scaling

X_train = scale.fit_transform(X_train)

X_test = scale.transform(X_test)



# Creating the ANN

import keras

from keras.models import Sequential

from keras.layers import Dense

ann_model = Sequential()

ann_model.add(Dense(activation="relu", input_dim=30, units=15, kernel_initializer="uniform"))

ann_model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

ann_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann_model.fit(X_train,y_train,batch_size=10,epochs=5)



# Prediction

y_pred = ann_model.predict(X_test) > 0.5



# Confusion Matrix

cm = confusion_matrix(y_test,y_pred)



print('\nModel:\t\t\tANN Classification')

print('Test Set Accuracy:\t' ,str(round(100*np.trace(cm)/np.sum(cm),1))+'%')