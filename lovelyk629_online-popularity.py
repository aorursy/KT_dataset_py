# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

data1=pd.read_csv('../input/OnlineNewsPopularity.csv')

data1.head(5)
data1['url'].value_counts().unique


data=data1.iloc[:,1:]

data.head(5)





b=['data_channel_is_lifestyle',

       ' data_channel_is_entertainment', 'data_channel_is_bus',

       'data_channel_is_socmed', 'data_channel_is_tech',

       'data_channel_is_world','weekday_is_monday', 'weekday_is_tuesday',

       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',

       'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend']

for i in b :

    data[i].value_counts().plot.bar(color='red',figsize=(20,20))

    

    
data['timedelta'].value_counts()




import matplotlib.pyplot as plt

import seaborn as sns

data['shares'].describe()

plt.figure(figsize=(10,10))

sns.distplot(data['shares'],color='g', bins=2, hist_kws={'alpha': 0.4})




import matplotlib.pyplot as plt

data.corr()['shares'].sort_values(ascending = False).plot(kind='bar',figsize=(20,20))

data.columns
import seaborn as sns

import matplotlib.pyplot as plot



col_names = ['timedelta', 'n_tokens_title', 'n_tokens_content','n_unique_tokens',]

fig, ax = plot.subplots(len(col_names), figsize=(10,10))



for i, col_val in enumerate(col_names):



    sns.boxplot(x=data[col_val], ax=ax[i])



col_names = [ 'n_non_stop_words', 'n_non_stop_unique_tokens','num_hrefs',

       'num_self_hrefs']



fig, ax = plot.subplots(len(col_names), figsize=(10,10))



for i, col_val in enumerate(col_names):



    sns.boxplot(x=data[col_val], ax=ax[i])
col_names = ['num_imgs', 'num_videos', 'average_token_length',

       'num_keywords']



fig, ax = plot.subplots(len(col_names), figsize=(10,10))



for i, col_val in enumerate(col_names):



    sns.boxplot(x=data[col_val], ax=ax[i])


sns.boxplot(x=data['shares'])



sns.boxplot(x=data['abs_title_subjectivity'])

plot.show()

sns.boxplot(x=data['abs_title_sentiment_polarity'])

plot.show()



fig, ax = plot.subplots(figsize=(10,10))

ax.scatter(data['n_tokens_content'],data['n_tokens_title'])
from scipy import stats

import numpy as np

z=np.abs(stats.zscore(data))

print(z)

data.shape
threshold=3  # threshold limit genearlly taken as 3 or -3

print(np.where(z>3))

print(z[0][22])
data_o=data[(z<3).all(axis=1)]

data_o .shape
sns.boxplot(x=data_o['n_tokens_title'])

plot.show()



fig, ax = plot.subplots(figsize=(10,10))

ax.scatter(data_o['n_tokens_content'],data_o['n_tokens_title'])
#standardalization



y=data_o['shares']

X=data_o.drop(columns=['shares'],axis=1)

from sklearn.preprocessing import MinMaxScaler

feature=X.columns.values

scaler=MinMaxScaler(feature_range=(0,1))

scaler.fit(X)

X=pd.DataFrame(scaler.transform(X))

X.columns=feature

X.head()

#PCA



from sklearn.decomposition import PCA

pca =PCA(n_components=2)



pc=pca.fit_transform(X)

principaldf= pd.DataFrame(data=pc ,columns=['Principal Component 1','Principal Component 2'])

principaldf.head()

finaldf = pd.concat([principaldf, y], axis = 1)

finaldf.head()

from sklearn.model_selection import train_test_split

principaldf_train,principaldf_test,y_train,y_test= train_test_split(principaldf,y,test_size=0.33 ,random_state=42)



print(principaldf_train.shape)

print(principaldf_test.shape)

print(y_train.shape)

print(y_test.shape)



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import  mean_squared_error ,r2_score





from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score





# Logistic regression



model=LogisticRegression()

output=model.fit(principaldf_train ,y_train)



Predict=model.predict(principaldf_test)

print("Logistc Regression Accuracy")

print(metrics.accuracy_score(y_test,Predict)*100)

model=LinearRegression()

output=model.fit(principaldf_train ,y_train)



Predict=model.predict(principaldf_test)

print("Mean square error")

print(mean_squared_error(y_test,Predict))

print("r2_score")

print(r2_score(y_test,Predict))
#support vector machine



from sklearn.svm import SVC  

model = SVC(kernel='linear') 





output=model.fit(principaldf_train ,y_train)



Predict=model.predict(principaldf_test)

print("SVM Accuracy")

print(metrics.accuracy_score(y_test,Predict)*100)

# Decision tree



from sklearn.tree import DecisionTreeRegressor

model1=DecisionTreeRegressor()

output=model1.fit(principaldf_train ,y_train)



Predict1=model1.predict(principaldf_test)

print("Decisipn tree Regression Accuracy")

print(metrics.accuracy_score(y_test,Predict1)*100)



from sklearn import metrics









from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33 ,random_state=42)



#print(X_train.shape)

#print(X_test.shape)

#print(y_train.shape)

#print(y_test.shape)







from sklearn.tree import DecisionTreeRegressor

model1=DecisionTreeRegressor()

output=model1.fit(X_train ,y_train)



Predict1=model1.predict(X_test)

print("Decisipn tree Regression Accuracy")

print(metrics.accuracy_score(y_test,Predict1)*100)
from sklearn.svm import SVC  

model = SVC(kernel='linear') 





output=model.fit(X_train ,y_train)



Predict=model.predict(X_test)

print("SVM Accuracy")

print(metrics.accuracy_score(y_test,Predict)*100)
from sklearn.neighbors import KNeighborsClassifier

model1=KNeighborsClassifier()

output=model1.fit(X_train ,y_train)



Predict1=model1.predict(X_test)

print("Decisipn tree Regression Accuracy")

print(metrics.accuracy_score(y_test,Predict1)*100)