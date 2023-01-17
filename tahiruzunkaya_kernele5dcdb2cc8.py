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
['WA_Fn-UseC_-Telco-Customer-Churn.csv']
cust=pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
cust.head()
cust.dtypes
cust.info
cust.describe
cust.shape
#totalcharges kolonunun numaric değere dönüştürülmesi
cust.TotalCharges = pd.to_numeric(cust.TotalCharges, errors='coerce')
cust.isnull().sum()
#Eksik değerlerin kaldırılması
cust.dropna(inplace = True)
#CustomerID kolonunun silinmesi
df2 = cust.iloc[:,1:]
#Değerlerin numaric yapılması
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Kategorik değişkenlerin dönüştürülmesi
df_dummies = pd.get_dummies(df2)
df_dummies.head()
#Veri ön işleme sonrası kontroller
df_dummies.dtypes
df_dummies.shape
#Histogram Gösterimi
import matplotlib.pyplot as plt
num_bins = 10
df_dummies.hist(bins=num_bins, figsize=(20,15))
#Korelasyon Gösterim 1
import matplotlib.pyplot as plt
plt.matshow(df_dummies.corr())
#Korelasyon gösterim 2
import seaborn as sns
corr = df_dummies.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#curn değerinin diğer değerlerle olan korelasyonu
plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

#churn değerini artı yönde en çok etkileyen kolon Contract_Month-to-month kolonudur

#bunun anlamı artı yönde olan değer arttıkça churn olma ihtimalinin artmasıdır.
#yinede bu 0.4 korelasyonu yeterli bir korelasyon değeri değildir

#churn değerini eksi yönde en çok etkileyen kolon tenure kolonudur.

#bunun anlamı eksi yönde olan değer azaltıkça churn olma ihtimalinin artmasıdır.
#Yinede bu -0.3 korelasyon değeri yeterli bir korelasyon değildir.
#Churn ile diğer kolonlar arasındaki korelasyon !düzmetin
df_dummies.corr()['Churn'].sort_values(ascending = False)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Eğitim  ve test verisini parcaliyoruz --> 80% / 20%
X = df_dummies.ix[:, df_dummies.columns != 'Churn']
Y = df_dummies['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#Sınıflandırma Modellerine Ait Kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Logistic Regression', LogisticRegression()))
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE

# Modelleri test edelim
for name, model in models:
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    #Accuracy değeri görelim
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_test, Y_pred)*100))
    
    #Confusion matris çizilmesi 
    report = classification_report(Y_test, Y_pred)
    print(report)
    
    
    
    #logistic regression da naive bayes e göre accuracy değeri daha yüksek olduğu için onun
    #kullanılması daha mantıklı olabilir.