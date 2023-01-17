

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


import matplotlib.pyplot as plt #gorselestirmek icin kullandigimiz kutuphane
import seaborn as sns #gorselestirmek icin kullandigimiz kutuphane(isi haritasi)
from sklearn import preprocessing #Veri Normalleştirme icin kullanilan kutuphane
from sklearn import  model_selection   #verileri ayirmak icin kullandigimiz kutuphane test,egitim( cross_validation)
df=pd.read_csv(r'../input/pokemon.csv') 
df.head(10) 
df.drop('#', axis = 1, inplace = True)

df.head(10) #ilk 10 satir
df.tail(10) 
df.shape
df.info()
df.describe()
df.columns
df[(df['Defense']>200) & (df['Attack']<300)].sort_values('Attack', axis=0, ascending=False)  

def Can(hp):
   
  return (hp>100)

df['YuksekCan'] = df['HP'].apply(Can)
df.head(20)
df.mean(axis=0,skipna=True)
df.dtypes
df2=df #dataframe kopyaladik
df2.corr() # korelasyon 
#correlation map
f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(df2.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)
plt.show() # 12 12 lik isi haritasi bize corelasyon renkli bir sekilde gosteriyor
#korelasyon tablosu,annot=içindeki sayıların görünürlüğü,linewidths=çerçeve fmt sondan iki basamak
df2['YuksekCan']=df2['YuksekCan'].astype('float') #YuksekCan niteligi float tipine cevirdik
df2.plot(kind='scatter', x='HP', y='YuksekCan',alpha = 0.5,color = 'red')
plt.xlabel('HP')             
plt.ylabel('YuksekCan')
plt.title('HP YuksekCan Scatter Plot')
df2['YuksekCan']=df2['YuksekCan'].astype('bool') #generation niteligi bool tipine cevirdik



df2["Attack"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.4,label="Attack",figsize=(18,8))
plt.legend()
plt.xlabel("Attack")
plt.ylabel("sıklık")
plt.title("Attack Sıklığı")
plt.show()

df2.isnull().sum() #Datamız içerisinde tanımlanmamış değerler 
df2.isnull().sum().sum()  #Datamız içerisinde toplam tanımlanmamış değerler 
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum() #eksik degerler
    mis_val_percent = 100 * df.isnull().sum()/len(df) #eksik degerlerin yuzdelik dilimi
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) #tabloda birlestir
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns

eksik_deger_tablosu(df2)
df3=df2
df3['HP'] = df3['HP'].transform(lambda x: x.fillna(x.mean()))
df3['Attack'] = df3['Attack'].transform(lambda x: x.fillna(x.mean()))
df3['Defense'] = df3['Defense'].transform(lambda x: x.fillna(x.mean()))
df3['Sp. Atk'] = df3['Sp. Atk'].transform(lambda x: x.fillna(x.mean()))
df3['Sp. Def'] = df3['Sp. Def'].transform(lambda x: x.fillna(x.mean()))
df3['Speed'] = df3['Speed'].transform(lambda x: x.fillna(x.mean()))



df3['Type 2'].fillna('empty',inplace=True) #empty ile doldur
df3['Name'].fillna('empty',inplace=True) #empty ile doldur

df3.head(10)

#Veri Normalleştirme

#Speed özniteliğini normalleştirmek için
x = df3[['Speed']].values.astype(float) #float tipine donüştürüyoruz

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
#speed niteligini 0-1 arasinda degerler indirgiyoruz.
min_max_scaler = preprocessing.MinMaxScaler((0,1))
x_scaled = min_max_scaler.fit_transform(x)
df3['Speed'] = pd.DataFrame(x_scaled)

df3



sns.boxplot(x=df3['Attack'])
plt.show()
den = np.percentile(df3.Attack, [0, 100])
den
new_df = df3[(df3.Attack > den[0]) & (df3.Attack < den[1])] 
new_df
df3['Legendary'].replace(False, 0, inplace=True) #burda legendary true false dan 1,0 konumuna getiriyoruz
df3['YuksekCan'].replace(True, 1, inplace=True) #burda YuksekCan true false dan 1,0 konumuna getiriyoruz
df3
num_bins = 10
df3.hist(bins=num_bins, figsize=(20,15))
df4=df3
df4.drop('Name', axis = 1, inplace = True)
df4.drop('Type 1', axis = 1, inplace = True)
df4.drop('Type 2', axis = 1, inplace = True)

df4
data1 = df3[df3['Legendary']==1]
data2 = df3[df3['Legendary']==0]
data1.shape #lengendary  olanlar(true)
data2.shape  #lengendary olmayanlar(false)
data = data1.append(data2[:65])
data.shape
X = data.ix[:, data.columns != 'Legendary']
Y = data['Legendary']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)






X_train = X_train.astype('float32') #float tipine ceviriyoruz
X_test = X_test.astype('float32') #float tipine ceviriyoruz
#ölçeklendirme
scaler = preprocessing.MinMaxScaler((-1,1))
scaler.fit(X)
XX_train = scaler.transform(X_train.values)
XX_test  = scaler.transform(X_test.values)
YY_train = Y_train.values 
YY_test  = Y_test.values
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Logistic Regression', LogisticRegression()))
models.append(('RandomForestClassifier', RandomForestClassifier())) 
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE

# Modelleri test edelim
for name, model in models:
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    #Accuracy değeri gör
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_test, Y_pred)*100))
    
    #Confusion matris görmek için aşağıdaki kod satırlarını kullanabilirsiniz   
    report = classification_report(Y_test, Y_pred)
    print(report)
