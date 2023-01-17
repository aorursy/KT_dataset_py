import pandas as pd                 

import numpy as np

import matplotlib.pyplot as plt    

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model



### 1) VERİLERİ CSV DOSYALARIMIZDAN OKUDUK

#train = pd.read_csv(r"C:\Users\hakanorcun\Desktop\GA2\DATAS\train.csv")

#test = pd.read_csv(r"C:\Users\hakanorcun\Desktop\GA2\DATAS\test.csv")



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape)

print(test.shape)



### 2) PLOT ÖZELLİKLERİNİ TANIMLADIK

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)







### 3) VERİ ÖZERLLİKLERİ

plt.hist(train.SalePrice, color='blue')

plt.show()





#np.log() kullanarak eğriyi hesaplayarak yeniden plot edelim. 

target = np.log(train.SalePrice)

plt.hist(target, color='blue')

plt.show()







### 4) NUMERİC DEGERLER AYRILDI

numeric_features = train.select_dtypes(include=[np.number])

#numeric_features.dtypes





### 5) KORELEASYON DURUMU İNCELENDİ

corr = numeric_features.corr()



#pozitif ve negatif koreleasyon var mı baktık

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])









#SalePrice ve GarageArea arasındaki grafik

plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()







### 6) UYGUNSUZ VERİLERİ ELİMİNE EDİYORUZ





#aykırı durumları çıkartarak yeni bir df oluştur

train = train[train['GarageArea'] < 1300]



#display the previous graph again without outliers

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))

plt.xlim(-200,1600) # This forces the same scale as before

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()







### 7)  NULL DEĞERLERİ AYIRDIK



#null değerlerin ayrıldığı yeni değer

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Degerler']

nulls.index.name = 'Özellik'





### 8) SAYISAL OLMAYAN DEĞERLERİDE AYIRDIK



categoricals = train.select_dtypes(exclude=[np.number])





### 9)  ONE-HOT-ENCODİNG YÖNTEMİ İLE BOOLEAN DEĞERLER OLUŞTURULUR. 

### enc_street ADLI YENİ BİR KOLON OLUŞTURDUK. KODLAMAYI pd.get_dummies() METODU YAPIYOR.



train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)



#SaleCondition dan bir pivot tablo oluşturduk.

condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()



#enc_condition adlı yeni bir tablo oluşturduk

def encode(x): return 1 if x == 'Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode)

test['enc_condition'] = test.SaleCondition.apply(encode)





#bu özelliğin SalePrice arasondali ilişkiyi anlamak için plot ettik

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()



### 10) KAYIP VERİLERİ ORTALAMA İLE DOLDUR

data = train.select_dtypes(include=[np.number]).interpolate().dropna()





### 11) LİNEAR MODELİ OLUŞTUR





y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)





X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=35)





### 12) LİNEAR REGRESSİON MODEL



lr = linear_model.LinearRegression()



### 13) MODEL FİTTİNG



model = lr.fit(X_train, y_train)



### 14) PERFORMANSI DEĞERLENDİRİN



print ("R^2 : \n", model.score(X_test, y_test))



### 15)  OLUŞTURDUĞUMUZ MODELİ TAHMİN İÇİN KULLANACAĞIZ



predictions = model.predict(X_test)





print ('RMSE : \n', mean_squared_error(y_test, predictions))





#tahminler ile verileri scatter plot ile görselleştiriyoruz.

actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.75,

            color='b') 

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()





### 16) OUTPUT ÜRETME



submission = pd.DataFrame()



submission['Id'] = test.Id



# Yukarıda yaptığımız gibi model için test verilerinin özelliklerini seçtik.

feats = test.select_dtypes(

    include=[np.number]).drop(['Id'], axis=1).interpolate()



#tahmin ürettik.

predictions = model.predict(feats)



#tahminleri doğrusal formlara dönüştür.

# np.exp() kullanıyoruz çünkü daha önce logarithm(np.log()) kullandık.

final_predictions = np.exp(predictions)







submission['SalePrice'] = final_predictions



print(submission.head())



### 17) TAHMİNLERİ CSV OLARAK DIŞARI AKTAR.

submission.to_csv('predictions_SalePrice.csv', index=False)






