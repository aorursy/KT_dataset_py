import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
data = data.loc[:,["date_block_num", "shop_id", "item_id", "item_cnt_day" ]]
data
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
test = test.drop("ID", axis=1)
pd.set_option('float_format', '{:.2f}'.format)
test
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
items.describe()
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
shops.describe()
data["date_block_num"]=((data["date_block_num"]-data["date_block_num"].min())/((data["date_block_num"].max()+1)-data["date_block_num"].min()))
data["shop_id"]=((data["shop_id"]-0)/(59-0))
data["item_id"]=((data["item_id"]-0)/(22169-0))
data
test["shop_id"]=((test["shop_id"]-0)/(59-0))
test["item_id"]=((test["item_id"]-0)/(22169-0))
test
## normalizasyon sonucu data_block_num 34 olacağı, 
## yani normalizasyonda direkt 1 olacağı için
## 1'lik dataframe oluşturuyoruz.
satistest = np.ones(214200)
satistest = pd.DataFrame(satistest)
satistest
#join ile bu dataframeleri birleştiriyoruz
test = satistest.join(test)
test
#Bu şekilde datasetimizi azaltabiliriz.
#Mevcut durumda bilgisayarın ekran kartı yetersizliğinden ve ram şişmesinden dolayı 3 milyon olan kümeyi 1.6 milyon alıyorum
data = data.sample(n=1600000, random_state=1)
egitimverisi, validationverisi = train_test_split(data, test_size=0.2)
egitimgirdi = egitimverisi.drop(["item_cnt_day"], axis=1)
egitimcikti = egitimverisi.item_cnt_day
egitimgirdi
egitimcikti
valgirdi = validationverisi.drop(["item_cnt_day"], axis=1)
valcikti = validationverisi.item_cnt_day
valgirdi
valcikti
test = test.rename(columns = {0: "date_block_num"})
test2 = test.sample(n=214200, random_state=1)
test2 = test2.to_numpy()
sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv",index_col='date',parse_dates=['date'])
sales["item_cnt_day"][:"2014-01-01"].plot(figsize=(16,10), legend= True, color = 'g')
sales["item_cnt_day"]['2014-01-01':'2015-01-01'].plot(figsize=(16,10), legend=True, color= 'b')
sales["item_cnt_day"]['2015-01-01':].plot(figsize=(16,10), legend = True, color = 'r')
plt.xlabel('Dates')
plt.ylabel('Number of Products Sold')
plt.title('Date vs Sold')
models = []
models.append(("LR",LogisticRegression()))
## models.append(("LDA",LinearDiscriminantAnalysis())) LR ile benzer olduğu için kaldırıldı.
models.append(("KNN",KNeighborsClassifier()))
models.append(("DCT",DecisionTreeClassifier()))
models.append(("GNB",GaussianNB()))
#models.append(("SVC",SVC()))  AŞIRI ZAMAN ALDIĞI İÇİN KALDIRILDI (YAKLAŞIK 10 SAAT) 
##### Your notebook tried to allocate more memory than is available. It has restarted.
#models.append(("GPC",GaussianProcessClassifier(1.0*RBF(1.0)))) AŞIRI ZAMAN ALDIĞI İÇİN KALDIRILDI (YAKLAŞIK 15 SAAT)
#models.append(("MLP",MLPClassifier()))                         AŞIRI ZAMAN ALDIĞI İÇİN KALDIRILDI
#models.append(("ADB",AdaBoostClassifier()))                    AŞIRI ZAMAN ALDIĞI İÇİN KALDIRILDI
m=0
modelCohorenceTrain = np.arange(4)
modelCohorenceValidation = np.arange(4)
for name, model in models:
    liste = np.arange(214200)
    i=0
    egitilmismodel = model.fit(egitimgirdi,egitimcikti)
    egitimsonuc = egitilmismodel.score(egitimgirdi,egitimcikti)
    valsonuc = egitilmismodel.score(valgirdi,valcikti)
    if name == "LR":
        print("Sonuclar:  %s Egitim Verilerindeki Coherence Oranı:     %f " %(name, egitimsonuc))
        print("Sonuclar:  %s Validation Verilerindeki Coherence Oranı: %f " %(name,valsonuc))
    else:
        print("Sonuclar:  %s Egitim Verilerindeki Coherence Oranı:     %f " %(name, egitimsonuc))
        print("Sonuclar:  %s Validation Verilerindeki Coherence Oranı: %f " %(name,valsonuc))
    modelCohorenceTrain[m] = egitimsonuc*100
    modelCohorenceValidation[m] = valsonuc*100
    for x in test2:
        liste[i]=(egitilmismodel.predict([[x[0],x[1],x[2]]]))
        i=i+1
        if m == 0 and i == 214200:
            df = pd.DataFrame(liste)
            df.to_csv('LR.csv', header=False, index=False)
            print("LR file saved")
        elif m == 1 and i == 214200:
            df = pd.DataFrame(liste)
            df.to_csv('KNN.csv', header=False, index=False)
            print("KNN file saved")
        elif m == 2 and i == 214200:
            df = pd.DataFrame(liste)
            df.to_csv('DCT.csv', header=False, index=False)
            print("DCT file saved")
        elif m == 3 and i == 214200:
            df = pd.DataFrame(liste)
            df.to_csv('GNB.csv', header=False, index=False) 
            print("GNB file saved")
    m = m+1
#modelCohorenceTrain = np.arange(9)
#modelCohorenceValidation = np.arange(9)
%matplotlib inline
ModelName = ['LogisticRegression','KNeighborsClassifier','DecisionTreeClassifier','GaussianNB']
fig = plt.figure(figsize = (10,10))
plt.plot(modelCohorenceTrain, color="red", ls="-.", marker="^", ms=10, label="Train Verilerinin Yüzdelik Coherence Oranı")
plt.plot(modelCohorenceValidation, color="green", ls="-.", marker="*", ms=10, label="Validation Verilerinin Yüzdelik Coherence Oranı")
plt.legend(loc = 'upper left', bbox_to_anchor=(1,1))
plt.xticks(list(range(4)), ModelName, rotation="horizontal")
plt.yticks( rotation= 0)
plt.show()
pd.set_option('float_format', '{:.0f}'.format)
lr = pd.read_csv("/kaggle/working/LR.csv", header = None)
lr.rename(columns={0:"satis_miktari"}, 
                 inplace=True)
testLR = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
testLR = testLR.join(lr)
testLR
fig = plt.figure(figsize = (15,15))
testLR.plot(x='shop_id',  y='satis_miktari',color="orange", style='^')
plt.title('SHOPS &  number of products sold LR')  
plt.xlabel('SHOPS')  
plt.legend(loc = 'upper left', bbox_to_anchor=(1,1)) #to show the labels at proper location
plt.ylabel(' number of products sold')  
plt.show()


knn = pd.read_csv("/kaggle/working/KNN.csv", header = None)
knn.rename(columns={0:"satis_miktari"}, 
                 inplace=True)
testKNN = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
testKNN = testKNN.join(knn)
testKNN
fig = plt.figure(figsize = (15,15))
testKNN.plot(x='shop_id', y='satis_miktari',color="navy", style='*')
plt.title('SHOPS &  number of products sold KNN')  
plt.xlabel('SHOPS')  
plt.legend(loc = 'upper left', bbox_to_anchor=(1,1)) #to show the labels at proper location
plt.ylabel(' number of products sold')  
plt.show()

dct = pd.read_csv("/kaggle/working/DCT.csv", header = None)
dct.rename(columns={0:"satis_miktari"}, 
                 inplace=True)
testDCT = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
testDCT = testDCT.join(dct)
testDCT
fig = plt.figure(figsize = (15,15))
testDCT.plot(x='shop_id', y='satis_miktari',color="blueviolet", style='+')
plt.title('SHOPS &  number of products sold DCT')  
plt.xlabel('SHOPS')  
plt.legend(loc = 'upper left', bbox_to_anchor=(1,1)) #to show the labels at proper location
plt.ylabel(' number of products sold')  
plt.show()

gnb = pd.read_csv("/kaggle/working/GNB.csv", header = None)
gnb.rename(columns={0:"satis_miktari"}, 
                 inplace=True)
testGNB = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
testGNB = testGNB.join(gnb)
testGNB
fig = plt.figure(figsize = (15,15))
testGNB.plot(x='shop_id', y='satis_miktari',color="deepskyblue", style='+')
plt.title('SHOPS &  number of products sold GNB')  
plt.xlabel('SHOPS')  
plt.legend(loc = 'upper left', bbox_to_anchor=(1,1)) #to show the labels at proper location
plt.ylabel(' number of products sold')  
plt.show()

