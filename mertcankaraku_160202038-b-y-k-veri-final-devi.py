#Kullanılan kütüphaneleri ekledik ve bir önişleme fonksiyonu tanımladık

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

def hazirla_sales_train(veri):
    veri=veri.groupby(['date_block_num','item_id','shop_id'], as_index = False).agg({'item_cnt_day': 'sum', 'item_price': 'max'})
    veri = veri[veri.item_price<60000]
    veri = veri[veri.item_cnt_day<12500]
    veri = veri.rename(columns={'item_cnt_day':'item_cnt_month', 'item_price':'max_item_price'})
    return veri

#Verilerimizi okuduk

egitim_verileri  = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
item_verileri    = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
test_verileri    = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
ciktilar         = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

print('test verileri :\n', test_verileri.head(20))

#Dataframe lerimizi istediğimiz hale getirmek için düzenledik.

egitim_verileri = hazirla_sales_train(egitim_verileri)
item_verileri    = item_verileri.drop(['item_name'],axis=1)

allTrainData=pd.merge(egitim_verileri,item_verileri)

allTestData= pd.merge(test_verileri,item_verileri)
allTestData=allTestData.drop(["ID"],axis=1)
allTestData['date_block_num'] = 34

df1= allTrainData[ ['max_item_price','item_id','shop_id'] ]

allTestData=pd.merge(df1, allTestData)


#Verimizi böldük.
x_train, x_test, y_train, y_test = train_test_split(allTrainData.drop('item_cnt_month', axis=1), allTrainData.item_cnt_month, test_size=0.33, random_state=0)
#AdaBoost algoritması ile regresyon denedik

from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(n_estimators=10,random_state=0)
abr.fit(x_train,y_train)
y_pred = abr.predict(x_test)

print("R2 Score:",r2_score(y_test,y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))


#GradientBoost Algoritması ile regresyon denedik.

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=10,random_state=0)
gbr.fit(x_train,y_train)
y_pred = gbr.predict(x_test)

print("R2 Score:",r2_score(y_test,y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
#Sonuçlarımızı karşılaştırıp GradientBoost algoritmasını seçtik ve onunla yolumuza devam ettik.

x=allTrainData.drop('item_cnt_month', axis=1)
y=allTrainData.item_cnt_month

gbr.fit(x,y)
y_pred = gbr.predict(allTestData)

prediction=pd.DataFrame(y_pred,columns=["item_cnt_month"])


#Çıktılarımızı ciktilar.csv dosyasına kaydettik.
ciktilar=ciktilar.drop(columns=['item_cnt_month'])
ciktilar=pd.concat([ciktilar,prediction],axis=1)
ciktilar=ciktilar.dropna()
ciktilar[['ID']]=ciktilar[['ID']].astype(int)
ciktilar.to_csv('ciktilar.csv', index=False)

print('Çıktılar dosyasının özeti :\n', ciktilar.head(15))