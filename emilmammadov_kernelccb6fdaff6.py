import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
dfTrain = pd.read_csv('../input/train.csv')

dfTest = pd.read_csv('../input/test.csv')

dfTestY = pd.read_csv('../input/sample_submission.csv')

dfTest['SalePrice'] = dfTestY.SalePrice
# Train setimizde toplamda 1460 kadar veri var. Bu yüzden 1000 üzerinde NaN değeri olan sütunlar işimize yaramaz

#Alley,PoolQC,Fence,MiscFeature kolonlarinda 1000 üzerinde NaN degeri oldugu icin siliyoruz.

for col in dfTrain.keys():

    #print('\n',col)

    #print('NaN  ',dfTrain[col].isna().sum())

    if dfTrain[col].isna().sum() >= 1000:

        dfTrain = dfTrain.drop(col, axis=1)

        

for col in dfTest.keys():

    #print('\n',col)

    #print('NaN  ',dfTrain[col].isna().sum())

    if dfTest[col].isna().sum() >= 1000:

        dfTest = dfTest.drop(col, axis=1)

        print(col,'silindi')

    # bu satir sadece butun unique degerleri yazdirmak icindir.

    #print (df[col].value_counts())
# Id değerleri anlam ifade etmediği için siliyoruz.

dfTrain = dfTrain.drop(['Id'], axis=1)

dfTest = dfTest.drop(['Id'], axis=1)
# Bütün kategorik değerleri numerik yapıyoruz

dfTrain = pd.get_dummies(dfTrain)

dfTest = pd.get_dummies(dfTest)
#Silinen 4 kolonun dışında yine NaN değerleri olan kolonlar var. Bu kolonları ortalama değerle dolduruyoruz.

dfTrain = dfTrain.fillna(dfTrain.mean())

dfTest = dfTest.fillna(dfTest.mean())
dfTrain = pd.get_dummies(dfTrain)

dfTest = pd.get_dummies(dfTest)

dfTrain.describe()
# bütün dataları numerik yaptıktan sonra train setinde 276, test setinde ise 260 sütun oldu.

# Veri iyi ayrılmadığı için böyle bir şey oldu.

# train verisinde bulunan 276 kolondan testte bulunmayan 16 kolonu siliyoruz.

dfTrain = dfTrain.drop(['GarageQual_Ex'], axis=1)

dfTrain = dfTrain.drop(['Electrical_Mix'], axis=1)

dfTrain = dfTrain.drop(['Utilities_NoSeWa'], axis=1)

dfTrain = dfTrain.drop(['Condition2_RRAe'], axis=1)

dfTrain = dfTrain.drop(['Condition2_RRAn'], axis=1)

dfTrain = dfTrain.drop(['Condition2_RRNn'], axis=1)

dfTrain = dfTrain.drop(['HouseStyle_2.5Fin'], axis=1)

dfTrain = dfTrain.drop(['RoofMatl_ClyTile'], axis=1)

dfTrain = dfTrain.drop(['RoofMatl_Membran'], axis=1)

dfTrain = dfTrain.drop(['RoofMatl_Metal'], axis=1)

dfTrain = dfTrain.drop(['RoofMatl_Roll'], axis=1)

dfTrain = dfTrain.drop(['Exterior1st_ImStucc'], axis=1)

dfTrain = dfTrain.drop(['Exterior1st_Stone'], axis=1)

dfTrain = dfTrain.drop(['Exterior2nd_Other'], axis=1)

dfTrain = dfTrain.drop(['Heating_Floor'], axis=1)

dfTrain = dfTrain.drop(['Heating_OthW'], axis=1)
# Train ve test verilerini ayırıyoruz.

X_train = dfTrain.iloc[:,:-1]

Y_train = dfTrain.SalePrice

X_test = dfTest.iloc[:,:-1]

Y_test = dfTest.SalePrice
# Train verilerimizi random forest algoritmasına girdi olarak veriyoruz.

rf_model = RandomForestRegressor()



rf_results = rf_model.fit(X_train, Y_train)





#print(rf_model.predict(X_test))
# Çıkan sonuç test verileri ile karşılaştırılıp scorunun belirlenmesi sağlanıyor. (Sadece bize verilen ornekteki degerlerle karsilastiriyoruz)

rf_model.score(X_test,Y_test)
#son = Y_test

predicted = pd.Series(np.array(rf_model.predict(X_test)))

#son['tahmin'] = 'predicted'

#Y_test['den'] = predicted

#print(predicted)



predicted.to_csv('deneme.csv', header=True)