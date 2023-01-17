# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



pd.options.display.max_columns = 10

pd.options.display.max_columns = 100

test = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/test.csv")

train = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/train.csv")





train_len = len(train)

#573075

#train[(train["ISLEM_TUTARI"]/train["ISLEM_ADEDI"]).isnull()].fillna()

train.loc[573075] = train.loc[573075].replace(0.0,1.0)
train["YIL_AY"] = train["YIL_AY"].astype(str)

test["YIL_AY"] = test["YIL_AY"].astype(str)
sektorler = train["SEKTOR"].value_counts()

sektorler.plot(kind='bar')
#for tur in pd.unique(train["ISLEM_TURU"]):

tur = 'PESIN'

for sektor in pd.unique(train["SEKTOR"]):

    p_sektor1 = train[(train["ISLEM_TURU"]==tur)&(train["SEKTOR"]==sektor)&(train["YIL_AY"].astype(str).str[0:4]>='2017')][["ISLEM_TURU","SEKTOR","YIL_AY","ISLEM_TUTARI","ISLEM_ADEDI"]]

    p_sektor1["ORT_TUTAR"] = p_sektor1["ISLEM_TUTARI"]/p_sektor1["ISLEM_ADEDI"]

    p_group1 = p_sektor1.groupby(["ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR':['mean']})

    p_group1.columns = ['ORT_TUTAR']

    p_group1.reset_index(inplace=True)

    p_group1["YIL_AY"] = p_group1["YIL_AY"].astype(str)

    fark = p_group1[p_group1["YIL_AY"]=='201801']['ORT_TUTAR'].values-p_group1[p_group1["YIL_AY"]=='201901']['ORT_TUTAR'].values

    yeni_deger = p_group1[p_group1["YIL_AY"]=='201802']['ORT_TUTAR'].values - fark

    p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    plt.figure(figsize=(10,8))

    p_group1.plot(x='YIL_AY',y='ORT_TUTAR')

    plt.xlabel('Yıl - Ay')

    plt.ylabel('Ortalama Tutar')

    title = "İşlem Türü: "+tur+" - Sektör: "+sektor

    plt.title(title)

    plt.show()
tahminler_test = pd.DataFrame(columns=["ISLEM_TURU","SEKTOR","YIL_AY","ORT_TUTAR"])



tur='PESIN'

for sektor in pd.unique(train["SEKTOR"]):

    p_sektor1 = train[(train["ISLEM_TURU"]==tur)&(train["SEKTOR"]==sektor)&(train["YIL_AY"].astype(str).str[0:6]<'201812')][["ISLEM_TURU","SEKTOR","YIL_AY","ISLEM_TUTARI","ISLEM_ADEDI"]]

    p_sektor1["ORT_TUTAR"] = p_sektor1["ISLEM_TUTARI"]/p_sektor1["ISLEM_ADEDI"]

    p_group1 = p_sektor1.groupby(["ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR':['mean']})

    p_group1.columns = ['ORT_TUTAR']

    p_group1.reset_index(inplace=True)

    p_group1["YIL_AY"] = p_group1["YIL_AY"].astype(str)

    fark = p_group1[p_group1["YIL_AY"]=='201711']['ORT_TUTAR'].values-p_group1[p_group1["YIL_AY"]=='201811']['ORT_TUTAR'].values

    yeni_deger = p_group1[p_group1["YIL_AY"]=='201712']['ORT_TUTAR'].values - fark

    p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201812', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    #plt.figure(figsize=(10,8))

    #p_group1.plot(x='YIL_AY',y='ORT_TUTAR')

    #plt.xlabel('Yıl - Ay')

    #plt.ylabel('Ortalama Tutar')

    #plt.title(("İşlem Türü: ",tur," - Sektör: ",sektor))

    #plt.show()

    tahminler_test = tahminler_test.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201812', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    

tur='TAKSITLI'

for sektor in pd.unique(train["SEKTOR"]):

    p_sektor1 = train[(train["ISLEM_TURU"]==tur)&(train["SEKTOR"]==sektor)&(train["YIL_AY"].astype(str).str[0:6]<'201812')][["ISLEM_TURU","SEKTOR","YIL_AY","ISLEM_TUTARI","ISLEM_ADEDI"]]

    p_sektor1["ORT_TUTAR"] = p_sektor1["ISLEM_TUTARI"]/p_sektor1["ISLEM_ADEDI"]

    p_group1 = p_sektor1.groupby(["ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR':['mean']})

    p_group1.columns = ['ORT_TUTAR']

    p_group1.reset_index(inplace=True)

    p_group1["YIL_AY"] = p_group1["YIL_AY"].astype(str)

    

    try:

        fark = p_group1[p_group1["YIL_AY"]=='201711']['ORT_TUTAR'].values-p_group1[p_group1["YIL_AY"]=='201811']['ORT_TUTAR'].values

        yeni_deger = p_group1[p_group1["YIL_AY"]=='201712']['ORT_TUTAR'].values - fark

        p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201812', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

        tahminler_test = tahminler_test.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201812', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    except:

        yeni_deger = p_group1['ORT_TUTAR'].mean()

        p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201812', 'ORT_TUTAR':yeni_deger}, ignore_index=True)

        tahminler_test = tahminler_test.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201812', 'ORT_TUTAR':yeni_deger}, ignore_index=True)    

    #plt.figure(figsize=(14,14))

    #p_group1.plot(x='YIL_AY',y='ORT_TUTAR')

    #plt.xlabel('Yıl - Ay')

    #plt.ylabel('Ortalama Tutar')

    #plt.title(("İşlem Türü: ",tur," - Sektör: ",sektor))

    #plt.show()
tahminler_test
train["ORT_TUTAR1"] = train["ISLEM_TUTARI"]/train["ISLEM_ADEDI"]

cust_dagilim = train.groupby(["CUSTOMER","ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR1':'mean'})

cust_dagilim.columns = ["CUST_ORT_ISLEM_TUTARI"]

cust_dagilim.reset_index(inplace=True)

cust_dagilim = cust_dagilim[(cust_dagilim["YIL_AY"]>'201807')&(cust_dagilim["YIL_AY"]<='201811')]



cust_dagilim = cust_dagilim.groupby(["CUSTOMER","ISLEM_TURU","SEKTOR"]).agg({'CUST_ORT_ISLEM_TUTARI':'mean'})

cust_dagilim.columns = ["CUST_ORT_ISLEM_TUTARI"]

cust_dagilim.reset_index(inplace=True)



cust_dagilim["YIL_AY"] = '201812'



train = pd.merge(train,cust_dagilim,on=["CUSTOMER","ISLEM_TURU","SEKTOR","YIL_AY"], how='left')
train_201812 = pd.merge(train,tahminler_test,on=["ISLEM_TURU","SEKTOR","YIL_AY"], how='inner')

#Müşterinin son 4 aydaki ortalama işlem tutarını tahmin ayındaki işlem adediyle çarpıp genele göre beklenen işlem tutarını hesaplıyoruz.

train_201812["PRED_ISLEM_TUTARI"] = train_201812["ORT_TUTAR"]*train_201812["ISLEM_ADEDI"]

#train_201812["PRED_ISLEM_TUTARI2"] = train_201812["ORT_TUTAR"]*(np.sqrt(train_201812["ISLEM_ADEDI"]))





#son 4 aylık verisi olmayan müşterilerde genel ortalama işlem tutarını kullandık.

train_201812["CUST_ORT_ISLEM_TUTARI"].loc[train_201812[train_201812["CUST_ORT_ISLEM_TUTARI"].isnull()].index.values]=train_201812["ORT_TUTAR"].loc[train_201812[train_201812["CUST_ORT_ISLEM_TUTARI"].isnull()].index.values].values

train_201812["CUST_ORT_ISLEM_TUTARI"] = train_201812["CUST_ORT_ISLEM_TUTARI"]*train_201812["ISLEM_ADEDI"]



#Mix

train_201812["PRED_ISLEM_TUTARI_MEAN"] = (train_201812["PRED_ISLEM_TUTARI"]*3 + train_201812["CUST_ORT_ISLEM_TUTARI"]*7)/10

print(np.sqrt(mean_squared_error(train_201812["ISLEM_TUTARI"],train_201812["PRED_ISLEM_TUTARI_MEAN"])))
tahminler = pd.DataFrame(columns=["ISLEM_TURU","SEKTOR","YIL_AY","ORT_TUTAR"])





### 201902 KISMI ###

tur='PESIN'

for sektor in pd.unique(train["SEKTOR"]):

    p_sektor1 = train[(train["ISLEM_TURU"]==tur)&(train["SEKTOR"]==sektor)&(train["YIL_AY"].astype(str).str[0:6]<'201902')][["ISLEM_TURU","SEKTOR","YIL_AY","ISLEM_TUTARI","ISLEM_ADEDI"]]

    p_sektor1["ORT_TUTAR"] = p_sektor1["ISLEM_TUTARI"]/p_sektor1["ISLEM_ADEDI"]

    p_group1 = p_sektor1.groupby(["ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR':['mean']})

    p_group1.columns = ['ORT_TUTAR']

    p_group1.reset_index(inplace=True)

    p_group1["YIL_AY"] = p_group1["YIL_AY"].astype(str)

    fark = p_group1[p_group1["YIL_AY"]=='201801']['ORT_TUTAR'].values-p_group1[p_group1["YIL_AY"]=='201901']['ORT_TUTAR'].values

    yeni_deger = p_group1[p_group1["YIL_AY"]=='201802']['ORT_TUTAR'].values - fark

    p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    #plt.figure(figsize=(10,8))

    #p_group1.plot(x='YIL_AY',y='ORT_TUTAR')

    #plt.xlabel('Yıl - Ay')

    #plt.ylabel('Ortalama Tutar')

    #plt.title(("İşlem Türü: ",tur," - Sektör: ",sektor))

    #plt.show()

    tahminler = tahminler.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    

tur='TAKSITLI'

for sektor in pd.unique(train["SEKTOR"]):

    p_sektor1 = train[(train["ISLEM_TURU"]==tur)&(train["SEKTOR"]==sektor)&(train["YIL_AY"].astype(str).str[0:6]<'201902')][["ISLEM_TURU","SEKTOR","YIL_AY","ISLEM_TUTARI","ISLEM_ADEDI"]]

    p_sektor1["ORT_TUTAR"] = p_sektor1["ISLEM_TUTARI"]/p_sektor1["ISLEM_ADEDI"]

    p_group1 = p_sektor1.groupby(["ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR':['mean']})

    p_group1.columns = ['ORT_TUTAR']

    p_group1.reset_index(inplace=True)

    p_group1["YIL_AY"] = p_group1["YIL_AY"].astype(str)

    

    try:

        fark = p_group1[p_group1["YIL_AY"]=='201801']['ORT_TUTAR'].values-p_group1[p_group1["YIL_AY"]=='201901']['ORT_TUTAR'].values

        yeni_deger = p_group1[p_group1["YIL_AY"]=='201802']['ORT_TUTAR'].values - fark

        p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

        tahminler = tahminler.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger[0]}, ignore_index=True)

    #except:

        #print("Something went wrong")

    except:

        yeni_deger = p_group1['ORT_TUTAR'].mean()

        p_group1 = p_group1.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger}, ignore_index=True)

        tahminler = tahminler.append({'ISLEM_TURU':tur ,'SEKTOR': sektor, 'YIL_AY':'201902', 'ORT_TUTAR':yeni_deger}, ignore_index=True)    

    #plt.figure(figsize=(10,8))

    #p_group1.plot(x='YIL_AY',y='ORT_TUTAR')

    #plt.xlabel('Yıl - Ay')

    #plt.ylabel('Ortalama Tutar')

    #plt.title(("İşlem Türü: ",tur," - Sektör: ",sektor))

    #plt.show()

    

train["ORT_TUTAR1"] = train["ISLEM_TUTARI"]/train["ISLEM_ADEDI"]

cust_dagilim = train.groupby(["CUSTOMER","ISLEM_TURU","SEKTOR","YIL_AY"]).agg({'ORT_TUTAR1':'mean'})

cust_dagilim.columns = ["CUST_ORT_ISLEM_TUTARI"]

cust_dagilim.reset_index(inplace=True)

cust_dagilim = cust_dagilim[(cust_dagilim["YIL_AY"]>'201808')&(cust_dagilim["YIL_AY"]<='201901')] #bunu bir de 201812 son olacak şekilde denemek gerek. çünkü senelik adet başına ort. işem tutarı değişimini pred'le yakalıyoruz.



cust_dagilim = cust_dagilim.groupby(["CUSTOMER","ISLEM_TURU","SEKTOR"]).agg({'CUST_ORT_ISLEM_TUTARI':'mean'})

cust_dagilim.columns = ["CUST_ORT_ISLEM_TUTARI"]

cust_dagilim.reset_index(inplace=True)



cust_dagilim["YIL_AY"] = '201902'





test = pd.merge(test,cust_dagilim,on=["CUSTOMER","ISLEM_TURU","SEKTOR","YIL_AY"], how='left')





test_201902 = pd.merge(test,tahminler,on=["ISLEM_TURU","SEKTOR","YIL_AY"], how='left')

test_201902["PRED_ISLEM_TUTARI"] = test_201902["ORT_TUTAR"]*test_201902["ISLEM_ADEDI"]

#test_201902["PRED_ISLEM_TUTARI2"] = test_201902["ORT_TUTAR"]*(np.sqrt(test_201902["ISLEM_ADEDI"]))



test_201902["CUST_ORT_ISLEM_TUTARI"].loc[test_201902[test_201902["CUST_ORT_ISLEM_TUTARI"].isnull()].index.values]=test_201902["ORT_TUTAR"].loc[test_201902[test_201902["CUST_ORT_ISLEM_TUTARI"].isnull()].index.values].values

test_201902["CUST_ORT_ISLEM_TUTARI"] = test_201902["CUST_ORT_ISLEM_TUTARI"]*test_201902["ISLEM_ADEDI"]

test_201902["PRED_ISLEM_TUTARI_MEAN"] = (test_201902["PRED_ISLEM_TUTARI"]*3 + test_201902["CUST_ORT_ISLEM_TUTARI"]*7)/10

submission=pd.DataFrame({'Id':test_201902["ID"], 'Predicted':test_201902["PRED_ISLEM_TUTARI_MEAN"]})

submission.to_csv('submission_ver11.csv', index=False)