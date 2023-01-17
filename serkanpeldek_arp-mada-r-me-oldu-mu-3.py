#Gerekli Kütüphaneleri Yüklenmesi Yapılıyor



import numpy as np # lineer cebir 

import pandas as pd # veri işleme



#Görselleştirme

import matplotlib.pyplot as plt

import seaborn as sns



#Makine Öğrenmesi kütüphaneleri

from sklearn.preprocessing import MinMaxScaler

from hep_ml.gradientboosting import UGradientBoostingClassifier

from hep_ml.losses import BinFlatnessLossFunction



#Sistem 

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))
train=pd.read_csv("../input/training.csv")

test=pd.read_csv("../input/test.csv")



print("train.shape:{} test.shape:{}".format(train.shape, test.shape))
train.head()
def show_missing_values(function_data):

#Veri setindeki eksik değerleri bulalım

    

    """

    shape veri matrisinin satır ve sütün sayısını verir.

    Satır sayısı shape[0]'da' ve sütün sayısı shape[1]'de tutulur

    """

    number_of_sample=function_data.shape[0]

    

    """

    isnull() fonksiyonu, skaler veya dizi benzeri bir nesne alır ve

    değerlerin eksik olup olmadığını gösterir.

    

    sum() fonksiyonu, istenen eksen için değerlerin toplamını hesaplar

    """    

    check_isnull=function_data.isnull().sum()

    

    """

    null değer sayısı toplamı, sıfır olan özelliker(sütünlar) çıkartılıyor.

    Kalan özellikler null değer sayısı çok olandan az olana doğru sıralanıyor.

    """

    check_isnull=check_isnull[check_isnull!=0].sort_values(ascending=False)



    #Eğer shape[0] 0 ise eksik değere sahip özellik yok demektir.

    if check_isnull.shape[0]==0:

        print("Veri setinde eksik bilgi yoktur")

        print(check_isnull)

    else:

        print(check_isnull)

        f, ax = plt.subplots(figsize=(15, 6))

        plt.xticks(rotation='90')

        sns.barplot(x=check_isnull.index, y=check_isnull)

        plt.title("Eksik veri içeren özellilere ait eksik veri sayısı")
#train veri seti için eksik bilgiler kontrol ediliyor. 

show_missing_values(train)
#test veri seti için eksik bilgiler kontrol ediliyor. 

show_missing_values(test)
def show_correlation(corr_data, corr_feature_name, n_most_correlated=12):

    """

    corr_data: korelasyon matrisi

    corr_feature_name: korelasyonu gösterilmek istenen özellik

    n_most_correlated: en yüksek korelasyona sahip özelliklerden gösterileceklerin sayısı

    """

    #abs() fonksiyonu negatif korelasyonları pozitif yapıyor

    corr=corr_data.corr().abs()

    most_correlated_feature=corr[corr_feature_name].sort_values(ascending=False)[:n_most_correlated].drop(corr_feature_name)

    most_correlated_feature_name=most_correlated_feature.index.values

    

    f, ax = plt.subplots(figsize=(15, 6))

    plt.xticks(rotation='90')

    sns.barplot(y=most_correlated_feature_name, x=most_correlated_feature)

    plt.title("{} ile en fazla korelasyona sahip özellikler".format(corr_feature_name))
show_correlation(corr_data=train, n_most_correlated=12, corr_feature_name='signal')
show_correlation(corr_data=train, n_most_correlated=12, corr_feature_name='production')
def add_features(df):

    # features used by https://www.kaggle.com/sionek/ugbc-gs

    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])

    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3

    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)

   

    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2

    # features from phunter

    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']

    df['NEW_IP_dira'] = df['IP']*df['dira']

    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']

    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']

    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)

    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)

    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)

    # My:

    # new combined features just to minimize their number;

    # their physical sense doesn't matter

    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']

    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']

    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']

    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']

    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']

    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']

    #My:

    # "super" feature changing the result from 0.988641 to 0.991099

    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']

    return df
train_added = add_features(train)

test_added = add_features(test)

print("Yeni özellikler eklendi")
show_correlation(corr_data=train_added, n_most_correlated=12, corr_feature_name='signal')


# Veri setinden çıkartılacak özellikler

filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',

              'SPDhits','CDF1', 'CDF2', 'CDF3',

              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',

              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',

              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',

              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',

              'p0_IP', 'p1_IP', 'p2_IP',

              'IP_p0p2', 'IP_p1p2',

              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',

              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',

              'DOCAone', 'DOCAtwo', 'DOCAthree']



#filter_out.remove('production')

features = list(f for f in train_added.columns if f not in filter_out)

print("Kullanılacak Özellikler")

print(features)
train_added[features].head()
X=train_added[features+['mass','production']]

y=train['signal']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3)
#UGradientBoostingClassifier eğitiliyor

loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0 , fl_coefficient=15, power=2)

ugbc = UGradientBoostingClassifier(loss=loss, n_estimators=550,

                                 max_depth=6,

                                 learning_rate=0.15,

                                 train_features=features,

                                 subsample=0.7,

                                 random_state=123)

print("Model eğitiliyor...")

ugbc.fit(X_train, y_train)

print("Model eğitildi")
y_pred=ugbc.predict(X_test)

print("Tahmin yapıldı")
from sklearn import metrics
print("Başarı oranı:\n",metrics.accuracy_score(y_test, y_pred))

print("Karışıklık Matrisi:\n",metrics.confusion_matrix(y_test, y_pred))

print("Sınıflandırma Raporu:\n",metrics.classification_report(y_test, y_pred))
#minMaxScaler.fit(test_added[features])





print ('----------------------------------------------')

print("Test veri seti üzerinde tahmin yapalıyor")

#yarışma sonlandığı için bu kısma gerek kalmadı. test verileri üzerinde 0.998195 sonuç vermiştir.

#X_test=test_added[features]

#test_probs = ugbc.predict_proba(X_test)[:,1]

#submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})

#submission.to_csv("sumbision_MinMax_mass_production.csv", index=False)

print('Tahmin yapıldı')
import pickle
file_name="ml_model.sav"

pickle.dump(ugbc, open(file_name, "wb"))
print(os.listdir("../working"))
loaded_model=pickle.load(open("../working/ml_model.sav",'rb'))
y_pred=loaded_model.predict(X_test)

print("Tahmin yapıldı")
print("Başarı oranı:\n",metrics.accuracy_score(y_test, y_pred))

print("Karışıklık Matrisi:\n",metrics.confusion_matrix(y_test, y_pred))

print("Sınıflandırma Raporu:\n",metrics.classification_report(y_test, y_pred))