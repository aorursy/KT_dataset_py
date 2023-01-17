# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import plotly.graph_objects as go

import seaborn as sns

import plotly.express as px



from sklearn import preprocessing

from sklearn import utils



from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import  linear_model

from sklearn.model_selection import KFold







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

earthquake = pd.read_csv("../input/earthquake/earthquake.csv")

print(earthquake)  # bütün verileri listeleme
df = pd.DataFrame(earthquake)  

df 
df.head() #ilk 5 sırayı listeleme

df.info() #df.info komutu kullandığımızda ekran çıktımızda verilerimizin 3.1 Mb yer kapladığını gördük
df[['richter','city']] # deprem büyüklükleri ve şehirleri yazma
df[(df['richter']>5) & (df['city'])] # büyüklüğü 5 den fazla olan depremleri sıralama
df.sort_values('richter', axis = 0, ascending = False) #depremleri büyükten küçüğe sıralama
df.groupby('city').size() #şehir bazında deprem sayısı görüntüleme
df['city'] = df['city'].astype(str) #none deger varsa onlarında satısını yazmak için dönüşüm

df.groupby('city').size()  #şehir bazında deprem sayısı görüntüleme 
df.groupby('city')['richter'].apply(lambda x: np.mean(x))

# şehir bazında alınan büyüklüklerin ortalamasını görüntüleme

df['richter'].mode() #deprem büyüklüklerinin modu
df.mean(axis = 0, skipna = True) # özniteliklerin aritmetik ortalaması
df['richter'].std() #büyüklüklerin standart sapması
df.cov() # özniteliklerin birbirleriyle olan ilişkilerini gösterir (kovaryans)
df.plot(x='city', y='richter', style='-') #şehir lerin büyüklüklerine göre tablolama
import seaborn as sns

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values) #Korelasyon gösterimi
df.isnull().sum().sum()

#Toplam kaç hücrede eksik değer (NaN ya da None) var?
df.isnull().sum()

# Özniteliklerin değer almadığı kaç satır var?

# Eksik Deger Tablosu Olusturalım



def eksik_deger_tablosu(df):

    eksik_deger=df.isnull().sum()

    eksik_deger_yuzde=100* df.isnull().sum()/len(df)

    eksik_deger_tablo= pd.concat([eksik_deger,eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son=eksik_deger_tablo.rename(

    columns = {0 : 'Eksik Değerler',1: '% Değeri'} )

    return eksik_deger_tablo_son



eksik_deger_tablosu(df)
# % 79 eksik deger oranı olan mw göze cok carpıyor,bu kadar eksik veri oldugu için bu öz nitelik büyük ihtimalle fazla işe yaramayacaktır.

# Bu öz niteliği veri kümeinden temizlemek mantıklı olabilir.

# Belirli bir eşik değerin üzerinde, örneğin %70, eksik değer olan öznitelikleri veri kümesinden kaldıralım.

# Bu stratejiyi hayata geçirmek için “df” dataframe’i üzerinde “dropna()” fonksiyonunu threshold (eşik) değeri alacak şekilde kullanabiliriz.

#%70 üzerinde null değer içeren kolonları sil

#tr = len(df) * .3

#df.dropna(thresh = tr, axis = 1, inplace = True)



#df

# diğer boş olan özniteliklerden area,direction ve dist yerine ne yazılabilr ?
#Apply fonksiyonu kullanarak sınav başarı durumunu yeni öznitelik olarak ekle

def siddet(richter):

    return (richter >= 6)



df['siddeti 6 dan büyük olanlar'] = df['richter'].apply(siddet)

df
# date özniteliğindeki yıl bilgisini kullanarak 'Yıl' isimli yeni bir öznitelik oluşturuyoruz

tarih = pd.to_datetime(df['date'])

df['Yıl'] = tarih.dt.year



df
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

df['Şiddet_Encoded']= label_encoder.fit_transform(df['siddeti 6 dan büyük olanlar'])



df
#richter özniteliğini ölçeklendirmek istiyoruz

from sklearn import preprocessing

x = df[['richter']].values.astype(float)



#Ölçeklendirme için MinMaxScaler fonksiyonunu kullanıyoruz.

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['richter2'] = pd.DataFrame(x_scaled)



df



import seaborn as sns

sns.boxplot(x=df['richter'])
Q1 = df.richter.quantile(0.25)

Q2 = df.richter.quantile(0.5)

Q3 = df.richter.quantile(0.75)

Q4 = df.richter.quantile(1)

IQR = Q3 - Q1



print("Q1-->", Q1)

print("Q3-->", Q3)

print("Q2-->", Q2)

print("Q4-->", Q4)

print("IQR-->", IQR)

print("Alt sınır: Q1 - 1.5 * IQR--->", Q1 - 1.5 * IQR)

print("Üst sınır: Q3 + 1.5 * IQR--->", Q3 + 1.5 * IQR)

def yeardate(x):

    return x[0:4]

df["yeardate"] = df.date.apply(yeardate)

#We must change object to integer.

df['yeardate'] = df.yeardate.astype(int)

print(df.yeardate.dtypes)

df.head(3)
#ÇIKARIMLAR

#Türkiye'de hangi YIL deprem en çok yaşandı?

#En büyük deprem neredeydi?

#Hangi ülke deprem oldu?

#Deprem ne kadar sürdü?

#En şiddetli deprem nerede ve ne zaman meydana geldi?
# Which year had the most earthquakes in Turkey?

df.yeardate.plot(kind = "hist" , color = "red" , edgecolor="black", bins = 100 , figsize = (12,12) , label = "Earthquakes frequency")

plt.legend(loc = "upper right")

plt.xlabel("Years")

plt.show()

# Where was the most earthquake?

df.city.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)

plt.xlabel("City",fontsize=18,color="blue")

plt.ylabel("Frequency",fontsize=18,color="blue")

plt.show()
# Which country-area was the most earthquake? Hangi Ülkede Daha Çok Deprem Oldu

df.country.value_counts().plot(kind = "bar" , color = "red" , figsize = (30,10),fontsize = 20)

plt.xlabel("Country",fontsize=18,color="blue")

plt.ylabel("Frequency",fontsize=18,color="blue")

plt.show()
#En Yüksek Şiddetli Deprem Nerede?

filtre = df.richter == df.richter.max()

df[filtre]
# En şiddetli deprem nerede ve ne zaman meydana geldi?

df.xm.max()

filtering = df.country == "turkey"

filtering2 = df.xm == 7.9

df[filtering & filtering2]
#Deprem - Büyüklük seviyesi

threshold = sum(df.xm) / len(df.xm)

df["magnitude-level"] = ["hight" if i > threshold else "low" for i in df.xm]

df.loc[:10,["magnitude-level","xm","city"]]
# derinliği en düşük olan deprem

#df[['date','yeardate','country','richter','depth']]



filtre = df.depth == df.depth.min()

#df[filtre]

df.loc[filtre,['date','yeardate','country','richter','depth']]
df.country.value_counts()

x = df.country.value_counts().index

y = df.country.value_counts().values 

f, (ax1) = plt.subplots(figsize=(15, 30), sharex=True)

sns.barplot(x=x, y=y, palette="rocket", ax=ax1)

plt.xticks(rotation = 90 , color='white')

ax1.axhline(0, color="k", clip_on=False)

bool_turkey = df.country == 'turkey'

turkey = df[bool_turkey]

turkey.city.value_counts()

f, (ax1) = plt.subplots(figsize=(16, 16), sharex=True)

sns.barplot(x=turkey.city.value_counts().index, y=turkey.city.value_counts().values, palette="ch:2.5,-.2,dark=.3", ax=ax1)

plt.xticks(rotation = 90 , color='black')

ax1.axhline(0, color="k", clip_on=False)
import plotly.express as px

turkey.direction

ax = sns.barplot(x=turkey.direction.value_counts().index, y=turkey.direction.value_counts().values,palette="ch:2.5,-.2,dark=.3" )

plt.xticks(rotation=90 , color ='black')

plt.yticks(rotation=0 , color ='black')

plt.show()
fig = px.scatter_mapbox(turkey, lat="lat", lon="long", hover_name="city", hover_data=["depth", "richter",'direction'],

                        color_discrete_sequence=["light green"], zoom=5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
turkey['date'] = pd.to_datetime(turkey['date'])

turkey['year'] = turkey['date'].dt.year

labels_turkey = turkey.year.value_counts().index

plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(turkey.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(turkey.year.value_counts().index) ,y = turkey.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()
boolen_mediterranean = df['country'] == 'mediterranean'

mediterranean = df[boolen_mediterranean]

mediterranean
#Akdeniz Deprem Yılları

mediterranean['date'] = pd.to_datetime(mediterranean['date'])

mediterranean['year'] = mediterranean['date'].dt.year

plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(mediterranean.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(mediterranean.year.value_counts().index) ,y = mediterranean.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()
# Akdeniz'in Deprem Haritası

fig = px.scatter_mapbox(mediterranean, lat="lat", lon="long", hover_name="city", hover_data=["depth", "richter",'direction'],

                        color_discrete_sequence=["light green"], zoom=5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
#Tüm Deprem Tarihleri



df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year

labels = df.year.value_counts().index

plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(df.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(df.year.value_counts().index) ,y = df.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()
#Ölçeklerin Tüm Değer yoğunlukları

long_data = zip(df.dist,df.depth,df.xm,df.md,df.richter,df.mw,df.ms,df.mb)

data_long = pd.DataFrame(long_data,columns=["dist","depth","xm","md","richter","mw","ms","mb"])

data_long.head()

data_long.dist = data_long.dist /max(data_long.dist)

data_long.depth = data_long.depth /max(data_long.depth)

data_long.xm = data_long.xm /max(data_long.xm)

data_long.md = data_long.md /max(data_long.md)

data_long.richter = data_long.richter /max(data_long.richter)

data_long.mw = data_long.mw /max(data_long.mw)

data_long.ms = data_long.ms /max(data_long.ms)

data_long.mb = data_long.mb /max(data_long.mb)



plt.figure(figsize = (32,32))

pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

sns.violinplot(data = data_long,palette = pal , inner = "points" )
df.loc[df['richter'] > 8, 'Sınıf'] = 'İyi'

df.loc[ (df['richter'] >= 7) & (df['richter'] < 7.9), 'Sınıf'] = 'Önemli'

df.loc[ (df['richter'] >= 6) & (df['richter'] < 6.9), 'Sınıf'] = 'Güçlü'

df.loc[ (df['richter'] >= 5.5) & (df['richter'] < 5.9), 'Sınıf'] = 'Ilımlı'
# Magnitude Class distribution



sns.countplot(x="Sınıf", data=df)

plt.ylabel('Sıklık')

plt.title('Büyüklük Sınıfı VS Sıklık')
# boş degerlerin hepsini sildim

df.dropna(how="any",inplace=True) 
df.isnull().sum()
df
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

df['Country_Encoded']= label_encoder.fit_transform(df['country'])

df['City_Encoded']= label_encoder.fit_transform(df['city'])

df['Area_Encoded']= label_encoder.fit_transform(df['area'])

df['Direction_Encoded']= label_encoder.fit_transform(df['direction'])

df['Magnitude-level_Encoded']= label_encoder.fit_transform(df['magnitude-level'])

df['Yıl_Encoded']= label_encoder.fit_transform(df['Yıl'])



df['Id_Encoded']= label_encoder.fit_transform(df['id'])

df['date_Encoded']= label_encoder.fit_transform(df['date'])

df['time_Encoded']= label_encoder.fit_transform(df['time'])



df['lat_Encoded']= label_encoder.fit_transform(df['lat'])

df['long_Encoded']= label_encoder.fit_transform(df['long'])

df['dist_Encoded']= label_encoder.fit_transform(df['dist'])

df['depht_Encoded']= label_encoder.fit_transform(df['depth'])

df['xm_Encoded']= label_encoder.fit_transform(df['xm'])

df['md_Encoded']= label_encoder.fit_transform(df['md'])

df['yeardate_Encoded']= label_encoder.fit_transform(df['yeardate'])



df['mw_Encoded']= label_encoder.fit_transform(df['mw'])

df['ms_Encoded']= label_encoder.fit_transform(df['ms'])

df['mb_Encoded']= label_encoder.fit_transform(df['mb'])

df['year_Encoded']= label_encoder.fit_transform(df['year'])



df['xm_Encoded']= label_encoder.fit_transform(df['xm'])

df['md_Encoded']= label_encoder.fit_transform(df['md'])

df['richter_Encoded']= label_encoder.fit_transform(df['richter'])









df
df =df.drop(columns ='country')

df =df.drop(columns ='city')

df =df.drop(columns ='area')

df =df.drop(columns ='direction')

df =df.drop(columns ='magnitude-level')

df =df.drop(columns ='yeardate')

df =df.drop(columns ='year')

df =df.drop(columns ='Yıl')

df =df.drop(columns ='date')

df =df.drop(columns ='time')

df =df.drop(columns ='siddeti 6 dan büyük olanlar')

df =df.drop(columns ='id')

df =df.drop(columns ='lat')

df =df.drop(columns ='long')

df =df.drop(columns ='dist')

df =df.drop(columns ='mw')

df =df.drop(columns ='ms')

df =df.drop(columns ='mb')

df =df.drop(columns ='depth')

df =df.drop(columns ='xm')

df =df.drop(columns ='md')

df =df.drop(columns ='richter')

df =df.drop(columns ='richter2')



df =df.drop(columns ='Country_Encoded')

#df =df.drop(columns ='City_Encoded')

df =df.drop(columns ='Area_Encoded')

df =df.drop(columns ='Direction_Encoded')

df =df.drop(columns ='Magnitude-level_Encoded')

#df =df.drop(columns ='Yıl_Encoded')

df =df.drop(columns ='date_Encoded')

df =df.drop(columns ='time_Encoded')

df =df.drop(columns ='Id_Encoded')

df =df.drop(columns ='lat_Encoded')

df =df.drop(columns ='long_Encoded')

df =df.drop(columns ='dist_Encoded')

df =df.drop(columns ='mw_Encoded')

df =df.drop(columns ='ms_Encoded')

df =df.drop(columns ='mb_Encoded')

#df =df.drop(columns ='depht_Encoded')

df =df.drop(columns ='xm_Encoded')

df =df.drop(columns ='md_Encoded')

df =df.drop(columns ='yeardate_Encoded')

df =df.drop(columns ='year_Encoded')

df =df.drop(columns ='Şiddet_Encoded')

#df =df.drop(columns ='richter_Encoded')







df
array = df.values

X = array[:,1:5]

y = array[:,0:1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)



print("Dataframe boyutu: ",df.shape)

print("Eğitim verisi boyutu: ",X_train.shape, Y_train.shape)

print("Test verisi boyutu: ",X_validation.shape, Y_validation.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

#Decision Trees

cellTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

print(cellTree) # it shows the default parameters

  #I fit the data with the training

cellTree.fit(X_train,Y_train)

  #now predictions

yhat_dt = cellTree.predict(X_validation)



  #Accuracy evaluation

acc = metrics.accuracy_score(Y_validation, yhat_dt)

print('karar agaci icin accuracy: ',acc)



#karar agaci icin confusion matrix ve metrik degerler

cellTree_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_dt = cross_val_score(cellTree_dt, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_dt)

print('cv_scores mean:{}'.format(np.mean(cv_scores_dt)))

from sklearn.metrics import classification_report

prec_dt = classification_report(yhat_dt,Y_validation)

print(prec_dt)
#call the models

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 3)

# fit the models

neigh = knn_model.fit(X_train,Y_train)

#predict the mode;

yhatknn=neigh.predict(X_validation)



  #Accuracy evaluation

accknn = metrics.accuracy_score(Y_validation, yhatknn)

print('en yakin komsular icin accuracy',accknn)



#knn=3 icin confusion matrix ve metrik degerler

knn_knn = KNeighborsClassifier(n_neighbors = 3)

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_knn = cross_val_score(knn_knn, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_knn)

print('cv_scores mean:{}'.format(np.mean(cv_scores_knn)))



#knn scores

from sklearn.metrics import classification_report

prec_knn = classification_report(yhatknn,Y_validation)

print(prec_knn)
#lojistik regresyon https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)

LR

#predict

yhatlr = LR.predict(X_validation)

#print('yhat', yhat)

  #Accuracy evaluation

acclr = metrics.accuracy_score(Y_validation, yhatlr)

print('lojistik regresyon icin accuracy',acclr)





#lojistik regresyon icin confusion matrix ve metrik degerler

lr_lr = LogisticRegression(C=0.01, solver='liblinear')

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_lr = cross_val_score(lr_lr, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_lr)

print('cv_scores mean:{}'.format(np.mean(cv_scores_lr)))





from sklearn.metrics import classification_report

prec_lr = classification_report(yhatlr,Y_validation)

print(prec_lr)
#support vector machines https://scikit-learn.org/stable/modules/svm.html

#SVM 

from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(X_train, Y_train) 

#predict

yhatsvm = clf.predict(X_validation)

#yhat [0:5]

accsvm = metrics.accuracy_score(Y_validation, yhatsvm)

print('svm icin accuracy',accsvm)







#svm icin confusion matrix ve metrik degerler

clf_svm = svm.SVC(kernel='rbf')

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_svm = cross_val_score(clf_svm, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_svm)

print('cv_scores mean:{}'.format(np.mean(cv_scores_svm)))





from sklearn.metrics import classification_report

prec_svm = classification_report(yhatsvm,Y_validation)

print(prec_svm)
#gaussian NB https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

#call the models

gnb = GaussianNB()

  #fit the model

gnb.fit(X_train, Y_train) 

  #predict

yhatgnb = gnb.predict(X_validation)

accgnb = metrics.accuracy_score(Y_validation, yhatgnb)

print('gaussian naive bayes icin accuracy',accgnb)





#gaussian naive bayes icin confusion matrix ve metrik degerler

clf_gnb = GaussianNB()

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_gnb = cross_val_score(clf_gnb, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_gnb)

print('cv_scores mean:{}'.format(np.mean(cv_scores_gnb)))



#klasifikasyon tablosu

from sklearn.metrics import classification_report

prec_gnb = classification_report(yhatgnb,Y_validation)

print(prec_gnb)
#linear discriminant analysis 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

#fit the model

lda.fit(X_train, Y_train) 

#predict

yhatlda = lda.predict(X_validation)

acclda = metrics.accuracy_score(Y_validation, yhatlda)

print('linear discriminant analiz icin accuracy',acclda)









#linear discrimant icin confusion matrix ve metrik degerler

clf_ld = LinearDiscriminantAnalysis()

#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik

cv_scores_ld = cross_val_score(clf_ld, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores_ld)

print('cv_scores mean:{}'.format(np.mean(cv_scores_ld)))



#klasifikasyon linear diskrimannt

from sklearn.metrics import classification_report

prec_lda = classification_report(yhatlda,Y_validation)

print(prec_lda)
#buda gordugun gibi calisiyor. tek tek calistir modelleri fonksiyon verme sonra karsilastir...



# RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(max_depth=5, n_estimators=100, max_features='auto')

rfc.fit(X_train, Y_train) 

#predict

yhat1 = rfc.predict(X_validation)

#yhat [0:5]

#evaluate



#create a new SVM model

rfc_cv = RandomForestClassifier(max_depth=5, n_estimators=100, max_features='auto')

#train model with cv of 10

cv_scores = cross_val_score(rfc_cv, X,y, cv=10)

#print each cv score (accuracy) and average them

print(cv_scores)

print('cv_scores mean:{}'.format(np.mean(cv_scores)))











from sklearn.metrics import classification_report, confusion_matrix

import itertools

from sklearn.metrics import f1_score

print('f1_score for Random Forest Classifier:',f1_score(Y_validation, yhat1, average='weighted'))

#print("Train set Accuracy for Random Forest Classifier: ", metrics.accuracy_score(Y_validation, rfc.predict(X_train)))

#print("Test set Accuracy for Random Forest Classifier: ", metrics.accuracy_score(Y_validation, yhat1))

from sklearn.metrics import classification_report

prec_rec = classification_report(yhat1,Y_validation)

print(prec_rec)