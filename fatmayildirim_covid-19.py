# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
world = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
world.head(-5)
# Satir Sayisi
print("Satır Sayısı:\n",world.shape[0:])

# Sutun Adlari
print("Sütun Adlari:\n",world.columns.tolist())

# Veri Tipleri
print("Veri Tipleri:\n",world.dtypes)
# Eksik veri sayıları ve veri setindeki oranları 
import matplotlib.pyplot as plt
import seaborn as sns
pd.concat([world.isnull().sum(), 100 * world.isnull().sum()/len(world)], 
              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
world["WHO Region"].fillna("Other", inplace = True)  
world.isnull().sum()
world['WHO Region'].unique()
world.describe().T
# Veri seti içerisinden belli alanlar seçilerek yeni bir veriseti oluşturuldu.
df1=pd.Series(world['Country/Region'],name="Country")
df2=pd.Series(world['Date'],name="Date")
df3=pd.Series(world['Confirmed'],name="Confirmed")
df4=pd.Series(world['Deaths'],name="Deaths")
df5=pd.Series(world['Recovered'],name="Recovered")
df_world=pd.concat([df1, df2,df3, df4,df5], axis=1)
#türkiye için korelasyon grafiği
turkey=df_world.copy()
turkey_values = (turkey['Country'] == 'Turkey').astype(int)
fields = list(turkey.columns[1:])  # everything except "country name"
correlations = turkey[fields].corrwith(turkey_values)
correlations.sort_values(inplace=True)
correlations
ax = correlations.plot(kind='bar')
ax.set(ylim=[0, 0.5], ylabel='turkey correlation');
plt.figure()
df_world.boxplot(column=['Confirmed','Deaths','Recovered'])

fig,axs=plt.subplots(2,2) 
axs[0, 0].boxplot(df_world['Confirmed'])
axs[0, 0].set_title('Hasta Sayısı')

axs[0, 1].boxplot(df_world['Recovered'])
axs[0, 1].set_title('İyileşen Hasta Sayısı')

axs[1, 0].boxplot(df_world['Deaths'])
axs[1, 0].set_title('Hayatını Kaybeden Hasta Sayısı')
# Enlem ve boylam değerlerinin de olduğu yeni bir dataframe oluşturuldu. 
# Değerler 1/22/2020 - 25/05/2020 tarihleri aralığını içermektedir.
df_1=df_world
df_2=pd.Series(world['Long'],name="Long")
df_3=pd.Series(world['Lat'],name="Lat")
df_4=pd.Series(world['WHO Region'],name="Region")
df_location=pd.concat([df_1,df_2,df_3,df_4], axis=1)

df_location.head()
# Zaman İçerisindeki Değişim
import plotly.express as px
fig = px.choropleth(df_location, locations="Country", locationmode='country names', color=np.log(df_location["Confirmed"]), 
                    hover_name="Country", animation_frame=df_location["Date"],
                    title='Zaman İçerisindeki Değişim', color_continuous_scale=px.colors.sequential.Purp)
fig.update(layout_coloraxis_showscale=False)
fig.show()
import folium
# World wide
temp = df_location[df_location['Date'] == max(df_location['Date'])]
m = folium.Map(location=[0, 0], titles='Dünya Haritası Üzerinde Değerler',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(temp)):
    folium.Circle(
        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],
        color='crimson', fill='crimson',
        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country'])+
                    '<li><bold>Province : '+str(temp.iloc[i]['Region'])+
                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths']),
        radius=int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)
m
import plotly.express as px
fig = px.bar(df_location.sort_values("Confirmed"),
            x='Region', y="Confirmed",
            hover_name="Region",
            hover_data=["Recovered","Deaths","Confirmed"],
            title='COVID-19: Test Sonucu Pozitif Olan Hasta Sayısı Bölgelere Göre',
)
fig.update_xaxes(title_text="Region")
fig.update_yaxes(title_text="Positif Test Sayısı(%)")
fig.show()
fig = px.bar(df_location.sort_values("Recovered"),
            x='Region', y="Recovered",
            hover_name="Region",
            hover_data=["Confirmed","Deaths","Recovered"],
            title='COVID-19: İyileşen Hasta Sayısı Bölgelere Göre',
)
fig.update_xaxes(title_text="Region")
fig.update_yaxes(title_text="İyileşen Hasta Sayısı")
fig.show()
fig = px.bar(df_location.sort_values("Deaths"),
            x='Region', y="Deaths",
            hover_name="Region",
            hover_data=["Confirmed","Recovered","Deaths"],
            title='COVID-19: Hayatını Kaybeden Hasta Sayısı Bölgelere Göre ',
)
fig.update_xaxes(title_text="Region")
fig.update_yaxes(title_text="Hayatını Kaybeden Hasta Sayısı")
fig.show()
from sklearn.model_selection import train_test_split


X = df_world.iloc[:,2:5]
y = df_world['Recovered']

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=0)
X_train = mms.fit_transform(X_train) 
X_test= mms.fit_transform(X_test)
print("Dataframe boyutu: ",df_world.shape)
print("Eğitim verisi boyutu: ",X_train.shape, y_train.shape)
print("Test verisi boyutu: ",X_test.shape,y_test.shape)
# type error için target typesı "Label Encoder" ile  multiclassa çevirdim.(Target=Y_train)
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(encoded))

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(y_train)

from sklearn    import metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import  linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Her bir modelin doğruluk değeri ,sınıflandırma raporu , karışıklık matrisi ve MSE(Ortalama Kare Hata Regresyon Oranı) değerlerini hesaplamak için import edildi.
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
# Lineer Regresyon
print("\nLineer Regresyon")
lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
y_true1 , y_pred1 =y_test,lm.predict(X_test)
print("\nTahmin değerleri: ",y_pred1)
plt.scatter(y_true1, y_pred1,c='orange')
plt.scatter(y_true1, y_test,c='green')
plt.xlabel("True Values")
plt.ylabel("Predictions")
#Lineer Regresyon
#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v = lab_enc.fit_transform(y_true1)
utils.multiclass.type_of_target(y_true1.astype('int'))
ypred1= lab_enc.fit_transform(y_pred1)
utils.multiclass.type_of_target(ypred1.astype('int'))
conf=confusion_matrix(encoded_v, ypred1)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v, ypred1))
print("\nClassification Report:\n",classification_report(encoded_v, ypred1))
print("MSE:",mean_squared_error(encoded_v, ypred1))
# SVR(Support Vector Regressions)
print("SVR(Support Vector Regressions)")
clf = svm.SVR(gamma="auto")
# modelimizi eğitim verilerimiz ve buna karşılık gelen Y_train(target ) değerleri ile eğittik
clf.fit(X_train, Y_train)
# test değerlerimize karşılık gelecek olan tahmin değerlerimizi oluşturduk
y_true2 , y_pred2 =y_test,clf.predict(X_test)
print("\nTahmin değerleri: ",y_pred2)
plt.scatter(y_true2, y_pred2,c='black')
plt.scatter(y_true2, y_test,c='green')
plt.xlabel("True Values")
plt.ylabel("Predictions")
#SVR
#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v1 = lab_enc.fit_transform(y_true2)
utils.multiclass.type_of_target(y_true2.astype('int'))
ypred2= lab_enc.fit_transform(y_pred2)
utils.multiclass.type_of_target(ypred2.astype('int'))
conf=confusion_matrix(encoded_v1, ypred2)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v1, ypred2))
print("\nClassification Report:\n",classification_report(encoded_v1, ypred2))
print("MSE:",mean_squared_error(encoded_v1, ypred2))
# GaussianNB
print("GaussianNB")
clf = GaussianNB()
clf.fit(X_train, Y_train)
y_true3 , y_pred3=y_test,clf.predict(X_test)
print("\nTahmin değerleri: ",y_pred3)
plt.scatter(y_true3, y_pred3,c='grey')
plt.scatter(y_true3, y_test,c='green')
plt.xlabel("True Values")
plt.ylabel("Predictions")
# GaussianNB
#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v2 = lab_enc.fit_transform(y_true3)
utils.multiclass.type_of_target(y_true3.astype('int'))
ypred3= lab_enc.fit_transform(y_pred3)
utils.multiclass.type_of_target(ypred3.astype('int'))
conf=confusion_matrix(encoded_v2, ypred3)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v2, ypred3))
print("\nClassification Report:\n",classification_report(encoded_v2, ypred3))
print("MSE:",mean_squared_error(encoded_v2, ypred3))
# Decision Tree Classifier
print("Decision Tree Classifier")
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_true5 , y_pred5=y_test,clf.predict(X_test)
print("\nTahmin değerleri: ",y_pred5)
plt.scatter(y_true5, y_pred5,c='brown')
plt.scatter(y_true5, y_test,c='green')
plt.xlabel("True Values")
plt.ylabel("Predictions")
#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v4 = lab_enc.fit_transform(y_true5)
utils.multiclass.type_of_target(y_true5.astype('int'))
ypred5= lab_enc.fit_transform(y_pred5)
utils.multiclass.type_of_target(ypred5.astype('int'))
conf=confusion_matrix(encoded_v4, ypred5)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v4, ypred5))
print("\nClassification Report:\n",classification_report(encoded_v4, ypred5))
print("MSE:",mean_squared_error(encoded_v4, ypred5))
# KNeighborsClassifier
print("KNeighbors Classifier")
clf = KNeighborsClassifier()
clf.fit(X_train, Y_train)
y_true7 , y_pred7=y_test,clf.predict(X_test)
print("\nTahmin değerleri: ",y_pred7)
plt.scatter(y_true7, y_pred7,c='blue')
plt.scatter(y_true7, y_test,c='green')
plt.xlabel("True Values")
plt.ylabel("Predictions")

#predictions multiclass olduğundan y_validation da multiclassa dönüştürüldü
encoded_v6 = lab_enc.fit_transform(y_true7)
utils.multiclass.type_of_target(y_true7.astype('int'))
ypred7= lab_enc.fit_transform(y_pred7)
utils.multiclass.type_of_target(ypred7.astype('int'))
conf=confusion_matrix(encoded_v6, ypred7)
print("\nConfusion matrix :\n",conf)
print("Accuracy score(Doğruluk değeri):\n",accuracy_score(encoded_v6, ypred7))
print("\nClassification Report:\n",classification_report(encoded_v6, ypred7))
print("MSE:",mean_squared_error(encoded_v6, ypred7))
