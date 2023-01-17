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

covid=pd.read_csv("../input/covid19-global-and-regional/covid_19_clean_complete.csv")

covid.head()
covid.isnull().sum()
# Satir Sayisi

print("Satır Sayısı:\n",covid.shape[0:])



# Sutun Adlari

print("Sütun Adlari:\n",covid.columns.tolist())



# Veri Tipleri

print("Veri Tipleri:\n",covid.dtypes)
# Eksik veri sayıları ve veri setindeki oranları 

import matplotlib.pyplot as plt

import seaborn as sns

pd.concat([covid.isnull().sum(), 100 * covid.isnull().sum()/len(covid)], 

              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
covid.shape
df1=pd.Series(covid['Date'],name="Date")

df2=pd.Series(covid['Country/Region'],name="Country_Region")

df3=pd.Series(covid['Lat'],name="Lat")

df4=pd.Series(covid['Long'],name="Long")

df5=pd.Series(covid['Recovered'],name="Recovered")

df6=pd.Series(covid['Deaths'],name="Deaths")

df7=pd.Series(covid['Confirmed'],name="Confirmed")

df_covid=pd.concat([df1, df2,df3, df4,df5,df6,df7], axis=1)

df_covid.describe().T
df_covid.head()
#Çin için korelasyon grafiği



chna=covid.copy()

y = (chna['Country/Region'] == 'China').astype(int)

fields = list(chna.columns[:-1])  # everything except "country name"

correlations = chna[fields].corrwith(y)

correlations.sort_values(inplace=True)

correlations
ax = correlations.plot(kind='bar')

ax.set(ylim=[-1, 1], ylabel='China Correlation');
plt.figure()

df_covid.boxplot(column=['Confirmed','Deaths','Recovered'])



fig,axs=plt.subplots(2,2) 

axs[0, 0].boxplot(df_covid['Confirmed'])

axs[0, 0].set_title('Hasta Sayısı')



axs[0, 1].boxplot(df_covid['Recovered'])

axs[0, 1].set_title('İyileşen Hasta Sayısı')



axs[1, 0].boxplot(df_covid['Deaths'])

axs[1, 0].set_title('Hayatını Kaybeden Hasta Sayısı')
who_region = {}



# African Region 

africa = "Algeria, Angola, Cabo Verde, Eswatini, Sao Tome and Principe, Benin, South Sudan, Western Sahara, Congo (Brazzaville), Congo (Kinshasa), Cote d'Ivoire, Botswana, Burkina Faso, Burundi, Cameroon, Cape Verde, Central African Republic, Chad, Comoros, Ivory Coast, Democratic Republic of the Congo, Equatorial Guinea, Eritrea, Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mauritius, Mozambique, Namibia, Niger, Nigeria, Republic of the Congo, Rwanda, São Tomé and Príncipe, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, Swaziland, Togo, Uganda, Tanzania, Zambia, Zimbabwe"

africa = [i.strip() for i in africa.split(',')]

for i in africa:

    who_region[i] = 'africa'

    

# Region of the Americas 

america = 'Antigua and Barbuda, Argentina, Bahamas, Barbados, Belize, Bolivia, Brazil, Canada, Chile, Colombia, Costa Rica, Cuba, Dominica, Dominican Republic, Ecuador, El Salvador, Grenada, Guatemala, Guyana, Haiti, Honduras, Jamaica, Mexico, Nicaragua, Panama, Paraguay, Peru, Saint Kitts and Nevis, Saint Lucia, Saint Vincent and the Grenadines, Suriname, Trinidad and Tobago, United States, US, Uruguay, Venezuela'

america = [i.strip() for i in america.split(',')]

for i in america:

    who_region[i] = 'america'



# South-East Asia Region 

asia = 'Bangladesh, Bhutan, North Korea, India, Indonesia, Maldives, Myanmar, Burma, Nepal, Sri Lanka, Thailand, Timor-Leste'

asia = [i.strip() for i in asia.split(',')]

for i in asia:

    who_region[i] = 'asia'



# European Region 

euro = 'Albania, Andorra, Greenland, Kosovo, Holy See, Liechtenstein, Armenia, Czechia, Austria, Azerbaijan, Belarus, Belgium, Bosnia and Herzegovina, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Georgia, Germany, Greece, Hungary, Iceland, Ireland, Israel, Italy, Kazakhstan, Kyrgyzstan, Latvia, Lithuania, Luxembourg, Malta, Monaco, Montenegro, Netherlands, North Macedonia, Norway, Poland, Portugal, Moldova, Romania, Russia, San Marino, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Tajikistan, Turkey, Turkmenistan, Ukraine, United Kingdom, Uzbekistan'

euro = [i.strip() for i in euro.split(',')]

for i in euro:

    who_region[i] = 'euro'



# Eastern Mediterranean Region 

emro = 'Afghanistan, Bahrain, Djibouti, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Pakistan, Palestine, West Bank and Gaza, Qatar, Saudi Arabia, Somalia, Sudan, Syria, Tunisia, United Arab Emirates, Yemen'

emro = [i.strip() for i in emro.split(',')]

for i in emro:

    who_region[i] = 'emro'



# Western Pacific Region 

wpro = 'Australia, Brunei, Cambodia, China, Cook Islands, Fiji, Japan, Kiribati, Laos, Malaysia, Marshall Islands, Micronesia, Mongolia, Nauru, New Zealand, Niue, Palau, Papua New Guinea, Philippines, South Korea, Samoa, Singapore, Solomon Islands, Taiwan, Taiwan*, Tonga, Tuvalu, Vanuatu, Vietnam'

wpro = [i.strip() for i in wpro.split(',')]

for i in wpro:

    who_region[i] = 'wpro'
df_covid['Region'] = df_covid['Country_Region'].map(who_region)

df_covid
import folium

# Dünya Geneli

temp = df_covid[df_covid['Date'] == max(df_covid['Date'])]

m = folium.Map(location=[0, 0], titles='Dünya Haritası Üzerindeki Değerler',

               min_zoom=1, max_zoom=4, zoom_start=1)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],

        color='crimson', fill='crimson',

        tooltip =   '<li><bold>Country_Region : '+str(temp.iloc[i]['Country_Region'])+

                    

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths']),

        radius=int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)

m
# Zaman İçerisindeki Değişim

import plotly.express as px

fig = px.choropleth(df_covid, locations="Country_Region", locationmode='country names', color=np.log(df_covid["Confirmed"]), 

                    hover_name="Country_Region", animation_frame=df_covid["Date"],

                    title='Zaman İçerisindeki Değişim', color_continuous_scale=px.colors.sequential.Purp)

fig.update(layout_coloraxis_showscale=False)

fig.show()
import plotly.express as px

fig = px.bar(df_covid.sort_values("Confirmed"),

            x='Region', y="Confirmed",

            hover_name="Region",

            hover_data=["Recovered","Deaths","Confirmed"],

            title='COVID-19: Test Sonucu Pozitif Olan Hasta Sayısı Bölgelere Göre',

)

fig.update_xaxes(title_text="Region")

fig.update_yaxes(title_text="Positif Test Sayısı(%)")

fig.show()

fig = px.bar(df_covid.sort_values("Recovered"),

            x='Region', y="Recovered",

            hover_name="Region",

            hover_data=["Confirmed","Deaths","Recovered"],

            title='COVID-19: İyileşen Hasta Sayısı Bölgelere Göre',

)

fig.update_xaxes(title_text="Region")

fig.update_yaxes(title_text="İyileşen Hasta Sayısı")

fig.show()

fig = px.bar(df_covid.sort_values("Deaths"),

            x='Region', y="Deaths",

            hover_name="Region",

            hover_data=["Confirmed","Recovered","Deaths"],

            title='COVID-19: Hayatını Kaybeden Hasta Sayısı Bölgelere Göre ',

)

fig.update_xaxes(title_text="Region")

fig.update_yaxes(title_text="Hayatını Kaybeden Hasta Sayısı")

fig.show()
from sklearn.model_selection import train_test_split





X = df_covid.iloc[:,2:5]

y = df_covid['Recovered']



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3, random_state=0)
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
import numpy as np

from sklearn    import metrics, svm

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import  linear_model
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