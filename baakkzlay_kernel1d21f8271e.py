# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

# Load libraries

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

from plotly.offline import init_notebook_mode, plot, iplot
import plotly as py
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the data
usa_data = pd.read_csv("../input/covid19-in-usa/us_counties_covid19_daily.csv")

# Data Glimpse
usa_data
usa_data.info()

usa_data[(usa_data['deaths']>50) & (usa_data['date'])]

usa_data['deaths'].mode()

usa_data['deaths'].std()

usa_data.cov()
#Korelasyon Gösterim
import seaborn as sns
corr = usa_data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
usa_data.plot(x='date', y='deaths', style='-')

usa_data.isnull().sum().sum()

#Özniteliklerin değer almadığı kaç satır var?
usa_data.isnull().sum()
#Eksik değer tablosu
def eksik_deger_tablosu(usa_data): 
    eksik_deger = usa_data.isnull().sum()
    eksik_deger_yuzde = 100 * usa_data.isnull().sum()/len(usa_data)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return eksik_deger_tablo_son
  
eksik_deger_tablosu(usa_data)
#%70 üzerinde null değer içeren kolonları sil
tr = len(usa_data) * .3
usa_data.dropna(thresh = tr, axis = 1, inplace = True)

usa_data
#Apply fonksiyonu 
def olum_durumu(deaths):
    return (deaths >= 100)

usa_data['yuksek_olum'] = usa_data['deaths'].apply(olum_durumu)
usa_data
#veri bilgisini 0 ve 1lere çevirdik.

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
usa_data['yuksek_olum_Encoded']= label_encoder.fit_transform(usa_data['yuksek_olum'])

usa_data
#deaths özniteliğini ölçeklendirmek istiyoruz
x = usa_data[['deaths']].values.astype(float)

#Ölçeklendirme için MinMaxScaler fonksiyonunu kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
usa_data['deaths2'] = pd.DataFrame(x_scaled)

usa_data
#Quartile (Kartiller) ve IQR ile Aykırı Değer Tespiti

import seaborn as sns
sns.boxplot(x=usa_data['deaths'])
Q1 = usa_data.deaths.quantile(0.25)
Q2 = usa_data.deaths.quantile(0.5)
Q3 = usa_data.deaths.quantile(0.75)
Q4 = usa_data.deaths.quantile(1)
IQR = Q3 - Q1

print("Q1-->", Q1)
print("Q3-->", Q3)
print("Q2-->", Q2)
print("Q4-->", Q4)
print("IQR-->", IQR)
print("Alt sınır: Q1 - 1.5 * IQR--->", Q1 - 1.5 * IQR)
print("Üst sınır: Q3 + 1.5 * IQR--->", Q3 + 1.5 * IQR)
from IPython.core.display import HTML
HTML('''<div class="flourish-embed" data-src="story/258632" data-url="https://flo.uri.sh/story/258632/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
state_details = pd.pivot_table(usa_data, values=['cases','deaths'], index='state', aggfunc='max')
state_details['Death Rate'] = round(state_details['deaths'] /state_details['cases'], 2)
state_details = state_details.sort_values(by='cases', ascending= False)
state_details.style.background_gradient(cmap='YlOrRd')
fig = px.bar(usa_data, x="date", y="total")

layout = go.Layout(
    title=go.layout.Title(
        text="ABD'de zaman içinde kümülatif Toplam COVID-19 testi sayısı",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
    xaxis_title = "Gözlem tarihi",
    yaxis_title = "Covid-19 testlerinin sayısı"
)

fig.update_layout(layout)
fig.show()
latest_data = usa_data[usa_data["date"] == max(usa_data["date"])].reset_index()
country_latest_data = latest_data.groupby('state').sum().reset_index().sort_values(by = 'cases',ascending = False).head(5)
fig = go.Figure(data=[
    #go.Bar(name='Confirmed', x=country_latest_data["state"], y=country_latest_data['cases'],marker_color = 'rgb(55, 83, 109)'),
    go.Bar(name='vaka', x=country_latest_data["state"], y=country_latest_data['cases'],marker_color = 'lightsalmon'),
    go.Bar(name = 'ölüm',x=country_latest_data["state"],y=country_latest_data['deaths'],marker_color = 'crimson' ),
    
])
fig.update_layout(barmode='group',title_text ='İlk 5 eyalet ')
fig.layout.template ='plotly_dark'
fig.show()
usa_data.loc[usa_data['deaths'] <10, 'Sınıf'] = 'İyi'
usa_data.loc[ (usa_data['deaths'] >= 100) & (usa_data['deaths'] < 300), 'Sınıf'] = 'Önemli'
usa_data.loc[ (usa_data['deaths'] >= 50) & (usa_data['deaths'] < 100), 'Sınıf'] = 'Güçlü'
usa_data.loc[ (usa_data['deaths'] >= 25) & (usa_data['deaths'] < 50), 'Sınıf'] = 'Ilımlı'
usa_data.dropna(how="any",inplace=True) 
# Magnitude Class distribution

sns.countplot(x="Sınıf", data=usa_data)
plt.ylabel('Sıklık')
plt.title('Önem derecesi VS Sıklık')
usa_data
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
usa_data['county_Encoded']= label_encoder.fit_transform(usa_data['county'])
usa_data['state_Encoded']= label_encoder.fit_transform(usa_data['state'])
usa_data['fips_Encoded']= label_encoder.fit_transform(usa_data['fips'])
usa_data['deaths_Encoded']= label_encoder.fit_transform(usa_data['deaths'])
usa_data['cases_Encoded']= label_encoder.fit_transform(usa_data['cases'])



usa_data
usa_data =usa_data.drop(columns ='county')
usa_data =usa_data.drop(columns ='state')
usa_data =usa_data.drop(columns ='fips')
usa_data =usa_data.drop(columns ='cases')
usa_data =usa_data.drop(columns ='deaths')
usa_data =usa_data.drop(columns ='yuksek_olum_Encoded')
usa_data =usa_data.drop(columns ='yuksek_olum')
usa_data =usa_data.drop(columns ='deaths2')
usa_data =usa_data.drop(columns ='date')





usa_data
array = usa_data.values
X = array[:,1:6]
y = array[:,0:1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print("Dataframe boyutu: ",usa_data.shape)
print("Eğitim verisi boyutu: ",X_train.shape, Y_train.shape)
print("Test verisi boyutu: ",X_validation.shape, Y_validation.shape)
from sklearn import preprocessing
from sklearn import utils
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
#lojistik regresyon
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
#gaussian NB 
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