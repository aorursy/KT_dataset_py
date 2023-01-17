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
covid = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
covid.sample(6)
covid.info()
covid['Confirmed'].mode()

covid['Confirmed'].std()

covid.cov()

#Korelasyon Gösterim
import seaborn as sns
corr = covid.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
covid.plot(x='Date', y='Confirmed', style='-')

covid.isnull().sum().sum()


covid.isnull().sum()
def eksik_deger_tablosu(covid): 
    eksik_deger = covid.isnull().sum()
    eksik_deger_yuzde = 100 * covid.isnull().sum()/len(covid)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return eksik_deger_tablo_son
  
eksik_deger_tablosu(covid)
tr = len(covid) * .3
covid.dropna(thresh = tr, axis = 1, inplace = True)

covid
def confirmed_durum(Confirmed):
    return (Confirmed >= 100)

covid['yuksek_vaka'] = covid['Confirmed'].apply(confirmed_durum)
covid
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
covid['yuksek_vaka_Encoded']= label_encoder.fit_transform(covid['yuksek_vaka'])

covid
x = covid[['Confirmed']].values.astype(float)

#Ölçeklendirme için MinMaxScaler fonksiyonunu kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
covid['Confirmed2'] = pd.DataFrame(x_scaled)

covid
import seaborn as sns
sns.boxplot(x=covid['Confirmed'])
Q1 = covid.Confirmed.quantile(0.25)
Q2 = covid.Confirmed.quantile(0.5)
Q3 = covid.Confirmed.quantile(0.75)
Q4 = covid.Confirmed.quantile(1)
IQR = Q3 - Q1

print("Q1-->", Q1)
print("Q3-->", Q3)
print("Q2-->", Q2)
print("Q4-->", Q4)
print("IQR-->", IQR)
print("Alt sınır: Q1 - 1.5 * IQR--->", Q1 - 1.5 * IQR)
print("Üst sınır: Q3 + 1.5 * IQR--->", Q3 + 1.5 * IQR)
latest = covid.loc[covid['Date'] == covid['Date'].max()].groupby('Country/Region').sum().reset_index()

latest = latest.sort_values(by=['Confirmed'], ascending=False).reset_index(drop=True)
top_10 = latest.loc[:9]
top_10_bar = top_10.set_index('Country/Region')[top_10.columns[3:]]
top_10_names = top_10['Country/Region']

(top_10_bar/1e3).plot.bar(figsize=(20,5))
plt.ylabel('Thousand Cases')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING
import plotly.express as px

fig = px.choropleth(latest, locations="Country/Region", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Confirmed", range_color=[1,50000], 
                    color_continuous_scale='Reds', 
                    title='Onaylanmış Vakaların Global Görünümü')
fig.show()
fig = px.choropleth(latest_european, locations="Country/Region", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Confirmed", range_color=[1,5000], 
                    color_continuous_scale='Reds', 
                    title='Mevcut Vakalara Avrupa Görüşü', scope='europe')#, height=800, width= 1400)
# fig.update(layout_coloraxis_showscale=False)
fig.show()
covid.loc[covid['Confirmed'] <100 , 'Sınıf'] = 'İyi'
covid.loc[ (covid['Confirmed'] >= 1000) & (covid['Confirmed'] < 2500), 'Sınıf'] = 'Yüksek'
covid.loc[ (covid['Confirmed'] >= 2500) & (covid['Confirmed'] < 20000), 'Sınıf'] = 'Çok_Yüksek'
covid.dropna(how="any",inplace=True) 

covid
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
covid['Province/State_Encoded']= label_encoder.fit_transform(covid['Province/State'])
covid['Country/Region_Encoded']= label_encoder.fit_transform(covid['Country/Region'])
covid['Lat_Encoded']= label_encoder.fit_transform(covid['Lat'])
covid['Long_Encoded']= label_encoder.fit_transform(covid['Long'])
covid['Date_Encoded']= label_encoder.fit_transform(covid['Date'])
covid['Confirmed_Encoded']= label_encoder.fit_transform(covid['Confirmed'])
covid['Deaths_Encoded']= label_encoder.fit_transform(covid['Deaths'])
covid['Recovered_Encoded']= label_encoder.fit_transform(covid['Recovered'])
covid['WHO Region_Encoded']= label_encoder.fit_transform(covid['WHO Region'])

covid
covid =covid.drop(columns ='Province/State')
covid =covid.drop(columns ='Country/Region')
covid =covid.drop(columns ='Lat')
covid =covid.drop(columns ='Long')
covid =covid.drop(columns ='Date')
covid =covid.drop(columns ='Confirmed')
covid =covid.drop(columns ='Deaths')
covid =covid.drop(columns ='Recovered')
covid =covid.drop(columns ='WHO Region')
covid =covid.drop(columns ='yuksek_vaka')
covid =covid.drop(columns ='yuksek_vaka_Encoded')
covid =covid.drop(columns ='Confirmed2')






covid
array = covid.values
X = array[:,1:6]
y = array[:,0:1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print("Dataframe boyutu: ",covid.shape)
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