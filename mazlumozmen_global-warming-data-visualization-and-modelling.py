import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns
import plotly.graph_objs as go


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from pandas import read_csv
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
data=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')
data.head().T
data.info()
data.describe().T
data2=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv', parse_dates=['dt'])
data2=data2.replace([np.inf, -np.inf], np.nan).dropna()
data2['maxAvgTemp']=data2['LandAverageTemperature']+data2['LandAverageTemperatureUncertainty']
data2['minAvgTemp']=data2['LandAverageTemperature']-data2['LandAverageTemperatureUncertainty']

data2=data2.groupby(data2['dt'].map(lambda x: x.year)).mean().reset_index()
min_year=data2['dt'].min()
max_year=data2['dt'].max()
data2.head(-5).T
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
_, ax=plt.subplots(figsize=(10, 6))

plt.plot(data2['dt'], data2['LandAverageTemperature'], color='black')
ax.fill_between(data2['dt'], data2['minAvgTemp'], data2['maxAvgTemp'], color='b')

plt.xlim(min_year, max_year)
ax.set_title('Belirsizlik dahil ortalama küresel sıcaklık')

plt.show()
global1=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')
global1=global1[['dt','LandAverageTemperature']]
global1.dropna(inplace=True)
global1['dt']=pd.to_datetime(global1.dt).dt.strftime('%d/%m/%Y')
global1['dt']=global1['dt'].apply(lambda x:x[6:])
global1=global1.groupby(['dt'])['LandAverageTemperature'].mean().reset_index()
trace=go.Scatter(
    x=global1['dt'],
    y=global1['LandAverageTemperature'],
    mode='lines',
    )
data1=[trace]
py.iplot(data1, filename='line-mode')
global_temp =pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')

global_temp['dt'] = pd.to_datetime(global_temp['dt'])
global_temp['year'] = global_temp['dt'].map(lambda x: x.year)
global_temp['month'] = global_temp['dt'].map(lambda x: x.month)

def get_season(month):
    if month >= 3 and month <= 5:
        return 'spring'
    elif month >= 6 and month <= 8:
        return 'summer'
    elif month >= 9 and month <= 11:
        return 'autumn'
    else:
        return 'winter'
    
min_year = global_temp['year'].min()
max_year = global_temp['year'].max()
years = range(min_year, max_year + 1)

global_temp['season'] = global_temp['month'].apply(get_season)

spring_temps = []
summer_temps = []
autumn_temps = []
winter_temps = []

for year in years:
    curr_years_data = global_temp[global_temp['year'] == year]
    spring_temps.append(curr_years_data[curr_years_data['season'] == 'spring']['LandAverageTemperature'].mean())
    summer_temps.append(curr_years_data[curr_years_data['season'] == 'summer']['LandAverageTemperature'].mean())
    autumn_temps.append(curr_years_data[curr_years_data['season'] == 'autumn']['LandAverageTemperature'].mean())
    winter_temps.append(curr_years_data[curr_years_data['season'] == 'winter']['LandAverageTemperature'].mean())
spring_temps
summer_temps
autumn_temps
winter_temps
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
f, ax = plt.subplots(figsize=(10, 6))
plt.plot(years, summer_temps, label='Yaz ortalama sıcaklık', color='orange')
plt.plot(years, autumn_temps, label='Sonbahar', color='r')
plt.plot(years, spring_temps, label='İlkbahar ortalama sıcaklık', color='g')
plt.plot(years, winter_temps, label='Kış ortalama sıcaklık', color='b')
plt.xlim(min_year, max_year)
ax.set_ylabel('Ortalama')
ax.set_xlabel('Yıllar')
ax.set_title('Her mevsimdeki ortalama sıcaklık')
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, borderpad=1, borderaxespad=1)
# NaN değerleri siliyoruz
data = data[data['LandAverageTemperature'].notna()]
data = data[data['LandAverageTemperatureUncertainty'].notna()]
data = data[data['LandMaxTemperature'].notna()]
data = data[data['LandMaxTemperatureUncertainty'].notna()]
data = data[data['LandMinTemperature'].notna()]
data = data[data['LandMinTemperatureUncertainty'].notna()]
data = data[data['LandAndOceanAverageTemperature'].notna()]
data = data[data['LandAndOceanAverageTemperatureUncertainty'].notna()]
print("Satır sayısı",data.shape[0])
del data['dt']
data.head().T
columns=data.columns
for x in columns:
    data[x] = data[x].astype(int)
data.info()
#Bölünmüş doğrulama veri kümesi
array = data.values
X = array[:,1:8]
y = array[:,0]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
X
y
# Algoritmalar uyguladım
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=4, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Algoritma karşılaştırması
pyplot.boxplot(results, labels=names)
pyplot.title('Algoritma karşılaştırması')
pyplot.show()
# Doğrulama veri kümesinde tahminlerde bulunma
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Tahminleri değerlendirdim
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))