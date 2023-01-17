import numpy as np #numerical python bilimsel hesaplamaları hızlı bir şekilde yapmamızı sağlayan bir matematik kütüphanesidir.
import pandas as pd #veri yapıları ve veri analiz araçları sağlayan açık kaynaklı bir BSD lisanslı kütüphanedir.
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
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
datac=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
#datas=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv')
#cityTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
#datam=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByMajorCity.csv')
print(data.shape)
print(datac.shape)
data.head(10).T
datac.head(10).T
# descriptions
data.describe()

datac.describe()
datac.groupby(datac.Country).size()
# NaN değerleri siliyoruz
data = data[data['LandAverageTemperature'].notna()]
data = data[data['LandAverageTemperatureUncertainty'].notna()]
data = data[data['LandMaxTemperature'].notna()]
data = data[data['LandMaxTemperatureUncertainty'].notna()]
data = data[data['LandMinTemperature'].notna()]
data = data[data['LandMinTemperatureUncertainty'].notna()]
data = data[data['LandAndOceanAverageTemperature'].notna()]
data = data[data['LandAndOceanAverageTemperatureUncertainty'].notna()]
# NaN değerleri siliyoruz
datac = datac[datac['AverageTemperature'].notna()]
datac = datac[datac['AverageTemperatureUncertainty'].notna()]
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()
datac.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histogramlar
data.hist()
plt.show()
# histogramlar
datac.hist()
plt.show()
scatter_matrix(data)
plt.show()
scatter_matrix(datac)
plt.show()
del data['dt']
data.head().T
del datac['dt']
datac.head().T
array = datac.values
X = array[:,0:2]
y = array[:,2]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
#her modeli sırayla değerlendir
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#Değerleri karşılaştıralım
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
# Doğrulama veri kümesi hakkında tahminler yapalım.
#NB, GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Tahminleri değerlendirelim.
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#KNN, KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#CART,  DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))