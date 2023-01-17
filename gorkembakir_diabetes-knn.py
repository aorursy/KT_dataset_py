from mlxtend.plotting import plot_decision_regions

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

#plt.style.use('ggplot')

#ggplot is R based visualisation package that provides better graphics with higher level of abstraction
# veri içeri alma

diabetes_data = pd.read_csv('../input/diabetes.csv')

# Pergnancies = gebelik sayısı

# BloodPressure = Kan Basıncı

# Skin Thickness = Cilt Kalınlığı

# Insulin= İnsülin 

# BMIBody mass index = Kütle Endeksi 

# Age = Yaş

# Outcome = 1(Diabet) - 0 (Diabet Değil)
# ilk 10 gözleme bakarak veri hakkında ilk izlenimi oluşturma

diabetes_data.head(10)
diabetes_data.info()

# 2 adet float  7 adet int dan oluşan veri setine sahibiz ve boş gözlemimiz bulunmamakta
diabetes_data.describe().T



# 0 değerleri NA  olarak değiştirildi

diabetes_data_copy = diabetes_data.copy(deep = True)

diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#null değerler tespit edildi

diabetes_data_copy.isnull().sum()
# copy datası histogramı

p = diabetes_data_copy.hist(figsize =( 15,15))
# datanın ilk halindeki histogramlar belirlenip aradaki değişim gözle görülebilir hale geldi

r = diabetes_data.hist(figsize = (20,20))
# uç değerler tespiti amaçlı çalışma

sz = (9, 9)

fig, ax = plt.subplots(figsize=sz)

sns.boxplot(ax=ax, data=diabetes_data_copy,  orient="h")

diabetes_data_copy.describe().T
#İnsulin verisi fazla uç ve anlamsız değer taşıdığından analiz güvenliği amacıyla çıkartıldı

diabetes_data_copy.drop(columns=['Insulin'],axis= 1, inplace=True )
diabetes_data_copy.tail()
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)

diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)

diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)

diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
#Glükoz değerleri anlamlı bir ayrıştırma sağlamaktadır

g = sns.catplot(x="Outcome", y="Glucose", data=diabetes_data_copy,

                height=6, kind="bar", palette="muted")
g = sns.catplot(x="Outcome", y="BloodPressure", data=diabetes_data_copy,

                height=6, kind="bar", palette="muted")
g = sns.catplot(x="Outcome", y="SkinThickness", data=diabetes_data_copy,

                height=6, kind="bar", palette="muted")
g = sns.catplot(x="Outcome", y="BMI", data=diabetes_data_copy,

                height=6, kind="bar", palette="muted")
g = sns.catplot(x="Outcome", y="DiabetesPedigreeFunction", data=diabetes_data_copy,

                height=6, kind="bar", palette="muted")
g = sns.catplot(x="Outcome", y="Age", data=diabetes_data_copy,

                height=6, kind="bar", palette="muted")
# Kan basıncı doşındaki değişkenlerimizde 0 vve 1 bazından anlamlı ayrışma gözlenmektedir.
from pandas.tools.plotting import scatter_matrix

p=scatter_matrix(diabetes_data,figsize=(25, 25))
p=sns.pairplot(diabetes_data_copy, hue = 'Outcome')
#korelasyon tespiti amaçlı headmap kullanıldı

plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diabetes_data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

# en yüksek korelasyon Glükoz ile en düşük korelasyon kan basıncı ile
y= diabetes_data_copy.Outcome.values

x_data = diabetes_data_copy.drop(["Outcome"], axis=1)
y
x_data.head()
#normalizasyon

x =( x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x.head()
#Train - Test ayrımı

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=44 )
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(3)

knn.fit(X_train,y_train)

prediction = knn.predict(X_train)

print(knn.score(X_train,y_train))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(3)

knn.fit(X_train,y_train)

prediction = knn.predict(X_test)

print(knn.score(X_test,y_test))
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(3)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
#import confusion_matrix

from sklearn.metrics import confusion_matrix

#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
y_pred = knn.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')