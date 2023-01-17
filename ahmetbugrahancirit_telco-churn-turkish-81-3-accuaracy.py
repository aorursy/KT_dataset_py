import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",sep=',',decimal='.')
df.rename(columns={'customerID':'Müşteri ID','gender':'Cinsiyet','SeniorCitizen':'Yaşlı','Partner':'Evli',

                   'Dependents':'Ekonomik Bağımlı','tenure':'Abonelik Süresi','PhoneService':'Telefon Hizmeti',

                   'MultipleLines':'Birden Fazla Hat','InternetService':'İnternet Servisi',

                   'OnlineSecurity':'Çevrimiçi Güvenlik','OnlineBackup':'Çevrimiçi Yedekleme',

                   'DeviceProtection':'Cihaz Koruma','TechSupport':'Teknik Destek','StreamingTV':'Televizyon',

                   'StreamingMovies':'Film','Contract':'Sözleşme Süresi','PaperlessBilling':'Çevrimiçi Fatura',

                   'PaymentMethod':'Ödeme Yöntemi','MonthlyCharges':'Aylık Ödeme','TotalCharges':'Toplam Ödeme',

                   'Churn':'Müşteri Kaybı'},inplace=True)
df.head(8000)
df.drop('Müşteri ID', axis=1, inplace=True)
df["Müşteri Kaybı"]= df["Müşteri Kaybı"].replace("No","Müşteri Kaybı Oluşmadı") 

df["Müşteri Kaybı"]= df["Müşteri Kaybı"].replace("Yes","Müşteri Kaybı Oluştu")
df["Yaşlı"]= df["Yaşlı"].replace(0, "No") 

df["Yaşlı"]= df["Yaşlı"].replace(1, "Yes") 
df.info()
df['Toplam Ödeme'] = pd.to_numeric(df['Toplam Ödeme'], errors='coerce')

df['Toplam Ödeme'] = df['Toplam Ödeme'].fillna(value=0)
df['Yaşlı'] = df['Yaşlı'].astype('object')
df.describe()
sns.countplot(x = "Müşteri Kaybı", data = df)

df.loc[:, 'Müşteri Kaybı'].value_counts()
df.isnull().sum()
Kategorik = df.select_dtypes(include='object').drop('Müşteri Kaybı', axis=1).columns.tolist()

Sayısal = df.select_dtypes(exclude='object').columns.tolist()
for c in Kategorik:

    print('Column {} unique values: {}'.format(c, len(df[c].unique())))
plt.figure(figsize=(18,18))

for i,c in enumerate(Kategorik):

    plt.subplot(5,4,i+1)

    sns.countplot(df[c], hue=df['Müşteri Kaybı'])

    plt.title(c)

    plt.xlabel('')
plt.figure(figsize=(20,5))

for i,c in enumerate(['Abonelik Süresi', 'Aylık Ödeme', 'Toplam Ödeme']):

    plt.subplot(1,3,i+1)

    sns.distplot(df[df['Müşteri Kaybı'] == 'Müşteri Kaybı Oluşmadı'][c], kde=True, color='blue', hist=False, kde_kws=dict(linewidth=2), label='Müşteri Kaybı Oluşmadı')

    sns.distplot(df[df['Müşteri Kaybı'] == 'Müşteri Kaybı Oluştu'][c], kde=True, color='Orange', hist=False, kde_kws=dict(linewidth=2), label='Müşteri Kaybı Oluştu')

    plt.title(c)
df.head(8000)
df.info()
import seaborn as sns

sns.boxplot(x=df['Abonelik Süresi'],y=df['Müşteri Kaybı'])
import seaborn as sns

sns.boxplot(x=df['Aylık Ödeme'],y=df['Müşteri Kaybı'])
import seaborn as sns

sns.boxplot(x=df['Toplam Ödeme'],y=df['Müşteri Kaybı'])
from sklearn.preprocessing import LabelEncoder

encoded = df.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)

encoded.head(8000)
Müşteri_Kaybı_Yaşandı=encoded.loc[encoded['Müşteri Kaybı'].abs()>0]

Müşteri_Kaybı_Yaşandı
Q1 = Müşteri_Kaybı_Yaşandı['Toplam Ödeme'].quantile(0.25)

Q3 = Müşteri_Kaybı_Yaşandı['Toplam Ödeme'].quantile(0.75)

IQR = Q3 - Q1

IQR
Q=Q3+(1.5*IQR)

Q
encoded_out = encoded[~((encoded['Toplam Ödeme'] < (Q3 + 1.5 * IQR)))&(encoded['Müşteri Kaybı']>0)]

encoded_out.head(8000)
encoded.drop(encoded[~((encoded['Toplam Ödeme'] < (Q3 + 1.5 * IQR)))&(encoded['Müşteri Kaybı']>0)].index, inplace=True)

encoded.head(8000)
Q1_A = Müşteri_Kaybı_Yaşandı['Abonelik Süresi'].quantile(0.25)

Q3_A = Müşteri_Kaybı_Yaşandı['Abonelik Süresi'].quantile(0.75)

IQR_A = Q3_A - Q1_A

IQR_A
Q_A=Q3_A+(1.5*IQR_A)

Q_A
encoded_A_out = encoded[~((encoded['Abonelik Süresi'] < (Q3_A + 1.5 * IQR_A)))&(encoded['Müşteri Kaybı']>0)]

encoded_A_out.head(8000)
encoded.drop(encoded[~((encoded['Abonelik Süresi'] < (Q3_A + 1.5 * IQR_A)))&(encoded['Müşteri Kaybı']>0)].index, inplace=True)

encoded.head(8000)
x = df.drop('Müşteri Kaybı', axis = 1)              

y = df['Müşteri Kaybı'] 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.85, random_state = 400)
x_test.head(8000)
x_train.head(8000)
y_test.head(8000)
y_train.head(8000)
x=encoded['Cinsiyet']

y=encoded['Müşteri Kaybı']

print('Cinsiyet:', x.corr(y)*100)

x=encoded['Yaşlı']

y=encoded['Müşteri Kaybı']

print('Yaşlı:', x.corr(y)*100)

x=encoded['Evli']

y=encoded['Müşteri Kaybı']

print('Evli:', x.corr(y)*100)

x=encoded['Ekonomik Bağımlı']

y=encoded['Müşteri Kaybı']

print('Ekonomik Bağımlı:', x.corr(y)*100)

x=encoded['Abonelik Süresi']

y=encoded['Müşteri Kaybı']

print('Abonelik Süresi:', x.corr(y)*100)

x=encoded['Telefon Hizmeti']

y=encoded['Müşteri Kaybı']

print('Telefon Hizmeti:', x.corr(y)*100)

x=encoded['Birden Fazla Hat']

y=encoded['Müşteri Kaybı']

print('Birden Fazla Hat:', x.corr(y)*100)

x=encoded['İnternet Servisi']

y=encoded['Müşteri Kaybı']

print('İnternet Servisi:', x.corr(y)*100)

x=encoded['Çevrimiçi Güvenlik']

y=encoded['Müşteri Kaybı']

print('Çevrimiçi Güvenlik:', x.corr(y)*100)

x=encoded['Çevrimiçi Yedekleme']

y=encoded['Müşteri Kaybı']

print('Çevrimiçi Yedekleme:', x.corr(y)*100)

x=encoded['Cihaz Koruma']

y=encoded['Müşteri Kaybı']

print('Cihaz Koruma:', x.corr(y)*100)

x=encoded['Teknik Destek']

y=encoded['Müşteri Kaybı']

print('Teknik Destek:', x.corr(y)*100)

x=encoded['Televizyon']

y=encoded['Müşteri Kaybı']

print('Televizyon:', x.corr(y)*100)

x=encoded['Film']

y=encoded['Müşteri Kaybı']

print('Film:', x.corr(y)*100)

x=encoded['Sözleşme Süresi']

y=encoded['Müşteri Kaybı']

print('Sözleşme Süresi:', x.corr(y)*100)

x=encoded['Çevrimiçi Fatura']

y=encoded['Müşteri Kaybı']

print('Çevrimiçi Fatura:', x.corr(y)*100)

x=encoded['Ödeme Yöntemi']

y=encoded['Müşteri Kaybı']

print('Ödeme Yöntemi:', x.corr(y)*100)

x=encoded['Aylık Ödeme']

y=encoded['Müşteri Kaybı']

print('Aylık Ödeme:', x.corr(y)*100)

x=encoded['Toplam Ödeme']

y=encoded['Müşteri Kaybı']

print('Toplam Ödeme:', x.corr(y)*100)
encoded.drop('Cinsiyet', axis=1, inplace=True)

encoded.drop('Telefon Hizmeti', axis=1, inplace=True)

encoded.drop('Birden Fazla Hat', axis=1, inplace=True)

encoded.drop('İnternet Servisi', axis=1, inplace=True)

encoded.drop('Televizyon', axis=1, inplace=True)

encoded.drop('Film', axis=1, inplace=True)
x=encoded.drop('Müşteri Kaybı',axis=1)

y=encoded['Müşteri Kaybı']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.85, random_state=43)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
Logistic_Regression = LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)

Logistic_Regression.fit(x_train, y_train)
y_pred=Logistic_Regression.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score
classification_report(y_true=y_test, y_pred=y_pred)
accuracy_score(y_test, y_pred)*100
confusion_matrix(y_test, y_pred)
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

NBG = GaussianNB()

NBG.fit(x_train, y_train)

y_tahmin = NBG.predict(x_test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.85, random_state=42)

print("x_train: ", x_train.shape)

print("x_test: ", x_test.shape)

print("y_train: ", y_train.shape)

print("y_test: ", y_test.shape)

print("Navy Bayes Gaussian Doğruluk Skoru :",accuracy_score(y_test, y_tahmin))

print("Karışıklık Matrisi :",confusion_matrix(y_test, y_tahmin))

print("Sınıflandırma Raporu :",classification_report(y_test, y_tahmin))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

NBM = MultinomialNB()

NBM.fit(x_train, y_train)

y_tahmin = NBM.predict(x_test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.85, random_state=42)

print("x_train: ", x_train.shape)

print("x_test: ", x_test.shape)

print("y_train: ", y_train.shape)

print("y_test: ", y_test.shape)

print("Navy Bayes Multinomial Doğruluk Skoru :",accuracy_score(y_test, y_tahmin))

print("Karışıklık Matrisi :",confusion_matrix(y_test, y_tahmin))

print("Sınıflandırma Raporu :",classification_report(y_test, y_tahmin))
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

NBB = BernoulliNB()

NBB.fit(x_train, y_train)

y_tahmin = NBB.predict(x_test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.85, random_state=42)

print("x_train: ", x_train.shape)

print("x_test: ", x_test.shape)

print("y_train: ", y_train.shape)

print("y_test: ", y_test.shape)

print("Navy Bayes Bernoulli Doğruluk Skoru :",accuracy_score(y_test, y_tahmin))

print("Karışıklık Matrisi :",confusion_matrix(y_test, y_tahmin))

print("Sınıflandırma Raporu :",classification_report(y_test, y_tahmin))
from sklearn.preprocessing import KBinsDiscretizer  

est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

encoded['Abonelik Süresi'] = est.fit_transform(encoded['Abonelik Süresi'].values.reshape(-1,1))
from sklearn.preprocessing import KBinsDiscretizer  

est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

encoded['Aylık Ödeme'] = est.fit_transform(encoded['Aylık Ödeme'].values.reshape(-1,1))
from sklearn.preprocessing import KBinsDiscretizer  

est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

encoded['Toplam Ödeme'] = est.fit_transform(encoded['Toplam Ödeme'].values.reshape(-1,1))
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix as KarışıklıkMatrisi



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.85, random_state=42)

Karar_Ağacı = DecisionTreeClassifier(max_depth = 4, random_state=42)

Karar_Ağacı.fit(x_train, y_train)
predictions = Karar_Ağacı.predict(x_test)

score = round(accuracy_score(y_test, predictions), 2)

KarışıklıkMatrisi = KarışıklıkMatrisi(y_test, predictions)

sns.heatmap(KarışıklıkMatrisi, annot=True, fmt=".0f")

plt.xlabel('Tahmin Değerler')

plt.ylabel('Gerçek Veriler')

plt.title('Doğruluk Skoru: {0}'.format(score), size = 15)

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, target_names=['Kayıp Olmayanlar', 'Kayıp Olanlar']))
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth = 4, random_state=42,min_weight_fraction_leaf=0.0)

clf = clf.fit(x, y)
tree.plot_tree(clf,fontsize=10) 
plt.figure(figsize=(16, 9))



from sklearn import ensemble



Karar_Ağacı2 = DecisionTreeClassifier(max_depth = 4, random_state=42)

Karar_Ağacı2.fit(x_train, y_train)

ranking = Karar_Ağacı2.feature_importances_

features = np.argsort(ranking)[::-1][:10]

columns = x.columns



plt.title("Modelin Özniteliklerinin Önem sıralamasının Analiz Edilmesi", y = 1.03, size = 18)

plt.bar(range(len(features)), ranking[features], color="lime", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

error = []



# Calculating error for K values between 1 and 40

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train, y_train)

    pred_i = knn.predict(x_test)

    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)

plt.title('Hata Oranı K İçin')

plt.xlabel('K Değeri')

plt.ylabel('Ortalama Hata')
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=1)

KNN.fit(x_train, y_train)
y_pred = KNN.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

from sklearn.datasets import make_classification

x, y = make_classification(n_samples=1000, n_classes=2, random_state=1)

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.85, random_state=42)

KNN_tahmin = [0 for _ in range(len(testy))]

NBG_tahmin = [0 for _ in range(len(testy))]

Karar_Ağacı_tahmin = [0 for _ in range(len(testy))]

Logistic_Regression_tahmin = [0 for _ in range(len(testy))]



model = NBG

model.fit(trainx, trainy)

model2 = KNN

model2.fit(trainx, trainy)

model3=Karar_Ağacı

model3.fit(trainx, trainy)

model4=Logistic_Regression

model4.fit(trainx, trainy)



NBG_tahmin = model.predict_proba(testx)

KNN_tahmin = model2.predict_proba(testx)

Karar_Ağacı_tahmin = model3.predict_proba(testx)

Logistic_Regression_tahmin= model4.predict_proba(testx)



NBG_tahmin = NBG_tahmin[:, 1]

KNN_tahmin = KNN_tahmin[:, 1]

Karar_Ağacı_tahmin = Karar_Ağacı_tahmin[:, 1]

Logistic_Regression_tahmin = Logistic_Regression_tahmin[:, 1]



KNN_hassasiyet = roc_auc_score(testy, KNN_tahmin)

NBG_hassasiyet = roc_auc_score(testy, NBG_tahmin)

Karar_Ağacı_hassasiyet = roc_auc_score(testy, Karar_Ağacı_tahmin)

Logistic_Regression_hassasiyet = roc_auc_score(testy, Logistic_Regression_tahmin)



print('KNN: ROC AUC=%.3f' % (KNN_hassasiyet))

print('Navy Bayes Gaussian: ROC AUC=%.3f' % (NBG_hassasiyet))

print('Karar Ağacı: ROC AUC=%.3f' % (Karar_Ağacı_hassasiyet))

print('Logistic Regresyon: ROC AUC=%.3f' % (Logistic_Regression_hassasiyet))



KNN_fpr, KNN_tpr, _ = roc_curve(testy, KNN_tahmin)

NBG_fpr, NBG_tpr, _ = roc_curve(testy, NBG_tahmin)

Karar_Ağacı_fpr, Karar_Ağacı_tpr, _ = roc_curve(testy, Karar_Ağacı_tahmin)

Logistic_Regression_fpr, Logistic_Regression_tpr, _ = roc_curve(testy, Logistic_Regression_tahmin)



pyplot.plot(KNN_fpr, KNN_tpr, linestyle='--', label='KNN')

pyplot.plot(NBG_fpr, NBG_tpr, marker='.', label='Navy Bayes Gaussian')

pyplot.plot(Karar_Ağacı_fpr, Karar_Ağacı_tpr, marker='.', label='Karar Ağacı')

pyplot.plot(Logistic_Regression_fpr, Logistic_Regression_tpr, marker='.', label='Logistic Regresyon')



pyplot.xlabel('Gerçek Müşteri Kaybı Oluştu')

pyplot.ylabel('Gerçek Müşteri Kaybı Oluşmadı')



pyplot.legend()



pyplot.show()