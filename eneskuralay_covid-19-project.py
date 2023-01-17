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
# Kütüphanelerimizi import ediyoruz.



import pandas as pd

import matplotlib.pyplot as plt

import re

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt; plt.rcdefaults()
# Sklearn içerisindeki modellerimizden Linear Regression, Logistic Regression

# ve Decision Tree Kütüphanelerini Projemize dahil ediyoruz

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

import warnings

warnings.filterwarnings('ignore')
agegrp=pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

covidindia=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospitalbeds=pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individualdetails=pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
agegrp.head()
agegrp.info()
hospitalbeds=hospitalbeds[:-2]

hospitalbeds.fillna(0,inplace=True)

hospitalbeds
hospitalbeds.info()
for col in hospitalbeds.columns[2:]:

    if hospitalbeds[col].dtype=='object':

        hospitalbeds[col]=hospitalbeds[col].astype('int64')
hospitalbeds.info()
covidindia.head()
covidindia['Date']=pd.to_datetime(covidindia['Date'])
covidindia.Date
gender=individualdetails.gender

gender.dropna(inplace=True)

gender=gender.value_counts()

per=[]

for i in gender:

    perc=i/gender.sum()

    per.append(format(perc,'.2f'))

plt.figure(figsize=(10,6))    

plt.title('Cinsiyete göre vaka karşılaştırması',fontsize=20)

plt.pie(per,autopct='%1.1f%%')

plt.legend(gender.index,loc='best',title='Cinsiyet',fontsize=15)
perc=[]

for i in agegrp['Percentage']:

    per=float(re.findall("\d+\.\d+",i)[0])

    perc.append(per)

agegrp['Percentage']=perc

plt.figure(figsize=(20,10))

plt.title('Yaş grubunda vaka yüzdesi',fontsize=20)

plt.pie(agegrp['Percentage'],autopct='%1.1f%%')

plt.legend(agegrp['AgeGroup'],loc='best',title='Yaş Grubu')
plt.figure(figsize=(20,10))

plt.style.use('ggplot')

plt.title('Farklı yaş gruplarında vaka karşılaştırması',fontsize=30)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Age Group',fontsize=20)

plt.ylabel('Confirmed Cases',fontsize=20)

plt.bar(agegrp['AgeGroup'],agegrp['TotalCases'],color=['Red','green','skyblue','orange','hotpink'],linewidth=3)

for i, j in enumerate(agegrp['TotalCases']):

    plt.text(i-.25, j,

              agegrp['TotalCases'][i], 

              fontsize=20 )
top=hospitalbeds.nlargest(20,'NumPrimaryHealthCenters_HMIS')



plt.figure(figsize=(15,10))

plt.title('Eyaletlere göre sağlık merkezi sayısı',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('birinci basamak sağlık merkezi sayısı',fontsize=25)

plt.ylabel('States',fontsize=25)

plt.barh(top['State/UT'],top['NumPrimaryHealthCenters_HMIS'],color='Purple',linewidth=1)
df1=covidindia.groupby('Date')[['Cured','Deaths','Confirmed']].sum()
plt.figure(figsize=(20,10))

plt.style.use('ggplot')

plt.title('Gözlemlenen Vaka',fontsize=30)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel('Tarih',fontsize=20)

plt.ylabel('Vaka Sayısı',fontsize=20)

plt.plot(df1.index,df1['Confirmed'],linewidth=3,label='Onaylanmış',color='blue')

plt.plot(df1.index,df1['Cured'],linewidth=3,label='Tedavi Edilmiş',color='green')

plt.plot(df1.index,df1['Deaths'],linewidth=3,label='Ölüm',color='red')

plt.legend(fontsize=20)
df2=covidindia.groupby('State/UnionTerritory')[['Cured','Deaths','Confirmed']].sum()
df2=df2.nlargest(20,'Confirmed')

plt.figure(figsize=(20,6))

plt.title('Doğrulanmış Vakalar',fontsize=20)

plt.xticks(rotation=90,fontsize=10)

plt.yticks(fontsize=15)

plt.xlabel('Eyalet',fontsize=15)

plt.ylabel('Vaka',fontsize=15)

plt.plot(df2.index,df2.Confirmed,marker='o',mfc='black',label='Onaylanmış',markersize=10,linewidth=1,color='blue')

plt.plot(df2.index,df2.Deaths,marker='o',mfc='black',label='Ölüm',markersize=10,linewidth=1,color='red')

plt.plot(df2.index,df2.Cured,marker='o',mfc='black',label='Tedavi Edilmiş',markersize=10,linewidth=1,color='green')

plt.legend(fontsize=20)
perc=[]

for i in df2.Confirmed:

    per=i/len(df2)

    perc.append(i)

plt.figure(figsize=(25,10))    

plt.title('Yüzde Dağılımı ile Onaylanmış Vakaları Olan Eyaletler ',fontsize=20)

plt.pie(perc,autopct='%1.1f%%')

plt.legend(df2.index,loc='upper right')
covidindia.isnull().sum()
covidindia["ConfirmedForeignNational"]=covidindia['ConfirmedForeignNational'].replace('-',0,inplace=True)

covidindia["ConfirmedIndianNational"]=covidindia['ConfirmedIndianNational'].replace('-',0,inplace=True)
covidindia['ConfirmedIndianNational']=covidindia['ConfirmedIndianNational'].astype('float64')

covidindia['ConfirmedForeignNational']=covidindia['ConfirmedForeignNational'].astype('float64')
df3=covidindia.groupby('State/UnionTerritory')[['ConfirmedIndianNational','ConfirmedForeignNational']].sum()
df4=df3.nlargest(20,'ConfirmedIndianNational')

df5=df3.nlargest(20,'ConfirmedForeignNational')
plt.figure(figsize=(30,25))

plt.subplot(311)

plt.title('Tedavi Edilmiş Vaka Sayısı',fontsize=28)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Cured,color='green',linewidth=5)

plt.figure(figsize=(30,25))

plt.subplot(311)

plt.title('Vefat Eden Vaka Sayısı',fontsize=28)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Deaths,color='blue',linewidth=5)
plt.figure(figsize=(30,25))

plt.subplot(311)

plt.title('Onaylanmış Vakalar',fontsize=28)

plt.xticks(rotation=90,fontsize=20)

plt.yticks(fontsize=25)

plt.bar(df2.index,df2.Confirmed,color='red',linewidth=5)





el=sns.catplot(x='State/UnionTerritory',y='Confirmed',kind='boxen',data=covidindia)

el.fig.set_figwidth(20)

el.fig.set_figheight(5)

el.set_xticklabels(rotation=90,fontsize=15)
el=sns.catplot(x='State/UnionTerritory',y='Cured',kind='boxen',data=covidindia)

el.fig.set_figwidth(20)

el.fig.set_figheight(5)

el.set_xticklabels(rotation=90,fontsize=15)
el=sns.catplot(x='State/UnionTerritory',y='Deaths',kind='boxen',data=covidindia)

el.fig.set_figwidth(20)

el.fig.set_figheight(5)

el.set_xticklabels(rotation=90,fontsize=15)
# Veriyi birebir sayısallaştırabilmek için label encoder kullandım. Kategorik her veriye sayısal değer atadık.

from sklearn.preprocessing import LabelEncoder

lbl=LabelEncoder()

covidindia['State/UnionTerritory']=lbl.fit_transform(covidindia['State/UnionTerritory'])
covidindia["ConfirmedForeignNational"]=covidindia['ConfirmedForeignNational'].fillna(0,inplace=False)

covidindia["ConfirmedIndianNational"]=covidindia['ConfirmedIndianNational'].fillna(0,inplace=False)
covidindia.isnull().sum()
covidindia['Date']=covidindia['Date'].astype('datetime64[ns]')
covidindia['date']=covidindia['Date'].dt.day

covidindia['month']=covidindia['Date'].dt.month
covidindia
linear=LinearRegression()

logistic=LogisticRegression(C=0.05, solver='liblinear')

tree=DecisionTreeRegressor()

neigh = KNeighborsClassifier(n_neighbors=3)

gnb = GaussianNB()

svm = svm.SVC(kernel='rbf')
from sklearn.model_selection import train_test_split

x=covidindia[['State/UnionTerritory','date','month','Cured','Deaths','ConfirmedIndianNational','ConfirmedForeignNational']]

y=covidindia['Confirmed']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


linear.fit(x_train,y_train)

logistic.fit(x_train,y_train)

tree.fit(x_train,y_train)

neigh.fit(x_train,y_train)

gnb.fit(x_train, y_train)

svm.fit(x_train, y_train) 





from sklearn.metrics import r2_score


predictionLin=linear.predict(x_test)

score1=r2_score(y_test,predictionLin)

print(score1)



predictionTree=tree.predict(x_test)

score2=r2_score(y_test,predictionTree)

print(score2)



predictionLog = logistic.predict(x_test)

score3 = r2_score(y_test,predictionLog)

print(score3)



predictionNeigh = neigh.predict(x_test)

score4 = r2_score(y_test,predictionNeigh)

print(score4)





predictionGnb = gnb.predict(x_test)

score5 = r2_score(y_test,predictionGnb)

print(score5)



predictionSvm = svm.predict(x_test)

score6 = r2_score(y_test,predictionSvm)

print(score6)
models = ['LinearRegression','DecisionTreeRegressor','LogisticRegression','KNeighboursClassifier',"GNB","SVM"]

y_pos = np.arange(len(models))

performance = [score1,score2,score3,score4,score5,score6]



plt.barh(y_pos, performance, align='center', alpha=0.5)

plt.yticks(y_pos, models)

plt.xlabel('Tahmin')

plt.title('Model Doğruluk Oranı')



plt.show()