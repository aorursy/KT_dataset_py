import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore') 



import os

import sys
df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv",sep = ",")

df.head()
df=df.rename(columns = {'Chance of Admit ':'Chance','GRE Score':'GRE','TOEFL Score':'TOEFL','University Rating':'URating'})
df.head()
#veri hakkında özet bilgi

df.info()
#boyutu

df.shape
#toplam boş değer sayısı..hangi değişkende kaç tane boş değer olduğuna bakıyorum.

df.isnull().sum()
#kolonlar

df.columns
#serial no değişkeninin analizime bir katkısı olmayacağından çıkarıyorum.

df.drop(["Serial No."],axis=1,inplace=True)
#temel istatistiki bilgiler

df.describe().T
#En önemli 3 kriter: CGPA, GRE ve TOEFL

#En az önemli 3 kriter: Research, LOR ve SOP



fig,ax = plt.subplots(figsize=(12,6))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap=colormap)

plt.show()
cor=df.corr()['Chance']

print(cor)
plt.figure(figsize=(15,12))

plt.subplot(2,2,1)

sns.distplot(df["GRE"], color='Orange')

plt.grid(alpha=0.5)



plt.subplot(2,2,2)

sns.distplot(df["TOEFL"], color='Orange')

plt.grid(alpha=0.5)



plt.subplot(2,2,3)

sns.distplot(df["SOP"], color='Orange')

plt.grid(alpha=0.5)



plt.subplot(2,2,4)

sns.distplot(df['CGPA'], color='Orange')

plt.grid(alpha=0.5)
#kabul değişkeni ile olan ilişkileri

sns.pairplot(df, x_vars=['GRE','TOEFL','CGPA','URating','SOP','LOR ','Research'], y_vars='Chance');
#TOEFL SCORE

y = np.array([df["TOEFL"].min(),df["TOEFL"].mean(),df["TOEFL"].max()])

x = ["Kötü","Orta","İyi"]

plt.bar(x,y,color="r")

plt.title("TOEFL Scores")

plt.xlabel("Düzey")

plt.ylabel("TOEFL Score")

plt.show()

print("Toefl Score Minumum:",df["TOEFL"].min())

print("Toefl Score Ortalama:",df["TOEFL"].mean())

print("Toefl Score Maksimim:",df["TOEFL"].max())
#GRE SCORE

y = np.array([df["GRE"].min(),df["GRE"].mean(),df["GRE"].max()])

x = ["Kötü","Orta","İyi"]

plt.barh(x,y,color="b")

plt.title("GRE Scores")

plt.xlabel("Düzey")

plt.ylabel("GRE Score")

plt.show()

print("GRE Score Minumum:",df["GRE"].min())

print("GRE Score Ortalama:",df["GRE"].mean())

print("GRE Score Maksimim:",df["GRE"].max())
#Üniversite Rating

plt.scatter(df["URating"],df.CGPA)

plt.title("CGPA Scores for University Ratings")

plt.xlabel("University Rating")

plt.ylabel("CGPA")

plt.show()
plt.hist(df.CGPA)

plt.title("CGPA Dağılım")

plt.ylabel("CGPA")

plt.show()
#İyi üniversitelerden mezun olanların kabul şansı daha yüksek, 4 rating olanlar en yüksek

s = df[df["Chance"] >= 0.75]["URating"].value_counts().head(5)

plt.title("University Ratings of Candidates with an 75% acceptance chance")

s.plot(kind='bar',figsize=(12,6))

plt.xlabel("University Rating")

plt.ylabel("Candidates")

plt.show()
#Araştırma yapanların kabul alma oranı daha yüksek gözükmekte.

sns.boxplot(x='Research',y='Chance',data=df);
#bağımlı ve bağımsız değişkeni belirleyip, veriyi eğitim ve test olmak üzere ayırıyorum.

x=df.drop(['Chance'],axis=1)

y=df['Chance'].values



from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict



x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])

x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])
import statsmodels.api as sm

lm=sm.OLS(y_train,x_train)

model=lm.fit()

model.summary()
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

model=lm.fit(x_train,y_train)
from sklearn.metrics import mean_squared_error,r2_score

#eğitim hatasına ulaşıyoruz.

egitim_rmse=np.sqrt(mean_squared_error(y_train,model.predict(x_train)))  
egitim_rmse
#test hatasına ulaşıyoruz.

test_rmse=np.sqrt(mean_squared_error(y_test,model.predict(x_test)))
test_rmse
model.score(x_train,y_train)
#doğrulanmış r2 değerinin ortalaması

cross_val_score(model,x_train,y_train,cv=10,scoring="r2").mean()
np.sqrt(-cross_val_score(model,x_train,y_train,cv=10,

                scoring="neg_mean_squared_error")).mean()