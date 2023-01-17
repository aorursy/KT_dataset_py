import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head(10)
df.sample(10)
df.tail(10)
df.info()
df.columns
df.shape
df.isnull().sum()
df.describe().T
df.corr()
df_yeni=df.copy(deep=True)

df_yeni[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_yeni[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df_yeni.sample(10)
df_yeni.isnull().sum()
df_yeni.describe().T
df_yeni.corr()
df.hist(figsize = (20,20))

plt.show()
df_yeni.hist(figsize=(20,20))

plt.show()
df_yeni['Glucose'].fillna(df_yeni['Glucose'].median(), inplace = True)

df_yeni['BloodPressure'].fillna(df_yeni['BloodPressure'].median(), inplace = True)

df_yeni['SkinThickness'].fillna(df_yeni['SkinThickness'].median(), inplace = True)

df_yeni['Insulin'].fillna(df_yeni['Insulin'].median(), inplace = True)

df_yeni['BMI'].fillna(df_yeni['BMI'].median(), inplace = True)
df_yeni.isnull().sum()
df_yeni.describe().T
df_yeni.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df_yeni.corr(),annot=True,linewidths=5,fmt='.0%',ax=ax)

plt.show()
fig=plt.figure(figsize=(20,10))

sns.boxplot(data=df_yeni)

plt.show()
plt.figure(figsize=(18,5))

sns.scatterplot( x='Age', y='Pregnancies',color='red',data=df_yeni)

plt.title('AGE-PREGNANCİES')

plt.show()
plt.figure(figsize=(18,5))

sns.regplot( x='SkinThickness', y='BMI',data=df_yeni)

plt.title('SKİNTİCKNESS-BMI')

plt.show()
plt.figure(figsize=(18,5))

sns.lmplot( x= 'DiabetesPedigreeFunction', y='Pregnancies',data=df_yeni)

plt.title('Diabets Pedigree Function -Pregnancies')

plt.show()
df_yeni['Outcome'].value_counts()
fig1, ax1 = plt.subplots(1,2,figsize=(8,8))

sns.countplot(df_yeni['Outcome'],ax=ax1[0])

labels = 'Diabetic', 'Healthy'



df_yeni.Outcome.value_counts().plot.pie(labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

plt.show()
df_yeni['Age']=df_yeni['Age']

bins=[20,35,50,65,81]

labels=['Genç','Orta Yaş','Yetişkin','Yaşlı']

df_yeni['yas_grp']=pd.cut(df_yeni['Age'],bins,labels=labels)
df_yeni.head()
df_yeni.yas_grp.value_counts()
colors = ['green','yellow','orange','red']

labels = df_yeni.yas_grp.value_counts().index

plt.title('Hedef Yaş Grubu',color = 'blue',fontsize = 20)

plt.pie(df_yeni.yas_grp.value_counts(),colors=colors,autopct='%1.1f%%',labels=labels)

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "Pregnancies", data = df_yeni);

plt.title('YAŞ GRUBU- PREGNANCİES')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "Glucose", data = df_yeni);

plt.title('YAŞ GRUBU- GLUCOSE')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "SkinThickness", data = df_yeni);

plt.title('YAŞ GRUBU- SkinThickness')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "BloodPressure", data = df_yeni);

plt.title('YAŞ GRUBU- BLOODPRESSURE')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "Insulin", data = df_yeni);

plt.title('YAŞ GRUBU- INSULIN')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "BMI", data = df_yeni);

plt.title('YAŞ GRUBU- BMI')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "DiabetesPedigreeFunction", data = df_yeni);

plt.title('YAŞ GRUBU- DIABET PEDIGREE FUNCTION')

plt.show()
plt.figure(figsize=(18,5))

sns.violinplot(x = "yas_grp", y = "Outcome", data = df_yeni);

plt.title('YAŞ GRUBU- OUTCOME')

plt.show()


sns.jointplot(x = df_yeni["Outcome"], y = df_yeni["Glucose"],kind='kde', color = "red");
plt.figure(figsize=(18,5))

sns.scatterplot( x='Age', y='Pregnancies',hue='Outcome',data=df_yeni)

plt.title('AGE-PREGNANCİES')

plt.show()
plt.figure(figsize=(18,5))

sns.scatterplot( x= 'SkinThickness', y='BMI',hue='Outcome',data=df_yeni)

plt.title('SKİNTİCKNESS-BMI')

plt.show()
plt.figure(figsize=(18,5))

sns.scatterplot( x='Insulin', y='Glucose',hue='Outcome',data=df_yeni)

plt.title('INSULIN-GLUCOSE')

plt.show()
y=df_yeni["Outcome"].values
x_data = df_yeni.drop(["Outcome","yas_grp"],axis=1)
x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x.head()
import statsmodels.api as sm

stmodel = sm.OLS(y, x).fit()

stmodel.summary()
from sklearn.model_selection import train_test_split

x_train, x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print(x.shape)

print(x_train.shape)

print(x_test.shape)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',random_state=0)

dt.fit(x_train,y_train)
dtpre=dt.predict(x_test)

dtpre
y_test
from sklearn.metrics import confusion_matrix

dtmatrix=confusion_matrix(y_test,dtpre)

dtmatrix
print("score:",dt.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

nb.fit(x_train,y_train)
deneme=nb.predict(x_test)

deneme
y_test
cm=confusion_matrix(y_test,deneme)

cm
nb.score(x_test,y_test)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)
knn.score(x_test,y_test)
knn1= KNeighborsClassifier(n_neighbors=5)

knn1.fit(x_train,y_train)
knn1.score(x_test,y_test)
score_list=[]

for each in range (1,15):

    knn85=KNeighborsClassifier(n_neighbors=each)

    knn85.fit(x_train,y_train)

    score_list.append(knn85.score(x_test,y_test))

plt.plot(range(1,15),score_list)
from sklearn.model_selection import GridSearchCV

grid={"n_neighbors":np.arange(1,15)}

knn2=KNeighborsClassifier()

knn2_cv = GridSearchCV(knn2,grid,cv=10)

knn2_cv.fit(x_train,y_train)
knn2_cv.best_score_
knn2_cv.best_params_
knn3=KNeighborsClassifier(n_neighbors=13)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=knn3,X=x_train,y=y_train,cv=10)
np.mean(accuracies)
np.std(accuracies)
accuracies