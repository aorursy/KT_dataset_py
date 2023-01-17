import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import  DecisionTreeClassifier



from sklearn import utils

from sklearn import preprocessing

from sklearn import metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv')

df.head()
df.head(10)
df.info()
df.dropna(inplace=True)
negara = df.groupby('country')['suicides_no'].sum()

negara



plt.barh(negara.index, negara)

plt.xlabel('Angka bunuh diri')

plt.ylabel('Negara')

plt.title('Angka bunuh diri di tiap negara')

plt.show()
negara_max = df.groupby('country')['suicides_no'].sum().nlargest(10)

negara_max



plt.barh(negara_max.index, negara_max)

plt.xlabel('Angka bunuh diri')

plt.ylabel('Negara')

plt.title('10 negara dengan angka bunuh diri tertinggi')

plt.show()
tahun = df.groupby(['year'])['suicides_no'].sum()

tahun





plt.bar(tahun.index, tahun)

plt.ylabel('Angka bunuh diri')

plt.xlabel('Tahun')

plt.title('Angka bunuh diri per tahun')

plt.show()
tahun_max = df.groupby(['year'])['suicides_no'].sum().nlargest(10)

tahun_max





plt.bar(tahun_max.index, tahun_max)

plt.ylabel('Angka bunuh diri')

plt.xlabel('Tahun')

plt.title('10 tahun dengan angka bunuh diri tertinggi')

plt.show()
jenis_kelamin = df.groupby('sex')['suicides_no'].sum()

jenis_kelamin



plt.bar(jenis_kelamin.index, jenis_kelamin)

plt.ylabel('Angka bunuh diri')

plt.xlabel('Jenis kelamin')

plt.title('Angka bunuh diri berdasarkan jenis kelamin')

plt.show()
umur = df.groupby('age')['suicides_no'].sum()

umur



plt.barh(umur.index, umur)

plt.xlabel('Angka bunuh diri')

plt.ylabel('Umur')

plt.title('Angka bunuh diri berdasarkan umur')

plt.show()
sns.pointplot(x='age', y='suicides_no', hue='sex', data=df)
fig, ax = plt.subplots()

ax.scatter(df["suicides_no"], df["population"])

ax.set_xlabel("populasi")

ax.set_ylabel("angka bunuh diri")

plt.show()
sns.relplot(x="year", y="suicides_no", data=df, kind="line")

plt.show()
df1=df

df1=df1.drop(['country'],axis=1)



df1['age']=df1['age'].replace('5-14 years',0)

df1['age']=df1['age'].replace('15-24 years',1)

df1['age']=df1['age'].replace('25-34 years',2)

df1['age']=df1['age'].replace('35-54 years',3)

df1['age']=df1['age'].replace('55-74 years',4)

df1['age']=df1['age'].replace('75+ years',5)

df1['sex']=df1['sex'].replace('male',0)

df1['sex']=df1['sex'].replace('female',1)



df1['suicides/100kPopulation']=(df1.suicides_no/df1.population)/100000

df1['fatality_rate']=np.where(df1['suicides/100kPopulation']>df1['suicides/100kPopulation'].mean(),0,1)
x = df1.drop(['fatality_rate', 'suicides/100kPopulation'], 1)

y = df1.fatality_rate

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)



label_enc=preprocessing.LabelEncoder()

trs=label_enc.fit_transform(y_train)



print("Shape of x_train: ",x_train.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of x_test: ",x_test.shape)

print("Shape of y_test: ",y_test.shape)
def age_cat(age) :

    if age < 2:

        return 'Young'

    elif 2 < age < 4:

        return 'Adult'

    else:

        return 'Senior'

    

# df1['age'] = df1.fillna(df1['age'].mean())

df1['age_cat'] = df1['age'].apply(age_cat)

df1.head(20)

    
sns.barplot(x='age_cat', y='suicides_no', data=df1)
sns.barplot(x='age_cat', y='suicides_no', data=df1)
logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

acc_linear_svc
gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

acc_gaussian
decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machine', 'Linear SVC', 'Gaussian Naive Bayes', 'Decision Tree Regressor', 'Random Forest Classifier'],

    'Score': [acc_log, acc_svc, acc_linear_svc, acc_gaussian, acc_decision_tree, acc_random_forest]

})

models.sort_values(by='Score', ascending=False)