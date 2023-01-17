# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
veri = pd.read_csv("/kaggle/input/covid19-in-bangladesh/COVID-19_in_bd.csv")
veri.head() 
veri.columns.tolist() 
veri.describe() 
ax = plt.axes() 
ax.scatter(veri["Deaths"], veri["Recovered"])

ax.set(xlabel='Ölen Kişi',
       ylabel='Kurtarılan Kişi',
       title='Ölen ve Kurtarılan Kişi');
plt.figure(figsize=(16,8))
plt.plot(veri['Confirmed'], label='Confirmed cases');
import seaborn as sns     
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(veri.corr());
sns.pairplot(veri);
df=veri.filter(['Confirmed','Deaths','Recovered'])
df["Confirmed"]=pd.to_numeric(df["Confirmed"])
df["Recovered"]=pd.to_numeric(df["Recovered"])
df["Deaths"]=pd.to_numeric(df["Deaths"])
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score 
GNB = GaussianNB()
nb = {'gaussian': GaussianNB(),
      'bernoulli': BernoulliNB(),
      'multinomial': MultinomialNB()}
scores = {}

for key, model in nb.items():
    s = cross_val_score(model, X, y, cv=5, n_jobs=10, scoring='accuracy')
    scores[key] = np.mean(s)
scores
from sklearn.linear_model import LinearRegression
dataframe = veri.dropna(how='any',axis=0)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=135)
lr = LinearRegression()
r_fit = lr.fit(X_egitim, y_egitim)

y_pred = lr.predict(X_test)
r_fit.score(X_test, y_test)*100
from sklearn.tree import DecisionTreeClassifier
X_egitim,X_test,y_egitim,y_test = train_test_split(X,y, random_state=135)
agac = DecisionTreeClassifier(random_state=135,min_samples_split=2)
agac.fit(X_egitim, y_egitim)
agac.score(X, y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_egitim)
X_egitim_std = sc.transform(X_egitim)
X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_egitim_std, y_egitim)

prediction = lr.predict(X_test_std)
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))
plt.figure(figsize=(10, 10))
plt.plot(veri.Date.index, df.Deaths, color='blue')
plt.plot(veri.Date.index, df.Recovered, color='green')
plt.legend(['Ölümler', 'Kurtulanlar'], loc='best' , fontsize=20)
plt.title('Kornavirüs Vakaları', size=20)
plt.xlabel('Gün', size=20)
plt.ylabel('Vakalar', size=20)
plt.xticks(size=15)
plt.show()
