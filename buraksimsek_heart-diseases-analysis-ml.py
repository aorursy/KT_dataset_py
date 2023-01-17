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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.describe()
df.target.value_counts()
df.hist(figsize = (15, 15))
plt.show()
plt.subplots(figsize = (9, 6))
sns.countplot(x = "target", data = df, palette = "YlGn")

plt.title("Risk Grubunun Sayisal Dagilimi", fontsize = 14)
plt.xlabel("Risk Grubu (0 = Risk Tasimiyor    1 = Risk Tasiyor)", fontsize = 12)
plt.ylabel("Kisi Sayisi", fontsize = 12)
plt.show()

plt.subplots(figsize = (7, 5))
df.target.value_counts().plot.pie(explode = [0.1,0.0], autopct = '%1.1f%%', cmap = "YlGn")
fig = plt.figure(figsize = (12,12))
ax1 = fig.add_subplot(2,1,1)
ax1.set_title("Risk Tasiyan Grubun Yas Dagilimi", fontsize = 13)
ax1.tick_params(labelbottom =  "off", axis = "x")
sns.countplot(df[df.target == 1].age);
plt.xlabel("Yaslar", fontsize = 12)
plt.ylabel("Kisi Sayisi", fontsize = 12)


ax2 = fig.add_subplot(2,1,2)
ax2.set_title("Risk Tasimayan Grubun Yas Dagilimi", fontsize = 13)
sns.countplot(df[df.target == 0].age);
plt.xlabel("Yaslar", fontsize = 12)
plt.ylabel("Kisi Sayisi", fontsize = 12)
plt.subplots(figsize = (10, 8))
sns.countplot(x = "target", hue = "sex", data = df, palette = "PuBuGn")

plt.title("Risk Grubunun Cinsiyetlere Gore Dagilimi", fontsize = 14)
plt.xlabel("Risk Grubu (0 = Risk Tasimiyor    1 = Risk Tasiyor)", fontsize = 12)
plt.ylabel("Kisi Sayisi", fontsize = 12)
plt.show()
plt.subplots(figsize = (10, 8))
sns.countplot(x = "target", hue = "cp", data = df, palette = "PuRd")

plt.title("Risk Grubunun Gogus Agrisi Tipine Gore Dagilimi", fontsize = 14)
plt.xlabel("Risk Grubu (0 = Risk Tasimiyor    1 = Risk Tasiyor)", fontsize = 12)
plt.ylabel("Kisi Sayisi", fontsize = 12)
plt.show()
plt.subplots(figsize=(15,7))
sns.lineplot(y = "thalach", x = "age", data = df, hue = "target", style = "target", palette = "magma", markers = True, dashes = False, err_style = "bars", ci=68)
plt.title("Yas ve Kalp Atis Hizi Tablosu")
plt.subplots(figsize = (10, 8))
sns.countplot(x = "ca", data = df, hue = "target", palette = "BuPu")
plt.subplots(figsize = (10, 8))
sns.countplot(x = "slope", hue = "target", data = df, palette = "GnBu", linewidth = 3)
plt.subplots(figsize = (16, 14))
sns.heatmap(df.corr(), annot = True, cmap = "PuBu")
plt.title("Veri Setinin Korelasyon Isi Haritasi", fontsize = 15)
plt.show()
col = df.corr()
pd.DataFrame(col.target).sort_values(by = "target", ascending = False).style.background_gradient(cmap = "PuBu")
kalp_atis_der = pd.cut(df.thalach, 3, labels = ["71-114", "115-158", "159-202"])
pd.crosstab([df.slope[df.slope == 2], df.target],kalp_atis_der).style.background_gradient(cmap = "PuBu")
pd.crosstab([df.cp[df.cp == 2], df.target],kalp_atis_der).style.background_gradient(cmap = "PuBu")
pd.crosstab([df.cp[df.cp == 2], df.slope[df.slope == 2], df.target],kalp_atis_der).style.background_gradient(cmap = "PuBu")
X = df.drop(["target"], axis=1)
y = df.target

print(X.shape)
X = (X - X.min())/(X.max()-X.min())
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred )
plt.subplots(figsize = (8, 6))
sns.heatmap(cm, annot=True)
plt.plot()
accuracies = {}

acc = lr.score(X_test,y_test)*100

accuracies['Logistic Regression'] = acc
print("LR Dogruluk Degeri {:.2f}%".format(acc))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize = (8, 6))
sns.heatmap(cm, annot=True)
plt.plot()
acc = knn.score(X_test,y_test)*100

accuracies['KNeighbors Classifier'] = acc
print("KNN Dogruluk Degeri {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)
y_pred = nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize = (8, 6))
sns.heatmap(cm, annot=True)
plt.plot()
acc = nb.score(X_test, y_test)*100

accuracies['Naive Bayes'] = acc
print("NB Dogruluk Oranı {:.2f}%".format(acc))
from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize = (8, 6))
sns.heatmap(cm, annot=True)
plt.plot()
acc = svm.score(X_test, y_test)*100

accuracies['Suppot Vector Machine'] = acc
print("SVM Dogruluk Oranı {:.2f}%".format(acc))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize = (8, 6))
sns.heatmap(cm, annot=True)
plt.plot()
acc = rf.score(X_test, y_test)*100

accuracies['Random Forest'] = acc
print("RF Dogruluk Oranı {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize = (8, 6))
sns.heatmap(cm, annot=True)
plt.plot()
acc = dtc.score(X_test, y_test)*100

accuracies['Decision Tree Classifier'] = acc
print("DTC Dogruluk Oranı {:.2f}%".format(acc))
colors = ["red", "pink", "blue", "yellow", "green", "purple"]

sns.set_style("whitegrid")
plt.figure(figsize = (15,8))
plt.yticks(np.arange(0,100,10))
plt.xlabel("Algoritmalar")
plt.ylabel("Dogruluk Yuzdesi")
sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()), palette = colors)
plt.show()