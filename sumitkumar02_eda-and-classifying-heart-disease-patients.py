# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/heart.csv')
df.head()
df.info()
df.shape[0]
df.shape[1]
Null=df.isnull()

Null.sum()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

male =len(df[df['sex'] == 1])

female = len(df[df['sex']== 0])

total=len(df.sex)

total_male_percent=(male/total)*100

total_female_percent=(female/total)*100



#plot

labels = ['male', 'female']

values = [total_male_percent, total_female_percent]



plt.figure(figsize=(8,6))

plt.title('Sex Percentage')

plt.xlabel('sex')

plt.ylabel('percentage')

plt.bar(labels, values,color=('r','b'))

plt.show()
plt.figure(figsize=(8,6))



# Data to plot

labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'

values = [len(df[df['cp'] == 0]),len(df[df['cp'] == 1]),

         len(df[df['cp'] == 2]),

         len(df[df['cp'] == 3])]

colors = ['blue', 'green','orange','red']

 

# Plot

plt.title('Chest pain')

plt.xlabel('Types')

plt.ylabel('values')



plt.bar(labels,values, color=colors) 

plt.show()
plt.figure(figsize=(8,6))



# Data to plot

labels = 'fasting blood sugar < 120 mg/dl','fasting blood sugar > 120 mg/dl'

sizes = [len(df[df['fbs'] == 0]),len(df[df['cp'] == 1])]

colors = ['skyblue', 'yellowgreen','orange','gold']

explode = (0.1, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

 

plt.axis('equal')

plt.show()
plt.figure(figsize=(8,6))



# Data to plot

labels = 'No','Yes'

sizes = [len(df[df['exang'] == 0]),len(df[df['exang'] == 1])]

colors = ['skyblue', 'yellowgreen']

explode = (0.1, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=90)

 

plt.axis('equal')

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')

plt.show()
print(df.dtypes)
X= df.drop('target',axis=1)

y=df['target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.fit_transform(X_test)

X_test = pd.DataFrame(X_test_scaled)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

dpen = []

for i in range(5,11):

    model = XGBClassifier(max_depth = i)

    model.fit(X_train,y_train)

    target = model.predict(X_test)

    dpen.append(accuracy_score(y_test, target))

    print("accuracy : ",dpen[i-5])

print("Best accuracy: ",max(dpen))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)  # n_neighbors means k

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)



print("{} NN Score: {:.2f}%".format(7, knn.score(X_test, y_test)*100))
# try ro find best k value

scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(X_train, y_train)

    scoreList.append(knn2.score(X_test, y_test))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()





print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(X_train, y_train)

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(X_test,y_test)*100))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train, y_train)

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(X_test,y_test)*100))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(X_test,y_test)*100))