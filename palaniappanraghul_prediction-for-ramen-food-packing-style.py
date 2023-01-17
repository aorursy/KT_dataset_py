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
df = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')

df.head()
df = df.drop(columns=['Top Ten'])
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
df.describe(include='all')
x = df.drop(columns=['Style'])

x
y = df['Style']

y
import matplotlib.pyplot as plt

# create figure and axis

fig, ax = plt.subplots()

# plot histogram

ax.hist(df['Style'])

# set title and labels

ax.set_title('Style of Presenting')

ax.set_xlabel('Types')

ax.set_ylabel('Frequency')
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

x
import seaborn as sns

sns.distplot(x['Stars']);
y= label_encoder.fit_transform(y)

y
import seaborn as sns

sns.jointplot(x=x['Brand'], y=x['Stars'], kind="kde")
features=['Style', 'Country'] # Subplot for count plot

fig=plt.subplots(figsize=(25,20))

for i, j in enumerate(features):

    plt.subplot(4, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.countplot(x=j,data = df)

    plt.xticks(rotation=90)

    plt.title("Ramen")

    

plt.show()
import seaborn as sns

sns.kdeplot(data=x['Brand'], shade=True)

sns.kdeplot(data=x['Country'], shade=True)

sns.kdeplot(data=x['Stars'], shade=True)
plt.figure(figsize=(12,6))

sns.boxplot(x="Country", y="Brand", data=x)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
# data normalization with sklearn

from sklearn.preprocessing import MinMaxScaler



# fit scaler on training data

norm = MinMaxScaler().fit(x_train)



# transform training data

X_train_norm = norm.transform(x_train)





# transform testing dataabs

X_test_norm = norm.transform(x_test)
# fit scaler on training data

norm = MinMaxScaler().fit(x_train)



# transform training data

X_train_norm = norm.transform(x_train)

print("Scaled Train Data: \n\n")

print(X_train_norm)
# transform testing dataabs

X_test_norm = norm.transform(x_test)

print("\n\nScaled Test Data: \n\n")

print(X_test_norm)
from sklearn.ensemble import RandomForestClassifier

# random forest model creation

rfc = RandomForestClassifier()

rfc.fit(X_train_norm,y_train)
# predictions

rfc_predict = rfc.predict(X_test_norm)



print("Accuracy:",accuracy_score(y_test, rfc_predict))
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, n_init = 10, random_state=251)

kmeans.fit(x)
centroids = kmeans.cluster_centers_

centroid_df = pd.DataFrame(centroids, columns = list(x) )

centroid_df = pd.DataFrame(centroids, columns = list(x) )

df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))
snail_df_labeled = x.join(df_labels)

df_analysis = (snail_df_labeled.groupby(['labels'] , axis=0)).head(4177) 

df_analysis.head()
df_analysis.isnull().sum()
df_analysis = df_analysis.dropna()

df_analysis.isnull().sum()
from sklearn.model_selection import train_test_split  

X= df_analysis.drop('labels',axis =1)

y= df_analysis['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)
# predict Model

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

accuracy_score(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

rclf = RandomForestClassifier(n_estimators= 100)

rclf.fit(X_train,y_train)

y_pred = rclf.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))
from sklearn.neighbors import KNeighborsClassifier 

classifier = KNeighborsClassifier(n_neighbors= 5)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)

predicted = model.predict(X_test)

print('Predicted Value',predicted)
cm = confusion_matrix(y_true=y_test,y_pred=predicted)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,predicted))