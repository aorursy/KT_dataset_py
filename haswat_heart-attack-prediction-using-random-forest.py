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
df=pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df.head()
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

numeric_index=['age','trestbps','chol','thalach','oldpeak']

df.hist(column=numeric_index, figsize=(10,30), layout=(5,1))

plt.show()
plt.figure(figsize=(25,8))

total = float(len(df) )



ax = sns.countplot(x="sex", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(25,8))

total = float(len(df) )



ax = sns.countplot(x="cp", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(25,8))

total = float(len(df) )



ax = sns.countplot(x="fbs", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(25,8))

total = float(len(df) )



ax = sns.countplot(x="restecg", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(25,8))

total = float(len(df) )



ax = sns.countplot(x="exang", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(12,10))

total = float(len(df["target"]) )



ax = sns.countplot(x="slope", hue="target", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(12,10))

total = float(len(df["target"]) )



ax = sns.countplot(x="thal", hue="target", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(12,10))

total = float(len(df["target"]) )



ax = sns.countplot(x="ca", hue="target", data=df)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
x=df.drop(['target'],axis=1)

y=df['target']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y, test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42,

                                   max_features = 'auto', max_depth = 10)

classifier.fit(X_train, Y_train)
fitur=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

tmp = pd.DataFrame({'Feature':fitur , 'Feature importance': classifier.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()  
from sklearn.metrics import confusion_matrix

predicted = classifier.predict(X_test)

predicted_proba = classifier.predict_proba(X_test)



matrix = confusion_matrix(Y_test, predicted)

print(matrix)
from sklearn.metrics import accuracy_score, make_scorer

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(y_true = Y_test, y_pred = predictions)