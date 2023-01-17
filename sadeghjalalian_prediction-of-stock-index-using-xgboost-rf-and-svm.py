# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix

from sklearn import metrics
df = pd.read_csv('/kaggle/input/stock-index-prediction-both-labels-and-features/SP500.csv')

df.head(10)
df.info()
import missingno as msno

msno.matrix(df)
plt.figure(figsize=(12,8))

sns.heatmap(df.describe()[1:].transpose(),

            annot=True,linecolor="w",

            linewidth=2,cmap=sns.color_palette("Set2"))

plt.title("Data summary")

plt.show()


cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
corr=df.corr()

corr.sort_values(by=["LABEL"],ascending=False).iloc[0].sort_values(ascending=False)
explode = (0.1,0)  

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['LABEL'].value_counts(), explode=explode,labels=['Down','Up'], autopct='%1.1f%%',

        shadow=True)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
plt.figure(figsize=(10,9))

sns.scatterplot(x='InterestRate',y='ExchangeRate',data=df,palette='Set1', hue = 'LABEL');
df = df.drop('Date',axis=1)

df.head()
X = df.drop('LABEL',axis=1)

y = df['LABEL']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
#prediction the test set result

y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test,y_pred)

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="RdGy" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#fitting random forest classification into training set

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)

rf.fit(X_train, y_train)
#prediction the test set result

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="RdGy" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(X_train,y_train)
pred = svc_model.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix

cm = confusion_matrix(y_test,pred)

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BuPu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))