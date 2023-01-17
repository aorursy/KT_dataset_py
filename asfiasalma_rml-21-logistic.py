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
#importing the file

data = pd.read_csv("/kaggle/input/glass/glass.csv")

data.head()
#data preprocessing

data.info()

data.describe()
import seaborn as sns

sns.countplot(x="Type", data=data)
data.Type.value_counts().sort_index()
# glass_type 1, 2, 3 are window glass

# glass_type 5, 6, 7 are non-window glass

data['Glass_type'] = data.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

data.head()
import matplotlib.pyplot as plt

features = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type","Glass_type"]

#features.head()



mask = np.zeros_like(data[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(data[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
#selecting the feaures & spilting into train & test

from sklearn.model_selection import train_test_split

#X = data.drop("Type", axis=1)

X = data[["Na","Al","Ba"]]

Y = data["Glass_type"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=100)
from sklearn.linear_model import LogisticRegression

Logreg = LogisticRegression()

Logreg.fit(X_train, Y_train)

Y_pred = Logreg.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# cnf_matrix = metrics.multilabel_confusion_matrix(Y_test, Y_pred)

print(cnf_matrix)

print(Y_test.groupby(Y_test).count())
class_names=[1,2,3,5,6] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
# print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

# print("Precision:",metrics.precision_score(Y_test, Y_pred, average='weighted'))

# print("Recall:",metrics.recall_score(Y_test, Y_pred, average='weighted'))

print (metrics.classification_report(Y_test, Y_pred))