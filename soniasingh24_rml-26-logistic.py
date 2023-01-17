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
df = pd.read_csv('/kaggle/input/glass/glass.csv')

df.head(5)
df.info()

df.describe()
import seaborn as sns

sns.countplot(x="Type", data=df)
import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn import metrics
df['glass_type'] = df.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

df.head()
y = df.glass_type

X = df.loc[:,['Al','Mg','Ba']]

df.Type.value_counts().sort_index()



model_logr = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=200)

output_model = model_logr.fit(X_train, y_train)

output_model
score = output_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = output_model.predict(X_test)
model_logr.fit(X_train,y_train)

y_predict = model_logr.predict(X_test)

y_predict
cnf_matrix = metrics.confusion_matrix(y_test,y_predict)

cnf_matrix

%matplotlib inline

class_names=[0,1] # name  of classes

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