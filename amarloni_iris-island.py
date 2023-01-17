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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
dataset = pd.read_csv("/kaggle/input/iris/Iris.csv", sep = ',')

dataset.head()
dataset.info()
dataset.describe()
le= LabelEncoder()

dataset['Species'] = le.fit_transform(dataset['Species'])
df = pd.DataFrame(dataset)

df.head()
df = df.drop(['Id'],axis = 1)
names = df.columns.unique()

for name in names:

    sns.boxplot(x = name, data = df)

    plt.show()

sns.pairplot(df, hue = 'Species')
X = df.drop(['Species'], axis = 1)

y = df['Species']
sc = StandardScaler()

X_scaled = sc.fit_transform(X)

#df_scaled = pd.DataFrame(X_scaled,y)

#df_scaled.tail()



X_scaled_df = pd.DataFrame(X_scaled,columns=['SepalLength','SepalWidth',

                                             'PetalLength','PetalWidth'])

df_scaled = pd.concat([X_scaled_df,y],axis=1)

df_scaled.head()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn import metrics



lr = LogisticRegression()

sgd = SGDClassifier()

knn = KNeighborsClassifier()

gnb = GaussianNB()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

svc = SVC()



models = [lr,sgd,knn,gnb,dt,rf ]

for model in models:

    mod = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    print('\n', model, 'accuracy score:', accuracy,'\n')
from sklearn.metrics import confusion_matrix, classification_report

for model in models:

    cm = confusion_matrix(y_test, y_pred)

    print('\n', model,'\n', 'Confusion matrix:','\n', cm,'\n')



from sklearn.metrics import confusion_matrix, classification_report

for model in models:

    print('\n',model,'\n', classification_report(y_test, y_pred))

    #print('\n', model,'\n', 'Confusion matrix:','\n', cm,'\n')