# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
print(data.iloc[:10,:])
plt.figure(figsize=(12,12))

sns.heatmap(data.corr(),

            vmin=-1,

            cmap='coolwarm',

            annot=True);
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



data['quality'].values[data['quality']<7] = 0 

data['quality'].values[data['quality']>=7] = 1



x = data.loc[:,data.columns != 'quality']

y = data.loc[:,data.columns == 'quality']

x = np.array(x)

y = np.array(y)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .25, random_state = 0)



logreg = LogisticRegression(max_iter = 7000)

logreg.fit(x_train,np.ravel(y_train))

y_pred = logreg.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix



plt.title('Confusion Matrix for Wine Quality')

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot = True, cmap="gist_earth")

plt.show()



from sklearn.metrics import classification_report



print('\nClassification Report\n')

print(classification_report(y_test, y_pred))