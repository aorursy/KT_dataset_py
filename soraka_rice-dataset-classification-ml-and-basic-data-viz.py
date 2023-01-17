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
df = pd.read_csv('../input/rice-dataset-gonenjasmine/Rice-Gonen andJasmine.csv')
df.head()
print(df.info())

print('=='*50)

print(df.shape)
df['Class'].value_counts()
class_map = {'jasmine':1,'Gonen':0}

df['Class'] = df['Class'].map(class_map)
df.head()
import seaborn as sns

import matplotlib.pyplot as plt
features = ['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea',

            'EquivDiameter','Extent','Perimeter','Roundness','AspectRation']

for graph in features:

    sns.scatterplot(data = df,x=np.arange(0,df[graph].shape[0]),y=df[graph],hue=df['Class'])

    plt.xlabel(graph)

    plt.show()
for hist in features:

    sns.distplot(df[hist])

    plt.xlabel(hist)

    plt.show()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
def box(arr):

    q1 = arr.quantile(0.25)

    q3 = arr.quantile(0.75)

    ıqr = q3-q1

    mini = q1-(1.5*ıqr)

    maxi = q3+(1.5*ıqr)

    arr = arr[(arr>mini)&(arr<maxi)]

    

def standart(arr):

    plus = arr.mean()+3*arr.std()

    minus = arr.mean()-3*arr.std()

    arr = arr[(arr>minus)&(arr<plus)]
box(df['AspectRation'])

box(df['Roundness'])

standart(df['Perimeter'])

standart(df['Extent'])

box(df['EquivDiameter'])

box(df['ConvexArea'])

box(df['Eccentricity'])

box(df['MinorAxisLength'])

standart(df['MajorAxisLength'])
df.head()
df['TotalAxisLength'] = df['MajorAxisLength']+df['MinorAxisLength']
df.corr()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

import sklearn.neural_network as nn

from sklearn.metrics import accuracy_score
y = df['Class']

X = df.drop('Class',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,test_size = 0.2)
rf = RandomForestClassifier()

rf_model = rf.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)
accuracy_score(y_pred,y_test)