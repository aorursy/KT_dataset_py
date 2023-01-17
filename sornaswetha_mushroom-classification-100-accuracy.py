# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

df.shape
df.head()
df.tail()
df.describe().transpose()
df.info()
df.isnull().sum() [df.isnull().sum() > 0]
print(df["class"].value_counts())

sns.countplot(df["class"],palette="Reds" )

plt.show()
def draw_countplot(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        sns.countplot(x=feature, hue="class", data=df, ax=ax, palette="Reds")

        ax.set_title(feature+" vs class",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

draw_countplot(df.iloc[:, 1:],df.iloc[:, 1:].columns,6,4)
from sklearn.preprocessing import LabelEncoder

Encoder_y=LabelEncoder()

df["class"] = Encoder_y.fit_transform(df["class"])
df = pd.get_dummies(df, drop_first=True)
df.head()
X=df.drop('class',axis=1) #Predictors

y=df['class'] #Response
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
model = DecisionTreeClassifier()

model.fit(X_train,y_train)

print("Testing Accuracy :", accuracy_score(y_test , model.predict(X_test)))

print("Confusion Matrix:\n" , confusion_matrix(y_test , model.predict(X_test ) ))

print("Classification Report:\n", classification_report(y_test , model.predict(X_test)))
model = RandomForestClassifier()

model.fit(X_train,y_train)

print("Testing Accuracy : ", accuracy_score(y_test , model.predict(X_test)))

print("Confusion Matrix:\n" , confusion_matrix(y_test , model.predict(X_test ) ))

print("Classification Report:\n", classification_report(y_test , model.predict(X_test)))