import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

df=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.isna().values.any()
df.head(5)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

l1=[]

l1=df.columns

for i in l1:

    df[i]=le.fit_transform(df[i])
x=df.drop('class',axis=1)

y=df['class']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)
from sklearn.ensemble import ExtraTreesClassifier

etc=ExtraTreesClassifier()

etc.fit(x_train,y_train)

y_pred=etc.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import f1_score

f1_score(y_test,y_pred)