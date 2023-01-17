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
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
#There are 0 values in the dataset in the Glucose,BloodPressure,SkinThickness, Insulin and BMI, we need to replace them with the NAN 



df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0, np.NaN)
df.isnull().any()
df.isna().sum()
#Replacing the null values with the mean and median respectively



df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

df['BloodPressure'].fillna(df['BloodPressure'].mean(),inplace=True)

df['SkinThickness'].fillna(df['SkinThickness'].median(),inplace=True)

df['Insulin'].fillna(df['Insulin'].median(),inplace=True)

df['BMI'].fillna(df['BMI'].median(),inplace=True)
df.hist(figsize=(12,10));
import seaborn as sns

sns.pairplot(df,hue='Outcome')
sns.heatmap(df.corr(),annot = True);
df.info()
df.describe()
from sklearn.preprocessing import StandardScaler 

from keras.utils import to_categorical
sc = StandardScaler()

X = sc.fit_transform(df.drop('Outcome',axis=1))

y = df['Outcome'].values

y_cat = to_categorical(y)
#X
X.shape
#y_cat
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y_cat,random_state=22,test_size=0.2)
from keras.models import Sequential 

from keras.layers import Dense

from keras.optimizers import Adam
model = Sequential()

model.add(Dense(32,input_shape=(8,),activation ='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(32,activation='relu'))

#model.add(Dense(32,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(Adam(lr=0.05),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,verbose=2,validation_split=0.1)
model.summary()
y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1)

y_pred_class = np.argmax(y_pred,axis=1)
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
print('Accuracy of model is:',accuracy_score(y_test_class,y_pred_class))
print(classification_report(y_test_class,y_pred_class))
confusion_matrix(y_test_class,y_pred_class)
pd.Series(y_test_class).value_counts()
pd.Series(y_test_class).value_counts()/len(y_test_class)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



for mod in [RandomForestClassifier(),SVC(),GaussianNB()]:

    mod.fit(X_train,y_train[:,1])

    y_pred = mod.predict(X_test)

    print("="*80)

    print(mod)

    print("_"*80)

    print("Accuracy score:{:0.3}".format(accuracy_score(y_test_class,y_pred)))

    print("Confusion Matrix:")

    print(confusion_matrix(y_test_class,y_pred))
from sklearn.ensemble import RandomForestClassifier 

model= RandomForestClassifier(n_estimators=100,random_state=0)

X=df[df.columns[:8]]

Y=df['Outcome']

model.fit(X,Y)

pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)