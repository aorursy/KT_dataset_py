# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/diabetes.csv')

print(df.isnull().any())

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
print(df.describe())
#check for the missing values

print("Number of rows with 0 values for each variable")

for col in df.columns:

    missing_rows = df.loc[df[col]==0].shape[0]

    print(col + ": " + str(missing_rows))

import numpy as np



df['Glucose'] = df['Glucose'].replace(0, np.nan)

df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)

df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)

df['Insulin'] = df['Insulin'].replace(0, np.nan)

df['BMI'] = df['BMI'].replace(0, np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())

df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())

df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())

df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())

df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
print(df.describe())
from sklearn import preprocessing

df_scaled=preprocessing.scale(df)

df_scaled=pd.DataFrame(df_scaled, columns=df.columns)

df_scaled['Outcome']=df['Outcome']

df=df_scaled

print(df.describe().loc[['mean', 'std','max'],].round(2).abs())
from sklearn.model_selection import train_test_split

x=df.loc[:,df.columns !='Outcome']

y=df.loc[:,'Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2)
from keras.models import Sequential

from keras.layers import Dense

model=Sequential()

model.add(Dense(units=32,activation='relu', input_dim=8))

model.add(Dense(16,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=200)



print("Training Accuracy:"+ str((model.evaluate(x_train, y_train)[1]*100).round(2)))



scores = model.evaluate(x_test, y_test)

print("Testing Accuracy:" + str((scores[1]*100).round(2)))
from sklearn.metrics import confusion_matrix

import seaborn as sns



y_test_pred = model.predict_classes(x_test)

c_matrix = confusion_matrix(y_test, y_test_pred)

ax = sns.heatmap(c_matrix, annot=True, 

                 xticklabels=['No Diabetes','Diabetes'],

                 yticklabels=['No Diabetes','Diabetes'], 

                 cbar=False, cmap='Greens')

ax.set_xlabel("Prediction")

ax.set_ylabel("Actual")
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt



y_test_pred_probs = model.predict(x_test)

FPR,TPR,_=roc_curve(y_test,y_test_pred_probs)

plt.plot(FPR,TPR)

plt.plot([0,1],[0,1],'--',color='Black')

plt.title('ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')