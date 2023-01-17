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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data                  
data.isnull().sum()
f = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

for _ in f:
    print(_,"= ",(data[_]==0).sum())   
data['BMI']=data.BMI.mask(data.BMI == 0,(data['BMI'].mean(skipna=True)))
data['SkinThickness']=data.SkinThickness.mask(data.SkinThickness == 0,(data['SkinThickness'].mean(skipna=True)))
data['BloodPressure']=data.BloodPressure.mask(data.BloodPressure == 0,(data['BloodPressure'].mean(skipna=True)))
data['Glucose']=data.Glucose.mask(data.Glucose == 0,(data['Glucose'].mean(skipna=True)))
print(data.head(15))
data['Age']=data['Age'].astype(int)
data.loc[data['Age'] <= 16, 'Age']= 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[data['Age'] > 64, 'Age'] = 4

data['Glucose']=data['Glucose'].astype(int)
data.loc[data['Glucose'] <= 80, 'Glucose']= 0
data.loc[(data['Glucose'] > 80) & (data['Glucose'] <= 100), 'Glucose'] = 1
data.loc[(data['Glucose'] > 100) & (data['Glucose'] <= 125), 'Glucose'] = 2
data.loc[(data['Glucose'] > 125) & (data['Glucose'] <= 150), 'Glucose'] = 3
data.loc[data['Glucose'] > 150, 'Glucose'] = 4

data['BloodPressure']=data['BloodPressure'].astype(int)
data.loc[data['BloodPressure'] <= 50, 'BloodPressure']= 0
data.loc[(data['BloodPressure'] > 50) & (data['BloodPressure'] <= 65), 'BloodPressure'] = 1
data.loc[(data['BloodPressure'] > 65) & (data['BloodPressure'] <= 80), 'BloodPressure'] = 2
data.loc[(data['BloodPressure'] > 80) & (data['BloodPressure'] <= 100), 'BloodPressure'] = 3
data.loc[data['BloodPressure'] > 100, 'BloodPressure'] = 4

data
data.drop(['Insulin'], axis = 1)
def bar_chart(feature):
    Positive = data[data['Outcome']==1][feature].value_counts()
    Negative = data[data['Outcome']==0][feature].value_counts()
    df = pd.DataFrame([Positive,Negative])
    df.index = ['Positive','Negative']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
bar_chart('Glucose')
bar_chart('BloodPressure')
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True )
plt.show()
X = data.iloc[: , :-1].values
y = data.iloc[: , -1].values
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size = 0.2,random_state=0 )
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
q = []
i=[3,4,5,6]
j=[3,4,5,6]

for a in i:
    for b in j:
        #initialising the ANN
        classifier = Sequential()

        #adding the input layer and the first hidden layer
        classifier.add(Dense(activation="relu", input_dim=8, units=a, kernel_initializer="uniform"))

        #adding the second hidden layer
        classifier.add(Dense(output_dim = b, init ='uniform', activation="relu"))

        #adding the output layer
        classifier.add(Dense(output_dim = 1, init ='uniform', activation="sigmoid"))

        #compiling the ANN
        classifier.compile(optimizer = "adam", loss ="binary_crossentropy" , metrics = ['accuracy'])

        #fitting the ANN to the training set
        classifier.fit(X_train,y_train , batch_size = 10 , nb_epoch = 100)
        
        y_pred = classifier.predict(X_test)

        y_pred = (y_pred>0.5)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test,y_pred)
        accuracy=((cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+cm[0,0]+cm[1,0]))
        q.append(accuracy)
        print(q)
q