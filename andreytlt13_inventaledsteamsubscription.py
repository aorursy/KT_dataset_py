# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as metrics





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# mapping_Contacto = {'telephone': 0, 'cellular': 1}

# data = data.replace({'Contacto': mapping_Contacto})
data = pd.read_csv("../input/dl-term-deposit/train.csv")

data.head()
sampleSubmission = pd.read_csv("../input/dl-term-deposit/sampleSubmission.csv")

sampleSubmission.head()
segment = data['Tipo_Trabajo'].value_counts(). reset_index()

segment.columns = ['Tipo_Trabajo', 'Count'] # Changed the column names

plt.figure(figsize= (20,5)) # Make a plot size

trace = sns.barplot(x = segment['Tipo_Trabajo'], y = segment['Count'], data = segment)

# Adding values on the top of the bars

for index, row in segment.iterrows():

    trace.text(x = row.name, y = row.Count+ 2, s = str(row.Count),color='black', ha="center" )

plt.show()
segment = data['Mes'].value_counts(). reset_index()

segment.columns = ['Mes', 'Count'] # Changed the column names

plt.figure(figsize= (20,5)) # Make a plot size

trace = sns.barplot(x = segment['Mes'], y = segment['Count'], data = segment)

# Adding values on the top of the bars

for index, row in segment.iterrows():

    trace.text(x = row.name, y = row.Count+ 2, s = str(row.Count),color='black', ha="center" )

plt.show()
data.groupby('y').size()

data.columns

catdf = pd.get_dummies(data[['Tipo_Trabajo','Estado_Civil','Educacion', 'Incumplimiento', 'Vivienda', 'Consumo', 'Resultado_Anterior']])

data = pd.concat([data, catdf],axis=1, sort=False)

data.head()
data.drop(columns = ['Tipo_Trabajo','Estado_Civil','Educacion', 'Incumplimiento', 'Vivienda', 'Consumo', 'Resultado_Anterior', 'ID'], inplace = True)

data['Contacto'] = pd.Categorical(data['Contacto'])

data['Contacto'] = data['Contacto'].cat.codes



data['Mes'] = pd.Categorical(data['Mes'])

data['Mes'] = data['Mes'].cat.codes



data['Dias'] = pd.Categorical(data['Dias'])

data['Dias'] = data['Dias'].cat.codes
data.head()
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import RandomOverSampler
x_train, x_test, y_train, y_test = train_test_split(data, data['y'], random_state=7)

x_train = x_train.drop(columns = ['y'])

x_test = x_test.drop(columns = ['y'])

x_train.columns

x_train.shape
# Create the classifier and fit it to our training data

model = RandomForestClassifier(random_state=7, n_estimators=100)

model.fit(x_train, y_train)

# Predict classes given the validation features

y_pred = model.predict(x_test)



# Calculate the accuracy as our performance metric

accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)
# Calculate the confusion matrix itself

confusion = metrics.confusion_matrix(y_test, y_pred)

print(f"Confusion matrix:\n{confusion}")





# Normalizing by the true label counts to get rates

print(f"\nNormalized confusion matrix:")

for row in confusion:

    print(row / row.sum())
probs = model.predict_proba(x_test)

print(probs)
test_raw = pd.read_csv("../input/dl-term-deposit/test.csv")

catdf = pd.get_dummies(test_raw[['Tipo_Trabajo','Estado_Civil','Educacion', 'Incumplimiento', 'Vivienda', 'Consumo', 'Resultado_Anterior']])

test = pd.concat([test_raw, catdf],axis=1, sort=False)
test.drop(columns = ['Tipo_Trabajo','Estado_Civil','Educacion', 'Incumplimiento', 'Vivienda', 'Consumo', 'Resultado_Anterior', 'ID'], inplace = True)

test['Contacto'] = pd.Categorical(test['Contacto'])

test['Contacto'] = test['Contacto'].cat.codes



test['Mes'] = pd.Categorical(test['Mes'])

test['Mes'] = test['Mes'].cat.codes



test['Dias'] = pd.Categorical(test['Dias'])

test['Dias'] = test['Dias'].cat.codes



test.head()
y_test = model.predict(test)

submission = pd.DataFrame(test_raw['ID'],columns=['ID'])

submission['y'] = y_test

submission.head()
submission.to_csv('MySubmission.csv',index=False)