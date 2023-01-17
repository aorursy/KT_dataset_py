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
# Cargamos los datos de entrenamiento y test
test_data = pd.read_csv("../input/titanic/test.csv").fillna("")
train_data = pd.read_csv("../input/titanic/train.csv").fillna("")

train_data.head()
# Exploramos los datos
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", round(rate_men,1)*100,"%")
print("% of women who survived:", round(rate_women,1)*100,"%")
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier


y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model =  KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)
predictions = model.predict(X_test)
#model.score(predictions, y_test)
   
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('mi_prediccion.csv', index=False)
print("Mi predicción ha sido grabada :)!")