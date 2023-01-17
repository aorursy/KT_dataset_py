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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import tensorflow.keras as tfk
from matplotlib import pyplot as plt 
import seaborn as sns
# Load csv file
df = pd.read_csv("/kaggle/input/titanic/train.csv")
# Set PassengerId as index of the dataframe
df.set_index("PassengerId", inplace=True)
df.head()
df.describe()
df.info()
df.isnull().sum()
df.corr()
# Drop the name, the ticket and cabine number
data = df.drop(["Name", "Ticket", "Cabin"], axis=1)
data.head()
# Search for NaN values
data.isnull().sum()
mean_by = data.groupby(["Pclass", "Sex"]).median()
mean_by
def calculate_age(row):
    '''Function that replace the NaN value in the age colum by the mean age calculate by age, passenger class and sex'''
    if np.isnan(row["Age"]):
        return mean_by.loc[(row["Pclass"], row["Sex"])]["Age"]
    else:
        return row["Age"]
# Replace the age column
data["Age"] = data.apply(calculate_age, 1)
# Check for NaN
data.isnull().sum()
data[data["Embarked"].isnull()]
# Drop the 2 rows with Embarked
# data.dropna(inplace=True)
#  Check for NaN
# Find thanks to google that the 2 passengers embarked at Southampton
data.fillna("S", inplace=True)
data.isnull().sum()
# Create dummy columns for Embarked column
# Create dummy columns for sex column and Pclass column
data_enc = pd.get_dummies(data, columns=["Sex", "Pclass", "Embarked"])
data_enc
# Data standardization
data_scaled = pd.DataFrame(preprocessing.scale(data_enc), columns=data_enc.columns, index=data_enc.index)
data_scaled["Survived"] = data["Survived"]
data_scaled
X = data_scaled.drop("Survived", axis=1)
Y = data_scaled["Survived"]
X
Y
model = tfk.models.Sequential()
model.add(tfk.layers.Dense(64, activation='relu', input_dim=X.shape[1]))
model.add(tfk.layers.Dense(32, activation='relu'))
model.add(tfk.layers.Dense(16, activation='relu'))
#model.add(tfk.layers.Dense(8, activation='relu'))
#model.add(tfk.layers.Dense(4, activation='relu'))
#model.add(tfk.layers.Dense(2, activation='relu'))
#output_layer
model.add(tfk.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X, Y, epochs=300)
scores = model.evaluate(X, Y)
print(f"Training Accuracy: {scores[1] * 100}\n")
# Load csv file
dft = pd.read_csv("/kaggle/input/titanic/test.csv")
# Set PassengerId as index of the dataframe
dft.set_index("PassengerId", inplace=True)
datat = dft.drop(["Name", "Ticket", "Cabin"], axis=1)
# Create dummy columns for sex column
datat
datat.isnull().sum()
# Replace the age column
datat["Age"] = datat.apply(calculate_age, 1)
# Check for NaN
datat.isnull().sum()
datat[datat["Fare"].isnull()]
# Mean calculation on the entire dataset groupby Pclass, Sex and embarked
meant = data.drop("Survived", axis=1).append(datat).groupby(["Pclass", "Sex", "Embarked"]).median()
meant
# Replace the fare NaN value
datat.fillna(meant.loc[(datat.loc[1044]["Pclass"], datat.loc[1044]["Sex"], datat.loc[1044]["Embarked"])]["Fare"], inplace=True)
datat.isnull().sum()
datat
# Create dummy columns for Embarked column
# Create dummy columns for sex column and Pclass column
datat_enc = pd.get_dummies(datat, columns=["Sex", "Pclass", "Embarked"])
datat_enc
# Data standardization
datat_scaled = pd.DataFrame(preprocessing.scale(datat_enc), columns=datat_enc.columns, index=datat_enc.index)
datat_scaled
prediction = pd.DataFrame(model.predict_classes(datat_scaled), index=datat.index, columns=["Survived"])
prediction
prediction.to_csv("prescription.csv")
Y_test_pred_probs = model.predict(X)
FPR, TPR, _ = roc_curve(Y, Y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')