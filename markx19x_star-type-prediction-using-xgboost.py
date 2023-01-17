# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#plotting libs

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D



#sklearn libs

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
star_data = pd.read_csv("../input/star-dataset/6 class csv.csv")

star_data.head()
star_types = {0:"Brown Dwarf", 1:"Red Dwarf", 2:"White Dwarf", 3:"Main Sequence", 4:"Supergiant", 5:"Hypergiant"}

star_data["Star type Decoded"] = star_data["Star type"].map(star_types) 
star_data.isnull().sum()
star_data["Spectral Class"].value_counts()
star_data["Star color"].value_counts()
star_data["Star color"] = star_data["Star color"].str.lower()

star_data["Star color"] = star_data["Star color"].str.replace(' ','')

star_data["Star color"] = star_data["Star color"].str.replace('-','')

star_data["Star color"] = star_data["Star color"].str.replace('yellowwhite','whiteyellow')

star_data["Star color"].value_counts()
le_specClass = LabelEncoder()

star_data["SpecClassEnc"] = le_specClass.fit_transform(star_data["Spectral Class"])

print("Encoded Spectral Classes: " + str(le_specClass.classes_))
le_starCol = LabelEncoder()

star_data["StarColEnc"] = le_starCol.fit_transform(star_data["Star color"])

print("Encoded Star colors: " + str(le_starCol.classes_))
sns.pairplot(star_data.drop(["Star color", "Spectral Class"], axis=1), hue="Star type Decoded", diag_kind=None)

plt.show()
sns.catplot(x="Spectral Class", y="Absolute magnitude(Mv)", data=star_data, hue="Star type Decoded", order=['O', 'B', 'A', 'F', 'G', 'K', 'M'], height=9)

plt.gca().invert_yaxis()
#select all features for learning and the feature we want to predict

x = star_data.select_dtypes(exclude="object").drop("Star type", axis=1)

y = star_data["Star type"]



#split out dataset into a training and a test dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=star_data["Star type"])



scaler = StandardScaler()



#use the scaler to scale our data

x_train_sc = scaler.fit_transform(x_train)

x_test_sc = scaler.transform(x_test)



#since we want to have our dataframe we need to replace it in the corresponding sets

x_train = pd.DataFrame(x_train_sc, index=x_train.index, columns=x_train.columns)

x_test = pd.DataFrame(x_test_sc, index=x_test.index, columns=x_test.columns)

xgb = XGBClassifier(n_estimators=1000, n_jobs=-1, random_state=42)



xgb.fit(x_train, y_train)



y_pred = xgb.predict(x_test)

predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))