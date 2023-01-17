from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model 
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

fires = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv')
fires_test = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_nrt_M6_96619.csv')
fires.dataframeName = 'fire_archive_M6_96619.csv'
fires.dataframeName = 'fire_nrt_M6_96619.csv'

fires.head()
fires.head()
confidence = fires['confidence']
dangerlvl = fires.drop(columns = 'confidence')
fires['confidence'] = np.log(fires['confidence'] + 1)
fires.max()
from sklearn.preprocessing import LabelEncoder

categorical = list(fires.select_dtypes(include = ["object"]).columns)
for i, column in enumerate(categorical) :
    label = LabelEncoder()
    fires[column] = label.fit_transform(fires[column])
fires['satellite'].value_counts()
fires['acq_date'].value_counts()
fires.head()
confidence = fires['confidence']
dangerlvl = fires.drop(columns = 'confidence')
fires['confidence'] = np.log(fires['confidence'] + 1)
X_train, X_test, y_train, y_test = train_test_split(dangerlvl, confidence, test_size = 0.15, random_state = 196)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
X_train.head()
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mean_squared_error(y_test, predictions)
r2_score(y_test, predictions)