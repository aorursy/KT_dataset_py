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
# dataframe 1 (Generation data): clean and mean data by DATE_TIME

df1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)

df1 = df1.drop(columns=['SOURCE_KEY','PLANT_ID'])

df1 = df1.set_index('DATE_TIME').resample('D')['DAILY_YIELD','DC_POWER','AC_POWER'].mean()

df1.info()

df1.head(10)
# dataframe 2 (Weather sensor data): clean and mean data by DATE_TIME

df2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)

df2 = df2.drop(columns=['SOURCE_KEY','PLANT_ID'])

df2 = df2.set_index('DATE_TIME').resample('D')['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION'].mean()

df2.info()

df2.head(10)

# merge dataframe 1 and dataframe 2 by DATE_TIME

df = pd.merge(df1, df2,on='DATE_TIME', how='outer', indicator=True)

df.info()

df.head(10)
X = df[["DC_POWER","AC_POWER","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"]]

y = df[["DAILY_YIELD"]]
X.head()
y.head(20)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=100)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



model = LinearRegression()



# Train the model using the training sets

model.fit(X_train, y_train)



# Make predictions using the testing set

y_pred = model.predict(X_test)



# The mean squared error

print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))



from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=10)

scores  
# 10-fold

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=1, shuffle=True)

print('fold = ',kf.get_n_splits(X))
for train_index, test_index in kf.split(X):

    print("TRAIN:", train_index, "TEST:", test_index)

n = np.array(y)

n