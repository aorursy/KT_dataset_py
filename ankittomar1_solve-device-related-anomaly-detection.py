# load the dataset 

import pandas as pd

import numpy as np



df = pd.read_csv('../input/sensor.csv')

df.head(2)
df.tail()



# 01-Apr-2018 to 31-Aug-2018 

# apr, may, jun, jul, aug - 5 months every min data is collected 
del df['Unnamed: 0']
# convert time into index 

df['index'] = pd.to_datetime(df['timestamp'])

df.index = df['index']
# delete the colunmns 

del df['index']

del df['timestamp']
df.head(2)
df['sensor_15'].nunique() # no unique - complete zero

# drop the column 

df.drop(['sensor_15'], axis=1, inplace = True)

df.shape
df.info()
# machine status - no null 

# we will drop na in whole dataframe 

df['sensor_00'].isna().sum()
# machine status

df['machine_status'].unique()#'NORMAL', 'BROKEN', 'RECOVERING' 

df['machine_status'].value_counts()
# draw a countplot for machine status 

import seaborn as sns

sns.countplot(y = df['machine_status'])
# apply label encoder to encode the machine status

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['machine_status'] = le.fit_transform(df['machine_status'])

df['machine_status'].value_counts()



# 1 - normal 

# 2 - recovering 

# 0 - broken
#  look on complete data frame when device is broken

df_broken = df[df.machine_status ==0]

df_broken



# there is no nan value corellation for broken device 

#
import matplotlib.pyplot as plt 

plt.plot(df['sensor_02'])
# imputation for null values 

df['sensor_04'].hist()

# data is skewwed so we need to use median value to fill the data
# let us figureout NaN values 

df['sensor_00'].isna().sum()
df['sensor_50'].isna().sum()
# used ffill method to fill the missing values

df = df.fillna(method='ffill')
X = df.drop(['machine_status'], axis=1)

X.shape
Y = df['machine_status']

Y.shape
# apply the logitic regression 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)
# apply 

logit = LogisticRegression()

model = logit.fit(X_train, y_train)
# predict

y_pred = model.predict(X_test)
# evaluate the model

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cm = pd.crosstab(y_test,y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

cm
# accuracy is not a good metrics for Anomaly detection and imblaanced dataset

accuracy = accuracy_score(y_test, y_pred)

accuracy
# Classification Report

cr = classification_report(y_pred, y_test)

print(cr)
df.shape # look on the shape of the dataset
df1 = df.copy()

df1 = df[(df1.machine_status ==1) | (df1.machine_status ==0)]

df1.shape