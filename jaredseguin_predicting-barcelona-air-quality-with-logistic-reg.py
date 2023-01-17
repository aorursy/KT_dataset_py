import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

air_quality = pd.read_csv('../input/barcelona-data-sets/air_quality_Nov2017.csv')
air_quality.columns.values
#From observing the data just on kaggle, it seems that the Quality

#indicators are redundent for O3, No2, and PM10, as they simply

#hit as 'good' whenever Value has a value/isn't NaN.



dataframe = air_quality[['O3 Value', 'NO2 Value', 'PM10 Value',

                        'Air Quality']]
#From a quick look, we need to clear NaN entries

dataframe.head()
#Cleaning: Let's remove all the NaN entries



dataframe2 = dataframe.dropna(axis = 0)
dataframe2.head()
#Cleaning: collecting the Air Quality variable that will be 

#converted to a dummy



not_num = dataframe2.select_dtypes(include = ['object']).columns

dataframe2[not_num].head()



dataframe2[not_num].describe()
dummies = pd.get_dummies(dataframe2[not_num], drop_first = True)

dummies.head()
# Now, let's swap the air quality measure with the dummy.

# 1 indicated moderate, 0 indicates good. We saw in output 54 

# that moderate and good are the only 2 unique ratings



dataframe3 = dataframe2.drop(not_num, axis = 1)

dataframe3 = pd.merge(dataframe3, dummies, left_index = True, right_index = True)

dataframe3.head()
# Let's define our X and y sets

X = dataframe3.drop('Air Quality_Moderate', axis = 1)

y = dataframe3['Air Quality_Moderate']
# Let's get our training test splits and run the logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                   test_size = 0.2,

                                                   random_state = 0)
Log_reg = LogisticRegression()

Log_reg.fit(X_train, y_train)
y_pred = Log_reg.predict(X_test)
from sklearn import metrics

metrics.confusion_matrix(y_test, y_pred)
# As you can see, the model does very well with unseen data

# When the model predicted the air quality is GOOD, the air quality was infact good 550 times out of 561. 

# When the model predicted the air quality is MODERATE, the air quality was infact moderate 9 times out of 10

# Overall, the model predicts correctly 97.8% of the time



# Naturally, O3, NO3, and PM10 are main components of the Air Quality Index so they should predict the Air Quality quite well

# However, PM2.5, SO2 and CO are also measures that contribute to the Index

# So, 3 of the 5 or 6 components of the index are on their own accurate almost 98% of the time to see if the Air is Moderate or Good