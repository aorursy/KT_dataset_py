# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

print('lets get this started...yeah')
page_stats_data_path = '../input/Kaggle_course_on_ML_pageStats.csv'
page_stats_data = pd.read_csv(page_stats_data_path)


#page_stats_data.describe() #checking
#print(page_stats_data.describe()) # checking 2
#print(page_stats_data.columns)

y = page_stats_data.Views # prediction target
moi_predictors = ['Forks','Comments','Voters','Versions', 'Data']
X = page_stats_data[moi_predictors] # predictors DataFrame

print(X.describe())

from sklearn.ensemble import RandomForestRegressor
RFR_model = RandomForestRegressor()
RFR_model.fit(X, y)
print('\n \n the page count predictions are')
print(RFR_model.predict(X))



#validation
from sklearn.metrics import mean_absolute_error

predicted_pageViews = RFR_model.predict(X)
print('\n \n Validation: the mean_absolute_error is')
print(mean_absolute_error(y, predicted_pageViews))



# further validation using train_test_split technique
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 1)
RFR_model_tts = RandomForestRegressor()
RFR_model_tts.fit(train_X, train_y)
val_predictions = RFR_model_tts.predict(val_X)
print('\n\n train test split technique''s MAE is')
print(mean_absolute_error(val_y, val_predictions))
