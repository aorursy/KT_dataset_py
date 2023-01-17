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
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
data_path = '/kaggle/input/world-happiness/2017.csv'
data = pd.read_csv(data_path)
data.describe()
data.head(5)
data.columns
data = data.dropna(axis=0)
y = data['Happiness.Score']
features = ['Economy..GDP.per.Capita.','Health..Life.Expectancy.','Freedom']
X = data[features]
X.head()
y.head()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestRegressor(random_state=1)
model.fit(train_X,train_y)
predicted = model.predict(val_X)
mae = mean_absolute_error(predicted,val_y)
print(mae)
model_full = RandomForestRegressor(random_state=1)
model_full.fit(X,y)
predicted_full = model_full.predict(X)
mae_full = mean_absolute_error(predicted_full,y)
print(mae_full)
data_path_2018 = '/kaggle/input/world-happiness/2018.csv'
data_2018 = pd.read_csv(data_path_2018)
data_2018.describe()
data_2018.columns
y_2018 = data_2018.Score
countries = ['Country or region']
country_2018 = data_2018[countries]
features_2018 = ['GDP per capita','Healthy life expectancy','Freedom to make life choices']
X_2018 = data_2018[features_2018]
X_2018.head()
country_2018.head()
predicted_2018 = model.predict(X_2018)
mae_2018 = mean_absolute_error(predicted_2018,y_2018)
print(mae_2018)
output = pd.DataFrame({'Actual_Score': y_2018, 'Predicted_Score': predicted_2018})
output.insert(2,'country',country_2018,True)
output.to_csv('WorldHappinessReport.csv', index=False)
Final_result = pd.read_csv('WorldHappinessReport.csv')
Final_result.describe()
Final_result.head(10)
compression_opts = dict(method='zip',
                        archive_name='out.csv')  
Final_result.to_csv('out.zip', index=False,
          compression=compression_opts)  
os.listdir()

data_path_output = 'WorldHappinessReport.csv'
data_output = pd.read_csv(data_path_output)
data_output.describe()
data_output.head()
compression_opts = dict(method='zip',
                        archive_name='final_output.csv')  
data_output.to_csv('final_output.zip', index=False,
          compression=compression_opts)  
data_output.to_csv('WorldHappinessReport_output.csv', index=False)
data_output.head(156)