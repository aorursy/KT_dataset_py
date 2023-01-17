# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sea


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fString = "/kaggle/input/nfl-example/2019_nfl_preclean.csv" # use the built-in Kaggle code printed file names for input data

teams = pd.read_csv(fString) # create a pandas dataframe and assign it to the teams data
teams.head() # preview the data and what it looks like
teams.info() # get a summary of our preview dataset
fileString = "/kaggle/input/all-nfl/all_nfl.csv" # file location for our complete dataset
data = pd.read_csv(fileString) # use the pandas dataframe to read the csv
data.head() # preview the data to make sure it looks right
data.info() # get a summary of our complete datset to ensure that nothing seems out of the place
plt.figure(figsize=(20,10)) # set our figure plot size
sea.pairplot(data=data[data.columns[2:9]], hue='W') # use the offensive total columns after team and year and plot them using seaborn pairwise plotting function
plt.figure(figsize=(30,15)) # set our figure plot size larger since there are more columns to account for
offensive_data = data.iloc[:, np.r_[2,10:27]] # based on our earlier data, grab the index strings using numpy's slice concatenation abizlity
sea.pairplot(data=offensive_data[offensive_data.columns[:]], hue='W') # plot again
plt.figure(figsize=(20,10)) # set our figure plot size small again
total_defensive_data = data.iloc[:, np.r_[2,28:35]] # slice for new defensive total columns
sea.pairplot(data=total_defensive_data[total_defensive_data.columns[:]], hue='W') # plot defensive totals
plt.figure(figsize=(20,15)) # set our figure plot size
ind_defensive_data = data.iloc[:, np.r_[2,36:52]] # slice
sea.pairplot(data=ind_defensive_data[ind_defensive_data.columns[:]], hue='W') # plot individual defensive data
data.info() # grab the column titles against based on our conclusions and what we want to evaluate against
features = data.iloc[:, np.r_[3:5, 7, 28,42]] # create a new dataframe with the all rows for the columns we are interested in
features # preview the dataframe
target = data['W'] # isolate the wins column
target # preview
line_reg = LinearRegression() # create our linear regression model
line = line_reg.fit(features, target) # train it based on our 2015-2019 data
r2 = line.score(features, target) # get the R^2 value for our model
print(r2)


est = line_reg.predict(features) # use our model to calculate the expected wins
est_data = round(data['W']-est) # round our subtraction
est_data.values
est_data.value_counts() # find out what the distribution of our values is
file2014 = "/kaggle/input/2014-nfl/2014_nfl.csv" # file location for our 2014 data
data_2014 = pd.read_csv(file2014) # read csv
features_2014 = data_2014.iloc[:, np.r_[3:5, 7, 28,42]]
est_2014 = line_reg.predict(features_2014) # use our model to calculate the expected wins
est14_data = round(data_2014['W']-est_2014) # round our subtraction
est14_data.values
est14_data.value_counts() # find out what the distribution of our values is
features_all = data.iloc[:, np.r_[3:53]] # create features out of everything
features_all

all_reg = LinearRegression() # create our linear regression model
over = all_reg.fit(features_all, target) # train it based on our 2015-2019 data
r22 = over.score(features_all, target) # get the R^2 value for our model
print(r22)
est_all = all_reg.predict(features_all) # use our new overfit model
est_all_data = round(data['W']-est_all) # round the predictions
est_all_data.values
est_all_data.value_counts() # find out what the distribution of our values is
features_all_2014 = data_2014.iloc[:, np.r_[3:53]] # try the 2014 data now
over_est_2014 = all_reg.predict(features_all_2014) 
over_est14_data = round(data_2014['W']-over_est_2014) # round
over_est14_data.values
over_est14_data.value_counts()