# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # date time
import matplotlib.pyplot as plt # for plotting

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Add dates and grid number you are interested in

start_date = dt.datetime(2019,12,1)
end_date = dt.datetime(2020,5,31)
grid_nr = 1
grid_col = f'grid{grid_nr}-loss'
# Load the test dataset. Training dataset is not needed here as Persistence model just returns the grid loss values from the week before, hence no training is needed.

test_data = pd.read_csv('../input/grid-loss-time-series-dataset/test.csv', index_col=0)
x_test = test_data[grid_col]
display(x_test.head())
# Defining the Persistence model, in line with other sklearn models

class PersistenceModel:
    def __init__(self):
        pass
    def train(self, x_train, y_train):
        # No training needed
        pass
    def predict(self, x_test):
        # returns the values shifted back by 7 days (i.e. 24*7 hourly values)
        return x_test.shift(24*7)
## Other helful functions

# Calculating the model performance. MAE, RMSE and MAPE are calculated.
def calculate_error(pred, target):
    target = target.loc[pred.index[0]: pred.index[-1]]
    metrics = {
        "mae": np.mean(np.abs(pred - target)), # Mean absolute error
        "rmse": np.sqrt(np.mean((pred - target) ** 2)), # Root mean squared error
        "mape": 100 * np.sum(np.abs(pred - target)) / np.sum(target)} # Mean absolute percentage error
    return metrics

# Visualizing the target and predictions
def plot_predictions(pred, target):
    target.plot(figsize=(30,10), label='target', linewidth=2)
    pred.plot(label='prediction', linewidth=2)
    plt.title('Persistence model performance', fontsize=20)
    plt.xlabel('Date and time', fontsize=18)
    plt.ylabel('Grid loss', fontsize=18)
    plt.xticks(fontsize=14)
    plt.legend()
# Initializing the model
model = PersistenceModel() 

# Returns the last week's values
y_test = model.predict(x_test)   

# Check model's performance
error_metrics = calculate_error(x_test, y_test) 
print(f'Model performance for predicting loss for grid {grid_nr} is: {error_metrics}')

# Visualize the performance
plot_predictions(y_test, x_test)