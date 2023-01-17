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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error as mae



%matplotlib inline

plt.rcParams["figure.figsize"] = (10,5)
DATA_POINTS = 50

data = {

    'features': np.arange(DATA_POINTS),

    'target' : np.arange(DATA_POINTS) + np.random.randn(DATA_POINTS)*5

}

df = pd.DataFrame(data)

df.head()
# Plotting mean-line and data-points

df['mean'] = df['target'].mean()

plt.scatter(df['features'], df['target'], label='data points')

plt.plot(df['mean'], 'r', label='mean')

plt.legend()

plt.title(f"MAE : {mae(df['mean'], df['target'])}")

plt.show()
#Drawing 3 straight lines and observing the MAE



line_slope = [0.1, 0.5, 1]

line_intercept = [1, 2, -3]



f = plt.figure(figsize=(15, 6))

gs = f.add_gridspec(1, len(line_slope))



for i in range(len(line_slope)):

    ax = f.add_subplot(gs[i])

    line = df['features'].apply(lambda x: x*line_slope[i] + line_intercept[i])

    plt.scatter(df['features'], df['target'], label='data points')

    plt.plot(line, 'y', label=f'LINE slope({line_slope[i]}), intercept({line_intercept[i]})')

    plt.legend()

    plt.title(f"MAE : {mae(line, df['target'])}")

    plt.plot()

    
# Drawing MAE for different values of slopes



slopes_mae = []

INTERCEPT = 0.3

for slope in range(-10, 10):

    line = df['features'].apply(lambda x: x*slope + INTERCEPT)

    slopes_mae.append(mae(line, df['target']))

plt.plot(slopes_mae, label='cost function: MAE')

plt.ylabel('ERROR')

plt.legend()

plt.show()
# Drawing MSE for different values of slopes



from sklearn.metrics import mean_squared_error as mse



slopes_mse = []

INTERCEPT = 0.3

for slope in range(-10, 10):

    line = df['features'].apply(lambda x: x*slope + INTERCEPT)

    slopes_mse.append(mse(line, df['target']))

plt.plot(slopes_mse, label='cost function: MSE')

plt.ylabel('ERROR')

plt.legend()

plt.show()