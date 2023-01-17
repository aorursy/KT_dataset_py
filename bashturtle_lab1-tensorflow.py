# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import math



from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_io, estimator



# This line incrasing the amount of logging when there is an error.  You can

# remove it if you want less logging

tf.logging.set_verbosity(tf.logging.ERROR)



print ("Done with the imports.")
# Set the output display to have one digit for decimal places, for display

# readability only and limit it to printing 15 rows.

pd.options.display.float_format = '{:.2f}'.format

pd.options.display.max_rows = 15
# Provide the names for the columns since the CSV file with the data does

# not have a header row.

cols = ['symboling', 'losses', 'make', 'fuel-type', 'aspiration', 'num-doors',

        'body-style', 'drive-wheels', 'engine-location', 'wheel-base',

        'length', 'width', 'height', 'weight', 'engine-type', 'num-cylinders',

        'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',

        'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']





# Load in the data from a CSV file that is comma seperated.

car_data = pd.read_csv('https://storage.googleapis.com/ml_universities/cars_dataset/cars_data.csv',

                        sep=',', names=cols, header=None, encoding='latin-1')



print ("Data set loaded.")
car_data[1:10]
car_data.describe()
car_data.info()
car_data
#there are some ? in the output, let us use the coerce function of pandas, replace ? with nan values



car_data['price'] = pd.to_numeric(car_data['price'], errors='coerce')

car_data['horsepower'] = pd.to_numeric(car_data['horsepower'], errors='coerce')

car_data['peak-rpm'] = pd.to_numeric(car_data['peak-rpm'], errors='coerce')

car_data['city-mpg'] = pd.to_numeric(car_data['city-mpg'], errors='coerce')

car_data['highway-mpg'] = pd.to_numeric(car_data['highway-mpg'], errors='coerce')

car_data['losses'] = pd.to_numeric(car_data['losses'], errors='coerce')

car_data.describe()
car_data.info()
car_data[1:100]
# Replace nan by the mean storing the solution in the same table (`inplace')

car_data.fillna(0, inplace=True)

car_data.info()
#using a scatter plot to visualize the dimensions

INPUT_FEATURE = "horsepower"

LABEL = "price"



plt.ylabel(LABEL)

plt.xlabel(INPUT_FEATURE)

plt.scatter(car_data[INPUT_FEATURE], car_data[LABEL], c='black')

plt.show()
import seaborn as sns

sns.relplot(x="horsepower", y="price", data=car_data);
#before gradient descent , trying to find out the optimal lne which fits the curve

x = car_data[INPUT_FEATURE]

y = car_data[LABEL]

opt = np.polyfit(x, y, 1)

y_pred = opt[0] * x + opt[1]

opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))

slope = opt[0]

bias = opt[1]

print ("Optimal RMSE =", opt_rmse, "for solution", opt)
#to viisualize performance of the linear regressor, use the scatter plot, fit family of lines of corresponding slopes and biases

def make_scatter_plot(dataframe, input_feature, target,

                      slopes=[], biases=[], model_names=[]):

  """ Creates a scatter plot of input_feature vs target along with the models.

  

  Args:

    dataframe: the dataframe to visualize

    input_feature: the input feature to be used for the x-axis

    target: the target to be used for the y-axis

    slopes: list of model weight (slope) 

    bias: list of model bias (same size as slopes)

    model_names: list of model_names to use for legend (same size as slopes)

  """      

  # Define some colors to use that go from blue towards red

  colors = [cm.coolwarm(x) for x in np.linspace(0, 1, len(slopes))]

  

  # Generate the Scatter plot

  x = dataframe[input_feature]

  y = dataframe[target]

  plt.ylabel(target)

  plt.xlabel(input_feature)

  plt.scatter(x, y, color='black', label="")



  # Add the lines corresponding to the provided models

  for i in range (0, len(slopes)):

    y_0 = slopes[i] * x.min() + biases[i]

    y_1 = slopes[i] * x.max() + biases[i]

    plt.plot([x.min(), x.max()], [y_0, y_1],

             label=model_names[i], color=colors[i])

  if (len(model_names) > 0):

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
make_scatter_plot(car_data,INPUT_FEATURE, LABEL,

                  [slope], [bias], ["initial model"])
INPUT_FEATURE = "price"

LABEL = "losses"



# Fill in the rest of this block.

#before gradient descent , trying to find out the optimal lne which fits the curve

x = car_data[INPUT_FEATURE]

y = car_data[LABEL]

opt = np.polyfit(x, y, 1)

y_pred = opt[0] * x + opt[1]

opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))

slope = opt[0]

bias = opt[1]

print ("Optimal RMSE =", opt_rmse, "for solution", opt)



make_scatter_plot(car_data,INPUT_FEATURE, LABEL,

                  [slope], [bias], ["fit on losses"])
# Load in the data from a CSV file that is comma seperated.

car_data_v2 = pd.read_csv('https://storage.googleapis.com/ml_universities/cars_dataset/cars_data.csv',

                           sep=',', names=cols, header=None, encoding='latin-1')

car_data_v2['price'] = pd.to_numeric(car_data_v2['price'], errors='coerce')

car_data_v2['losses'] = pd.to_numeric(car_data_v2['losses'], errors='coerce')
car_data_v2.info()

car_data_nonfilled=car_data_v2
# Fill in what you want to do with the nan here

#compute the mean of the columns 

car_data_v2['losses'].fillna(car_data_v2['losses'].mean(),inplace=True);

car_data_v2['price'].fillna(car_data_v2['price'].mean(),inplace=True);

INPUT_FEATURE = "price"

LABEL = "losses"



# Fill in the rest of this block.

#before gradient descent , trying to find out the optimal lne which fits the curve

x = car_data_v2[INPUT_FEATURE]

y = car_data_v2[LABEL]

opt = np.polyfit(x, y, 1)

y_pred = opt[0] * x + opt[1]

opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))

slope = opt[0]

bias = opt[1]

print ("Optimal RMSE =", opt_rmse, "for solution", opt)



make_scatter_plot(car_data_v2,INPUT_FEATURE, LABEL,

                  [slope], [bias], ["fit on losses"])
#comparing the scatter plots for the mean case and non filled(0) vaues

# Fill in what you want to do with the nan here



car_data_nonfilled['losses'].fillna(car_data_nonfilled['losses'].mode(),inplace=True);

car_data_nonfilled['price'].fillna(car_data_nonfilled['price'].mode(),inplace=True);

INPUT_FEATURE = "price"

LABEL = "losses"



# Fill in the rest of this block.

#before gradient descent , trying to find out the optimal lne which fits the curve

x = car_data_nonfilled[INPUT_FEATURE]

y = car_data_nonfilled[LABEL]

opt = np.polyfit(x, y, 1)

y_pred = opt[0] * x + opt[1]

opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))

slope = opt[0]

bias = opt[1]

print ("Optimal RMSE =", opt_rmse, "for solution", opt)



make_scatter_plot(car_data_nonfilled,INPUT_FEATURE, LABEL,

                  [slope], [bias], ["fit on losses"])