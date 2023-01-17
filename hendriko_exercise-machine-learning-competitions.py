# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
#nice corr plots



from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np





def heatmap(x, y, **kwargs):

    if 'color' in kwargs:

        color = kwargs['color']

    else:

        color = [1]*len(x)



    if 'palette' in kwargs:

        palette = kwargs['palette']

        n_colors = len(palette)

    else:

        n_colors = 256 # Use 256 colors for the diverging color palette

        palette = sns.color_palette("Blues", n_colors) 



    if 'color_range' in kwargs:

        color_min, color_max = kwargs['color_range']

    else:

        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation



    def value_to_color(val):

        if color_min == color_max:

            return palette[-1]

        else:

            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range

            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1

            ind = int(val_position * (n_colors - 1)) # target index in the color palette

            return palette[ind]



    if 'size' in kwargs:

        size = kwargs['size']

    else:

        size = [1]*len(x)



    if 'size_range' in kwargs:

        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]

    else:

        size_min, size_max = min(size), max(size)



    size_scale = kwargs.get('size_scale', 500)



    def value_to_size(val):

        if size_min == size_max:

            return 1 * size_scale

        else:

            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range

            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1

            return val_position * size_scale

    if 'x_order' in kwargs: 

        x_names = [t for t in kwargs['x_order']]

    else:

        x_names = [t for t in sorted(set([v for v in x]))]

    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}



    if 'y_order' in kwargs: 

        y_names = [t for t in kwargs['y_order']]

    else:

        y_names = [t for t in sorted(set([v for v in y]))]

    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}



    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid

    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot



    marker = kwargs.get('marker', 's')



    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [

         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'

    ]}



    ax.scatter(

        x=[x_to_num[v] for v in x],

        y=[y_to_num[v] for v in y],

        marker=marker,

        s=[value_to_size(v) for v in size], 

        c=[value_to_color(v) for v in color],

        **kwargs_pass_on

    )

    ax.set_xticks([v for k,v in x_to_num.items()])

    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')

    ax.set_yticks([v for k,v in y_to_num.items()])

    ax.set_yticklabels([k for k in y_to_num])



    ax.grid(False, 'major')

    ax.grid(True, 'minor')

    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)

    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)



    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])

    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    ax.set_facecolor('#F1F1F1')



    # Add color legend on the right side of the plot

    if color_min < color_max:

        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot



        col_x = [0]*len(palette) # Fixed x coordinate for the bars

        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars



        bar_height = bar_y[1] - bar_y[0]

        ax.barh(

            y=bar_y,

            width=[5]*len(palette), # Make bars 5 units wide

            left=col_x, # Make bars start at 0

            height=bar_height,

            color=palette,

            linewidth=0

        )

        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle

        ax.grid(False) # Hide grid

        ax.set_facecolor('white') # Make background white

        ax.set_xticks([]) # Remove horizontal ticks

        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max

        ax.yaxis.tick_right() # Show vertical ticks on the right 





def corrplot(data, size_scale=500, marker='s'):

    corr = pd.melt(data.reset_index(), id_vars='index')

    corr.columns = ['x', 'y', 'value']

    heatmap(

        corr['x'], corr['y'],

        color=corr['value'], color_range=[-1, 1],

        palette=sns.diverging_palette(20, 220, n=256),

        size=corr['value'].abs(), size_range=[0,1],

        marker=marker,

        x_order=data.columns,

        y_order=data.columns[::-1],

        size_scale=size_scale

    )
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

def findCorrelations():

    for i in range(0,80,16):

        #i=32

        myrange = list(range(i,i+16))

        myrange.append(80)

        print(myrange)

        mycolumns = np.array(home_data.columns)[myrange]

        print(mycolumns)

        plt.figure()

        corrplot(home_data[mycolumns].corr())

        

def createDataFromFeatures(features):

    X = home_data[features]



    # Split into validation and training data

    return train_test_split(X, y, random_state=1)



def testModelOnTestData():

    # Define the model. Set random_state to 1

    rf_model = RandomForestRegressor(random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



def trainModelOnFullData():

    # To improve accuracy, create a new Random Forest model which you will train on all training data

    rf_model_on_full_data = RandomForestRegressor(random_state=1)



    # fit rf_model_on_full_data on all data from the training data

    rf_model_on_full_data.fit(X,y)
new_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','OverallQual', 'TotalBsmtSF','GrLivArea','GarageCars','FullBath']

train_X, val_X, train_y, val_y = createDataFromFeatures(features)

#train_X.columns

testModelOnTestData()

train_X, val_X, train_y, val_y = createDataFromFeatures(new_features)

#train_X.columns

testModelOnTestData()
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
step_1.check()

#step_1.solution()