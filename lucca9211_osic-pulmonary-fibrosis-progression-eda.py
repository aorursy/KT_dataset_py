import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('fivethirtyeight')
# Read the Dataset

train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv", index_col="Patient")

test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv", index_col="Patient")

submit_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv", index_col="Patient_Week")
# First Five training data

train_df.head()
# First Five test data

test_df.head()
print('There are  {:} rows in training data.'.format(len(train_df)))

train_df.isna().sum()
# describe training data

train_df.describe()

def plot_fn(df, feature):



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
plot_fn(train_df, 'Weeks')
plot_fn(train_df, 'FVC')
plot_fn(train_df, 'Percent')
plot_fn(train_df, 'Age')