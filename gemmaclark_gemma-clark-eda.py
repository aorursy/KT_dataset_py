# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import johnsonsu





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# EDA Tutorial: https://www.kaggle.com/pavansanagapati/a-simple-tutorial-on-exploratory-data-analysis
# OUR GOAL: PREDICT THE CONCENTRATION OF PARTICLE MATTER USING FRAMES FROM THE VIDEO FEED AND SENSOR DATA





# Step 1: Reading in the csv files

test_data = pd.read_csv('/kaggle/input/pollutionvision/test_data.csv')

    # Tabular data test set

sample = pd.read_csv('/kaggle/input/pollutionvision/sample.csv')

train_data = pd.read_csv('/kaggle/input/pollutionvision/train_data.csv')

    # Tabular data training set
# Step 2: Looking at the training set

train_data.head()

    # 64961 rows, 20 columns

    

# Column nmae meanings

    # Temp = ambient temperature

    # Pressure = air pressure

    # Rel. Humidity = relative humidity

    # Errors = if air measurement equipment has error during sampling (0 = no)

    # Alarm Triggered = if any instrumental warning shows during sampling (0 = no)

    # Dilution factor = instrumental parameter that should be close to 1

    # Dead time = instrumental parameter that should be close to 0

    # Median, mean, geo mean, mode, and geo st dev describe particle sizes which can be ignored

    # Total Conc. = an output variable from the instrument that should not be used

    # Image_file = visual information of traffic condition, corresponding to an image in the frames directory

    # wind_speed = wind velocity during sampling

    # Distance_to_road = distance between the camera and the road

    # Elevation = elevation between camera and breathing zone

    # Total = total measured particle number concentration(#/cm3) = the dependent variable
# Step 3: Cleaning up the dataset to contain only variables we care about

    # Unnamed: 0 seems to be an old indexing column

    # The Median thru Geo. St. Dev. columns describe particle sizes which is superfluous for our goal.

    # Total Conc. (#/cm3) is an instrument output variable which shouldn't be used

    # image_file refers to the image in the frames directory that the data were taken from but is not needed for predicting PM matter



train_data = train_data[['Temp(C)', 'Pressure(kPa)', 'Rel. Humidity', 'Errors', 'Alarm Triggered' , 'Dilution Factor', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation', 'Total']]



train_data.head()
# Step 4: Look at a description of the data

train_data.describe()

    # Observations:

        # Ambient air temperature was regularly warm to hot (25.3C – 40.9C)

        # Pressure barely varied with a coeff of var of 0.376/99.51 = 0.0038

        # Relative humidity was always 0

        # Most (over 75% of) sampling events had no errors (0)

        # The alarm was never triggered so there were no instrumental warnings during any sampling events

        # The dilution factor was always 1 (which is the ideal instrumental parameter)

    # Conclusion from initial observations:

        # We can remove 'relative humidity', 'Alarm Triggered', and 'Dilution Factor' because they were constant across the training dataset
# Step 5: Further narrow down the dataset

train_data = train_data[['Temp(C)', 'Pressure(kPa)', 'Errors', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation', 'Total']]

train_data.head()
# Step 6: Create an array with independent variables so we can loop through them for creating graphs

ind_var = ['Temp(C)', 'Pressure(kPa)', 'Errors', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation']
# Step 7: Create scatter plots to visualize how total PM varies with independent variables

for indy in ind_var:

    plt.scatter(train_data[indy], train_data['Total'])

    plt.xlabel(indy)

    plt.ylabel('Total PM')

    plt.show()



# Observations:

    # Temperature and Total PM may have a very weak positive correlation

    # Pressure and Total PM may have a somewhat weak negative correlation

    # Errors and Total PM may have a very weak negative correlation

    # Dead Time and Total PM have a very strong positive linear correlation

    # Wind Speed and Total PM may have a weak positive correlation

    # Distance to Road and Total PM may have a weak negative correlation

    # Camera angle and Total PM has a moderately strong positive linear correlation

    # Elevation and Total PM has a moderately strong negative linear correlation
# Step 8: Plot a heat map of linear correlation coefficients to quickly visualize and confirm or reject observations from scatter plots

    # Use absolute value to determine any linear correlation, regardless of positive/negative direction



correlation_matrix = train_data.corr()

sns.heatmap(abs(correlation_matrix), cmap = 'coolwarm')

    # Observations

        # Total PM is closely correlated with Dead Time 

        # Total PM is somewhat correlated with cameran angle and pressure

        # Total PM is slightly correlated with distance to road and elevation

        # Total PM is least correlated with temperature, errors, and wind_speed

# Step 9: Create boxplots and histograms to find outliers among variables and visualize the spread of the independent variables



plot_dat = ['Temp(C)', 'Pressure(kPa)', 'Errors', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation', 'Total']



for dat in plot_dat:

    # Create a figure of two plots side by side (1 row, 2 columns)

    fig, (ax1, ax2) = plt.subplots(1,2)

    fig.suptitle(dat)

    fig.tight_layout(pad = 5.0)

    fig.set_figwidth(15)

    

    # Create a horizontal boxplot and add the mean to the figure

    ax1.boxplot(train_data[dat], vert = False, showmeans = True, 

                flierprops = dict(markerfacecolor = 'darkorange', marker = 'o'), 

                meanprops = dict(markerfacecolor = 'royalblue', markeredgecolor = 'royalblue', marker = '^', markersize = 8))

    ax1.set(xlabel = dat)

    

    # Create a histogram of the data

    ax2.hist(train_data[dat], color = 'skyblue')

    ax2.set(xlabel = dat, ylabel = 'counts')

    

    # Plot each histogram/boxplot

    plt.show()





# Observations:

    # The temperature data are skewed left with some outliers at the lower end of the temperature range

    # The pressure data are slightly left skewed but more normally distributed and have no outliers

    # The errors data are overwhelmingly skewed right (value = 0 so no errors) but there are a few outliers where errors occurred

    # The dead time data are very right-skewed with most of the observations low but quite a few outliers to the right of the majority of the data

    # The wind speed data are slightly skewed right but have no outliers

    # The distance_to_road data are somewhat left skewed but also have no outliers

    # The camera_angle data appear to be very normally distributed (but have outliers on both the upper and lower ends)

    # The elevation data are somewhat left skewed but have no outliers

    # The Total PM data are heavily right skewed with most observations at the lower range but many outliers above the majority of the data



# Many of the variables including the dependent data have outliers

    # It would likely be inappropriate to remove the outliers when creating the model because we would miss edge cases
# Step 10: Examine skewness and kurtosis of data



print(train_data.skew())

print('\n')

print(train_data.kurt())



plt.figure(1)

sns.distplot(train_data.skew(), color='blue', axlabel ='Skewness')

plt.show()



plt.figure(2, figsize = (12,8))

sns.distplot(train_data.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)



# Observations: 

    # The skew data align with observations from the boxplots and histograms

    # The dependent variable, 'Total' (PM), is not normally distributed
# Step 11: Try transforming the dependent variable to see if a transformed dataset can be normally distributed

dep = train_data['Total']



# Johnson Distribution

plt.figure(1); plt.title('Johnson SU')

sns.distplot(dep, kde = False, fit = stats.johnsonsu)



# Normal Distribution

plt.figure(2); plt.title('Normal')

sns.distplot(dep, kde = False, fit = stats.norm)



# Log Normal Distribution

plt.figure(3); plt.title('Log Normal')

sns.distplot(dep, kde = False, fit = stats.lognorm)



# Exponential Distribution

plt.figure(4); plt.title('Exponential')

sns.distplot(dep, kde = False, fit = stats.expon)





# Observations

    # None of them fit super well but the Johnson SU or Exponential distribution are probably the best fits among those attempted
# Step 12: Create a PairPlot between 'Total' PM and correlated variables to see how they are related

    # This goes beyond the correlation coefficient values and plots pairwise relationships in a dataset

    # The diagonal is a repeat of the histograms (where the x and y axis has the same variable)

    # The rest of the plots are scatterplots showing the relationship between the variables



sns.set()

columns = ['Temp(C)', 'Pressure(kPa)', 'Errors', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation', 'Total']

sns.pairplot(train_data[columns], height = 2, kind = 'scatter')

plt.show()



# Observations:

    # The only strong linear relationship is between dead time and total PM

    # The dotplots on the 'Total' PM axis are the same as those shown in Step 7 when I made scatter plots of the independent variables to total PM

    # There are no strong linear relationships between any of the independent variables
# Step 13: Analyzing coefficient of variation

coef_var = train_data.std()/train_data.mean()

print(coef_var)



# Observations

    # Errors has a massive coefficient of variation

    # Dead time and 'Total' PM have large coefficients of variation
# Step 14: Address Missing Values



columns = ['Temp(C)', 'Pressure(kPa)', 'Errors', 'Dead Time', 'Wind_Speed', 'Distance_to_Road', 'Camera_Angle', 'Elevation', 'Total']



for col in columns:

    print(col, np.round(train_data[col].isnull().mean(), 4), '% of Missing Values')



# Observation: there are no missing values in the dataset columns/variables of interest