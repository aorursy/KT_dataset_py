import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



### Visualization Libraries

from matplotlib import pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")

#sets the gridstyle for plots made 

%matplotlib inline  

#no need for plt.show()
df = pd.read_csv("../input/data.csv")

df.head()
print(df.keys())
def make_histogram(column, bins=None, kde=False, norm_hist=False):

    """

    This function returns a seaborn histogram based on an inputted dataset column.

    :param column: column of dataset

    :param bins: list of bin values of the histogram

    :param kde: boolean of fitting kernel density estimate 

    :param norm_hist: boolean of normalizing histogram

    :returns: histogram of the column

    """

    return sns.distplot(df[column], bins=bins, kde=kde, norm_hist=norm_hist);



#sns.distplot(df["Age"], bins=[15, 20, 25, 30, 35, 40, 45], kde=False, norm_hist=False)

age_histogram = make_histogram("Age")
age_histogram = make_histogram("Age", [15, 20, 25, 30, 35, 40, 45])
uneven_bins = make_histogram("Age", [15, 20, 30, 35, 45])
uneven_bins_normalized = make_histogram("Age", [15, 20, 30, 35, 45], norm_hist=True)
def make_barplot(x_column, y_column, data, x_inches, y_inches, hue=None):

    """

    This function returns a seaborn barplot based on the data columns passed in.

    :param x_column: x-axis column as a string

    :param y_column: y-axis column as a string

    :param hue: hue column as a string

    :param data: dataframe containing above columns

    :returns: barplot of the columns

    """

    #set size of plot bigger to fit the display

    fig = plt.gcf() #create the graph figure

    fig.set_size_inches(x_inches, y_inches) #set figure to x inches and y inches

    return sns.barplot(x=x_column, y=y_column, hue=hue, data=data);



position_longpassing = make_barplot("Position", "LongPassing", df, 20, 10, "Preferred Foot")
def make_scatterplot(x_column, y_column, data, hue=None, regression=False):

    """

    This function returns a seaborn barplot based on the data columns passed in.

    :param x_column: x-axis column as a string

    :param y_column: y-axis column as a string

    :param data: dataframe containing above columns

    :param hue: hue column as a string

    :param regression: boolean of whether to plot regression 

    :returns: barplot of the columns

    """

    if not regression:

        return sns.relplot(x=x_column, y=y_column, hue=hue, data=data);

    else:

        assert hue is None, "Can't have Hue with Regression Plot"

        return sns.regplot(x=x_column, y=y_column, data=data);



acc_stam_regression = make_scatterplot("Stamina", "Acceleration", df, "Preferred Foot")
#Regression Plot

acc_stam_regression = make_scatterplot("Stamina", "Acceleration", df, regression=True)
sns.relplot(x="Stamina", y="Acceleration", hue="Preferred Foot", data=df)

plt.xlabel("Player Stamina Rating")

plt.ylabel("Player Acceleration Rating")

plt.title("FIFA Players' Stamina vs. Acceleration Ratings")
sns.relplot(x="Stamina", y="Acceleration", data=df);

plt.xlim(20, 60)

#plt.ylim(y1, y2) 

#plt.axis([min x, max x, min y, max y])
plt.plot(df["Overall"], df["Potential"]) #plt.scatter

plt.plot(df["Overall"], df["Age"])

plt.xlim(50, 80)

plt.legend(["Potential", "Age"])

plt.xlabel("Overall")
plt.figure()

fig = plt.gcf() #creates the plot figure object

fig.set_size_inches(10, 10) #sets the plot size

plt.subplot(2, 2, 1) #first subplot, 2 rows and 2 columns

sns.distplot(df["Age"], bins=[15, 20, 25, 30, 35, 40, 45], kde=False, norm_hist=False) #make plot

plt.subplot(2, 2, 2) #second subplot, 2 rows and 2 columns

sns.distplot(df["Potential"], kde=False, norm_hist=False) #make plot

plt.subplots_adjust(left=0) #shift to make plots legible