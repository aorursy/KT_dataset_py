import pandas as pd

import numpy as np 

import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

import warnings # ignore warnings



warnings.filterwarnings("ignore")
iris = pd.read_csv('../input/iris/Iris.csv') # loading iris dataset
iris.head() # seeing the first 6 rows from iris dataset
plt.figure(figsize=(10,6)) # difyning plt size

sns.distplot(iris['SepalWidthCm']) # plotting distribution

plt.title('SepalWidthCm Distribution'); # difyning a tittle
iris['SepalWidthCm'].describe() # seeing the principals descriptive statistics using .describe()
# to see the mode we just need to call .mode() in front of our variable

iris['SepalWidthCm'].mode()
# to see the variance we just need to call .var() in front of our variable

iris['SepalWidthCm'].var()
plt.figure(figsize=(10,6)) # defyning the plotsize



plt.plot(iris['SepalWidthCm'] # defyning the data

         , color='darkblue'# defyning the color

         ,label='SepalWidthCm'

         , marker='o' # defyning the type of marker, in this case, i use a dot

         ,linewidth=1 # defyning the line width

         , markersize=2) # defyning the markersize



plt.ylabel('SepalWidth in Cm') # defyning ylabel title

plt.xlabel('Time') # defyning the xlabel title

plt.legend()

plt.title('SepalWidthCm Line Plot'); # defyning the title
plt.figure(figsize=(10,6)) # defyning the plotsize



plt.plot(iris['PetalWidthCm'] # defyning the data

         , color='darkred'# defyning the color

         ,label='PetalWidthCm'

         , marker='o' # defyning the type of marker, in this case, i use a dot

         ,linewidth=1 # defyning the line width

         , markersize=2) # defyning the markersize



plt.ylabel('PetalWidth in Cm') # defyning ylabel title

plt.xlabel('Time') # defyning the xlabel title

plt.legend()

plt.title('PetalWidthCm Line Plot'); # defyning the title
plt.figure(figsize=(10,6)) # creating the figure

plt.hist(iris['SepalWidthCm'] # plotting the histogram

         ,bins=20 # defyning number of bars

         ,label='SepalWidhCm'# add legend

        ,color='darkgreen') # defyning the color



plt.xlabel('SepalWidh in Cm') # add xlabel

plt.ylabel('frequency') # add ylabel

plt.legend()

plt.title('SepalWidhCm distribution');
plt.figure(figsize=(10,6)) # difyning plt size

sns.distplot(iris['SepalWidthCm']

            ,label='SepalWidthCm') # plotting distribution



plt.xlabel('SepalWidh in Cm') # add xlabel

plt.ylabel('frequency') # add ylabel

plt.legend()

plt.title('SepalWidhCm distribution');
Q1 = iris['SepalWidthCm'].quantile(0.25) # calculating Q1

Q3 = iris['SepalWidthCm'].quantile(0.75) # calculating Q3



IQR = Q3 - Q1 # calculating IQR



print("The IQR is:",IQR) # printing IQR
plt.figure(figsize=(14,8)) # defyning plotsize

sns.boxplot(iris['SepalWidthCm'] # plotting boxplot

           ,color='green') # color

plt.xlabel('Distribution') # defyning xlabel title

plt.title('SepalWidthCm Distribution'); # defyning a title
plt.figure(figsize=(6,8)) # defyning plotsize

sns.boxplot(iris['PetalLengthCm'] # plotting boxplot

           ,color='orange'

           ,orient='v') # color

plt.xlabel('Distribution') # defyning xlabel title

plt.title('PetalLenghtCm Distribution'); # defyning a title
plt.figure(figsize=(10,6)) # difyning plot size

iris['Species'][1:125].value_counts().plot(kind='bar' # here i use .value_counts() to count the frequency that each category occurs of dataset

                                    ,color=['darkblue','darkgreen','darkred']) #  counting category values and plotting



plt.ylabel('Frequency') # defyning ylabel title

plt.xticks(rotation=45) # defyning the angle of xlabel text

plt.title('Our class categorys frequency'); # difyning a title
plt.figure(figsize=(10,6)) # difyning plot size

iris['Species'][1:110].value_counts().plot(kind='barh' # here i use .value_counts() to count the frequency that each category occurs of dataset

                                    ,color=['darkblue','darkgreen','darkred']) #  counting category values and plotting



plt.xlabel('Frequency') # defyning ylabel title

plt.title('Our class categorys frequency'); # difyning a title
plt.figure(figsize=(10,10)) # difyning plot size

iris['Species'][1:125].value_counts().plot(kind='pie' # here i use .value_counts() to count the frequency that each category occurs of dataset

                                    ,colors=['yellowgreen', 'lightcoral', 'lightskyblue'] # colors

                                          ,autopct='%1.1f%%',shadow=True, startangle=140) #  putting percentages

plt.legend(); # plotting the legend
plt.figure(figsize=(10,6)) # defyning plotsize

sns.scatterplot(x='SepalLengthCm' # defynin x_axis

                ,y='PetalLengthCm' # defyning y_axis

               ,data=iris) # defyning the data base

plt.title('Relation Between SepalLengthCm and PetalLengthCm'); # defyning a title
sns.lmplot(x='SepalLengthCm' # defynin x_axis

                ,y='PetalLengthCm' # defyning y_axis

               ,data=iris # defyning the data base

          ,aspect=0.3*5) # defyning plotsize

plt.title('Relation Between SepalLengthCm and PetalLengthCm'); # defyning a title
plt.figure(figsize=(12,6)) # defyning plotsize

sns.boxplot(x='Species', # defyning x axis

            y='SepalLengthCm' # defyning y axis

            ,data=iris # defyning the dataset

            ,palette="bright") # defyning the color palette

plt.title('Distribution with boxplots of SepalLengthCm across the Species groups'); # setting a title
plt.figure(figsize=(12,7)) # defyning plotsize

sns.boxplot(data=iris[['SepalLengthCm','PetalLengthCm']] # creating the plot

            ,palette="colorblind") # defyning the color palette

plt.title('Parallel Boxplot of SepalLengthCm and PetalLengthCm'); # setting a title
plt.figure(figsize=(12,6)) # defyning plotsize

sns.barplot(x='Species', # defyning x axis

            y='SepalLengthCm' # defyning y axis

            ,data=iris # defyning the dataset

            ,palette="pastel") # defyning the color palette

plt.title('Distribution with bars of SepalLengthCm across the Species groups'); # setting a title
config = dict(marker='o', linewidth=1, markersize=2) # defyning configs for all plots



plt.figure(figsize=(14,8)) # defyning the plotsize

plt.plot(iris['PetalWidthCm'], **config, label= 'PetalWidthCm')



plt.plot(iris['SepalWidthCm'], **config, label= 'SepalWidthCm')



plt.plot(iris['SepalLengthCm'], **config, label= 'SepalLengthCm')



plt.plot(iris['PetalLengthCm'], **config, label= 'PetalLengthCm')



plt.ylabel('variance') # defyning ylabel title

plt.xlabel('Time') # defyning the xlabel title



# defyning legend config

plt.legend(loc = "upper left"

           , frameon = True

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)



plt.title('Numerical Variables Line plots'); # defyning the title
config = dict(histtype = 'stepfilled', alpha = 0.7, density = True, bins = 20) # defyning configs for all plots



plt.figure(figsize=(14,8)) # defyning the plotsize

plt.hist(iris['PetalWidthCm'], **config, label= 'PetalWidthCm')



plt.hist(iris['SepalWidthCm'], **config, label= 'SepalWidthCm')



plt.hist(iris['SepalLengthCm'], **config, label= 'SepalLengthCm')



plt.hist(iris['PetalLengthCm'], **config, label= 'PetalLengthCm')



plt.ylabel('frequency') # defyning ylabel title

#plt.xlabel('Time') # defyning the xlabel title



# defyning legend config

plt.legend(loc = "upper left"

           , frameon = True

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)



plt.title('Numerical Variables Distributions'); # defyning the title
num_vars = ['SepalLengthCm','PetalLengthCm' # selecting variables

          ,'SepalWidthCm','PetalWidthCm']



plt.figure(figsize=(16,8)) # defyning plotsize

sns.boxplot(data=iris[num_vars], palette="bright") # plotting multiple boxplots

plt.title('Numerical features boxplots distributions and outliers analysis'); # defyning a title
sns.pairplot(iris[num_vars],aspect=0.3*5); # pair plot with numerical variables in iris dataset
plt.figure(figsize=(10,6)) # defyning plotsize

sns.scatterplot(x="SepalLengthCm"

                , y="PetalLengthCm"

                , hue="Species" # defyning the variable to group our plot

                ,data=iris)

plt.title('Simple scatterplot clustering');
sns.lmplot(x='SepalLengthCm' # defynin x_axis

                ,y='PetalLengthCm' # defyning y_axis

           ,hue='Species' # defyning the variable to group our plot

               ,data=iris # defyning the data base

          ,aspect=0.25*6) # defyning plotsize

plt.title('Lm plot separated by Species'); # defyning a title
var = ['SepalLengthCm','PetalLengthCm' # selecting variables

          ,'SepalWidthCm','PetalWidthCm','Species']

sns.pairplot(iris[var],hue='Species',aspect=0.3*5); # pair plot with numerical variables in iris dataset
var = ['SepalLengthCm','PetalLengthCm','SepalWidthCm','PetalWidthCm'] # selecting numerical variables to correlation plot



colormap = plt.cm.RdBu # defyning colormap

plt.figure(figsize=(12,12)) # difyning plot size

plt.title('Pearson Correlation between numerical variables', y=1.05, size=15) # defyning a title to our plot

sns.heatmap(iris[var].astype(float).corr(),linewidths=0.5,vmax=1, 

            square=True, cmap=colormap, linecolor='white', annot=True);