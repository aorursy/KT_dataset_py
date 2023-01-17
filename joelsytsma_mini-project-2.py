## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

import sklearn.metrics as sm

from numpy.polynomial.polynomial import polyfit

from sklearn.metrics import r2_score

from scipy import stats

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#happiness2015 = pd.read_csv("../input/world-happiness/2015.csv")

#happiness2016 = pd.read_csv("../input/world-happiness/2016.csv")

#happiness2017 = pd.read_csv("../input/world-happiness/2017.csv")

#happiness2018 = pd.read_csv("../input/world-happiness/2018.csv")

#happiness2019 = pd.read_csv("../input/world-happiness/2019.csv")

#happinessList= [happiness2015,happiness2016,happiness2017,happiness2018,happiness2019] 

#happinessMash = pd.DataFrame()

#(above) putting all these dictionaries in a list so I can iterate through and do the same action on all of them

#year= 2015

# (above) setting variable to equal the year that represents the first year of the data set I'm looking at.

#for column in happinessList:

#    columnList= list(column.columns)

#    #for header in columnList:

#        #if 'Country'in header:

#            #print(year,header)

#            #x= column[header]

#            #print(x)

#        #if 'Score' in header:

#            #print(year, header)

#(above) I realized that the different years have different column names, so now I need to clean that up. I'm thinking I will just keep rank and score.

#after a couple hours trying to iterate through the column names and then dropping the unselected columns, I'm realizing there may be an easier way here

#    column['Year']= year

#    happiness2015 = happiness2015['Country']

#    year = year+1

#adding the year variable to the data set so that when I combine these dataframes they are able to report the year of each data



#happinessMash
happiness2015 = pd.read_csv("../input/world-happiness/2015.csv")

happiness2016 = pd.read_csv("../input/world-happiness/2016.csv")

happiness2017 = pd.read_csv("../input/world-happiness/2017.csv")

happiness2018 = pd.read_csv("../input/world-happiness/2018.csv")

happiness2019 = pd.read_csv("../input/world-happiness/2019.csv")

happinessList= [happiness2015,happiness2016,happiness2017,happiness2018,happiness2019] 

#(above) putting all these dictionaries in a list so I can iterate through and do the same action on all of them

year= 2015

# (above) setting variable to equal the year that represents the first year of the data set I'm looking at.

for column in happinessList:

    column['Year']= year

    year = year+1

#adding the year variable to the data set so that when I combine these dataframes they are able to report the year of each data







#reordering the column names and only including data I want in the final piece of data.



happiness2015= happiness2015[['Year','Country','Happiness Rank','Happiness Score']]

happiness2016= happiness2016[['Year','Country','Happiness Rank','Happiness Score']]

happiness2017= happiness2017[['Year','Country','Happiness.Rank','Happiness.Score']]

happiness2018= happiness2018[['Year','Country or region','Overall rank','Score']]

happiness2019= happiness2019[['Year','Country or region','Overall rank','Score']]



#changing the column names to match. I originally just changed the column names by reassigning the column names. df = df.columns ['before':'after']

#but this didn't change the table at a deep enough level. I kep on getting an index reference error. So I tried this and it seemed to work.

happiness2017_2 = happiness2017.set_axis(['Year', 'Country', 'Happiness Rank', 'Happiness Score'], axis=1, inplace=False)

happiness2018_2 = happiness2018.set_axis(['Year', 'Country', 'Happiness Rank', 'Happiness Score'], axis=1, inplace=False)

happiness2019_2 = happiness2019.set_axis(['Year', 'Country', 'Happiness Rank', 'Happiness Score'], axis=1, inplace=False)







#putting the cleaned datasets in a list

happinessList_2 = [happiness2015, happiness2016, happiness2017_2, happiness2018_2, happiness2019_2]

#combining all the happiness dataframes into one

h_mash= pd.concat(happinessList_2).reset_index(drop=True)

h_mash





#converting the happinesse rank and happiness score to a numpy so I can start to plot the intersection of these two

X= h_mash[['Happiness Rank','Happiness Score']].to_numpy()
#labeling every dot on the plot

labels = range(0, 781)

#setting the width and height of the display

plt.figure(figsize = (20, 7))

#I'm honestly not sure what this is doing. I've messed with the numbers to see what happens but I get unexpected errors.

plt.subplots_adjust(bottom = 0.1)

#Assigning the values to each axis

plt.scatter(X[:,0],X[:,1], label = 'True Position')

#setting up a for loop that applies the label to every dot that is plotted. I've tried to make this only label every 50th dot but to no avail.

for label, x, y in zip(labels, X[:, 0], X[:, 1]):

   plt.annotate(

      label,xy = (x, y), xytext = (-3, 3),textcoords = 'offset points', ha = 'right', va = 'bottom')

plt.show()



#Creating a numpy array of every x value. In this case the happiness rank

hr = h_mash['Happiness Rank'].to_numpy()

#Creating a numpy array of every y value. In this case the happiness score.

hs = h_mash['Happiness Score'].to_numpy()
#using the scypy library to define everything about the line of best fit. I love libraries!

slope, intercept, r_value, p_value, std_err = stats.linregress(hr,hs)

#setting where the line should draw

def linefitline(b):

    return intercept + slope * b

line1 = linefitline(hr)



#plotting the line we just made

plt.figure(figsize = (20, 7))

plt.scatter(hr,hs)

plt.plot(hr,line1, c = 'g')

plt.show()
#drawing the figure large

plt.figure(figsize = (20, 7))

#setting the line to draw the length of the x2 array and at the height of the mean of the y data.

line2 = np.full(len(hr),[hs.mean()])



average = hs.mean()

print('the average happiness score is', average)

#drawing the scatter plot again

plt.scatter(hr,hs)

plt.plot(hr,line2, c = 'r')

plt.show()
#calculating the difference between each point and the line of best fit

differences_line1 = linefitline(hr)-hs

line1sum = 0

for i in differences_line1:

    line1sum = line1sum + (i*i)



#calculating the difference between each point and y intercept line

differences_line2 = line2 - hs

line2sum = 0

for i in differences_line2:

    line2sum = line2sum + (i*i)

r2 = r2_score(hs, linefitline(hr))

print('The rsquared value is: ' + str(r2))
forest = pd.read_csv("../input/forest-area-of-land-area/forest_area.csv")

forest.tail()
#Setting the forest dataframe to only show the information that I'm interested in. Country name and the year 2015

forest = forest[['CountryName','2015']]

#Taking that truncated dataframe and renaming the column names so that I can join the two dataframes on this key column

simpleF = forest.set_axis(['Country','Forest Percent'], axis=1, inplace=False)

#Setting the happiness report to only show the data I'm interested in. Mainly happiness score.

simpleHR= happiness2015[['Country', 'Happiness Score']]



#Creating a new dataframe that is the forest and happiness dataframes merged.

FandHR = pd.merge(left=simpleF, right=simpleHR, left_on='Country', right_on='Country')

#Printing that to new dataframe to verify the merge worked.

print(FandHR)
f= FandHR['Forest Percent'].to_numpy()

hr2015= FandHR['Happiness Score'].to_numpy()
#using the scypy library to define everything about the line of best fit. I love libraries!

slope, intercept, r_value, p_value, std_err = stats.linregress(hr2015,f)

#setting where the line should draw

def linefitline(b):

    return intercept + slope * b

line1 = linefitline(hr2015)



#plotting the line we just made

plt.figure(figsize = (20, 7))

#using the numpy arrays I've set up earlier to plot a graph.

plt.scatter(hr2015,f)

plt.plot(hr2015,line1, c = 'g')

#describing the graph

plt.title('Forests and happiness')

plt.ylabel('Percentage of the country that is forest')

plt.xlabel('Happiness Score')



plt.show()
#drawing the figure large

plt.figure(figsize = (20, 7))

#setting the line to draw the length of the x2 array and at the height of the mean of the y data.

line2 = np.full(len(hr2015),[f.mean()])



average = f.mean()

print('the average forest percentage score is', average)

#drawing the scatter plot again

plt.scatter(hr2015,f)

plt.plot(hr2015,line2, c = 'r')

plt.show()
#calculating the difference between each point and the line of best fit

differences_line1 = linefitline(hr2015)-f

line1sum = 0

for i in differences_line1:

    line1sum = line1sum + (i*i)



#calculating the difference between each point and y intercept line

differences_line2 = line2 - f

line2sum = 0

for i in differences_line2:

    line2sum = line2sum + (i*i)

r2 = r2_score(f, linefitline(hr2015))

print('The rsquared value is: ' + str(r2))
Military = pd.read_csv("../input/military-expenditure-of-countries-19602019/Military Expenditure.csv")

Military.head()
#Creating new DF that only reference the columns that have matching data in the happiness report

Military2015 = Military[['Name','2015']]

Military2016 = Military[['Name','2016']]

Military2017 = Military[['Name','2017']]

Military2018 = Military[['Name','2018']]

#adding the year column to each of the new DFs

year = 2015

MilitaryList = [Military2015, Military2016, Military2017,Military2018]

for column in MilitaryList:

    column['Year']= year

    year = year+1

#renaming the columns so that I cna do a join

M2015= Military2015.set_axis(['Country', 'Military Spending', 'Year',], axis=1, inplace=False)

M2016= Military2016.set_axis(['Country', 'Military Spending', 'Year',], axis=1, inplace=False)

M2017= Military2017.set_axis(['Country', 'Military Spending', 'Year',], axis=1, inplace=False)

M2018= Military2018.set_axis(['Country', 'Military Spending', 'Year',], axis=1, inplace=False)



#stacking all the DFs I created to create a long DF that I can join with the happiness CSV

ML = [M2015, M2016, M2017, M2018]

M_mash= pd.concat(ML).reset_index(drop=True)

M_mash = M_mash[['Year', 'Country', 'Military Spending']]

#doing an inner join with happiness DF I created way back in box three of this notebook.

MandHS = pd.merge(left=M_mash, right=h_mash, left_on=['Year','Country'], right_on=['Year','Country'])



MandHS
MandHS2= MandHS[['Military Spending','Happiness Score']]

MandHS2 = MandHS2.apply (pd.to_numeric, errors='coerce')

MandHS2 = MandHS2.dropna()

MandHS2
HappScore= MandHS2['Happiness Score'].to_numpy()

MilSpend= MandHS2['Military Spending'].to_numpy()



#using the scypy library to define everything about the line of best fit. I love libraries!

slope, intercept, r_value, p_value, std_err = stats.linregress(HappScore,MilSpend)

#setting where the line should draw

def linefitline(b):

    return intercept + slope * b

line2 = linefitline(HappScore)



#plotting the line we just made

plt.figure(figsize = (20, 7))

#using the numpy arrays I've set up earlier to plot a graph.

plt.scatter(HappScore,MilSpend)

plt.plot(HappScore,line2, c = 'g')

#describing the graph

plt.title('Happiness and Military spending')

plt.ylabel('Military Spending')

plt.xlabel('Happiness Score')



plt.show()
outliers = MandHS.sort_values(by='Military Spending', ascending=False)

outliers.reset_index(drop=True,inplace=True)

outliers.head(20)
variance = np.var(MilSpend)

mu = np.mean(MilSpend)



sigma = math.sqrt(variance)

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.plot(x, stats.norm.pdf(x, mu, sigma))

plt.show()
#I'm doing this the easy way by just dropping the first 8 rows of my sorted database rather and creating a new DF from that.

#This is instead of going through each row and looking for "China" or "US"

#The easy way means I need to drop each non-number again.

noUSAorChina = outliers.iloc[8:]

MandHS3= noUSAorChina[['Military Spending','Happiness Score']]

MandHS3 = MandHS3.apply (pd.to_numeric, errors='coerce')

MandHS3 = MandHS3.dropna()

MandHS3

NOHappScore= MandHS3['Happiness Score'].to_numpy()

NOMilSpend= MandHS3['Military Spending'].to_numpy()



#using the scypy library to define everything about the line of best fit. I love libraries!

slope, intercept, r_value, p_value, std_err = stats.linregress(NOHappScore,NOMilSpend)

#setting where the line should draw

def linefitline(b):

    return intercept + slope * b

line2 = linefitline(NOHappScore)



#plotting the line we just made

plt.figure(figsize = (20, 7))

#using the numpy arrays I've set up earlier to plot a graph.

plt.scatter(NOHappScore,NOMilSpend)

plt.plot(NOHappScore,line2, c = 'g')

#describing the graph

plt.title('No China or USA: Happiness and Military spending')

plt.ylabel('No China or USA: Military Spending')

plt.xlabel('No China or USA: Happiness Score')



plt.show()
#drawing the figure large

plt.figure(figsize = (20, 7))

#setting the line to draw the length of the x2 array and at the height of the mean of the y data.

line4 = np.full(len(NOHappScore),[NOMilSpend.mean()])



average = NOMilSpend.mean()

print('the average military spend without the USA or China is:', average)

#drawing the scatter plot again

plt.scatter(NOHappScore,NOMilSpend)

plt.plot(NOHappScore,line4, c = 'r')

plt.show()
r2 = r2_score(NOMilSpend, linefitline(NOHappScore))

print('The rsquared value is: ' + str(r2))