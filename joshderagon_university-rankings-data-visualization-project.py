# First, import pandas, a data analysis tool for visualizations

import pandas as pd



# We'll also import seaborn, a Python statistical data visualization library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns



#For calculating the slope of the regression line

from scipy.stats import linregress



# Next, we'll load the College World Rankings Dataset.

college = pd.read_csv("cwurData.csv") # the College World Rankings dataset is now a pandas dataframe.



#filter on the year 2015 to condense the data

college2015 = college[college['year']==2015]

#filter on just the tp 100 colleges to condense the data further

college2015 = college2015[college2015['world_rank']<101]

# Now we will take a look at the college2015 DataFrame we have created to see if we imported it correctly.

college2015.head()
#Use the seaborn library to make a linear regression plot. 

#Here we want to see if there is a relationship between the world ranking of a university and its quality of education ranking

p = sns.regplot(x="world_rank", y="quality_of_education", data=college2015, color = "blue")

#You can see that with the top 100 schools, there does seem to be a relationship between quality of education and the world rankings.



#we need to get the points in the graph so we can calculate the slope (relationship) of the regression line.

#this will be the same code for the next few graphs - the only thing we will be changing is the color of the plot and the y measure value

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)
#Use the seaborn library to make a linear regression plot. 

 

#Here we want to see if there is a relationship between the world ranking of a university and its alumni employment ranking

p=sns.regplot(x="world_rank", y="alumni_employment", data=college2015, color = "green")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)

#A slope that is closer to 1 indicates a stronger relationship between the world rank and the measure value.
#Use the seaborn library to make a linear regression plot. 

#Here we want to see if there is a relationship between the world ranking of a university and its quality of faculty ranking

p = sns.regplot(x="world_rank", y="quality_of_faculty", data=college2015, color = "red")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)

#Here we have the strongest relationship between a universities world ranking and it a measure value, that measure value being quality of faculty.

#This means that quality of faculty has the most direct influence on a universities world ranking.
#Here we want to see if there is a relationship between the world ranking of a university and its publications ranking

p=sns.regplot(x="world_rank", y="publications", data=college2015, color = "purple")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)
#Here we want to see if there is a relationship between the world ranking of a university and its broad impact ranking

p = sns.regplot(x="world_rank", y="broad_impact", data=college2015, color = "orange")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)

#The relationship is there, but is not as strong as other measures
#Here we want to see if there is a relationship between the world ranking of a university and its influence ranking

p = sns.regplot(x="world_rank", y="influence", data=college2015, color = "cyan")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)

#The relationship is there, but is not as strong as other measures
#Here we want to see if there is a relationship between the world ranking of a university and its patents ranking

p = sns.regplot(x="world_rank", y="patents", data=college2015, color = "black")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)

#The relationship is there, but is not as strong as other measures
#Here we want to see if there is a relationship between the world ranking of a university and its citations ranking

p = sns.regplot(x="world_rank", y="citations", data=college2015, color = "pink")

x=p.get_lines()[0].get_xdata()

y=p.get_lines()[0].get_ydata()

linregress(x, y)

#The relationship is there, but is not as strong as other measures
#Let's further filter the data on just top 50 schools to see the distribution of world ranks for each country in the top 50

college2015 = college2015[college2015['world_rank']<51]

ax = sns.boxplot(x="country", y="world_rank", data=college2015)

ax = sns.stripplot(x="country", y="world_rank", data=college2015, edgecolor="gray")

#you can see that the US has the most amount of universities in the top 50, but is spread out throughout the top 50

#The UK has less schools in the top 50, but the median rank is a higher rank than any other country listed.
#Now let's refilter the original data to just get universities in the US. 

collegeUSA = college[college['country']=='USA']

#Let's further filter our data to just the top 100 schools

collegeUSA = collegeUSA[collegeUSA['world_rank']<101]

#We will do a boxplot of each year to see the distribution per year of US schools in the top 100 world rankings

ax = sns.boxplot(x="year", y="world_rank", data=collegeUSA)

ax = sns.stripplot(x="year", y="world_rank", data=collegeUSA, jitter=True, edgecolor="gray")

#The plot shows us that US schools typically ranks from top 5 schools all the way to 100 with the exception of 2014.

#In 2014 the distribution was a bit more condensed than other years, but the median ranking was still around 40.

#Another observation from the boxplot is that the median ranking got worse from about 40ish in 2012 to around 45ish in 2015.

#Though only a small change, this is an interesting note, indicating that universities from other countries have surpassed the US's midtier universities.
#Finally, to see the distribution of scores from year to year we will compare them with a Facetgrid and kdeplot.

#Here we have a kdeplot with each color representing a different year and that years' score distribution.

sns.FacetGrid(collegeUSA, hue="year", size=6).map(sns.kdeplot, "score", shade=True).add_legend()