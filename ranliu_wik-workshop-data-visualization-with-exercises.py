# First we'll import libraries we need
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
salary = pd.read_csv("../input/philly_salary_with_gender_clean.csv")
salary.head()
# We'll transform the dataset a little bit before plotting. 
# Select year 2018 only
salary2018 = salary[salary.calendar_year==2018]
salary2018.head()
# Select columns and rename them
salary2018 = salary2018[["last_name", "first_name", "gender", "title", "department", "annual_salary"]]
salary2018.columns = ["last_name","first_name", "gender", "title", "department", "salary2018"]
# Get the precentage of female employees in each department
dummy = pd.get_dummies(salary2018['gender'])
salary2018[["female", "male"]]=dummy
salary2018["female_percent"] = salary2018.groupby("department")["female"].transform("mean")
# Get the top 5 departments by number of employees
top5dep = salary2018['department'].value_counts().index[range(0,5)]
top5 = salary2018[salary2018['department'].isin(top5dep)]
# From now on we'll mainly use the new dataset top5 
top5.head()
# Bar plot with counts
top5["department"].value_counts().plot.bar()
# Bar plot with percentages: just devide the count with the total count
(top5['department'].value_counts() / len(top5)).plot.bar()
# I'm putting department on the y axis (a horizontal bar plot)
# to make it easier to read the department names
sns.countplot(y='department', data=top5, order=top5dep)
# Exercise: make a simple count plot using seaborn for gender
# Try to put gender on y and x axises respectively and see the differences

top5["salary2018"].value_counts().sort_index().plot.line()
# A basic kdeplot on salary2018
sns.kdeplot(top5.salary2018)
#notice that there is a very long tail. You can limit the data range before plotting:
sns.kdeplot(top5.query('salary2018 < 150000').salary2018)
top5["salary2018"].plot.hist()
#kde=True will draw the kde line over the histogram. 
sns.distplot(top5.query('salary2018 < 150000').salary2018, bins=10, kde=True)
# Exercise: draw a histogram showing the distribution of female_percent


# Plot the distribuion of department by gender
sns.countplot(y='department', hue="gender", data=top5, order=top5dep)
# Salary distribution by gender
sns.boxplot(x='gender', y='salary2018', data=top5[top5.salary2018<150000])
#Exercise: draw a boxplot showing distribution of salary2018 by department
#Hint: you can put department on the y axis because the department names are long

# Add another categorical variable by specifying "hue"
sns.boxplot(x='salary2018', y='department', hue="gender", data=top5[top5.salary2018<150000])
sns.violinplot(x='gender', y='salary2018', data=top5)
#Exercise: draw violinplot for salary2018 by department

# Exercise: draw a violinplot of salary2018 by department and gender
# Hint: adding hue="gender"

# If the third categorical variable is binary, we can actually show both categories in one violin by stating "split=True"
sns.violinplot(x="salary2018", y="department", hue="gender", data=top5, split=True)
# jitter=True means that points will be spread out
sns.stripplot(x="salary2018", y="gender", data=top5, jitter=True)
# Exercise: plot distribution of salary2018 by department with a strip plot
# Mean salary by department
sns.barplot(y="department", x="salary2018", data=top5)
# You can also add gender by using hue:
sns.barplot(y="department", x="salary2018", hue="gender", data=top5)
# Exercise: Plot mean salary by department and gender
# This time use y=gender, hue=department

# A scatter plot
sns.jointplot(x="female_percent", y='salary2018', data=top5)
# A hex plot
sns.jointplot(x="female_percent", y="salary2018", kind="hex", data=top5)
# Exercise: change kind="kde" to create a ked joint plot
# Isn't this great?!

sns.pairplot(top5)
import matplotlib.pyplot as plt
g = sns.PairGrid(top5, diag_sharey=False)
g.map_lower(sns.kdeplot) #Lower half of the plot will be kde plots
g.map_upper(plt.scatter) #Upper half of the plot will be scatter plots
g.map_diag(sns.kdeplot)  #diagnal will be kdeplots. 
# Exercise: subset the two true continous variables 
# (salary2018 and female_percent) and plot a pairplot for them
# Change the plot types for upper and lower plots using PairGrid
sns.factorplot(y='department', kind="count", col="gender", data=top5, order=top5dep)
# Exercise: change kind=strip and see what happens
# Change to bar plot, this time gender inside each plot and department for each column.
# You can use col_wrap to decide how many subplots you want in one row
sns.factorplot(x="gender", y="salary2018", col="department", col_wrap=3,
                data=top5, kind="bar")
# Exercise: Draw violin plots of salary2018 by gender, column by department, three plots in a row

sns.lmplot(x='female_percent', y='salary2018', data=top5)
# Use hue to draw seperate lines for different groups, 
# Which can be used to show a possible interaction effect 
sns.lmplot(x='female_percent', y='salary2018', hue='gender', data=top5)
# it's also possible to change markers for each group: 
sns.lmplot(x='female_percent', y='salary2018', hue='gender', 
           markers=['o', '*'], data=top5)

# And split your plot into two columns based on gender: 
sns.lmplot(x='female_percent', y='salary2018', hue="gender",   
           col='gender', data=top5)
sns.set_style("darkgrid")
sns.violinplot(x="salary2018", y="department", hue="gender", data=top5, split=True)
#Exercise: choose another theme and see how it changes
sns.set_style("white")
sns.violinplot(x="salary2018", y="department", hue="gender", data=top5, split=True)
sns.despine()
# You can choose which spine to remove:
sns.violinplot(x="salary2018", y="department", hue="gender", data=top5, split=True)
sns.despine(left=True)
# You can also push the spine away from the data
sns.violinplot(x="salary2018", y="department", hue="gender", data=top5, split=True)
sns.despine(offset=10, trim=True)
# Exercise: draw a box plot to show salary distribution by department and hue=gender
# choose the white theme, remove left, top, and right spines
# Push the bottom spine away from the data
sns.set_style("white")
sns.boxplot(x="salary2018", y="department", hue="gender", data=top5)
sns.despine(left=True, offset=10, trim=True)
violin = sns.violinplot(x="salary2018", y="department", hue="gender", data=top5, split=True)
violin.set(xlabel='Salary in 2018', ylabel='', title="Salary by Department")
sns.despine(left=True, offset=10, trim=True)
# Exercise: plot anything you like 
# and change the x & y axes labels and the title as you like
sns.palplot(sns.color_palette("dark"))
# Exercise: check the colorblind palette
sns.palplot(sns.color_palette("Blues"))
sns.palplot(sns.color_palette("Paired"))
sns.set_palette("Paired")
sns.factorplot(y='department', kind="count", col="gender", data=top5, order=top5dep)
# Exercise: choose a different color palette for your plot
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import random
random = random.sample(range(1, len(top5)), 1000)
random_sample=top5.iloc[random]
iplot([go.Scatter(x=random_sample['department'], y=random_sample['salary2018'], mode='markers')])
# Exercise: create an interactive scatterplot showing salary2018 by title
# Hint: just change "department" into "title"

# We can also use go.Bar instead of go.Scatter to create an interactive bar plot:
iplot([go.Bar(x=random_sample['department'], y=random_sample['salary2018'])])
# something more fancy: Histogram2dContour
iplot([go.Histogram2dContour(x=random_sample['female_percent'], 
                             y=random_sample['salary2018'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=random_sample['female_percent'], y=random_sample['salary2018'], mode='markers'
       )])

from plotnine import *
(ggplot(top5, aes("salary2018", "female_percent", color='department'))+
    geom_point()+
    geom_smooth(method="lm"))