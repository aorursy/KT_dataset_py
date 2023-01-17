#import packages

import numpy as np # linear algebra & matrix functions

import pandas as pd # data frames



#for these kaggle notebooks this shows what files are on the machine

import os

print(os.listdir("../input"))



#This data comes from donors choose. it is a set of applications for projects to receive donations
#Read a csv into a pandas data frame. This file has a lot of text in it so it takes a few seconds to read. 

train = pd.read_csv("../input/train.csv")



#See the first 5 rows 

train.head()
#looks like there is a distinct ID for each submission. let's make that column the index of this data frame

train.set_index(['id'], inplace = True)



#get info about the df

train.info()
#most of the columns are objects, which is a generic data type and not memory efficient. There are a few ints though so this will show some summary stats:

train.describe()
# let's look at just a single column. You can call that by 

train.project_is_approved.head()
#you can also call a single column like this:

train['project_is_approved'].head()
#the .column_name way is convenient, but it can't be used to assign a new column. for example, you have to do:



#assign like this.........call like this

train['half_approved'] = train.project_is_approved/2



train.half_approved.head()
# you can also use a list of columns in the df[] notation

train[['half_approved', 'project_is_approved']].head()
#the half approved column is nonsense so lets just drop it. 

#The axis = 1 means you are dropping a column rather than a row. if you did axis = 0 you would pass an index value rather than a column name

train.drop(['half_approved'], axis = 1, inplace = True)
# in order to clean up the data, group all of the columns by type. Each group is stored in a list

cat_cols = ['teacher_prefix', 'school_state','project_grade_category','project_subject_categories','project_subject_subcategories']

text_cols = ['project_title','project_essay_1','project_essay_2','project_essay_3','project_essay_4', 'project_resource_summary']

num_cols=['teacher_number_of_previously_posted_projects','project_is_approved']



#use the list to view a subset of columns:

train[num_cols].head()
#you can iterate through lists sequentially using a for loop. You can also start with a blank list and append to it.

new_empty_list = []





#use a for loop to go through each item in the list

for item in num_cols: #you could call 'item' anything, it's just a generic variable

    #take the mean of each column and append it to the new list. 

    new_empty_list.append((round(train[item].mean(),2))) 



#the .format() method is a clean way to insert variables into strings.   

print("the mean for {} is {} and the mean for {} is {}".format(num_cols[0],new_empty_list[0],num_cols[1],new_empty_list[1]))
#use a list to change the data type of some columns. 

"""

This for loop goes though every column in the categorical list, and takes the unique values in that column,orders them, and saves those to col_cats for each iteration

It stores each of those lists of unique values in another list, cat_list. This is to reference the categories for each column

Lastly, convert the column to a categorical column, also passing through the unique values 

"""

cat_list = []

for col in cat_cols:

    #the .astype(str) converts the values in that columns to strings. you could also do int, float or category.

    col_cats = pd.unique(train[col].astype(str)) 

    cat_list.append(col_cats)

    #converting to categorical this way instead of just doing .astype(category) since we can pass through the unique categories rather than relying on them to be inferred correctly

    train[col] = pd.Categorical(train[col], categories = col_cats)

    

#compare the size of the df here to what it was before. We shaved like 20% of it off by converting to cats.

train.info()
# the column 'project_submitted_datetime' is obviously a date so lets convert it

train['project_submitted_datetime'] = pd.to_datetime(train.project_submitted_datetime, utc=True, format='%Y-%m-%d %H:%M:%S')



#that column is a timestamp, but maybe we are more interested in the month that it was submitted

#sometimes there are functions that work on a single value but not an entire column. 

#you can apply those functions to the whole column by using a for loop to go through every value in the column

#below is the cleanest way of doing that, it's called a list comprehension



train['month_submitted'] = [date.month for date in train.project_submitted_datetime]



#lets make sure it checks out:

train[['project_submitted_datetime','month_submitted']].head()
# I wonder if the month submitted has any relationship to the acceptance rate

# here we take the mean of the approved column (which amounts to the acceptance rate since it is binary) and group by month_submitted



months = train[['month_submitted','project_is_approved']].groupby(['month_submitted']).mean()

print(months)
#ok but that averages over several years. Let's look at the monthly trend over time. 

#We'll need to keep the year in this set, so let's truncate these timestamps to months



train['month_datetime'] = train['project_submitted_datetime'].values.astype('<M8[M]') 



#view the col

train.month_datetime.head()
#cool, lets make another df with monthly acceptance rates

months_timestamps = train[['month_datetime','project_is_approved']].groupby(['month_datetime']).mean()



#sort the df by the timestamps

months_timestamps.sort_values(['month_datetime'], ascending = True, inplace = True)

months_timestamps.head()
#ok lets see how it looks over time with a graph

import matplotlib.pyplot as plt #graphing package



#lets set our x and y values

x = months_timestamps.index.values # you can call the index like you can call a column name

y = months_timestamps.project_is_approved.values # the .values method takes just the values in the column, and does not carry along any other information



plt.plot(x,y)

plt.xlabel('month')

plt.ylabel('acceptance rate')

plt.title('project acceptance rate over time')

plt.show()
#lets go back to the text columns and covert them to strings

for col in text_cols:

    train[col] = train[col].astype(str)
#you could also do it without a for loop by applying to all cols at once

train[text_cols] = train[text_cols].astype(str)
#ok lets play around with the text a bit. I wonder if essays that talk about tech are more likely to get approved



train['mentioned_tech'] = train['project_essay_1'].str.contains('technology') #returns true if the essay contains the word poverty



#aggregate and group by the new col

train[['mentioned_tech', 'project_is_approved']].groupby(['mentioned_tech']).mean()
# there was another CSV included in this set that shows the resources requested



resources = pd.read_csv("../input/resources.csv")

resources.set_index(['id'], inplace = True)

resources.head()

#lets look at the total cost of each project

resources['cost'] = resources.quantity * resources.price



# aggregate to the project level

project_costs = resources[['cost','quantity']].groupby(by = 'id').sum()



project_costs.head()
#plot a histogram, but limit the range to remove crazy outliers

plt.hist(project_costs.cost, range = (0,3000)) 

plt.xlabel('cost')

plt.ylabel('count of projects')

plt.title('project cost distribution')

plt.show()
#alright lets look at cost vs. approval rate. we need to join the dfs together



train_cost = train.merge(project_costs, left_on = 'id', right_on = 'id', how = 'left')

train_cost.head(1)
#check some summary stats

train_cost[['project_is_approved','cost','quantity']].describe()
#find average costs of approved and not approved

train_cost[['project_is_approved','cost','quantity']].groupby(['project_is_approved']).mean()