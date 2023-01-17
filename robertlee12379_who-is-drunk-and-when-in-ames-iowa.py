# import pandas

import pandas as pd



# read the data into your workspace

ba_data = pd.read_csv('../input/breath_alcohol_ames.csv')



# quickly inspect the data

print(ba_data)



# obtain counts for each year 

ba_year = ba_data['year'].value_counts()

ba_year
# use value_counts to tally up the totals for each department

pds = ba_data['location'].value_counts()

pds
%matplotlib inline



# count by hour 

hourly = ba_data.groupby(['hour']).size()

print(hourly)



# create a vertical bar graph of the arrest count by hour

hourly.plot.bar(x=hourly)
# count by month and arrange by descending frequency

monthly = ba_data.groupby(['month']).size()



# use plot.bar to make the appropriate bar chart

monthly.plot.bar(x='monthly')
# Check the column names

# print(ba_data.columns)



# count by gender

counts_gender = ba_data['gender'].value_counts()



# Men took more tests

print(counts_gender)



# create a dataset with no NAs in gender 

gen = ba_data.dropna(subset=['gender'])



# create a mean test result variable

mean_bas = gen.assign(meanRes=(gen.Res1+gen.Res2)/2)



# # create side-by-side boxplots to compare the mean blood alcohol levels of men and women

mean_bas.boxplot(['meanRes'], by = 'gender')
# Filter the data

duis = ba_data[(ba_data.Res1 > 0.08) | (ba_data.Res2 > 0.08)]



# proportion of tests that would have resulted in a DUI

p_dui = duis.shape[0] / ba_data.shape[0]

p_dui
# Create date variable

ba_data['date'] = pd.to_datetime(ba_data[['year', 'month', 'day']])



# Create a week variable

ba_data['week'] = ba_data['date'].dt.week



# Check your work

ba_data.head()
# choose and count the variables of interest

timeline = ba_data.groupby(['week','year']).count()['Res1']



# unstack and plot

timeline.unstack().plot(title='VEISHEA DUIs', legend=True)
## Was it right to permanently cancel VEISHEA? TRUE or FALSE?  

canceling_VEISHEA_was_right = True