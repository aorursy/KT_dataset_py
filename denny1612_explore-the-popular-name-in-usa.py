import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import bq_helper
import os
#https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
query = """SELECT year, gender, name, sum(number) as count FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
data = usa_names.query_to_pandas_safe(query)
data.to_csv("usa_names_data.csv")
list(data.columns.values)
data.sample(5)
BabyByGender = data.sort_values(by=['year']).groupby(['year', 'gender'])['count'].agg('sum')
BabyGender = BabyByGender.reset_index()
BabyGender
plt.plot(BabyGender['year'].unique(), BabyGender['count'].where(BabyGender['gender'] == 'F').dropna(), color='g', label='Female')
plt.plot(BabyGender['year'].unique(), BabyGender['count'].where(BabyGender['gender'] == 'M').dropna(), color='orange', label='Male')
plt.xlabel('Year')
plt.ylabel('Number of baby')
plt.title('Male & Female babies in USA till 2014')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

def plot_yearly_count(inputname):
    BabyByName = data.sort_values(by=['year']).groupby(['year', 'name'])['count'].agg('sum')
    dataset = BabyByName.reset_index()
    dataframe = dataset.where(dataset['year']>=1995).dropna()
    
    merge = pd.merge(dataframe[['year','count']].where(dataframe['name'] == inputname).dropna(), dataframe['year'].drop_duplicates().reset_index(),on='year',how='right').sort_values(by=['year'])
    
    objects = dataframe['year'].unique()
    y_pos = np.arange(len(objects))
    performance = merge['count']
    
    plt.subplots(figsize=(25,5)) # set the size that you'd like (width, height)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel("# of Applicants")
    plt.xlabel('Year')
    plt.title('Popularity of name '+str(inputname))
    plt.show()
plot_yearly_count("Denny")
PopularName = data.where(data['year'] == 2016).sort_values(by='count', ascending=False).dropna()

#top 5
top20 = PopularName[['name','count']].head(20)

#others
other_names = pd.DataFrame(data = {
    'name' : ['Other'],
    'count' : [PopularName['count'][20:].sum()]
})

df2 = pd.concat([top20, other_names])


# Data to plot

plt.subplots(figsize=(15,15)) # set the size that you'd like (width, height)

labels = top20['name']
sizes = top20['count']


explode = []
for x in range(top20['name'].count()):
    if x == 0:
        explode.append(0.1)
    else:
        explode.append(0.0)

# Plot
plt.pie(sizes,  explode=explode, labels=labels,
        autopct='%1.1f%%', startangle=140)
 
plt.axis('equal')
plt.show()
