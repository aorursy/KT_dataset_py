%autosave 10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

        
%matplotlib inline
filename = "/kaggle/input/videogamesales/vgsales.csv"
df = pd.read_csv(filename)
df.head()
df.info()
df.shape
nullvalues_percentage = df.isna().sum()*100 / df.shape[0]
nullvalues_percentage
df.dropna(inplace = True)
nullvalues_percentage = df.isna().sum()*100 / df.shape[0]
nullvalues_percentage
# Uncomment the below line and install jupyterthemes. 
#!pip install jupyterthemes
# To show all available theme
!jt -l
# To apply a theme  #do not forget to uncomment

#!jt -t chesterish
# To go back to the default theme 

#!jt -r
df.dtypes
convert = {'Year':int}

df = df.astype(convert)
df.head(2)
df['Year'].value_counts().sort_values(ascending = True).head()
# Dropping rows with the year 2017 and 2020
drop = df.drop(df[(df['Year'] == 2017) | (df['Year'] == 2020)].index, inplace= True)
df['Genre'].unique()
# Figure style
plt.figure(figsize=[11,5], dpi= 95)
sns.set_style(style= 'whitegrid')

# Plot
sns.countplot(x=df['Genre'])

# Axis and title label
plt.xlabel('Genre',fontsize=12, color='black')
plt.ylabel('Count',fontsize=12, color='black')
plt.title('The Number of Games by Genre', fontsize= 15, color= 'blue')

plt.tight_layout()
plt.figure(figsize=[11,4], dpi= 95)

sns.countplot(x=df['Year'], palette='viridis', order= df['Year'].value_counts().index)

plt.xticks(rotation = 90)
plt.title('Number of Produced Games in Each Year', fontsize= 15, color= 'blue')
plt.figure(figsize=[14,4], dpi= 95)

sns.countplot(x='Year',data=df, hue= 'Genre', palette="dark")

plt.xticks(rotation=90)
plt.figure(figsize=[14,5], dpi= 95)

sns.countplot(x='Year',data=df, hue= 'Genre', palette="dark", order= df['Year'].value_counts().head().index, 
              hue_order=df['Genre'].value_counts().index)

# Axis label
plt.xlabel('Year',fontsize=12, color='black')
plt.ylabel('Count',fontsize=12, color='black')

# Axis ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Title
plt.title('Number of Released Games by Genre in Each Year', fontsize= 15, color= 'blue')

# Legend
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize= 'x-large')
# Number of entries for each genre to find out the number of games
genre_count_by_year = df.groupby(by= ['Year','Genre']).count()

# Unstack method
genre_count_by_year = genre_count_by_year['Rank'].unstack()
genre_count_by_year.head(3)
# Replacing the NaN values with 0
genre_count_by_year.fillna(0, inplace= True)

# Dropping the rows of between year 1980 to 1999 inclusive
genre_count_by_year.drop(range(1980, 1995), axis= 0, inplace= True)
genre_count_by_year.head(2)
# Evolution of Action games only
# Figure style
sns.set_style('darkgrid')
fig_dpi=90
plt.figure(figsize=(14,6), dpi=fig_dpi)

# Line plot
for column in genre_count_by_year.drop('Action', axis=1):
    plt.plot(genre_count_by_year[column], marker= '', color= 'grey', linewidth=1, alpha=0.4)
    
# The highlighted plot with specific genre
plt.plot(genre_count_by_year['Action'], marker= '', color= 'green', linewidth=4, alpha=0.7)

# Increasing the xlimit as we need to add annotation
plt.xlim(1994,2019)

# Adding annotation
plt.text(2016.5, genre_count_by_year.Action.tail(1), 'Action', horizontalalignment='left', size='large', color='green')

# Titles and axis label
plt.title("Evolution of Action games VS others", loc='left', fontsize=12, fontweight=0, color='green')
plt.xlabel("Year")
plt.ylabel("Number of Games")
# Evolution of one game vs others
# Figure style
plt.style.use('seaborn-darkgrid')
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 900/my_dpi), dpi=my_dpi)

# Color palette
palette = plt.get_cmap('tab10')

# Line plot
num = 0
for column in genre_count_by_year:
    num += 1
    
#     if num==10:
#         break
    
    plt.subplot(4,3, num)
    
    # Generating all lineplot
    for item in genre_count_by_year.drop(column, axis= 1):
        plt.plot(genre_count_by_year[item], marker='', color='grey', linewidth=1.2, alpha=0.3)
    
    # Line plot for the expeceted one
    plt.plot(genre_count_by_year[column], marker='', color= palette(num), linewidth=2.4, alpha=0.9, label=column)
    
    # x-axis limit for subplot
    plt.xlim(1994,2017)
    
    # Sub plot title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color= palette(num))
    
    # Axis label for the figure
    if column == 'Sports':
        plt.xlabel('Year(1995-2016)', fontsize= 15)
    if column == 'Misc':
        plt.ylabel('Number of Games', fontsize= 15)
    
    
# Figure title    
plt.suptitle(' Evolution of Each Genre Compare to Others ', fontsize=15, fontweight=0, color='blue', y= 1.03)


plt.tight_layout()
publisher = df['Publisher'].value_counts().sort_values(ascending= False)[:10]
publisher
# we need this to add text in the graph
x = list(publisher)
y = list(publisher.index)

#plotting figure
plt.figure(figsize= [15,5])
fig = publisher.plot(kind='bar', color = 'green')

# labeling the bar
style = dict(ha= 'center', size= 13, color = 'black')

x_position = 0
for i in range(0,len(x)):
    fig.text(x_position, x[i]+2, str(x[i]), **style)
    x_position += 1

# Styling axis
plt.ylabel('Count',fontsize=14, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Title
plt.title('Top Ten Publisher in Number of Produced Games', fontsize= 15, color= 'blue')
# Index of publisher dataframe
top_ten_publisher = list(publisher.index)
print(top_ten_publisher)
# Creating a Dataset
g_1 = df.groupby(by = ['Publisher','Genre'])

# Dictionary to make another dataframe
dic = {'Publisher':[],'Genre':[], 'Count':[]}

for item in top_ten_publisher:
    sub_publisher = g_1.count().loc[item,'Year'].sort_values(ascending = False).head(3)
    dic['Publisher'].extend((item, item, item))  # to keep the publisher name next to one another
    dic['Genre'] += list(sub_publisher.index)    # Appendong genre
    dic['Count'] += list(sub_publisher.values)   # Number of games for each genre 
# Creating a dataset of top three genre for each publisher.
sub_df = pd.DataFrame(dic) 
sub_df.head(5)
#bar height
height = list(sub_df['Count'])

# xtick label 
bars = list(sub_df['Genre'])

# index position
y_pos = np.arange(len(bars))

#color for all the bar values
color = []
name = ['blue','red','orange','green','purple','violet','olive','gray','brown','cyan']
for item in name:
    color.extend((item, item, item))

# Ticks for second axis
label = list(sub_df['Publisher']) 
ax_label = []
for item in label:
    if item not in ax_label:
        ax_label.append(item)
# Figure Style
sns.set_style(style='dark')
plt.figure(figsize=[17,5])

# Plot
plt.bar(y_pos, height, color=color)


plt.xticks(y_pos, bars, rotation = 90, fontsize=14)  # we did this in an usual way.
plt.yticks(fontsize= 14)

# Secons axis
axes1 = plt.gca()
axes2 = axes1.twiny()                 # to get another axis here we used object oriented.

axes2.set_xticks(list(range(0,30,3))) # indexing for xticks.
axes2.set_xticklabels(ax_label, fontsize= 14, rotation= 90)

# Axis label
axes1.set_xlabel("Productions")
axes2.set_xlabel("Games")

# Axis limit
axes1.set_xlim(-.5,29.5)
axes2.set_xlim(0,30)

# Title
plt.title('Top Three Games of Publishers', fontsize= 15, color= 'blue')
plt.show()
# Figure Size
plt.figure(figsize=[17,5], dpi= 95)

# to sort the bar in descending order
order = df.groupby(by = 'Year').sum()['Global_Sales'].sort_values(ascending= False).index

# plot
sns.barplot(x='Year',y='Global_Sales',data=df,estimator= sum, palette="dark", order= order, ci= 0)

# Axis ticks
plt.xticks(fontsize=12, rotation= 90)
plt.yticks(fontsize=12)

# Title
plt.title('Highest Global Sales by Year', fontsize= 16, color= 'blue')

plt.tight_layout()
sns.set_style('whitegrid')
plt.figure(figsize=[17,5], dpi= 95)

tick = ['NA_Sales', 'EU_Sales','JP_Sales', 'Other_Sales']

for item in tick:
    sns.lineplot(x= 'Year', y= df[item], data= df, estimator= 'sum', label= item, ci=None, marker='o', linewidth= 3)

# Axis label and ticks    
plt.xlabel('Year',fontsize=14, color='black')
plt.ylabel('Sales in Different Continent',fontsize=14, color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Title and legend
plt.title('Sales in Different Region', fontsize= 18, color= 'blue')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize= 'x-large')
# Top three genre in number of produced games.
df['Genre'].value_counts().head(3)
# Dataframe for sales of different genre in different continents
sales_by_genre = df.groupby(by= ['Year','Genre']).sum()[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].unstack()

# Replacing NaN Value 
sales_by_genre.fillna(0, inplace= True)

# Dropping some rows
sales_by_genre.drop(range(1980,1996), inplace= True)
sales_by_genre.head(2)
# Top 3 Genre index list
index = list(df['Genre'].value_counts().head(3).index)


# Figure style
sns.set_style('darkgrid')
plt.figure(figsize=[14,6], dpi=95)

# Color palette
palette = plt.get_cmap('tab20')

# Multi plot
num = 0
col_pal = 0
for name in index:
    num += 1
    plt.subplot(1,3,num)
    
    # Using .xs method to get the region sales with specific genre
    data_frame = sales_by_genre.xs(key=name, level=1, axis=1)
    
    # Line plot
    for column in data_frame:
        col_pal += 1
        plt.plot(data_frame[column], marker='', color=palette(col_pal), linewidth=2.4, alpha=0.9, label=column)
    
    # Subplot title, legend and axis ticks
    plt.title(name, loc='left', fontsize=14, fontweight=0, color='red')
    plt.legend(fontsize=12)
    plt.xticks([1995,2000,2005,2010,2015])
    
    # Figure x-axis and y-axis
    if name == 'Sports':
        plt.xlabel('Year(1995-2016)', fontsize= 15)
    if name == 'Action':
        plt.ylabel('Sales in Million Dollar', fontsize= 15)

# Figure title
plt.suptitle('Sales of Top Three Genre in Different Continents', fontsize= 15, y = 1.05, color= 'blue')

plt.tight_layout()

genre_heatmap = df.groupby(by= 'Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
genre_heatmap.head()
# Fig style
plt.figure(figsize= [8,6], dpi= 99)

# Plot
ax = sns.heatmap(genre_heatmap, annot= True, fmt= '.1f', linecolor= 'white', linewidths= 1.2, cmap= 'gist_earth_r')

# There was some kind of issue with matplotlib version. That is why i had to use the below line. Otherwise it should work fine.
# Getting the current y limit and then resizing
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + .5, top - .5)

# Title
plt.title('Heatmap for Sales of All Genre by Region', color= 'blue')
plt.show()
plt.figure(figsize=[5,5], dpi= 120)

# parameters
labels = 'NA_Sales','JP_Sales','Other_sales', 'EU_Sales'
sizes = [df['NA_Sales'].sum(),df['JP_Sales'].sum(), df['Other_Sales'].sum(), df['EU_Sales'].sum()]
colors = ['green','yellowgreen','red','lightskyblue']
explode = (0.1,0,0.1,0) # explode the highest and lowest slice.

# Pie plot
plt.pie(sizes, explode= explode, labels= labels, colors= colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')  # to get a circle shape
plt.title('Sales Percentage by Region', color= 'blue', y= 1.02, x= .45)
plt.show()
sns.pairplot(df[['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales']])
# Figure style
plt.figure(figsize=[8,5], dpi= 95)

# PLot
ax = sns.heatmap(df.corr(), annot= True)

# There was some kind of issue with matplotlib version. That is why i had to use the below line. Otherwise it should work fine.
# Getting the current y limit and then resizing
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + .5, top - .5)

# Title
plt.title('Correlation Between Columns', fontsize= 14, color= 'blue')
