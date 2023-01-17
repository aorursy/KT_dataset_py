# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import df and libraries
%matplotlib inline
!pip install
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import os

if not os.path.exists("images"):
    os.mkdir("images")
from sklearn import linear_model
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import chisquare
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot
%matplotlib inline
%matplotlib notebook
#import df and get rid of null values and check header for first few values to make sure df was imported properly
data = pd.read_csv('../input/video-game-sales-and-ratings/Video_Game_Sales_as_of_Jan_2017.csv')
data_new = data.dropna()
data_new.head()
#check all columns to understand what data I have to work with
data_new.info()
#I need to know lower limit of year since I know from kaggle that the upper limit is January 2017
data.Year_of_Release.min()
#to see bias versus platforms
data_new.Platform.unique()
data_new.Publisher.unique()
# Omitting video games released in 2017.
data_new = data_new.loc[data.Year_of_Release < 2017]

data_new = data_new.dropna()

# organizing all platforms
sony = ('PS','PS2','PS3','PS4' ,'PSP','PSV')
microsoft = ('PC','X360','XB','XOne', 'X')
nintendo = ('3DS','DS','GBA','GC', 'Wii','WiiU', 'GBA', 'GC')
sega = ('DC')

# using df.loc to create new column "Company"
data_new.loc[data['Platform'].isin(['PS','PS2','PS3','PS4' ,'PSP','PSV']), 'Company'] = 'Sony'
data_new.loc[data['Platform'].isin(['PC','X360','X','XOne']), 'Company'] = 'Microsoft'
data_new.loc[data['Platform'].isin(['3DS','DS','GBA','GC','N64','Wii','WiiU', 'GB', 'GBA', 'GC']), 'Company'] = 'Nintendo'
data_new.loc[data['Platform'].isin(['DC']), 'Company'] = 'Sega'

# view result
data_new.info()
print(data_new.groupby('Company').count())
#created company column was a success
# Calculate correlations
corr = data_new.corr()
# Heatmap
plt.rcParams['font.size'] = 20
ax.set_title('Correlations of Video Game Sales Since 2017')
fig, ax = plt.subplots(figsize=(15,10)) 
sns.heatmap(corr, ax = ax, cmap = 'coolwarm')

fig=plt.figure(figsize=(29,14))
plt.subplots_adjust(left=0.25, wspace=0.20, hspace=0.35)
sns.set_style("white")

plt.subplot(2, 2, 1)
plt.title('Gross Sales of Different Genre Games in Europe',fontdict={'fontsize':16})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.barplot(y='Genre', x='EU_Sales', data=data_new.groupby('Genre').sum().EU_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Genre',fontdict={'fontsize':16})
plt.xlabel('Sales in Europe',fontdict={'fontsize':16})

plt.subplot(2, 2, 2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Sales of Different Genre Games in North America',fontdict={'fontsize':16})
sns.barplot(y='Genre', x='NA_Sales', data=data_new.groupby('Genre').sum().NA_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('',fontdict={'fontsize':16})
plt.xlabel('Sales in North America',fontdict={'fontsize':16})

plt.subplot(2, 2, 3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Sales of Different Genre Games in Japan',fontdict={'fontsize':16})
sns.barplot(y='Genre', x='JP_Sales', data=data_new.groupby('Genre').sum().JP_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Genre',fontdict={'fontsize':16})
plt.xlabel('Sales in Japan',fontdict={'fontsize':16})


plt.subplot(2, 2, 4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Sales of Different Genre Games in Other Countries',fontdict={'fontsize':16})
sns.barplot(y='Genre', x='Other_Sales', data=data_new.groupby('Genre').sum().Other_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('',fontdict={'fontsize':16})
plt.xlabel('Sales in Other Countries',fontdict={'fontsize':16})

fig=plt.figure(figsize=(24.5,22))
plt.subplot2grid((3,1), (1,0))
sns.set_style("white")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Global Sales of Different Genre Games',fontdict={'fontsize':16})
sns.barplot(y='Genre', x='Global_Sales', data=data_new.groupby('Genre').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Genre',fontdict={'fontsize':16})
plt.xlabel('Global Sales',fontdict={'fontsize':16});
# to see and base outliers
data_new[['User_Score', 'Critic_Score', 'Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales','Other_Sales']].describe()
bins = np.linspace(0, 1, 100)
pyplot.figure(figsize=[10,5])
pyplot.hist(data_new['JP_Sales'], bins, alpha=0.5, label='Japan')
pyplot.hist(data_new['EU_Sales'], bins, alpha=0.5, label='Europe')
pyplot.hist(data_new['NA_Sales'], bins, alpha=0.5, label='North America')
pyplot.hist(data_new['Other_Sales'], bins, alpha=0.5, label='Other')
pyplot.legend(loc='upper right')
pyplot.ylabel('Count')
pyplot.xlabel('Sales by Country')
pyplot.title('Histogram of Sales by Country')
pyplot.show()
# because looking at multiple columns the data doesn't appear normallly distributed run kruskal-Wallis to test significance
stats.kruskal(data_new['Other_Sales'], data_new['JP_Sales'], data_new['NA_Sales'], data_new['EU_Sales'])
# reduced critic scores from 1-100 to 1-10 to make them more comparable to user scores that are from 1-10
data_new['Rev_Critic_Score'] = data['Critic_Score'] * .1
#printing calculation to verify column works
data_score = data_new.groupby('Genre')['Rev_Critic_Score', 'User_Score'].mean().reset_index()
print(data_new.groupby('Genre')['Rev_Critic_Score', 'User_Score'].mean().reset_index())
#groupby and graph to see how they compare visually
data_score.plot( kind = 'bar',legend = True)
ax = plt.axes()
plt.title('Average Critic and User Scores by Genre', y = 1.08)
ax.set_xticklabels (data_score['Genre'])
plt.legend(loc="upper right", bbox_to_anchor=(2.4, 1.0), ncol=2)
plt.show()
print(data_new.info())
pts = 1000
a = data_new['Rev_Critic_Score']
b = data_new['User_Score']
x = np.concatenate((a, b))
k2, p = stats.normaltest(x)
alpha = 1e-3
print("p = {:g}".format(p))
p = 3.27207e-11
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")

plt.hist(data_new['Rev_Critic_Score'], alpha = .5)
plt.hist(data_new['User_Score'], alpha = .5)
plt.show()

print(stats.describe(data_new['Rev_Critic_Score']))
print(stats.describe(data_new['User_Score']))
stats.kruskal(data_new['Rev_Critic_Score'], data_new['User_Score'])
fig=plt.figure(figsize=(29,14))
plt.subplots_adjust(left=0.25, wspace=0.20, hspace=0.35)
sns.set_style("white")

plt.subplot(2, 2, 1)
plt.title('Gross Sales of Company Consoles in Europe',fontdict={'fontsize':16})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.barplot(y=data_new['Company'], x=data_new['EU_Sales'], data=data_new.groupby('Genre').sum().EU_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Company',fontdict={'fontsize':16})
plt.xlabel('Sales in Europe',fontdict={'fontsize':16})

plt.subplot(2, 2, 2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Sales of Company Consoles in North America',fontdict={'fontsize':16})
sns.barplot(y=data_new['Company'], x=data_new['NA_Sales'], data=data_new.groupby('Genre').sum().NA_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Company',fontdict={'fontsize':16})
plt.xlabel('Sales in North America',fontdict={'fontsize':16})

plt.subplot(2, 2, 3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Sales of Company Consoles in Japan',fontdict={'fontsize':16})
sns.barplot(y=data_new['Company'], x=data_new['JP_Sales'], data=data_new.groupby('Genre').sum().JP_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Company',fontdict={'fontsize':16})
plt.xlabel('Sales in Japan',fontdict={'fontsize':16})


plt.subplot(2, 2, 4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Sales of Company Consoles in Other Countries',fontdict={'fontsize':16})
sns.barplot(y=data_new['Company'], x=data_new['Other_Sales'], data=data_new.groupby('Genre').sum().Other_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Company',fontdict={'fontsize':16})
plt.xlabel('Sales in Other Countries',fontdict={'fontsize':16})

fig=plt.figure(figsize=(24.5,22))
plt.subplot2grid((3,1), (1,0))
sns.set_style("white")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Gross Global Sales of Different Genre Games',fontdict={'fontsize':16})
sns.barplot(y=data_new['Company'], x=data_new['Global_Sales'], data=data_new.groupby('Genre').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');
plt.ylabel('Company',fontdict={'fontsize':16})
plt.xlabel('Global Sales',fontdict={'fontsize':16});
#plot sales versus year to see any bias
global_sales_genre = ggplot(data_new, aes(x = 'Year_of_Release', y = 'Global_Sales', fill = 'Genre')) + \
    geom_bar(stat='identity')
print(global_sales_genre + ggtitle("Global Sales by Genre Over Time"))
#there appears to be a bias towards sales after 1996
# make variable total sales
Total_Sales = sum(data_new['Global_Sales'])
#compare sales to platforms
platform_sales = ggplot(data_new, aes(x = 'Year_of_Release', y = 'Total_Sales', fill = 'Platform')) + \
    geom_bar(stat='identity')
print(platform_sales + ggtitle("Global Platform Sales"))
#seems after PS2 was released sony has dominated the market in general
# make new df to look at sales and PS2 % of them
PS2_2001 = data_new.query('Year_of_Release == "2001"')
#calculate % of global sales is PS2
PS2_2001['Percentage'] = PS2_2001['Global_Sales'] / PS2_2001['Global_Sales'].sum() 
print(PS2_2001.groupby('Platform')['Percentage'].sum().reset_index())
# make new df to look at sales and Wii % of them
Wii_2006 = data_new.query('Year_of_Release == "2006"')
#calculate % of global sales is Wii
Wii_2006['Percentage'] = Wii_2006['Global_Sales'] / Wii_2006['Global_Sales'].sum() 
print(Wii_2006.groupby('Platform')['Percentage'].sum().reset_index())
#Goal: Create Heatmap to look at sales patterns by Genre

#new df for calculations
genre_yr = data
# get rid of null values
genre_yr = genre_yr[genre_yr.Year_of_Release.notnull()]
genre_yr = genre_yr.loc[genre_yr.Year_of_Release < 2017]
# Pulling only the data from 1996 to 2016
# There is a large outlier spike in 1995 
genre_yr = genre_yr.loc[genre_yr.Year_of_Release >= 1996]


# Creating a table of the total global sales for each genre and year with pivot
Sales_Gen_Yr = pd.pivot_table(genre_yr,index=['Year_of_Release'],
                     columns=['Genre'],values=['Global_Sales'],aggfunc=np.sum)
Sales_Gen_Yr.columns = Sales_Gen_Yr.columns.get_level_values(1)

# Plotting the heat map of global sales for games released each year by genre
plt.figure(figsize=(20,10))
sns.heatmap(Sales_Gen_Yr,annot = True, fmt = '.2f', cmap = 'Blues')
plt.tight_layout()
plt.ylabel('Year of Release',fontdict={'fontsize':16})
plt.xlabel('Genre', fontdict={'fontsize':16})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Global Sales (in millions) of Games Released Each Year by Genre', fontdict={'fontsize':20})

plt.show()
# Check for distribution with a Histogram of global sale
plt.figure(figsize=(9,5))
data.Global_Sales.hist(bins=75)
plt.show()
# Calculate the median sales value by genre and year. Need to make year of release the index and genres the columns.
Med_Sales_Gen_Yr = pd.pivot_table(genre_yr,index=['Year_of_Release'],
                     columns=['Genre'],values=['Global_Sales'],aggfunc=np.median)
Med_Sales_Gen_Yr.columns = Med_Sales_Gen_Yr.columns.get_level_values(1)

Med_Sales_Gen_Yr.head()
def Linear_Regression_Plot(genre_yr):
    Regr_Coeff = []
    Regr_MSE = []
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20,18))

    x_data = np.transpose(np.matrix(genre_yr.index))

    count = 0
    
    for genre in genre_yr.columns:
        axs = axes[count//3,count%3]
        y_data = genre_yr[genre].to_frame()
    
        # Linear regression
        regr = linear_model.LinearRegression()
        regr.fit(x_data,y_data)
        
        # Mean Squared Error
        MSE = np.mean((regr.predict(x_data)-y_data)**2)
        
        Regr_Coeff.append(regr.coef_[0][0])
        Regr_MSE.append(MSE[0])

        genre_yr[genre].plot(ax=axs)
        axs.plot(x_data,regr.predict(x_data), color='black')

        y_lims = axs.get_ylim()
        
        
        txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,MSE)
        y_loc = 0.80*(y_lims[1]-y_lims[0])+y_lims[0]
        axs.text(2005,y_loc,txt)

        axs.set_title(genre)
        axs.set_xlabel('Year')
        axs.set_ylabel('Median')
        count+=1
    fig.tight_layout()
    
    return [Regr_Coeff,Regr_MSE]
    
[Regr_Coeff,Regr_MSE] = Linear_Regression_Plot(Med_Sales_Gen_Yr)
Med_Sales_by_Yr = pd.pivot_table(genre_yr,index=['Year_of_Release'],
                     values=['Global_Sales'],aggfunc=np.median)


plt.figure(figsize=(80,40))
Med_Sales_by_Yr.plot()

x_data = np.transpose(np.matrix(Med_Sales_by_Yr.index))
y_data = Med_Sales_by_Yr
regr = linear_model.LinearRegression()
regr.fit(x_data,y_data)

plt.plot(x_data,regr.predict(x_data), color='black')

txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,np.mean((regr.predict(x_data[0])-y_data)**2))

plt.text(2008,0.65*Med_Sales_by_Yr.max(),txt)

plt.title('Median Global Sales')
plt.xlabel('Year')
plt.ylabel('Median Sales (in millions)')
jpdf = data_new
jpdf = jpdf.dropna()
g = sns.barplot( x = 'Company', y = 'JP_Sales',  data = jpdf, ci = 68)
g.set_title('Japan Sales by Company')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
#make new df
jpdf = data_new
jpdf = jpdf.dropna()
print(jpdf.head())
#show % sales are role-playing
jpdf['Percentage'] = jpdf['JP_Sales'] / jpdf['JP_Sales'].sum()
print(jpdf.groupby('Genre')['Percentage'].sum().reset_index())
jpdf = data_new
jpdf = jpdf.dropna()
g = sns.barplot( x = "Genre", y = 'JP_Sales',  data = jpdf, ci = 68)
g.set_title('Japan Sales by Genre')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
#make barplot of JP_Sales and Company filtering by genre
g = sns.catplot(x = 'Genre', y = 'JP_Sales', kind = 'bar', hue = 'Company', height=5, 
                aspect = 4.5, data = jpdf).fig.suptitle('Japan Sales by Genre Filtered by Company')
plt.show()
#new df for calculations
genre_yr_JP = data
# get rid of null values
genre_yer_JP = genre_yr_JP.dropna()
#genre_yr_JP = genre_yr_JP[genre_yr_JP.Year_of_Release.notnull()]
genre_yr_JP = genre_yr_JP.loc[genre_yr_JP.Year_of_Release < 2017]
# Pulling only the data from 1996 to 2016
# There is a large outlier spike in 1995 
genre_yr_JP = genre_yr_JP.loc[genre_yr_JP.Year_of_Release >= 1996]


# Creating a table of the total global sales for each genre and year with pivot
Sales_Gen_Yr_JP = pd.pivot_table(genre_yr_JP,index=['Year_of_Release'],
                     columns=['Genre'],values=['JP_Sales'],aggfunc=np.sum)
Sales_Gen_Yr_JP.columns = Sales_Gen_Yr_JP.columns.get_level_values(1)

# Plotting the heat map of global sales for games released each year by genre
plt.figure(figsize=(14,12))
sns.heatmap(Sales_Gen_Yr_JP,annot = True, fmt = '.2f', cmap = 'Blues')
plt.tight_layout()
plt.ylabel('Year of Release')
plt.xlabel('Genre')
plt.title('Japan Sales (in millions) of Games Released Each Year by Genre')
plt.show()
Med_Sales_Gen_Yr_JP = pd.pivot_table(genre_yr_JP,index=['Year_of_Release'],
                     columns=['Genre'],values=['JP_Sales'],aggfunc=np.median)
Med_Sales_Gen_Yr_JP.columns = Med_Sales_Gen_Yr_JP.columns.get_level_values(1)

Med_Sales_Gen_Yr_JP.head()

def Linear_Regression_Plot(genre_yr_JP):
    Regr_Coeff = []
    Regr_MSE = []
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20,18))

    x_data = np.transpose(np.matrix(genre_yr_JP.index))

    count = 0
    
    for genre in genre_yr_JP.columns:
        axs = axes[count//3,count%3]
        y_data = genre_yr_JP[genre].to_frame()
    
        # Linear regression
        regr = linear_model.LinearRegression()
        regr.fit(x_data,y_data)
        
        # Mean Squared Error
        MSE = np.mean((regr.predict(x_data)-y_data)**2)
        
        Regr_Coeff.append(regr.coef_[0][0])
        Regr_MSE.append(MSE[0])

        genre_yr_JP[genre].plot(ax=axs)
        axs.plot(x_data,regr.predict(x_data), color='black')

        y_lims = axs.get_ylim()
        
        
        txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,MSE)
        y_loc = 0.80*(y_lims[1]-y_lims[0])+y_lims[0]
        axs.text(2003,y_loc,txt)

        axs.set_title(genre)
        axs.set_xlabel('Year')
        axs.set_ylabel('Median')
        count+=1
    fig.tight_layout()
    
    return [Regr_Coeff,Regr_MSE]
    
[Regr_Coeff,Regr_MSE] = Linear_Regression_Plot(Med_Sales_Gen_Yr_JP)
#make new df and drop null values
nadf = data_new
nadf = nadf.dropna()
#make barplot
g = sns.barplot(x = 'Company', y = 'NA_Sales', data = nadf)
g.set_title('North American Sales by Company')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
#make new df and drop null values
nadf = data_new
nadf = nadf.dropna()
#make sns barplot by genre
g = sns.barplot(x = 'Genre', y = 'NA_Sales', data = nadf)
g.set_title('North American Sales by Genre')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
#make new df and drop null values
nadf = data_new
nadf = nadf.dropna()
#make sns barplot by company
g = sns.catplot(x = 'Genre', y = 'NA_Sales', kind = 'bar', hue = 'Company', height=5, aspect = 5, data = nadf).fig.suptitle('North American Sales by Genre Filtered by Company')
plt.show()
#new df for calculations
genre_yr_NA = data
# get rid of null values
genre_yr_NA = genre_yr_NA[genre_yr_NA.Year_of_Release.notnull()]
genre_yr_NA = genre_yr_NA.loc[genre_yr_NA.Year_of_Release < 2017]
# Pulling only the data from 1996 to 2016
# There is a large outlier spike in 1995 
genre_yr_NA = genre_yr_NA.loc[genre_yr_NA.Year_of_Release >= 1996]


# Creating a table of the total global sales for each genre and year with pivot
Sales_Gen_Yr_NA = pd.pivot_table(genre_yr_NA,index=['Year_of_Release'],
                     columns=['Genre'],values=['NA_Sales'],aggfunc=np.sum)
Sales_Gen_Yr_NA.columns = Sales_Gen_Yr_NA.columns.get_level_values(1)

# Plotting the heat map of global sales for games released each year by genre
plt.figure(figsize=(12,12))
sns.heatmap(Sales_Gen_Yr_NA,annot = True, fmt = '.1f', cmap = 'Blues')
plt.tight_layout()
plt.ylabel('Year of Release')
plt.xlabel('Genre')
plt.title('North American Sales (in millions) of Games Released Each Year by Genre')
plt.show()
Med_Sales_Gen_Yr_NA = pd.pivot_table(genre_yr_NA,index=['Year_of_Release'],
                     columns=['Genre'],values=['NA_Sales'],aggfunc=np.median)
Med_Sales_Gen_Yr_NA.columns = Med_Sales_Gen_Yr_NA.columns.get_level_values(1)

Med_Sales_Gen_Yr_NA.head()
def Linear_Regression_Plot(genre_yr_NA):
    Regr_Coeff = []
    Regr_MSE = []
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18,18))

    x_data = np.transpose(np.matrix(genre_yr_NA.index))

    count = 0
    
    for genre in genre_yr_NA.columns:
        axs = axes[count//3,count%3]
        y_data = genre_yr_NA[genre].to_frame()
    
        # Linear regression
        regr = linear_model.LinearRegression()
        regr.fit(x_data,y_data)
        
        # Mean Squared Error
        MSE = np.mean((regr.predict(x_data)-y_data)**2)
        
        Regr_Coeff.append(regr.coef_[0][0])
        Regr_MSE.append(MSE[0])

        genre_yr_NA[genre].plot(ax=axs)
        axs.plot(x_data,regr.predict(x_data), color='black')

        y_lims = axs.get_ylim()
        
        
        txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,MSE)
        y_loc = 0.8*(y_lims[1]-y_lims[0])+y_lims[0]
        axs.text(2005,y_loc,txt)

        axs.set_title(genre)
        axs.set_xlabel('Year')
        axs.set_ylabel('Median')
        count+=1
    fig.tight_layout()
    
    return [Regr_Coeff,Regr_MSE]
    
[Regr_Coeff,Regr_MSE] = Linear_Regression_Plot(Med_Sales_Gen_Yr_NA)
#make new df and drop null values
eudf = data_new
eudf = eudf.dropna()
#make barplot
g = sns.barplot(x = 'Company', y = 'EU_Sales', data = eudf)
g.set_title('European Sales by Company')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
#make sns barplot by genre  c
g = sns.barplot(x = 'Genre', y = 'EU_Sales', data = eudf)
g.set_title('European Sales by Genre')
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()
#make sns barplot by company
g = sns.catplot(x = 'Genre', y = 'EU_Sales', kind = 'bar', hue = 'Company', height=5, aspect = 4.5, data = eudf).fig.suptitle('European Sales by Genre Filtered by Company')
plt.show()
#new df for calculations
genre_yr_eu = data
# get rid of null values
genre_yr_eu = genre_yr_eu[genre_yr_eu.Year_of_Release.notnull()]
genre_yr_eu = genre_yr_eu.loc[genre_yr_eu.Year_of_Release < 2017]
# Pulling only the data from 1996 to 2016
# There is a large outlier spike in 1995 
genre_yr_eu = genre_yr_eu.loc[genre_yr_eu.Year_of_Release >= 1996]


# Creating a table of the total global sales for each genre and year with pivot
Sales_Gen_Yr_eu = pd.pivot_table(genre_yr_eu,index=['Year_of_Release'],
                     columns=['Genre'],values=['EU_Sales'],aggfunc=np.sum)
Sales_Gen_Yr_eu.columns = Sales_Gen_Yr_eu.columns.get_level_values(1)

# Plotting the heat map of global sales for games released each year by genre
plt.figure(figsize=(12,12))
sns.heatmap(Sales_Gen_Yr_eu,annot = True, fmt = '.1f', cmap = 'Blues')
plt.tight_layout()
plt.ylabel('Year of Release')
plt.xlabel('Genre')
plt.title('Europe Sales (in millions) of Games Released Each Year by Genre')
plt.show()
Med_Sales_Gen_Yr_eu = pd.pivot_table(genre_yr_eu,index=['Year_of_Release'],
                     columns=['Genre'],values=['EU_Sales'],aggfunc=np.median)
Med_Sales_Gen_Yr_eu.columns = Med_Sales_Gen_Yr_eu.columns.get_level_values(1)

Med_Sales_Gen_Yr_eu.head()
def Linear_Regression_Plot(genre_yr_eu):
    Regr_Coeff = []
    Regr_MSE = []
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20,24))

    x_data = np.transpose(np.matrix(genre_yr_eu.index))

    count = 0
    
    for genre in genre_yr_eu.columns:
        axs = axes[count//3,count%3]
        y_data = genre_yr_eu[genre].to_frame()
    
        # Linear regression
        regr = linear_model.LinearRegression()
        regr.fit(x_data,y_data)
        
        # Mean Squared Error
        MSE = np.mean((regr.predict(x_data)-y_data)**2)
        
        Regr_Coeff.append(regr.coef_[0][0])
        Regr_MSE.append(MSE[0])

        genre_yr_eu[genre].plot(ax=axs)
        axs.plot(x_data,regr.predict(x_data), color='black')

        y_lims = axs.get_ylim()
        
        
        txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,MSE)
        y_loc = 0.85*(y_lims[1]-y_lims[0])+y_lims[0]
        axs.text(2005,y_loc,txt)

        axs.set_title(genre)
        axs.set_xlabel('Year')
        axs.set_ylabel('Median')
        count+=1
    fig.tight_layout()
    
    return [Regr_Coeff,Regr_MSE]
    
[Regr_Coeff,Regr_MSE] = Linear_Regression_Plot(Med_Sales_Gen_Yr_eu)
# check for bias by company means
data_new.groupby('Company').median()
# it appears for Sega they are not focusing on what they are selling more recently compared to the other 3 companies
# and sega has a lower critic review count
#create new df
data_chi = data_new
#make new df and encode genres to test distirbution significance with Genres and Companies
genre_new = {'Genre' : {'Action': 1, 'Adventure': 2, 'Fighting': 3, 'Misc': 4, 'Platform': 5, 'Puzzle': 6, 'Racing': 7, 
                              'Role-Playing': 8 , 'Shooter': 9,'Simulation': 10,'Sports': 11, 'Strategy' : 12}}
labels = data_chi['Genre'].astype('category').cat.categories.tolist()
new_genre = {'Genre' : {k: v for k, v in zip(labels, list (range(1, len(labels)+1)))}}
data_chi.replace(new_genre, inplace = True)
# same for company
company_new = {'Company' : {'Microsoft' : 1, 'Nintendo': 2, 'Sega': 3, 'Sony' : 4}}
labels = data_chi['Company'].astype('category').cat.categories.tolist()
new_company = {'Company' : {k: v for k, v in zip(labels, list (range(1, len(labels)+1)))}}
data_chi.replace(new_company, inplace = True)
print(data_new.groupby('Company').count())
#set up Chi-Square
genre_table= pd.crosstab(data_new['Genre'], columns = 'count')
print (genre_table)
company_table = pd.crosstab(data_new['Company'], columns = 'count')
print (company_table)
#for genres and companies
observed = genre_table

company_ratios = company_table/len(data_new['Company'])  # get ratios

expected = company_ratios * len(data_new['Genre']) # get expected results

chi_squared_stat = (((observed-expected)**2/expected).sum())

print(chi_squared_stat)

crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 33)   # Df = number of variable categories - 1

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=33)
print("P value")
print(p_value)