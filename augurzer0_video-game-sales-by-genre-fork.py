%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model



data = pd.read_csv('../input/Video_Game_Sales_as_of_Jan_2017.csv')

data.head()
# Remove NaN data from Year of Release

data = data[data.Year_of_Release.notnull()]



# Omitting video games released in 2017.

data = data.loc[data.Year_of_Release < 2017]



# Converting year of release to integers

data.Year_of_Release = data['Year_of_Release'].astype(int)
# Creating a table of the total global sales for each genre and year

Sales_by_Gen_and_Yr = pd.pivot_table(data,index=['Year_of_Release'],

                     columns=['Genre'],values=['Global_Sales'],aggfunc=np.sum)

Sales_by_Gen_and_Yr.columns = Sales_by_Gen_and_Yr.columns.get_level_values(1)



# Finding the yearly totals and cumulative proportion of yearly global sales

Yearly_Tots = Sales_by_Gen_and_Yr.sum(axis=1)

Yearly_Tots = Yearly_Tots.sort_index()

YT1_cumsum = Yearly_Tots.cumsum()/Yearly_Tots.sum()



# Plotting the yearly totals and cumulative proportions

fig = plt.figure(figsize=(12,5))

ax1=fig.add_subplot(121)

ax2=fig.add_subplot(122)

sns.barplot(y = Yearly_Tots.values, x = Yearly_Tots.index,ax=ax1)

ax1.set_title('Total Yearly Global Sales')

plt.setp(ax1.get_xticklabels(),rotation=90)

ax1.set_xlabel('Years')

ax1.set_ylabel('Number of games sold (in millions)')



sns.barplot(y = YT1_cumsum.values, x = YT1_cumsum.index, ax=ax2)

ax2.set_title('Cumulative Proportion of Yearly Global Sales')

plt.setp(ax2.get_xticklabels(),rotation=90)

ax2.set_xlabel('Years')

ax2.set_ylabel('Cummulative Proportion')

ax2.yaxis.set_ticks(np.arange(0,1,0.05))

fig.tight_layout()



# Plotting the heat map of global sales for games released each year by genre

plt.figure(figsize=(10,10))

sns.heatmap(Sales_by_Gen_and_Yr,annot = True, fmt = '.2f', cmap = 'Blues')

plt.tight_layout()

plt.ylabel('Year of Release')

plt.xlabel('Genre')

plt.title('Global Sales (in millions) of Games Released Each Year by Genre')

plt.show()
# Histogram of global sales

plt.figure(figsize=(9,5))

data.Global_Sales.hist(bins=50)

plt.show()
# Pulling only the data from 1991 to 2016

data = data.loc[data.Year_of_Release >= 1991]



# Finding the median sales value by genre and year

Med_Sales_by_Gen_and_Yr = pd.pivot_table(data,index=['Year_of_Release'],

                     columns=['Genre'],values=['Global_Sales'],aggfunc=np.median)

Med_Sales_by_Gen_and_Yr.columns = Med_Sales_by_Gen_and_Yr.columns.get_level_values(1)



Med_Sales_by_Gen_and_Yr.head()
def Linear_Regression_Plot(Data):

    Regr_Coeff = []

    Regr_MSE = []

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10,12))



    x_data = np.transpose(np.matrix(Data.index))



    count = 0

    

    for genre in Data.columns:

        axs = axes[count//3,count%3]

        y_data = Data[genre].to_frame()

    

        # Linear regression

        regr = linear_model.LinearRegression()

        regr.fit(x_data,y_data)

        

        # Mean Squared Error

        MSE = np.mean((regr.predict(x_data)-y_data)**2)

        

        Regr_Coeff.append(regr.coef_[0][0])

        Regr_MSE.append(MSE[0])



        Data[genre].plot(ax=axs)

        axs.plot(x_data,regr.predict(x_data), color='black')



        y_lims = axs.get_ylim()

        

        

        txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,MSE)

        y_loc = 0.85*(y_lims[1]-y_lims[0])+y_lims[0]

        axs.text(2007,y_loc,txt)



        axs.set_title(genre)

        axs.set_xlabel('Year')

        axs.set_ylabel('Median')

        count+=1

    fig.tight_layout()

    

    return [Regr_Coeff,Regr_MSE]

    

[Regr_Coeff,Regr_MSE] = Linear_Regression_Plot(Med_Sales_by_Gen_and_Yr)
Med_Sales_by_Gen_and_Yr = Med_Sales_by_Gen_and_Yr.loc[Med_Sales_by_Gen_and_Yr.index >= 1995]



[Regr_Coeff_After_95,Regr_MSE_After_95] = Linear_Regression_Plot(Med_Sales_by_Gen_and_Yr)
Linear_Regression_Results = pd.DataFrame({'Regression Coeff After 1991':Regr_Coeff,

                                         'MSE After 1991':Regr_MSE,

                                         'Regression Coeff After 1995':Regr_Coeff_After_95,

                                         'MSE After 1995':Regr_MSE_After_95},

                                        index = list(Med_Sales_by_Gen_and_Yr.columns))

Column_Order = ['Regression Coeff After 1991','MSE After 1991','Regression Coeff After 1995',

                'MSE After 1995']



# Printing the linear regression results

Linear_Regression_Results[Column_Order].head(n=len(list(Med_Sales_by_Gen_and_Yr.columns)))
Med_Sales_by_Yr = pd.pivot_table(data,index=['Year_of_Release'],

                     values=['Global_Sales'],aggfunc=np.median)





fig = plt.figure(figsize=(13,5))

Med_Sales_by_Yr.plot()



x_data = np.transpose(np.matrix(Med_Sales_by_Yr.index))

y_data = Med_Sales_by_Yr

regr = linear_model.LinearRegression()

regr.fit(x_data,y_data)



plt.plot(x_data,regr.predict(x_data), color='black')



txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,np.mean((regr.predict(x_data)-y_data)**2))



plt.text(2011,0.8*Med_Sales_by_Yr.max(),txt)



plt.title('Median Global Sales')

plt.xlabel('Year')

plt.ylabel('Median Sales (in millions)')