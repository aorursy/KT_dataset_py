import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats # to plot normal distribution

import matplotlib.pyplot as plt # this is basically matplotlib as we know it

import seaborn as sns 

# seaborn is a wrapper library which uses matplotlib under the hood to make graphs more beautiful

from pylab import rcParams # to change the parameters globally
SEED = 13

np.random.seed(SEED) # so that you can re create exactly whats in Notebook



rcParams['figure.figsize'] = 7,4

plt.style.use('seaborn')



def get_cmap(n, name='hsv',return_cmap=True):

    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 

    RGB color; the keyword argument name must be a standard mpl colormap name.

    Can throw error if number of colors exceeds the limit of cmap'''

    cmap = plt.cm.get_cmap(name, n)

    if return_cmap:

        return cmap 

    else:

        return cmap.colors
df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

df.drop('#',axis=1,inplace=True) # there is this '#' column which is nothing but index

df.head()
figure = plt.figure(figsize=(13,6))



plt.title('TITLE: This Figure (13,6) demonstrate general attributes with color given in hex (#343ff3) and size=15',color='#343ff3',size=15)



plot1 = plt.plot([4,1,4,6,9],label='First Line Shows Something inside legend')

plot2 = plt.plot([5,6,2,4,5], label='Second line has different meaning')

plot3 = plt.plot([3,7,6,4,0],label='Third Line is of different color')



plt.xlabel('This is X-axis with default size and Red Color [1,0,0] and it shows number (count) of values in our figure as x-large alphabets in teal color'

           ,color=[1,0,0])



plt.ylabel('Y-Label with large "g" green fonts, and padding of 40 tells us the range of values',

           color='g',labelpad=40, size='large')



plt.xticks([0,1,2,3,4],['a','b','c','d','e'],color='teal',size='x-large')



plt.legend(loc='best') # you can do a lot more with legend

# plt.grid() # show a grid in background to show major, minor or both lines and whether for x, y or both axis

# with seaborn style grid is true automatically

plt.show()
stocks = pd.read_csv('/kaggle/input/national-stock-exchange-time-series/infy_stock.csv',index_col='Date')

stocks.head()
# Default Figure Size



sample_stock = stocks.iloc[:20,:] # First 20 months stocks Prices details only to show clearly



plt.plot(sample_stock['High'],label='Highest (No Marker)') # default attributes

plt.plot(sample_stock['Low'],label='Lowest',ls='dotted',color='green',lw=2,marker='*') 

plt.plot(sample_stock['Close'],label='Closed (No Line)',linestyle='none',marker='o',color='m',markersize=7)



plt.xlabel('Month',size='large')

plt.ylabel('Value',size='large')

plt.xticks(rotation=60)

plt.title('Monthly Share Price Details',size='x-large',color='maroon')



plt.legend()

plt.show()
# you can do all of the things using plt.hist() too

f,ax = plt.subplots(2,2,figsize=(14,10)) # generate 4 subplots

ax = ax.ravel()

f.suptitle('Super Title: Different visualizations of Histogram')



# matplotlib's histogram

n,bins,patches = ax[0].hist(df['Attack'],bins=20,color="#87ceeb",) # try changing the bins to have a different shape

ax[0].set_title('Histogram of Attack vs Number of Pokemons')

ax[0].set_xlabel('Attack Values')

ax[0].set_ylabel('No of Pokemons having Attack Attributes')



# seaborn's distplot # you can also use sns.distplot() withoot 'ax' param if not potting within any subplot

ax[1] = sns.distplot(df['Attack'],kde=True,bins=20,ax=ax[1],color="#87ceeb",fit=stats.norm,label='KDE Curve')

ax[1].set_ylabel('Count of Pokemons')

ax[1].set_title('Seaborn: distplot | Black curve shows KDE curve if data was NORMALLY Distributed')

ax[1].legend()



# seaborn's kdeplot

ax[2] = sns.kdeplot(df['Attack'], shade=True, color="#87ceeb",ax=ax[2])

ax[2].set_ylabel('Probabilities')

ax[2].set_xlabel('Distribution of Attack')

ax[2].set_title('Seaborn: kdeplot')



# Probability Plots

stats.probplot(df['Attack'], dist=stats.norm, plot=ax[3]) # If data was normal, it would be a straight line



plt.subplots_adjust(wspace=0.25, hspace=0.33) # change the spacing

plt.show()
num_cols = ['Attack','Defense','Speed'] # numerical column names

num_df = df.loc[:,num_cols] # matplotlib uses lists or numpy array for multiple boxplots at once



#set Figure and axes

f,ax = plt.subplots(3,2,figsize=(14,14))

ax = ax.ravel()



# matplotlib box

ax[0].boxplot(num_df.values,labels=num_cols,patch_artist=True,showmeans=True,meanline=True)

ax[0].set_ylabel('Range of Values')

ax[0].set_title('Matplotlib Boxplot | Red Line:Mean, Green Line:Median')



# seaborn box

ax[1] = sns.boxplot(data=df[num_cols],showmeans=True,meanline=True,ax=ax[1])

ax[1].set_ylabel('Range of Values')

ax[1].set_title('Seaborn Boxplot | Red Line:Mean, Black Line:Median')



# matplotlib Violin

ax[2].violinplot(num_df.values,showmeans=True,showmedians=True)

ax[2].set_ylabel('Range of Values')

ax[2].set_xticks(np.arange(1, len(num_cols) + 1)) # set ticks on x-axis

ax[2].set_xticklabels(num_cols) # give the ticks labels as the column names

ax[2].set_title('Matplotlib Violinplot with Mean and Median')



# seaborn violin

ax[3] = sns.violinplot(data=df[num_cols],ax=ax[3])

ax[3].set_ylabel('Range of Values')

ax[3].set_title('Seaborn Violinplot')



# overlay first and third plots

ax[4].boxplot(num_df.values,labels=num_cols,patch_artist=True,showmeans=True,meanline=True)

ax[4].violinplot(num_df.values,showmeans=True,showmedians=True)

ax[4].set_ylabel('Range of Values')

ax[4].set_title('Overlaid Box and Violin | Just for demonstration')



f.delaxes(ax[-1])

plt.subplots_adjust(hspace=0.25)

plt.show()
counted_unique_values = df['Type 1'].value_counts() # this returns a dataframe. Please check how it looks

counted_unique_values.plot(kind='pie',autopct='%.2f%%',radius=2.2) # use the Pandas plotting Function itself

# autopct tells us how to plot the % signs. Here 2 digits are used after decimal (float) and then a % sign

plt.show()



#Please Trye using 

# plt.pie(counted_unique_values,autopct='%1.2f%%',radius=2.2) # You'll find the same thing
cmap = get_cmap(18,'Pastel2',return_cmap=False) # 18 different colors (list of RGBA tuples) from Pastel2 style



df['Type 1'].value_counts(normalize=True).plot(kind='bar',color=cmap)

plt.ylabel('Proportion of the the total Pokemon Population')

plt.xlabel('Types of Pokemon | Type 1')

plt.show()
f,ax = plt.subplots(3,2,figsize=(15,19))

ax = ax.ravel()

sample = df.sample(300,random_state=SEED) # random 300 samples



# first plot. Simple single Plot

ax[0].scatter(x=sample['Attack'],y=sample['Speed'],facecolor='green')

ax[0].set_ylabel('Speed',size='large')

ax[0].set_xlabel('Attack',size='large')

ax[0].set_title(' Default | 2 Attributes',color='green',size='x-large')



# multiple scatter plots in one 

ax[1].scatter(x=sample['Attack'],y=sample['Speed'],s=40,facecolor=None,edgecolor='teal',marker='*',alpha=0.85,

            label='Attack Vs Speed') #  Teal colored Stars with less opacity and size=40



ax[1].scatter(x=sample['Attack'],y=sample['HP'],facecolor='none',edgecolor='black',linewidth=2,

            label='Attack Vs HP') # black hollow circles with thick boundry



ax[1].set_xlabel('Attack',size='large')

ax[1].set_ylabel('Speed / HP',size='large')

ax[1].set_title('Two Plots in one | 3 Attributes | ',color='green',size='x-large')

ax[1].legend()



# Three variables in single Plot. Size shows the Legendary of Pokemon

ax[2] = sns.scatterplot(x="Attack", y="Speed", size="Type 1",hue='Type 1',data=sample,ax=ax[2],sizes=(10, 200))

# change color of dots by using args of plt.scatter()

ax[2].set_xlabel('Attack',size='large')

ax[2].set_ylabel('Speed',size='large')

ax[2].set_title('3 Attributes in Single Plot | Marker "Size" Show Type 1',color='green',size='x-large')



# Three variables in single Plot. Size shows the Type 1 of Pokemon

ax[3] = sns.scatterplot(x="Attack", y="Speed", hue="Legendary",data=sample,ax=ax[3],s=100)

# s is the size of dots

ax[3].set_xlabel('Attack',size='large')

ax[3].set_ylabel('Speed',size='large')

ax[3].set_title('3 Attributes in Single Plot | Marker "Hue" Show Legendary',color='green',size='x-large')





# Three variables in single Plot. Marker shows the Legendary of Pokemon

ax[4] = sns.scatterplot(x="Attack", y="Speed", style="Legendary",data=sample,ax=ax[4],facecolor='m',s=85)

ax[4].set_xlabel('Attack',size='large')

ax[4].set_ylabel('Speed',size='large')

ax[4].set_title('3 Attributes in Single Plot | Marker "Style" Show Legendary',color='green',size='x-large')



f.delaxes(ax[-1])

plt.subplots_adjust(hspace=0.25)



plt.show()
f,ax = plt.subplots(1,2,figsize=(15,11))

ax = ax.ravel()



# Four variables in single Plot. Size shows the Legendary of Pokemon and Hue  shows  Type 1

ax[0] = sns.scatterplot(x="Attack", y="Speed", size="Type 1",hue='Type 1',style='Legendary',data=sample,ax=ax[0])

ax[0].set_xlabel('Attack',size='large')

ax[0].set_ylabel('Speed',size='large')

ax[0].set_title('4 Attributes in Single Plot | "Size" Show Type 1 | "Style" Show Legendary',

                color='green',size='large')





# Five variables in single Plot. Size shows the Legendary of Pokemon, Hue  shows  Type 1 Size Shows Type 2

ax[1] = sns.scatterplot(x="Attack", y="Speed",size='Type 2',sizes=(10, 200), hue="Type 1",style='Legendary',

                        data=sample,ax=ax[1])

ax[1].set_xlabel('Attack',size='large')

ax[1].set_ylabel('Speed',size='large')

ax[1].set_title('5 Attributes in Single Plot | "Hue" Show Type 1 | "Size" shows Type 2 | "Style" Show Legendary',

                color='green',size='large')



plt.show()
# plt.scatter() and plt.line() arguments can be passed here

scatter_kws = {'edgecolor':'red','s':100,'alpha':0.85,'linewidth':1}

sns.regplot(x='Attack', y='Speed', data=sample, ci=90, color='green',marker='*',scatter_kws=scatter_kws)

plt.title('Attack vs Speed Regression Plot with 90% Confidence Interval',size='x-large',color='blue')

plt.show()
j = sns.jointplot('Attack','Defense',data=sample,color='green',kind='reg')



plt.subplots_adjust(top=0.9) # there are 3 subplots so to adjust the title

j.fig.suptitle('Attack vs Defense Joint-Regression Plot with Probability Distributions',size='x-large',

               color='maroon')

plt.show()
f,axs = plt.subplots(2,1,figsize=(10,10))



axs[0] = pd.crosstab(df['Type 1'],df['Legendary']).plot(kind='bar',width=0.85,ax=axs[0],cmap='Set2',

                                                        stacked=True)

axs[0].set_title('Stacked (Count) Bar Plot Using Pandas Plotting (built in plt.bar())',color='green',

                 size='x-large')



axs[1] = sns.countplot(x="Type 1", hue="Legendary", data=df,ax=axs[1],palette='husl')

axs[1].set_title('Count Plot Using Seaborn',color='green',size='x-large')



for ax in axs: # set tick labels and Y labels for both of the sublots at once

    plt.setp(ax.get_xticklabels(), rotation=45)

    ax.set_ylabel('Count')



plt.subplots_adjust(hspace=.4)

plt.show()

f,ax = plt.subplots(2,1,figsize=(15,9))

ax = ax.ravel()



ax[0] = sns.boxplot(x="Type 1", y="Attack", data=df,ax=ax[0],dodge=True,meanline=True,showmeans=True)

ax[0].set_title('Attack Box Plot for Each Type 1')



ax[1] = sns.violinplot(x="Type 1", y="Attack", data=df,ax=ax[1],dodge=True)

ax[1].set_title('Attack Violin Plot for Each Type 1')



plt.subplots_adjust(hspace=.33)

plt.show()
f,ax = plt.subplots(1,2,figsize=(15,5))





ax[0] = sns.distplot(df[df['Legendary'] == True]['Attack'],kde=True,ax=ax[0],color="#87ceeb",

                    label='Legendary')

ax[0] = sns.distplot(df[df['Legendary'] == False]['Attack'],kde=True,ax=ax[0],color="#FFB6C1",

                    label='Not Legendary')

ax[0].set_ylabel('Count of Pokemons')

ax[0].set_title('Seaborn: distplot | Distributon of attack in Legendary and Not Legendary')

ax[0].legend()



# seaborn's kdeplot

ax[1] = sns.kdeplot(df[df['Legendary'] == True]['Attack'], shade=True, color="#87ceeb",ax=ax[1], label='Legendary')

ax[1] = sns.kdeplot(df[df['Legendary'] == False]['Attack'], shade=True, color="#FFB6C1",ax=ax[1],label='Not Legendary')

ax[1].set_ylabel('Proportion of the Whole Data')

ax[1].set_xlabel('Attack ')

ax[1].set_title('Seaborn: kdeplot | Distributon of attack in Legendary and Not Legendary')



plt.show()
f,ax = plt.subplots(2,2,figsize=(15,11))

ax = ax.ravel()



ax[0] = sns.boxplot(x="Legendary", y="Attack", data=df,ax=ax[0],color='0.8')

ax[0] = sns.stripplot(x="Legendary", y="Attack", data=df,ax = ax[0],jitter=True)

ax[0].set_title('Stip with Box Plot')



ax[1] = sns.violinplot(x="Legendary", y="Attack", data=df,ax=ax[1],color='0.8')

ax[1] = sns.stripplot(x="Legendary", y="Attack", data=df,ax = ax[1],jitter=0.25)

ax[1].set_title('Stip with Violin Plot')



ax[2] = sns.boxplot(x="Legendary", y="Attack", data=df,ax=ax[2],color='0.8')

ax[2] = sns.swarmplot(x="Legendary", y="Attack", data=df,ax = ax[2],size=4)

ax[2].set_title('Swarm with Box Plot')



ax[3] = sns.violinplot(x="Legendary", y="Attack", data=df,ax=ax[3],color='0.8')

ax[3] = sns.swarmplot(x="Legendary", y="Attack", data=df,ax = ax[3],size=4)

ax[3].set_title('Swarm with Violin Plot')



plt.show()
fig = plt.figure(figsize=(15,6))



plt.title('Swarm Plot with Hue as Type 1')

sns.swarmplot(x="Type 1", y="HP", hue='Legendary', data=df,size=7,palette='Set2')

plt.show()
tips = sns.load_dataset("tips")

tips.head()
f,ax = plt.subplots(1,2,figsize=(15,4.5))



ax[0] = sns.pointplot(x="time", y="tip", data=tips,ax=ax[0]) # default mean with 95% confidence interval

ax[0].set_title('Tip given vs Time of Day',size='x-large')



ax[1] = sns.pointplot(x="day", y="tip", hue="smoker",data=tips,estimator=np.median,ci=99,dodge=True,

                   markers=["o", "x"],linestyles=["-", "--"],palette='husl',ax=ax[1],seed=SEED)

# dodge is to seperate the two so that they do not overlab

ax[1].set_title('Tip vs Time of Day given whether there was a Smoker or not',size='x-large')



plt.show()
label_df = df.drop(['Type 1','Type 2','Name','Generation',],axis=1)

label_df.head()
sns.pairplot(label_df.dropna(),kind='scatter',diag_kind='kde',hue='Legendary',palette='husl')

# plot reression plot for every numerical attribute with 'kde' plot for diagonals where the class separation is

# according to Legendary



plt.show()

# you should most probabily save the figure if it not feasible to show
sns.relplot(x="total_bill", y="tip",row='size',col="day", hue="time", style="smoker",kind="scatter", data=tips,

           height=3.3, aspect=1,s=133)

# you can put the respective matplotlib args for the respective 'kind'

plt.show()
# pass one of either X or Y in count plot 

sns.catplot(x="size",row='smoker',col="day", hue="time",kind="count", data=tips,height=3.3)

plt.show()
sns.catplot(y='tip', x="size",row='smoker',col="day", hue="time",kind="swarm",data=tips,height=3.3)

plt.show()