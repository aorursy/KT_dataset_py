import numpy as np

import pandas as pd
#Suppress warnings in the IPython Notebook

import warnings

warnings.simplefilter('ignore')
# Reading the csv file.

concrete = pd.read_csv('../input/compressive-strength-of-concrete/npvproject-concrete.csv')  # We can replace the file name with path if the file location differs.
# Printing DataFrame 

print(concrete)      # Print the contents of Data Frame
# Printing shape of the dataset

print(concrete.shape)    # Returns total number of Rows and Columns
# Retrieving Data in first 5 Rows 

concrete.head() # We can mention number of rows to retrieve inside braces too
# Retrieving columns names of DataFrame.

concrete.columns # Returns the column names in the Concrete Dataset
# Using info() method 

concrete.info()     # Outputs some general information about the dataframe
# Using describe() method 

concrete.describe()      # Basic statistical characteristics of each numerical column 
#Applying Functions to Cells, Columns and Rows

concrete.apply(np.max) # Returns the Maximum values of all columns
concrete.isna().sum()  #We use isna() function. It returns dataframe of Boolean values with a True value if NaNs are found.
duplicate_rows = concrete[concrete.duplicated()]

print("There are {} duplicate rows in the dataset".format(duplicate_rows.shape[0]))

duplicate_rows
concrete.drop(index=[2,5,1026,1029],columns=['slag','coarseagg'],inplace=False)
concrete.sort_values(by='strength',ascending=False,inplace=False).head(n=5)
concrete.sort_values(by=['water','strength'],ascending=[False,True],inplace=False).head(n=3)
concrete.iloc[2:10,2:9] #Fetch 2nd to 9th row and only 2nd to 8th columns
concrete.loc[4:20:2,  'water':'fineagg'] #Fetches values based on indices and columns specified
concrete[concrete['cement'] == 540]['strength'].mean() 
concrete[(concrete['slag']>130) & (concrete['age']>=100)]['strength'].min()
concrete.groupby(['strength'])[concrete.columns[:8]].agg([np.mean,np.min,np.max]).T
#Creating a spreadsheet-style pivot table as a DataFrame.



#The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) 

#on the index and columns of the result DataFrame.



df2 = pd.pivot_table(concrete, index = ['cement','slag'])

df2.iloc[:8] #Fetching only first 8 rows of pivoted table
#The sample() method selects rows or columns randomly.

#By default, one row is returned randomly.



concrete.sample(n=10)
#Correlation matrix is a table showing correlation coefficients between variables. 

#It is used to show the summarize data, as an input into a more advanced analysis, 

#and as a diagnostic for advanced analyses.

concrete.corr()  #Methods - Pearson(by default) or Spearman or Kendall
concrete.nsmallest(2,'strength')
#Loading the necessary packages

import matplotlib.pyplot as plt

import seaborn as sns
#Let's find what's the range of values in the Compressive strength. 

print("Maximum Strength achieved: ",concrete.strength.max())

print("Minimum Strength achieved: ",concrete.strength.min())

print("Range/Spread of the Strength of Concrete: ",concrete.strength.max() - concrete.strength.min())
#Which instances produce the maximum values in strength?

print("Maximum strength achieved in the instance(s): ")

concrete[concrete.strength == concrete.strength.max()]
#Which instances produce the minimum values in strength?

print("Minimum strength achieved in the instance(s): ")

concrete[concrete.strength == concrete.strength.min()]
#Let's also plot a box plot to know if there are any major outliers in the strength values.

#Alongide, lets also put a histogram of strength variation to know what type of distribtion it is



#Box Plot

plt.figure(figsize=(5,5), dpi=100, facecolor='cyan', edgecolor='#000000')

plt.boxplot(concrete.strength)

plt.text(x=1.1,y=concrete.strength.min() ,s="Min")

plt.text(x=1.1,y=concrete.strength.max(),s="Max")

plt.text(x=1.1,y=concrete.strength.median() ,s="Median")

plt.text(x=1.1,y=concrete.strength.quantile(0.25),s="Q1")

plt.text(x=1.1,y=concrete.strength.quantile(0.75),s="Q3")

plt.title("Distribution of strength of concrete across its range.")

plt.ylabel('Compressive Strength')



#Histogram

plt.figure(figsize=(5,5), dpi=100, facecolor='cyan', edgecolor='#000000')

plt.hist(concrete.strength,color='orange',rwidth=0.9)

plt.ylabel('Compressive strength')

plt.title("Histogram of Compressive strength across its range.")
#Let's plot the variation of strength against time. Here, strength is in MPa and time is in days

plt.figure(figsize=(5,5), dpi=100, facecolor='cyan', edgecolor='#000000')

plt.scatter(x=concrete.age,y=concrete.strength,marker='o',color='orange',alpha=0.5)

plt.xlabel('Age')

plt.ylabel('Compressive strength')

plt.title('Variation of Compressive Strength versus Ageing Time')
plt.figure(figsize=(10,8))

sns.heatmap(concrete.corr(), cmap ="YlGnBu", linewidths = 0.1,annot = True) #Plot contents of correlation matrix as a heatmap

plt.title("Heatmap - Relationship between all the attributes",fontdict={'fontsize': 16})  #Title of the plot

plt.show()
plt.figure(figsize=(15,8))

cs_d = sns.distplot(concrete.strength, color='mediumvioletred', rug=True,kde_kws={'shade':True})

cs_d.set_title("Compressive Strength Distribution")

plt.show()
concrete.plot(kind='density', subplots=True, layout=(5,2), sharex=False, sharey=False, figsize=(20,30))

plt.show()
sns.pairplot(concrete, x_vars = ['cement', 'slag','ash','water','superplastic','coarseagg','fineagg'], y_vars = 'strength',height=4, aspect=1.5)

plt.show()
fig, ax = plt.subplots(figsize=(15,8))

sns.scatterplot(y="strength", x="cement", hue="water", size="age", data=concrete, palette='inferno', ax=ax, sizes=(50, 300))

ax.set_title("Compressive Strength vs (Cement, Age, Water)")

plt.show()