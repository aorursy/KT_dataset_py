# Importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

# After uploading the file, we are able to see the name of the file saved as "World_Happiness_2015_2017.csv"

# Use pd.read_csv() to read the file and assign it to variable call "data"

data = pd.read_csv('../input/world-happiness/World_Happiness_2015_2017.csv')



# We then use data.head() to see the first 5 rows of data

data.head()
# Then what I do next is look into shape using data.shape(). This will tell me how many rows and columns there are.

data.shape
# Now lets see data types using data.dtypes

data.dtypes
# Now lets change the data type of Year from int to float 

data['Year'] = data['Year'].astype('float')



# Now let's check if we did it right

data.dtypes
# Have a look at data

data.head()
# Yikes! Year in decimals! Let's change it back to int!

data['Year'] = data['Year'].astype('int')
# Lets calculate the number of null values

data.isnull().sum()
g = sns.pairplot(data)

g.fig.suptitle('FacetGrid plot', fontsize = 20)

g.fig.subplots_adjust(top= 0.9);
# Creating a list of attributes we want (just copy the column name)

econ_happiness = ['Happiness Score','Economy (GDP per Capita)']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_happiness]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Economy (GDP per Capita)', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Economy")
data.head()
#Creating correlation dataframe for happiness score and family



Family_Happiness = ['Happiness Score', 'Family']

Family_corr = data[Family_Happiness]

Family_corr.corr()
#Plotting happiness score and family using linear regression



sns.regplot(data = Family_corr, x='Family', y='Happiness Score').set_title('Correlation Graph for Happiness score vs. Family')
#Creating R and R Squared for Happiness Score and Family



R = np.array(Family_corr.corr())

R2 = np.array(Family_corr.corr()**2)



print('The data shows we have an R value of: ' + str(R[1]) + ' and and R2 value of ' + str(R2[1]))
#Creating correlation dataframe for happiness score and health



Health_Happiness = ['Happiness Score', 'Health (Life Expectancy)']

Health_corr = data[Health_Happiness]

Health_corr.corr()
#Plotting happiness score and health using linear regression



sns.regplot(data = Health_corr, x='Health (Life Expectancy)', y='Happiness Score').set_title('Correlation Graph for Happiness score vs. Health')
#Creating R and R Squared for Happiness Score and Health



R = np.array(Health_corr.corr())

R2 = np.array(Health_corr.corr()**2)



print('The data shows we have an R value of: ' + str(R[1]) + ' and and R2 value of ' + str(R2[1]))
#Creating correlation dataframe for happiness score and freedom



Freedom_Happiness = ['Happiness Score', 'Freedom']

Freedom_corr = data[Freedom_Happiness]

Freedom_corr.corr()



#Plotting happiness score and freedom using linear regression



sns.regplot(data = Freedom_corr, x='Freedom', y='Happiness Score').set_title('Correlation Graph for Happiness score vs. Freedom')



#Creating R and R Squared for Happiness Score and Freedom



R = np.array(Freedom_corr.corr())

R2 = np.array(Freedom_corr.corr()**2)



print('The data shows we have an R value of: ' + str(R[1]) + ' and and R2 value of ' + str(R2[1]))
#Creating correlation dataframe for happiness score and trust



trust_Happiness = ['Happiness Score', 'Trust (Government Corruption)']

trust_corr = data[trust_Happiness]

trust_corr.corr()



#Plotting happiness score and trust using linear regression



sns.regplot(data = trust_corr, x='Trust (Government Corruption)', y='Happiness Score').set_title('Correlation Graph for Happiness score vs. Trust')



#Creating R and R Squared for Happiness Score and trust



R = np.array(trust_corr.corr())

R2 = np.array(trust_corr.corr()**2)



print('The data shows we have an R value of: ' + str(R[1]) + ' and and R2 value of ' + str(R2[1]))
#Creating correlation dataframe for happiness score and Generosity



Generosity_Happiness = ['Happiness Score', 'Generosity']

Generosity_corr = data[Generosity_Happiness]

Generosity_corr.corr()



#Plotting happiness score and Generosity using linear regression



sns.regplot(data = Generosity_corr, x='Generosity', y='Happiness Score').set_title('Correlation Graph for Happiness score vs. Generosity')



#Creating R and R Squared for Happiness Score and Generosity



R = np.array(Generosity_corr.corr())

R2 = np.array(Generosity_corr.corr()**2)



print('The data shows we have an R value of: ' + str(R[1]) + ' and and R2 value of ' + str(R2[1]))
#Creating correlation dataframe for happiness score and Dystopia Residual



Dystopia_Residual_Happiness = ['Happiness Score', 'Dystopia Residual']

Dystopia_Residual_corr = data[Dystopia_Residual_Happiness]

Dystopia_Residual_corr.corr()



#Plotting happiness score and Dystopia Residual using linear regression



sns.regplot(data = Dystopia_Residual_corr, x='Dystopia Residual', y='Happiness Score').set_title('Correlation Graph for Happiness score vs. Dystopia Residual')



#Creating R and R Squared for Happiness Score and Dystopia Residual



R = np.array(Dystopia_Residual_corr.corr())

R2 = np.array(Dystopia_Residual_corr.corr()**2)



print('The data shows we have an R value of: ' + str(R[1]) + ' and and R2 value of ' + str(R2[1]))
# Creating a correlation matrix for R values



all_data = ['Happiness Score', 'Economy (GDP per Capita)', 'Family', 	'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']

all_data_dataframe = data[all_data]

sns.heatmap(all_data_dataframe.corr(), annot=True).set_title("R values for 7 Happiness attributes")
# Creating a correlation matrix for R2 values



all_data = ['Happiness Score', 'Economy (GDP per Capita)', 'Family', 	'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']

all_data_dataframe = data[all_data]

sns.heatmap(all_data_dataframe.corr()**2, annot=True).set_title("R2 values for 7 Happiness attributes")