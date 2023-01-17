# Import all necessary data tools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Read the data file into Python enviornment.
Data = pd.read_csv("../input/data.csv")
# Some Information about the data. 
Data.shape
#The data has 5394 observation rows and 47 Coloumns. 
data_columns = np.array(list(Data.columns))
data_columns
# primary observation reveals, that many of the records in the years columns are not filled, up.
# Contratry to this ,  one indicator SP.POP.TOTL seems that all the columns are filled. 
# Let us to separate this indicator related data and observe the population trend across countries over the years

Data_population = Data.loc[Data['Indicator Code'] == 'SP.POP.TOTL']
Data_population = Data_population.drop('Unnamed: 46',axis = 1)
Data_population = Data_population.drop('Indicator Code', axis = 1)
Data_population = Data_population.drop('Indicator Name', axis = 1)
Data_population = Data_population.drop('Country Code', axis = 1)
Data_population = Data_population.set_index('Country Name').T

Data_population.head()
countries = np.array(list(Data_population.columns))

for m in range (1, len(countries), 2):
        
        country_data = Data_population[countries[m-1]]
        plt.subplot(1,2,1)
        country_data.plot(kind = 'line', title = countries[m-1], figsize = [15,5])
        country_data = Data_population[countries[m]]
        plt.subplot(1,2,2)
        country_data.plot(kind = 'line', title = countries[m])
        plt.tight_layout()
        plt.show()


#Countries where the population had shown decreasing trend. 
countries_lowerpop = []; Difference = []; Max_pop = []
for i in range (0, len(Data_population)):
    country_data = Data_population[countries[i]]
    max_population = int(country_data.max())
    population_2015 = int(country_data['2015'])
    if (max_population > population_2015):
        countries_lowerpop.append(country_data.name)
        Max_pop.append(max_population)
        Difference.append(max_population - population_2015)

        pop_decline = pd.DataFrame() 
countries_lowerpop = np.array(countries_lowerpop)
Difference = np.array(Difference)
Max_pop =  np.array(Max_pop)
pop_decrease = pd.DataFrame(countries_lowerpop,columns =['Country'])
pop_decrease['Max'] = Max_pop
pop_decrease['Diff'] = Difference
pop_decrease['Percent_decrease'] = pop_decrease['Diff']/pop_decrease['Max'] * 100
pop_decrease.sort_values(by='Percent_decrease',ascending = False)
pop_decrease
pop_decrease.plot(kind = 'bar', title = "Countries where the population decreased", figsize = [15,5], x = 'Country', y = 'Percent_decrease')
plt.show()
