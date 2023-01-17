import pandas as pd
# Importing the data
pop = pd.read_csv('../input/country_population.csv')
fer = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')
# Cleaning the data
population = pd.melt(pop, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='pop')
fertility = pd.melt(fer, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='fer_rate')
life_exp = pd.melt(life, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='life_exp')

s1 = fertility.iloc[:,5:6].copy()
s1.head()

s2 = life_exp.iloc[:,5:6].copy()
s2.head()
# Conactenating the columns s1 and s2 
new_table = pd.concat([population, s1], axis=1)
new_table.head()
final_table = pd.concat([new_table, s2], axis=1)
final_table.head()
# Deleting the unwanted columns from the final_table
del final_table['Indicator Name']
del final_table['Indicator Code']
final_table.head()
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
# Comparing the data for INDIA and CHINA
india = final_table.loc[final_table['Country Code'] == 'IND']
china = final_table.loc[final_table['Country Code'] == 'CHN']
china.head()
india.head()
# Comapring the population of INDIA and CHINA per year
x = range(1960, 2017)
y_india = india.iloc[:,3:4]
y_china = china.iloc[:, 3:4]

plt.title('China vs India')
plt.ylabel('population(billions)')
plt.xlabel('year')

plt.plot(x, y_india, label='India')
plt.plot(x, y_china, label='China')

plt.legend()

plt.show()
# Comapring the fertility rate for INDIA and CHINA per year
x = range(1960, 2017)
y_india = india.iloc[:,4:5]
y_china = china.iloc[:, 4:5]

plt.title('China vs India')
plt.ylabel('fertility rate(births per woman)')
plt.xlabel('year')

plt.plot(x, y_india, label='India')
plt.plot(x, y_china, label='China')

plt.legend()

plt.show()
# Comapring the life expectancy of people in INDIA and CHINA per year
x = range(1960, 2017)
y_india = india.iloc[:,5:6]
y_china = china.iloc[:, 5:6]

plt.title('China vs India')
plt.ylabel('life expectancy')
plt.xlabel('year')

plt.plot(x, y_india, label='India')
plt.plot(x, y_china, label='China')

plt.legend()

plt.show()
