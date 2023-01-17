import pandas as pd #importing pandas and giving it a nickname pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt #importing matplotlib.pyplot (the library that allows us to create graphs with python)
%matplotlib inline
import seaborn as sns #importing seaborn, the library that contains all the different types of graphs and plots
print("Setup Complete")
covid_filepath = "../input/uncover/UNCOVER/world_bank/total-covid-19-tests-performed-by-country.csv"
covid_data = pd.read_csv(covid_filepath, index_col="entity")
covid_data

plt.figure(figsize=(15,8))
sns.lineplot(x=covid_data['year'], y=covid_data['total_covid_19_tests'])
plt.xlabel("Age")
plt.ylabel("COVID-19 Tests")
plt.title("")

print("According to this dataset, this is the total number of COVID tests that have been conducted by all countries combined:")
covid_data.total_covid_19_tests.sum() 
country_test_nums_desc = covid_data.sort_values(by='total_covid_19_tests', ascending=False)
print("Here are the Top 10 countries/regions throughout the world for highest COVID tests conducted:")
country_test_nums_desc.head(10)
#New dataset
covid_sources_countries_filepath = "../input/uncover/UNCOVER/covid_tracking_project/covid-sources-for-counties.csv"
covid_sources = pd.read_csv(covid_sources_countries_filepath)
covid_sources