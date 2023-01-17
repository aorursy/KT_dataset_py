import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
print("Setup Complete")
file_path = '../input/novel-corona-virus-2019-dataset/covid_19_data.csv'

my_data = pd.read_csv(file_path, index_col= 'ObservationDate', parse_dates= True)
my_data.head()
# Create a plot
brazil = my_data[my_data['Country/Region'] == 'Brazil']

sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(data=brazil['Confirmed'], label='Confirmed')
sns.lineplot(data=brazil['Deaths'], label='Deaths');