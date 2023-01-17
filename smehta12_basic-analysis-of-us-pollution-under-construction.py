import pandas as pd

from pandas import Series, DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt



# import seaborn as sns



us_pollution_df = pd.read_csv(r"../input/pollution_us_2000_2016.csv")



#Rename the first column

us_pollution_df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)

# Remove the spaces from the column names

us_pollution_cols = [col.replace(' ', '_') for col in us_pollution_df.columns]

us_pollution_df.columns = us_pollution_cols



states_no2_1st_val = us_pollution_df.groupby(["State_Code"], as_index=False)["NO2_1st_Max_Value"].mean()



plt.bar( states_no2_1st_val['State_Code'], states_no2_1st_val['NO2_1st_Max_Value'])

plt.xlabel("State Code")

plt.ylabel("Avg. NO2 1st Max Value")

plt.title("State wise mean of the NO2 1st Max Value")

plt.show()


