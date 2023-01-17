import pandas as pd
unemployment_data = pd.read_csv("../input/us-unemployment-dataset-2010-2020/unemployment_data_us.csv")
unemployment_data_state = pd.read_csv("../input/us-unemployment-dataset-2010-2020/unemployment_data_us_state.csv")
unemployment_data.head(5)
unemployment_data.tail(5)
unemployment_data.info()
unemployment_data = unemployment_data.dropna()
unemployment_data['Date'] =pd.to_datetime(unemployment_data.Date)
unemployment_data = unemployment_data.sort_values(by='Date', ascending=True)
unemployment_data.isna().sum()
unemployment_data.describe()
import matplotlib.pyplot as plt
# multi-line plotting based on education

plt.figure(figsize=(20,10))
plt.plot( 'Date', 'Primary_School', data=unemployment_data, color='red')
plt.plot( 'Date', 'High_School', data=unemployment_data, color='yellow')
plt.plot( 'Date', 'Associates_Degree', data=unemployment_data, color='magenta')
plt.plot( 'Date', 'Professional_Degree', data=unemployment_data, color='blue')
plt.legend()
# multi-line plotting based on race

plt.figure(figsize=(20,10))
plt.plot( 'Date', 'White', data=unemployment_data, color='magenta')
plt.plot( 'Date', 'Black', data=unemployment_data, color='red')
plt.plot( 'Date', 'Asian', data=unemployment_data, color='blue')
plt.plot( 'Date', 'Hispanic', data=unemployment_data, color='yellow')
plt.legend()
# multi-line plotting based on gender

plt.figure(figsize=(20,10))
plt.plot( 'Date', 'Men', data=unemployment_data, color='blue')
plt.plot( 'Date', 'Women', data=unemployment_data, color='magenta')
plt.legend()
unemployment_data_state.head(5)
unemployment_data_state.tail(5)
unemployment_data_state.info()
unemployment_data_state = unemployment_data_state.dropna()
unemployment_data_state['Date'] =pd.to_datetime(unemployment_data_state.Date)
unemployment_data_state = unemployment_data_state.sort_values(by='Date', ascending=True)
unemployment_data_state.isna().sum()
unemployment_data_state.describe()
print(unemployment_data_state[unemployment_data_state.State == 'New York'])
print(unemployment_data_state[unemployment_data_state.State == 'California'])
print(unemployment_data_state[unemployment_data_state.State == 'North Carolina'])
print(unemployment_data_state[unemployment_data_state.State == 'New Jersey'])
plt.figure(figsize=(20,10))
plt.plot( 'Date', 'Unemployment_Rate', data=unemployment_data_state[unemployment_data_state.State == 'New York'], color='red')
plt.legend()
plt.figure(figsize=(20,10))
plt.plot( 'Date', 'Unemployment_Rate', data=unemployment_data_state[unemployment_data_state.State == 'California'], color='red')
plt.legend()
plt.figure(figsize=(20,10))
plt.plot( 'Date', 'Unemployment_Rate', data=unemployment_data_state[unemployment_data_state.State == 'North Carolina'], color='red')
plt.legend()