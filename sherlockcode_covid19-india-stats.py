import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as webreader

df = pd.read_excel("../input/covid-cases/Covid cases in India.xlsx")
#df2 = webreader.DataReader('Corona Cases in India', data_source='yahoo')
#df2
df['Total Cases']= df['Total Confirmed cases (Indian National)']+ df['Total Confirmed cases ( Foreign National )']
total_cases = df['Total Cases'].sum()
print(total_cases)

df.style.background_gradient(cmap='Reds')

#Finding out the total number of Active cases in India
df['Total Active Cases']= df['Total Cases']- df['Cured']+df['Death']

#Sorting the values according to the State
tot_case_statewise = df.groupby('Name of State / UT')['Total Active Cases'].sum().sort_values(ascending=False).to_frame()
tot_case_statewise.style.background_gradient(cmap='Reds')

#Ploting the Graph
plt.figure(figsize=(40,20))
x1 = df['Name of State / UT']
y2 = df['Total Active Cases']
#Printing all shits
plt.bar(x1, y2, width = 0.9, color = ['red'])
tot_case_statewise.style.background_gradient(cmap='Reds')
df.style.background_gradient(cmap='Reds')




