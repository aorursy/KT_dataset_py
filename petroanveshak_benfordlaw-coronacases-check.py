#Live Benford Law Check on Daily Corona Cases



import numpy as np # linear algebra

import pandas as pd # 



import seaborn as sns

import matplotlib.pyplot as plt
# URL to activate and run code with latest data

#url = 'https://covid19.who.int/WHO-COVID-19-global-data.csv'



# Read file into a DataFrame: df

# df = pd.read_csv(url)



df = pd.read_csv('../input/who-corona/WHO-COVID-19-global-data.csv')



df.head()
dataset = df[df['Date_reported'] == df['Date_reported'].max()]
dataset.head()
dataset.info()
dataset.reset_index(drop = 'index', inplace = True)
count_max = dataset[' Cumulative_cases'].count()

fd = np.arange(count_max)

i = 0

while i < count_max:

    fd[i] = int(str(dataset[' Cumulative_cases'][i])[:1])

    i = i+1

dataset['First Digit'] = fd
dataset.head()
bf_number = pd.Series([1,2,3,4,5,6,7,8,9])



bf_law = pd.Series([0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046])



data_count = dataset['First Digit'].value_counts().reset_index().sort_values('index')



frame = {'Number': bf_number, 'Benford Law': bf_law, 'Dataset_Count': data_count['First Digit']}



check_bf = pd.DataFrame (frame)



x = check_bf['Dataset_Count'].sum()



check_bf['World Corona Cases']=check_bf['Dataset_Count']/x



check_bf
fig, axes = plt.subplots()



axes.bar(check_bf['Number'], check_bf['Benford Law'], label = 'Benford Law', tick_label = check_bf['Number'])

axes.plot(check_bf['Number'], check_bf['World Corona Cases'],'o--', color = 'r')

axes.legend()