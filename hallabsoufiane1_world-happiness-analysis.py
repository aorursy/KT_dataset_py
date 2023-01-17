import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(os.path.join(dirname, filename))

# data = pd.read_csv('world-happiness-report-2019.csv')
data.head()
data.info()
data.isna().sum()
filled_data = data.fillna(method = 'ffill')



filled_data.isna().sum()
#Scatter plotting the six criterias used to assess happinness

criteria = filled_data.columns[5:11]



plt.figure(figsize=(16,10))

for i in range(criteria.shape[0]):

    plt.subplot(2,3,i+1)

    plt.scatter(x=filled_data['Ladder'],y=filled_data[criteria[i]])

    plt.xlabel('Ladder')

    plt.ylabel(criteria[i])
ten_best = filled_data[0:11]

ten_middle = filled_data[72:83]

ten_worst = filled_data[filled_data.shape[0]-11:-1]

plt.figure(figsize=(16,10))

for i in range(criteria.shape[0]):

    plt.subplot(2,3,i+1)

    plt.scatter(x=ten_best['Ladder'],y=ten_best[criteria[i]],c='green')

    plt.scatter(x=ten_middle['Ladder'],y=ten_middle[criteria[i]],c='blue')

    plt.scatter(x=ten_worst['Ladder'],y=ten_worst[criteria[i]],c='red')

    plt.xlabel('Ladder')

    plt.ylabel(criteria[i])

plt.suptitle("Green : Top ten, Blue : Middle ten, Red : Bottom ten");
reduced_data = filled_data[['Social support', 

                'Log of GDP\nper capita', 

                'Healthy life\nexpectancy']]

reduced_data.columns = ['Social', 'GDP', 'Health'] # Reducing the name of the columns



corr = reduced_data.corr()

sns.heatmap(corr,vmin=0.5,vmax=1)



print(corr)