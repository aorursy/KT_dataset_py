import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
file_folder = '../input/beer-consumption-in-20182019/'

data = pd.read_csv(f'{file_folder}/drink_dataset_money.csv')

print(data.shape)
data.head()
counts = data['drank'].sum()

print("I drank {} times over the last year, or every {:.2f} days.".format(counts, data.shape[0] / counts))
s = data['drank'].astype('bool')

print("Longest abstinence time: {} days".format(s.cumsum().value_counts().max() - 1))

print("Longest drinking streak: {} days".format((~s).cumsum().value_counts().max() - 1))
data['date'] = pd.to_datetime(data['date'])

data['day'] = data['date'].apply(lambda x: x.day)

data['month'] = data['date'].apply(lambda x: x.month)

data['year'] = data['date'].apply(lambda x: x.year)

data['weekday'] = data['date'].apply(lambda x: x.dayofweek)
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']



values = data.groupby('weekday')['drank'].agg('sum')

values.index = [day_names[x] for x in values.index]

values.plot(kind='bar', figsize=(15,7), grid=True)

plt.title("What day of the week do I prefer?")

plt.xlabel("Day of the week")

plt.ylabel("Frequency")

plt.show()
values = data.groupby('hour')['drank'].agg('sum')



for i in range(24):

    if i not in values:

        values[i] = 0

        

values.sort_index().plot(kind='bar', figsize=(15,10), grid=True)

plt.title("When do I usually start drinking?")

plt.xlabel("Hour")

plt.ylabel("Frequency")

plt.show()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



values = data.groupby('month')['drank'].agg('sum')

values.index = [month_names[x - 1] for x in values.index]

values.plot(kind='bar', figsize=(10,5), grid=True)

plt.title("Which month did I drink the most beer?")

plt.xlabel("Month")

plt.ylabel("Frequency")

plt.show()
data[data['drank'] == 1]['day'].value_counts().sort_index().plot(kind='bar', figsize=(15,7), grid=True)

plt.title("What day of the month is the most popular?")

plt.xlabel("Day of the month")

plt.ylabel("Frequency")

plt.show()
money_spent = data['money_spent'].sum()

print("I spent {:.0f}$ on alcohol over the last year, or {:.2f}$ a day.".format(money_spent, data.shape[0] / money_spent))
values = data.groupby('month')['money_spent'].agg('sum')

values.index = [month_names[x - 1] for x in values.index]

values.plot(kind='bar', figsize=(15,7), grid=True)

plt.title("How much money did I spend each month?")

plt.xlabel("Month")

plt.ylabel("$")

plt.show()
values = data.groupby('weekday')['money_spent'].agg('sum')

values.index = [day_names[x] for x in values.index]

values.plot(kind='bar', figsize=(15,7), grid=True)

plt.title("How much money did I spend each weekday?")

plt.xlabel("Day of the week")

plt.ylabel("$")

plt.show()
values = data[data['drank'] == 1]['money_spent']

values.value_counts().sort_index().plot(kind='bar', figsize=(15,7), grid=True)

plt.title("How much money do I usually spend every time?")

plt.xlabel("Amount spent $")

plt.ylabel("Frequency")

plt.show()
print("On average I spend {:.2f}$ every time I drink.".format(values.mean()))