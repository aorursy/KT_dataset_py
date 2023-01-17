import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import matplotlib.gridspec as gridspec
data = pd.read_csv('../input/creditcard.csv')
print(data.head())

print(data.shape)
data_class = pd.value_counts(data['Class'])

data_class.plot()

plt.show()

data_class.plot(kind = 'hist')

plt.show()

data_class.plot(kind = 'bar')

plt.title("Class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
timedelta = pd.to_timedelta(data['Time'], unit='s')

data['Time_min'] = (timedelta.dt.components.minutes).astype(int)

data['Time_hour'] = (timedelta.dt.components.hours).astype(int)
plt.figure(figsize=(12,5))

sns.distplot(data[data['Class'] == 0]["Time_hour"], 

             color='g')

sns.distplot(data[data['Class'] == 1]["Time_hour"], 

             color='r')

plt.title('Fraud x Normal Transactions by Hours', fontsize=17)

plt.xlim([-1,25])

plt.show()
df_fraud = data[data['Class'] == 1]

df_normal = data[data['Class'] == 0]



print("Fraud transaction statistics")

print(df_fraud["Amount"].describe())

print("\nNormal transaction statistics")

print(df_normal["Amount"].describe())
columns = data.iloc[:,1:29].columns



frauds = data.Class == 1

normals = data.Class == 0



grid = gridspec.GridSpec(14, 2)

plt.figure(figsize=(15,20*4))



for n, col in enumerate(data[columns]):

    ax = plt.subplot(grid[n])

    sns.distplot(data[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin

    sns.distplot(data[col][normals], bins = 50, color='r') #Will receive the "ocean" color

    ax.set_ylabel('Density')

    ax.set_title(str(col))

    ax.set_xlabel('')

plt.show()