import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/HR_comma_sep.csv")

data.head()
data.info()
correlation = data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True)



plt.title('Correlation between different fearures')
n_count = len(data)

n_promoted = data['promotion_last_5years'].value_counts()[1]

n_left = data['left'].value_counts()[1]



leave_rate = n_left * 100.0 / n_count



print("- {} People in this database".format(n_count))

print("- {} People got promoted in last 5 years".format(n_promoted))

print("- {} People have left the company".format(n_left))

print("--> This means that {:1.2f}% of all employees in this database left the company".format(leave_rate))



print("- Possible department values: {}".format(data['sales'].unique()))
sns.distplot(data['satisfaction_level'])
sns.boxplot(x="left", y="satisfaction_level", data=data,palette='rainbow')
data['satisfaction_level'].mean()
data['last_evaluation'].mean()
g = sns.FacetGrid(data, col="left",  row="promotion_last_5years",hue='salary')

g = g.map(plt.scatter, "average_montly_hours", "satisfaction_level").add_legend()
sns.jointplot(x='average_montly_hours',y='satisfaction_level',data=data,kind='hex', size=7)
sns.jointplot(x='average_montly_hours',y='last_evaluation',data=data,kind='hex', size=7)