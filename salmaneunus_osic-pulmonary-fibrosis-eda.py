import pandas as pd

data = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data.describe
data.head()
data.tail()
data
data.shape
print(data)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16,6))

sns.lineplot(x = data.Weeks,y=data.Percent)

#plt.show()
sns.distplot(data['FVC'],kde=False)

plt.title("Changes in Forced Vital Capacity")
sns.kdeplot(data=data['Percent'],label="Percentage distribution",shade=True)

plt.title("Changes in Percentage distribution")
sns.scatterplot(x=data['Weeks'],y=data['Percent'])

plt.title("Changes in Percentage over time")
sns.regplot(x=data['Weeks'],y=data['Percent'])
sns.scatterplot(x=data['Weeks'],y=data['Percent'],hue = data['SmokingStatus'])
sns.lmplot(x='Weeks',y='Percent',hue = 'SmokingStatus',data=data)
plt.figure(figsize=(20,20))

sns.barplot(x = data.Weeks, y= data.Percent)
sns.lmplot(x='Weeks',y='FVC',hue = 'SmokingStatus',data=data)

plt.title("Comparison of Forced Vital Capacity of Lungs over time among smokers and non smoker")

plt.ylabel("Forced Vital Capacity of Lungs")

plt.xlabel("Number of weeks")