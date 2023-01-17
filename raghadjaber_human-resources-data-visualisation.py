from collections import Counter

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

print("Setup Complete")
human_filepath = '../input/human-resources-data-set/HRDataset_v13.csv'

human_data = pd.read_csv(human_filepath)
human_data
human_data.columns
human_data.shape
nan_value = float("NaN") #Convert NaN values to empty string

human_data.replace("", nan_value, inplace=True)

human_data.dropna(subset = ["Employee_Name"], inplace=True)
print(human_data.shape)
human_data
human_data.describe()
print(human_data['Sex'].value_counts())

human_data['Sex'].value_counts().plot(kind='bar')
plt.figure(figsize=(16,5))

sns.countplot(x=human_data['Department'],hue=human_data['Sex'])
plt.figure(figsize=(16,5))

sns.countplot(x="ManagerName", data=human_data)

plt.yticks(np.arange(0, 25,2))

plt.xticks(rotation = 80)
plt.figure(figsize=(16,5))

sns.countplot(x="ManagerName", hue="Department", data=human_data)

import numpy as np

plt.yticks(np.arange(0, 22, 2))

plt.xticks(rotation = 80)
plt.figure(figsize=(16,5))

sns.countplot(x="ManagerName", hue="PerformanceScore", data=human_data)

import numpy as np

plt.yticks(np.arange(0, 22, 2))

plt.xticks(rotation = 80)
plt.figure(figsize=(16,5))

sns.countplot(x="ManagerName", hue="EmpSatisfaction", data=human_data)

import numpy as np

plt.yticks(np.arange(0, 12))

plt.xticks(rotation = 80)
plt.figure(figsize=(16,6))

#sns.distplot(a=human_data['PositionID'], kde=False)

sns.countplot(x="Position", data=human_data)

plt.yticks(np.arange(0, 150, 10))

plt.xticks(rotation = 80)
plt.figure(figsize=(16,5))

sns.swarmplot(y=human_data['PayRate'],

              x=human_data['Position'])

plt.xticks(rotation = 80)
plt.figure(figsize=(16,5))

sns.kdeplot(data=human_data['PayRate'],shade=True)

plt.xticks(np.arange(0, 100,5))
plt.figure(figsize=(12,5))

sns.countplot(y=human_data['RaceDesc'])
plt.figure(figsize=(16,8))

sns.swarmplot(x=human_data['PayRate'],

              y=human_data['RaceDesc'])

#plt.yticks(rotation = 80)
plt.figure(figsize=(12,6))

sns.regplot(x=human_data['PerfScoreID'],

            y=human_data['SpecialProjectsCount'])
plt.figure(figsize=(12,6))

sns.regplot(x=human_data['SpecialProjectsCount'],

            y=human_data['PayRate'])
plt.figure(figsize=(12,6))

sns.regplot(x=human_data['PayRate'],

            y=human_data['PerfScoreID'])
plt.figure(figsize=(16,6))

sns.lmplot(x='PayRate', y='PerfScoreID', hue='Sex', data=human_data)
plt.figure(figsize=(16,5))

sns.countplot(x="Position", hue="PerfScoreID", data=human_data)

import numpy as np

plt.yticks(np.arange(0, 120, 10))

plt.xticks(rotation = 80)
plt.figure(figsize=(16,4))

sns.countplot(x="Position", hue="EmpSatisfaction", data=human_data)

import numpy as np

plt.yticks(np.arange(0, 50, 5))

plt.xticks(rotation = 80)
plt.figure(figsize=(14,8))

sns.regplot(x=human_data['PayRate'],y=human_data['EmpSatisfaction'])
human_data['MaritalDesc'].value_counts().plot(kind='bar')
plt.figure(figsize=(16,6))

sns.countplot(x=human_data['EmploymentStatus'],hue=human_data['MaritalDesc'])
plt.figure(figsize=(16,5))

sns.countplot(x=human_data['RecruitmentSource'],hue=human_data['PerformanceScore'])

plt.xticks(rotation = 80)