# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

sns.set(style="darkgrid")
tips = sns.load_dataset('tips')
tips.info()
tips.head(10)
tips.describe().T
plt.figure(figsize=(10,7))

plt.title('Total Bill Amounts By Days')

sns.stripplot(x="day", y="total_bill", data=tips);
plt.figure(figsize=(10,7))

plt.title('Total Wage Distributions By Gender And Days')

sns.stripplot(x="sex", y="total_bill", hue="day", data=tips, marker="D");
plt.figure(figsize=(10,7))

plt.title('Total Bill Amounts By Days')

sns.swarmplot(x="day", y="total_bill", data=tips);
plt.figure(figsize=(10,7))

plt.title('Total Bill Amount by Days and Customer Smoking Status')

sns.swarmplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set2", dodge=True);
plt.figure(figsize=(20,7))

plt.title('Total Bill')

sns.boxplot(x=tips["total_bill"]);
plt.figure(figsize=(20,7))

plt.title('Total Bill By Time And Day')

sns.boxplot(x="day", y="total_bill", hue="time", data=tips, palette="Set1");
plt.figure(figsize=(20,7))

plt.title('Total Bill')

sns.violinplot(x=tips["total_bill"]);
plt.figure(figsize=(20,7))

plt.title('Total Bill By Sex And Day')

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, palette="Set3_r");
plt.figure(figsize=(20,7))

plt.title('Total Bill')

sns.boxenplot(x=tips["total_bill"], color='g');
plt.figure(figsize=(20,7))

plt.title('Total Bill By Time')

sns.pointplot(x="time", y="total_bill", data=tips);
plt.figure(figsize=(20,7))

plt.title('Number Of Customers By Days')

sns.barplot(y=tips.day.index, x=tips.day, color='purple'); 
plt.figure(figsize=(20,7))

plt.title('Tip Total By Days')

sns.barplot(y='tip', x='day', data=tips, ci=77);
plt.figure(figsize=(20,7))

plt.title('Gender Of Customers Who Tip By Days')

sns.barplot(y=tips.tip, x=tips.day, hue=tips.sex); 
plt.figure(figsize=(20,7))

plt.title('Tip Distributions By Customer Group')

sns.barplot(x="size", y="total_bill", data=tips, palette="cubehelix");
plt.figure(figsize=(20,7))

plt.title('Customer Distributions By Sex')

sns.countplot(x="sex", data=tips);
plt.figure(figsize=(20,7))

plt.title('Tips by Total Bill')

sns.scatterplot(x="total_bill", y="tip", data=tips);
plt.figure(figsize=(20,7))

plt.title('Total Bill by Day')

sns.lineplot(x="day", y="tip", data=tips);
plt.figure(figsize=(20,7))

plt.title('Total Bill by Tip')

sns.regplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="total_bill", y="tip", data=tips);
plt.figure(figsize=(20,7))

sns.residplot(x=tips.tip, y=tips.total_bill, data=tips, scatter_kws={"s": 80});
plt.figure(figsize=(20,7))

sns.distplot(tips.total_bill);
plt.figure(figsize=(20,7))

sns.kdeplot(tips.total_bill);

sns.kdeplot(tips.tip);