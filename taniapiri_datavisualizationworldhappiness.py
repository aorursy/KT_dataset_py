import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
my_path="../input/world-happiness-report/2020.csv"
my_data=pd.read_csv(my_path)
my_data.head()
my_data.set_index('Country name',inplace=True)
my_data.head()
plt.figure(figsize=(25,6))

chart=sns.lineplot(x=my_data.index,y=my_data['Ladder score'])

chart.set_xticklabels(labels=my_data.index,rotation=90)
plt.figure(figsize=(20,6))

chart=sns.barplot(x=my_data.index,y=my_data["Ladder score"])

chart.set_xticklabels(labels=my_data.index,rotation=90)
plt.figure(figsize=(20,6))

chart=sns.barplot(x=my_data["Ladder score"],y=my_data["Social support"])

chart.set_xticklabels(labels=my_data["Ladder score"],rotation=90)
sns.regplot(x=my_data["Healthy life expectancy"],y=my_data["Ladder score"])
sns.lmplot(x="Healthy life expectancy",y="Ladder score",hue='Regional indicator',data=my_data)
sns.distplot(a=my_data["Ladder score"],kde=False)
sns.regplot(x=my_data["Perceptions of corruption"],y=my_data["Freedom to make life choices"])
sns.kdeplot(data=my_data["Dystopia + residual"],shade=True,label="Dystopia+Residual distribution")