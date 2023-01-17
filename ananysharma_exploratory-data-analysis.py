import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/telecom-churn-datasets/churn-bigml-80.csv')
data[:5]
data.nunique()
data["International plan"].value_counts()
data["Voice mail plan"].value_counts()
data["Churn"].value_counts()
dict = {False:0,True:1}
data["Churn"] = data["Churn"].map(dict)
data["Churn"][:20]
dict1 = {"No":0,"Yes":1}
data["International plan"] = data["International plan"].map(dict1)
data["Voice mail plan"]  = data["Voice mail plan"].map(dict1)
data.head()
plt.rcParams["figure.figsize"] = (15,8)
sns.scatterplot(data["Total day calls"],data["Total eve calls"],palette = "dark")
plt.title("Scatter Plot",fontsize=20)
plt.xlabel("Day Calls",fontsize=15)
plt.ylabel("Eve Calls",fontsize=15)
plt.show()
plt.rcParams["figure.figsize"] = (15,8)
sns.countplot(data["Area code"],palette = "dark")
plt.title("Count per Area",fontsize=20)
plt.xlabel("Area code",fontsize=15)
plt.show()
data.head()
data[(data["Churn"]==0 & (data["International plan"]==0))]["Total day calls"].mean()
data[(data["Churn"]==0 & (data["International plan"]==0))]["Total intl minutes"].mean()
data.apply(np.max)
data[data["State"].apply(lambda x: x[0]=="W")]
columns_to_show = ['Total day minutes', 'Total eve minutes', 
                   'Total night minutes']

data.groupby(['Churn'])[columns_to_show].describe(percentiles=[])
data.groupby(['Churn'])[columns_to_show].agg([np.mean,np.std])
data.groupby(['International plan'])[columns_to_show].agg([np.max,np.min,np.mean])

data.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
               ['Area code'], aggfunc='mean')
data.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
               ['Churn'], aggfunc='mean')
data.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
               ['International plan'], aggfunc='mean')
data["Total Calls"] = data["Total day calls"]+data["Total eve calls"]+data["Total night calls"]+data["Total intl calls"]
data[:5]
sns.countplot(data["International plan"],hue = data["Churn"],palette = "dark")
sns.countplot(data["Customer service calls"],hue = data["Churn"],palette = "dark")
data.loc[data["Churn"]==0]
data.iloc[-1]
sns.violinplot(x=data["Churn"],y=data["Total day calls"],palette = "Reds")
sns.violinplot(x=data["Churn"],y=data["Total Calls"],palette = "Reds")
sns.boxplot(x=data["Churn"],y=data["Total day calls"],palette = "Reds")
plt.rcParams["figure.figsize"] = (20,10)
sns.heatmap(data[:].corr(),annot = True)
plt.title("Heatmap of data",fontsize= 15)
plt.show()
