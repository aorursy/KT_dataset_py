

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# import packages 
# pandas for easy to use data structure 
# seaborn for data visualization 
# numpy for performing mathematicaloperation 
df=pd.read_csv("../input/haberman.csv")
# load the haberman data in to df 
df
# print the data 
print(df.shape)
# tell the rows and columns of the data
df.columns=['Age','Operation_Year','axil_nodes','surv_status']
# rename columns name according to given in kaggle platform
df
# print the data
print(df.columns)
# display columns name of dataframe 
df["surv_status"].value_counts()
# in this data i  filter the data on the basis of surv_status columns
#so there are 224  people whose survive status are equal to 1  means survive 5 years or more and
#there are 84 people whose survive status are equal to 2  means who 

df.head() # this is top 5 values  of my dataframe 
df.tail() # this is my last five  value of my data frame 
df.describe()
#plot the graph of df 
df.hist()
#  individual column wise histogram plot 
df.plot()

df.plot(kind='scatter', x='Age', y='axil_nodes') ;
plt.show()
# this is scatter plot b/w age and axil_nodes 
# as we observe the axil_nodes data is  below the 25 value is high and rare above that 

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4) \
   .map(plt.scatter, "Age", "axil_nodes") \
   .add_legend();
plt.show();

# 3d scatter plots 
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="surv_status", size=3);
plt.show()
sns.FacetGrid(df, hue="surv_status", size=5) \
   .map(sns.kdeplot, "Age") \
   .add_legend();
plt.show();

sns.FacetGrid(df, hue="surv_status", size=5) \
   .map(sns.kdeplot, "axil_nodes") \
.add_legend()
   
plt.show();
# as we can observe from the graph the survive chance is more for where axil node is -8 to 8 




cdf = sns.kdeplot(df['Age'], cumulative=True)
cdf= sns.kdeplot(df['Operation_Year'], cumulative=True)
cdf=sns.kdeplot(df['axil_nodes'], cumulative=True)
plt.show()
# cdf graph which is lies b/w 0to 1 


sns.boxplot(x='surv_status',y='Age', data=df)
plt.show()
# from this box plot we can observe the median is 52 and outliers is 30 and 77 (approx ) for surv_status=1
# from this box plot we can observe the median is 52 and outliers is 34 and 85 (approx ) for surv_status=2
sns.boxplot(x='surv_status',y='Operation_Year', data=df)
plt.show()
sns.boxplot(x='surv_status',y='axil_nodes', data=df)
plt.show()
#If no. of axillary nodes is less,than survival of patients is more
sns.violinplot(x="surv_status", y="axil_nodes", data=df, size=8)
plt.show()
sns.jointplot(x="axil_nodes", y="surv_status", data=df, kind='kde');
plt.show();

#conclusion 
#If no. of axillary nodes is less,than survival of patients is more
#Younger people has more chance of survival


