import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))
gdp = pd.read_table('../input/gdp_by_country', header = None)
gdp.head()
gdp_df = pd.DataFrame()

gdp_df["country"] = gdp[0]
gdp_df['last'] = gdp[1]
gdp_df.head()
print("highest gdp per capita : " + str(gdp_df['last'].max()))
print("lowesr gdp per capita : " + str(gdp_df['last'].min()))
print("average gdp per capita : " + str(gdp_df['last'].mean()))
plt.figure(figsize = (15,8))
sns.distplot(gdp_df['last'], 
             hist = True,
             kde = True, 
             rug = True)
plt.show()
plt.figure(figsize = (10,32))
sns.barplot(x = "last", y = "country", data = gdp_df , orient = "h")
plt.show()
gdp_per_capita = pd.read_table('../input/gdp_per_capita' , header = None)
gdp_per_capita.head()
print("highest gdp per capita : " + str(gdp_per_capita[1].max()))
print("lowesr gdp per capita : " + str(gdp_per_capita[1].min()))
print("average gdp per capita : " + str(gdp_per_capita[1].mean()))
plt.figure(figsize = (10,8))
sns.distplot(gdp_per_capita[1],
           hist = True,
           kde = True,
           rug = True)
plt.show()

plt.figure(figsize = (10,32))
sns.barplot(x = 1 , y = 0 , data = gdp_per_capita , orient = "h")
plt.show()
