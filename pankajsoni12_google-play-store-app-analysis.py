# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/googleplaystore.csv")
df.head(2)
plt.figure(figsize=(16,4))
df.Genres.value_counts()[:50].plot(kind="bar")
plt.show()
df.Rating.max()
df.Rating.min()
df.Rating.unique()
df1 =df[df.Rating.notnull()][["Rating","Type"]]
plt.figure(figsize=(16,4))
sns.boxplot(df1.Type,df1.Rating,)
# ABove figure tell that we have one value which is completely outlier for this data (i.e Rating 19). Lets remove them and plot again
df1 =df[(df.Rating.notnull()) & (df.Rating <19)][["Rating","Type"]]
plt.figure(figsize=(16,4))
sns.boxplot(df1.Type,df1.Rating)
plt.figure(figsize=(16,4))
df.Category.value_counts().plot(kind="bar")
plt.show()
df.head(1)
df.Category.value_counts()[:5]
Family_4_star = df[(df.Category == "FAMILY") & (df.Rating >= 3)].Category.count()
GAME_4_star = df[(df.Category == "GAME") & (df.Rating >= 3)].Category.count()
TOOLS_4_star = df[(df.Category == "TOOLS") & (df.Rating >= 3)].Category.count()
MEDICAL_4_star = df[(df.Category == "MEDICAL") & (df.Rating >= 3)].Category.count()
BUSINESS_4_star = df[(df.Category == "BUSINESS") & (df.Rating >= 3)].Category.count()
plt.figure(figsize=(16,4))
plt.bar("Family_4_star",Family_4_star)
plt.bar("GAME_4_star",GAME_4_star)
plt.bar("TOOLS_4_star",TOOLS_4_star)
plt.bar("MEDICAL_4_star",MEDICAL_4_star)
plt.bar("BUSINESS_4_star",BUSINESS_4_star)
plt.show()
df.Reviews.dtype
# pd.to_numeric(df.Reviews) 

# After analyzing thsi data, we see that their are some numbers 
# which has "M" for million, so it is not feasible to convert all of them to numeric. 
# !!!Someone can try here to add extra logic !!!
df.Type.unique()
df.Type.value_counts()
plt.figure(figsize=(16,4))
df.Type.value_counts().plot(kind="bar")
df.head(1)
rating_price_var = df[df.Rating >=3]
rating_price_var.Price.unique()
rating_price_var.Price.replace({"Everyone":0},inplace=True)
# Ignore the warning
rating_price_var.Price.unique()
# Now we can see "Everyone" is not present
rating_price_var.Price.replace({"\$":""},regex=True,inplace=True)
# Now we can see "Everyone" is not present
rating_price_var.Price.unique()
# Now no $ sign present
rating_price_var.Price = pd.to_numeric(rating_price_var.Price)
# Now no $ sign present
rating_price_var.Price.dtype
plt.figure(figsize=(16,4))
rating_price_var.Rating.sort_values().value_counts().plot(kind="bar")
plt.show()
# plt.figure(figsize=(16,4))
rate_price = rating_price_var[["Rating","Price"]].sort_values(by="Rating",ascending=False)
rate_price.head(2)
# plt.show()
plt.figure(figsize=(16,4))
plt.scatter(rate_price.Rating[1:],rate_price.Price[1:],alpha = .2,color="g")
plt.xticks(np.linspace(3,5,len(rate_price.Rating[1:]))[::400])
plt.show()
plt.figure(figsize=(16,4))
rating = rate_price.Rating[1:].tolist()
price = rate_price.Price[1:].tolist()
for i in range(rate_price.Rating[1:].size):
    plt.scatter(rating[i],price[i],alpha = .2,color="g")
    if price[i] >=250:
        plt.scatter(rating[i],price[i],alpha = .2,s=.012*price[i]**2,color="m",marker="*")
        plt.text(rating[i],price[i]+3,(rating[i],price[i]))
# plt.xticks(np.linspace(3,5,len(rate_price.Rating[1:]))[::400])
plt.show()