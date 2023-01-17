# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
df = pd.read_csv("../input/windows-store/msft.csv")
Total_null = df.isnull().sum()
df.dropna(inplace=True)
plt.figure(figsize=(16,16))
sns.countplot(df.Rating)
plt.title("Distribution of Rating",fontsize=20)
def price(x):
    chr = x.replace("₹","")
    if "Free" in chr:
        chr = chr.replace("Free","0")
    elif "," in chr:
        chr = chr.replace(",","")
    return chr
df.Price = df.Price.apply(lambda x : price(x))
df.Price = df.Price.astype("float")
def broad_price(x):
    if x>0:
        return "Paid"
    else:
        return "Free"
df["Broad_price"] = df.Price.apply(lambda x : broad_price(x))
Paid_Free_Percentage = df.Broad_price.value_counts().to_frame(name="Count")
Paid_Free_Percentage["Percentage"] = Paid_Free_Percentage / Paid_Free_Percentage.Count.sum() * 100
plt.figure(figsize=(16,16))
plt.pie(Paid_Free_Percentage.Percentage,labels=Paid_Free_Percentage.index.to_list(),autopct="%.2f%%")
plt.title("Percentage of Free&Paid Software",fontsize=20)
sns.set_style("whitegrid")
plt.figure(figsize=(16,16))
sns.kdeplot(df.Price,bw=0.5,shade=True,color="r")
plt.xlim(0,1000)
plt.xticks(np.arange(0,1000,step=100))
plt.title("Distribution of Price",fontsize=20)
Genre = df.Category.value_counts().to_frame(name="Count")
Genre["Percentage"] = Genre.Count / Genre.Count.sum() * 100
plt.figure(figsize=(16,16))
plt.pie(Genre.Percentage,labels=Genre.index.to_list(),autopct="%.2f%%")
plt.title("Percentage of Category",fontsize=20)
df["date_dt"] = pd.to_datetime(df.Date)
df["Year"] = df.date_dt.dt.year
def quarter(x):
    if 1<=x<=3:
        return 1
    elif 4<=x<=6:
        return 2
    elif 7<=x<=9:
        return 3
    else:
        return 4
df["Quarter"] = df.date_dt.dt.month.apply(lambda x : quarter(x))
plt.figure(figsize=(16,16))
sns.countplot(y=df.Quarter,orient="v",palette="Set2")
plt.title("Software released by Quarter",fontsize=20)
Rating_by_Price = df.groupby("Broad_price")["Rating"].mean()
print(Rating_by_Price)
plt.figure(figsize=(16,16))
sns.violinplot(x="Broad_price",y="Rating",data=df,linewidth=2)
plt.title("Distribution of Rating by Price",fontsize=20)
df.groupby("Category")["Price"].mean().sort_values(ascending=False)
df.groupby("Category")["Rating"].mean().sort_values(ascending=False)