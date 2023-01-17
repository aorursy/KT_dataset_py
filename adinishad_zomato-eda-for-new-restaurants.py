# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv") # import dataset
data.head(3) # first 3 row of dataset
data.shape # shape of data
data.info() # some import info of data
# rename columns
data = data.rename(columns={"approx_cost(for two people)": "AvarageCost", "listed_in(type)": "listed_in_type", "listed_in(city)":"listed_in_city"})
data.head(1)
# drop some unnecessary columns
drop_list = ["address", "phone", "url", "location"]
data.drop(drop_list, axis=1, inplace=True)
# Replace New by NaN
data["rate"] = data["rate"].replace("NEW", np.nan)
data.dropna(how="any", inplace=True)
# rate column string to integer convert
X = data
X["rate"] = X["rate"].astype(str)
X["rate"] = X["rate"].apply(lambda x: x.replace('/5',''))
X["rate"] = X["rate"].apply(lambda x: float(x))
X
rcParams["figure.figsize"] = 14, 8
loc = X["listed_in_city"].value_counts()
sns.set(style="whitegrid")
sns.barplot(y=loc, x=loc.index, palette="Set2")
plt.xticks(rotation=90)
plt.title("Different Location and Restaurent Count", fontweight="bold")
# Count of total online ad ofline order
sns.countplot(X["online_order"])
# Count of table booking 
sns.countplot(X["book_table"])
rcParams["figure.figsize"] = 14, 8

# Differnt Restaurent Type
res_type = X["rest_type"].value_counts()
sns.set(style="whitegrid")
sns.barplot(y=res_type, x=res_type.index, palette="Set2")
plt.xticks(rotation=90)
plt.title("Restaurent Type")
# Avarage cost replace comma
X["AvarageCost"] = X["AvarageCost"].apply(lambda x:x.replace(",",""))
X["AvarageCost"] = X["AvarageCost"].astype("int")
sns.distplot(X["AvarageCost"])
plt.title("Cost distribution for all restaurent")
X = X.drop_duplicates(subset="name", keep="first") # drop duplicates restaurent names
highRatYes = X[(X["rate"] >= 4.5) & (X["online_order"] == "Yes")]
ratOnline = highRatYes.shape[0]
b = highRatYes.max()
c = b["rate"]
a = b["name"]
print(f"Name of High Rated Restaurent: {a} and Rate is: {c}")
print(f"Total {ratOnline} restaurents take online order and their rating above 4.5")

highRatNo = X[(X["rate"] >= 4.5) & (X["online_order"] == "No")]
ratOffline = highRatNo.shape[0]
b = highRatNo.max()
c = b["rate"]
a = b["name"]
print(f"Name of High Rated Restaurent: {a} and Rate is: {c}")
print(f"Total {ratOffline} restaurents take offline order and their rating above 4.5")

print("*" * 50)

midRatYes = X[(X["rate"] >= 4.0) & (X["rate"] <= 4.4) & (X["online_order"] == "Yes")]
ratOnline = midRatYes.shape[0]
b = midRatYes.max()
c = b["rate"]
a = b["name"]
print(f"Name of High Rated Restaurent: {a} and Rate is: {c}")
print(f"Total {ratOnline} restaurents take online order and their rating between 4.0 to 4.4")

midRatNo = X[(X["rate"] >= 4.0) & (X["rate"] <= 4.4) & (X["online_order"] == "No")]
ratOffline = midRatNo.shape[0]
b = midRatNo.max()
c = b["rate"]
a = b["name"]
print(f"Name of High Rated Restaurent: {a} and Rate is: {c}")
print(f"Total {ratOffline} restaurents take offline order and their rating between 4.0 to 4.4")

print("*" * 50)

lowRatYes = X[(X["rate"] < 4.0) & (X["online_order"] == "Yes")]
ratOnline = lowRatYes.shape[0]
b = lowRatYes.max()
c = b["rate"]
a = b["name"]
print(f"Name of High Rated Restaurent: {a} and Rate is: {c}")
print(f"Total {ratOnline} restaurents take online order and their rating less 4.0")

lowRatNo = X[(X["rate"] < 4.0) & (X["online_order"] == "No")]
ratOfline = lowRatNo.shape[0]
b = lowRatNo.max()
c = b["rate"]
a = b["name"]
print(f"Name of High Rated Restaurent: {a} and Rate is: {c}")
print(f"Total {ratOfline} restaurents take offline order and their rating less 4.0")
list_loc = ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
       'Koramangala 4th Block', 'Koramangala 5th Block',
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
       'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
       'Old Airport Road', 'Rajajinagar', 'Residency Road',
       'Sarjapur Road', 'Whitefield']
i = 0
for i in range(len(list_loc)):
    c = X[X["listed_in_city"]==list_loc[i]]
    i+=1
    d = c.max()
    e = d["listed_in_city"]
    f = d["AvarageCost"]
    a = c["listed_in_type"].value_counts()
    g = c.min()
    h = g["AvarageCost"]
    print(f"Location: {e} ----> Avarage Cost between:{h} to {f}\ntypes of restaurent at {e}\n{a}")
# low budget restaurent
low_budget = X.groupby(['name','rest_type','cuisines', 'listed_in_city', 'rate', 'reviews_list', 'dish_liked'])['AvarageCost'].sum().sort_values(ascending=True).reset_index()
low_budget = low_budget[low_budget["AvarageCost"] <= 1500]

# mid budget restaurent
mid_budget = X.groupby(['name','rest_type','cuisines', 'listed_in_city', 'rate', 'reviews_list', 'dish_liked'])['AvarageCost'].sum().sort_values(ascending=True).reset_index()
mid_budget = mid_budget[(mid_budget["AvarageCost"] > 1500) & (mid_budget["AvarageCost"] <= 3000)]

# High budget restaurent
high_budget = X.groupby(['name','rest_type','cuisines', 'listed_in_city', 'rate', 'reviews_list', 'dish_liked'])['AvarageCost'].sum().sort_values(ascending=True).reset_index()
high_budget = high_budget[(high_budget["AvarageCost"] > 3000) & (high_budget["AvarageCost"] <= 6000)]
# Lowest Budget restaurent

low = low_budget["listed_in_city"].value_counts()
g = sns.barplot(y=low.values, x=low.index, palette="Set2")
plt.xticks(rotation=90)
plt.title("Lowest budget restaurent from different area")
for p in g.patches:
    g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.6, p.get_height()+1.3), ha='center', va='bottom', color= 'black', rotation=90)
# Mid budget restaurent

mid = mid_budget["listed_in_city"].value_counts()
g = sns.barplot(y=mid.values, x=mid.index, palette="Set2")
plt.xticks(rotation=90)
plt.title("Mid budget restaurent from different area")
for p in g.patches:
    g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+1.3), ha='center', va='bottom', color= 'black', rotation=90)
# High budget Restaurent


high = high_budget["listed_in_city"].value_counts()
g = sns.barplot(x=high.index, y=high.values, palette="plasma")
plt.xticks(rotation=90)
plt.title("HIghest budget restaurent from different area")
for p in g.patches:
    g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.45, p.get_height()+0.1), ha='center', va='bottom', color= 'black', rotation=90)
# Lowest Cost, High rated different restaurent

plt.subplots(figsize=(18,9))
lowbudVsHigrat = low_budget[low_budget["rate"] >= 4.5]
lowbudVsHigrat.iloc[42,0] ="Santa Spa Cusinies"
sns.barplot(lowbudVsHigrat["name"], lowbudVsHigrat["AvarageCost"], palette="Set2")
plt.xticks(rotation=90);
plt.title("Lowest budget restaurent name vs avarage cost")
# Cusines and mid budgets
print(mid_budget["cuisines"].value_counts()[:10])
# Low budgets and cuisines
print(low_budget["cuisines"].value_counts()[:10])
def cloud_word(budget):
    text = " ".join(dish for dish in budget["cuisines"])
    wc = WordCloud(max_font_size=100,colormap="summer", height=300, width=400, random_state=42, background_color='#151515')
    wc.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
# Low Avarage cost and cuisines
cloud_word(low_budget)
# medium Avarage cost and cuisines
cloud_word(mid_budget)
# High Avarage cost and cuisines
cloud_word(high_budget)
def dish_like(budget):    
    dish_liked = " ".join(f for f in budget["dish_liked"])
    wc_ = WordCloud(max_font_size=100,colormap="Set2", height=300, width=400, random_state=42, background_color='#151515')
    wc_.generate(dish_liked)
    plt.imshow(wc_, interpolation="bilinear")
    plt.axis("off")
# Low avarage Cost and dishes people liked in the restaurant
dish_like(low_budget)
# Medium avarage Cost and dishes people liked in the restaurant
dish_like(mid_budget)
# High avarage Cost and dishes people liked in the restaurant

dish_like(high_budget)
sns.heatmap(data.corr(), annot=True, linewidth=0.5, cmap="Blues_r") # heatmap
