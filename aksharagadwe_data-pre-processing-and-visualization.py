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
data = pd.read_csv("/kaggle/input/windows-store/msft.csv")
data.head()
data.describe()
data.info()
data =data.drop([5321],axis=0)
data.head()
data.iloc[5320].Price
data.Price = data.Price.replace("Free" , "0.00")
data.info()
data['New_Price'] = pd.Series(data['Price'], dtype="string")
data.info()
data["New_Price"]= data["New_Price"].apply(lambda x: x.replace("₹","").strip())
data = data.drop("Price",axis=1)
data["New_Price"]= data["New_Price"].apply(lambda x: x.replace(",","").strip())
data["New_Price"] = data["New_Price"].astype(str).astype(float)
data.info()
data = data.set_index("Name")
data.head()
data["Date_new"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date_new"].dt.year
data = data.drop("Date",axis=1)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,8))

data_ = data['Category'].value_counts() 
points = data_.index 
frequency = data_.values 
ax.bar(points, frequency) 
ax.set_title('Categrory Frequency') 
ax.set_xlabel('Category') 
ax.set_ylabel('Frequency')
most_reviewed = data.nlargest(10,"No of people Rated")
fig, ax = plt.subplots(figsize=(20,8))

data_ = most_reviewed.index
rating_ = most_reviewed["Rating"] 
ax.bar(data_, rating_) 
ax.set_title('Most Reviewed') 
ax.set_xlabel('Name') 
ax.set_ylabel('Rating')
yearwise = data.groupby('Year')["Rating"].nlargest(3)
names=[]
for i in range(len(yearwise)):
    names.append(yearwise.index[i][1])
    
names
name1 = names[0::3]
name2 = names[1::3]
name3 = names[2::3]

ratings1 = yearwise[0::3]
ratings2 = yearwise[1::3]
ratings3 = yearwise[2::3]

width = 0.25

r1 = np.arange(len(name1))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

fig, ax = plt.subplots(figsize=(25,8))

plt.bar(r1, ratings1, color='#f55353', width=width, edgecolor='white', label='1st')
plt.bar(r2, ratings2, color='#fc9338', width=width, edgecolor='white', label='2ns')
plt.bar(r3, ratings3, color='#fcd538', width=width, edgecolor='white', label='3rd')
 
plt.xticks([r for r in range(len(r1))],name1, rotation=90)



