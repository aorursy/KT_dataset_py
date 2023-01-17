import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
data.head()
data["Age"].hist(bins = 80, figsize=(15,8), rwidth=0.9)
plt.xlabel("Age of clients")
plt.ylabel("Counts")
plt.title("Histogram of client's age")
plt.rcParams['axes.axisbelow'] = True
h = data["Rating"].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(data["Rating"].unique(),h)
plt.xlabel("Rating")
plt.ylabel("Counts")
plt.title("Histogram of ratings")
plt.figure(figsize=(8,4))
ax.grid(True)
plt.rcParams['axes.axisbelow'] = True
means = data.groupby(by="Age").mean().reset_index()
sns.jointplot(x="Age", y="Positive Feedback Count", data=means, kind="reg", height=10, ratio=4);
plt.grid(True)
clothes = data['Class Name']
clothes.value_counts()
dresses = data[data['Class Name']=='Dresses']
means = dresses.groupby(by="Age").mean().reset_index()
sns.jointplot(x="Age", y="Positive Feedback Count", data=means, kind="reg", height=10, ratio=4)
plt.grid(True)