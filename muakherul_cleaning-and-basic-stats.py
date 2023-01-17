# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/books.csv', delimiter='|')
df.head(8).T
df.describe()
def calc(DP, D):
    P = (DP*100)/(100-D)
    return P

for i in range(0,1710):
    df.Price[i] = int(round(calc(df.DiscountedPrice[i], df.Discount[i])))
df.head(8).T
df.Price[151455] = 180
df.DiscountedPrice[151455] = 153
df = df.drop(['DiscountedPrice', 'Discount', 'ISBN', 'Edition'], axis=1)
#df.groupby(['Discount'])['Discount'].count().sort_values(ascending=False)
df.Reviews = df.Reviews.replace('No', '0')
df.Ratings = df.Ratings.replace('Not', '0')
df.RatingsNum = df.RatingsNum.replace('Write', '0')
df.Reviews = df.Reviews
df.urlID = df.urlID
df = df[df.Category != 'তুর্কি ক্যাপ']
df = df[df.Category != 'Shari']
#লেজার ভিশন, নিউ ইগল ভিডিও, ইমপ্রেস টেলিফিল্ম, জি সিরিজ প্রোডাকশন, জি-সিরিজ ও অগ্নিবীণা প্রোডাকশন, সিডি প্লাস, ইমপ্রেস অডিও ভিশন
dfpub = pd.read_csv('../input/publisher.csv', delimiter="\n")
#dfpub.groupby([' এলএলসি-ক্রিয়েট স্পেস'])[' এলএলসি-ক্রিয়েট স্পেস'].count()
dfpub.head()
dfpub = dfpub[dfpub.Publisher != 'Games World Books']
dfpub = dfpub[dfpub.Publisher != 'গেমস্‌  ওয়ার্ল্ড']
df = df[df.Publisher.isin(dfpub.Publisher)]
df.RatingsNum = df.RatingsNum.astype(int)
df.Reviews = df.Reviews.astype(int)
df.urlID = df.urlID.astype(int)

df.head().T
len(df)
df.groupby('Author')['Author'].count().sort_values(ascending=False).head(10)
df.groupby('Category')['Title'].count().sort_values(ascending=False).head(10)
#fig = plt.figure(figsize=(20,8), dpi = 200)

dendf = df[df.Author == 'হুমায়ূন আহমেদ'][['Title', 'Category', 'Ratings', 'Reviews']].sort_values(['Reviews', 'Ratings'], ascending=False).head(10)
dendf
print("Total books by হুমায়ূন আহমেদ: " + str(len(df[df.Author == 'হুমায়ূন আহমেদ'])))
#df[df.Author == 'হুমায়ূন আহমেদ'].groupby('Category').Title.count().sort_values(ascending=False)
nwt = df[df.Author == 'হুমায়ূন আহমেদ'][['Author', 'Category', 'Title']]

"""
# Import PhyloVega.
from phylovega import VegaTree

# Initialize a Vega Tree object.
vt = VegaTree(testdf)

# Display the tree.
vt.display()
"""
nwt.T
inse = len(df[(df.Ratings > 0) | (df.Reviews > 0)])
unin = len(df[(df.Ratings > 0) & (df.Reviews > 0)])
print("Total rows either Ratings or Reviews have value greater than 0: " + str(inse))
print("Total rows Ratings and Reviews both have value greater than 0: " + str(unin))
df.sort_values('Pages', ascending=False).head(10)
#set data range for Price and Pages
#eleminate zero and other hightest values
dff = df[['Pages', 'Price', 'Category']][(df.Price < 2000) & (df.Pages < 2000)] 

N = len(dff) 
x = dff.Pages
y = dff.Price
colors = np.random.rand(N)
groups = df.groupby('Category')

# Plot
#plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
#colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

#area = np.pi * (25 * np.random.rand(N))**2  # 0 to 15 point radii
fig = plt.figure(figsize=(20,10),dpi = 200)
plt.scatter(x, y,c=colors, alpha=0.5)
plt.xlabel('Page number')
plt.ylabel('Price')
plt.title('Price compare to Page number')
plt.show()
dff = df[['Pages', 'Price']][(df.Price < 2000) & (df.Price > 0) & (df.Pages < 2000) & (df.Pages > 0)] 
ranges = [1,50,100,200,300,500,1000,1500]

dff['ranges'] = pd.cut(dff['Price'], bins = ranges, labels=["1-50", "51-100", "101-200", "201-300", "301-500", "501-1000", "1000+" ])
dff = dff.groupby(['Price', 'ranges']).size().unstack(fill_value=0)
dff.head(5)
#dff.Price.plot(kind='pie', figsize=(8,8))
ranges = [1,50,100,150,200,300,400,500,1000,1500]

dff['ranges'] = pd.cut(dff['Pages'], bins = ranges, labels=["1-50", "51-100", "101-150","150-200" "201-300", "301-400","400-500" "501-1000", "1001-1500", "1500+" ])
dff = dff.groupby(['Pages', 'ranges']).size().unstack(fill_value=0)
dff.head(5)
#dff.Price.plot(kind='pie', figsize=(8,8))
dff[['1-50', '51-100', '101-200', '201-300', '301-500', '501-1000', '1000+']].sum().plot(kind='pie',autopct='%1.1f%%', figsize=(8,8))
