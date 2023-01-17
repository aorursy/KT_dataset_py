# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Ignore warnings :

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv("../input/videogamesales/vgsales.csv")

data=data.dropna()

convert={'Year':int}

data=data.astype(convert)

data
#shape of the data

data.shape
# to seek any null values

data.isnull().sum()
#info of the data

data.info()
#statistical features of the data

data.describe()
#correlation 

data1=data.corr()

sns.heatmap(data=data1,square=True,annot=True,linewidths=.5,fmt='.1f')
sns.factorplot(data=data,size=4,aspect=2.5,kind='box')
sns.kdeplot(data['Global_Sales'],shade=True,color='r')
#to seek sales max in year(1988,1992 etc)

chart=sns.factorplot(data=data,x='Year',y='Global_Sales',size=12,aspect=1,kind='box',ci=None)

chart.set_xticklabels(rotation=90)
#genres( sports have the maximum share followed by platform)

labels = data.Genre.unique().tolist()

sizes = data.Genre.value_counts().tolist()

colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7','#ff9999','#66b3ff','#007301','#ff0009','#67F9D9','#16F0A9']

explode = (0.1, 0.0, 0.1, 0, 0.1, 0, 0.1,0,0.1,0,0.1,0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=0)

plt.axis('equal')

plt.title("Percentage of Genre")

plt.plot()

fig=plt.gcf()

fig.set_size_inches(6,6)

plt.show()
#genre having max global sales i.e. platform followed by shooter

chart=sns.factorplot(data=data,x='Genre',y='Global_Sales',size=5,aspect=2.5,kind='bar',ci=None)

chart.set_xticklabels(rotation=45)
sns.jointplot(x='NA_Sales',y='Global_Sales',data=data,kind='regplot',size=5)

#corr:0.941269--> maximum correlation between these two
sns.jointplot(x='EU_Sales',y='Global_Sales',data=data,kind='regplot',size=5)

#corr:0.903264
sns.jointplot(x='JP_Sales',y='Global_Sales',data=data,kind='regplot',size=5)

#corr:0.612774
sns.jointplot(x='Other_Sales',y='Global_Sales',data=data,kind='regplot',size=5)

#corr:0.747964
#DS2,PS2 platforms are the highest

data[["Platform","Global_Sales"]].groupby('Platform').count().sort_values("Global_Sales",ascending=False).plot(kind="bar")
#as in genres PLATFORM has the maximum global sales, in this NES,GB,GEN are the high sale platforms.

platform=data[data['Genre'].str.contains("Platform")]

sns.factorplot(x="Platform",y="Global_Sales",data=platform,kind='bar',size=5,aspect=2.5,ci=None)
#in platforms,top 5 names of the video games are follows that are IN FOR producing global sales. 

top_sales_by_platform=platform.nlargest(5,["Global_Sales"])

top_sales_by_platform