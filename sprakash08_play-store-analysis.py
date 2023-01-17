# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Read the csv of playstore
playstore = pd.read_csv("../input/googleplaystore.csv")
playstore.head(10)
#Drop Duplicates
ps = playstore.drop_duplicates(subset='App')
#Drop Nulls
ss = ps.dropna()
plt.figure(figsize=(20,9))
plt.tight_layout()
#Which category apps are being developed
count_cat = sns.countplot(x='Category',data=ss)

#Rotation for labels
for item in count_cat.get_xticklabels():
    item.set_rotation(90)
#How many Free and paid apps are there of the dataset
sns.countplot(x='Type',data=ss)
ss['Installs'] = ss['Installs'].apply(lambda x: x.strip("+"))
#Top ten most installed and rating greater than 4.5 Apps
byInstalls = ss.sort_values(ascending=False,by='Installs').head(10)
byInstalls[byInstalls['Rating'] > 4.5]
#It is clear that there are no paid apps in the top category
#Most installed Games
byInstalls = ss.sort_values(ascending=False,by='Installs')
#byInstalls[byInstalls['Rating'] > 4.5]
#It is clear that there are no paid apps in the top category
topGames = byInstalls[(byInstalls['Category'] == "GAME") & (byInstalls['Type'] == "Free")].head(5)

#Free
sns.barplot(x='Rating',y='App',data=topGames,palette='viridis')
#paid
topGames = byInstalls[(byInstalls['Category'] == "GAME") & (byInstalls['Type'] == "Paid")].head(5)
sns.barplot(x='Rating',y='App',data=topGames,palette='viridis')
#How is the rating distributed
sns.distplot(ss['Rating'],bins=70,kde=False,color='r')

# See the distribution we can tell that most apps are rated between 4 to 4.5
