# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read Data 
df = pd.read_csv('../input/BlackFriday.csv')
# First 5 rows
df.head()
# Make columns lowercase
df.columns = [x.lower() for x in df.columns]
#Summary info of the data
df.info()
# Find which occupation spend more money
df_puramount = df[['occupation','purchase']]
df_puramount.occupation = df_puramount.occupation.astype(str)
occupation_list = list(df_puramount.occupation.unique())
purchase_sum = []

# Find total purchase for each occupations
for occu in occupation_list:
    purchase_sum.append(sum(df_puramount[df_puramount.occupation == occu].purchase))

# Create a new dataframe 
df_occupur = pd.DataFrame({'occupation':occupation_list,'total_purchase':purchase_sum})
df_occupur_sorted = df_occupur.sort_values(by=['total_purchase'],ascending = False)
# Seaborn BarPlot visulazation
plt.figure(figsize=(10,8))
sns.barplot(x = 'occupation',y = 'total_purchase', 
            order=df_occupur_sorted.occupation,
            color="salmon", saturation=.7, 
            data = df_occupur_sorted)
plt.title('Total Purchase for Each Occupation')
plt.xlabel('Occupation')
plt.ylabel('Total Purchase')
plt.show()
# Gender and occupation spend more money
df_genpuramount = df[['occupation', 'gender', 'purchase']]
df_genpuramount.occupation = df_genpuramount.occupation.astype(str)
occulist = list(df_genpuramount.occupation.unique())
male = []
female = []
for occu in occulist:
    df_genpuramount_temp = df_genpuramount[df_genpuramount.occupation == occu]
    male.append(df_genpuramount_temp[df_genpuramount_temp.gender=='M']['purchase'].sum())
    female.append(df_genpuramount_temp[df_genpuramount_temp.gender=='F']['purchase'].sum())
df_occupurgen = pd.DataFrame({'occupation':occulist,'male':male, 'female':female})
#Visualization
plt.subplots(figsize = (9,8))
sns.barplot(x = df_occupurgen.occupation, y = df_occupurgen.male, color='blue',
            alpha = 0.5, label = 'Male')
sns.barplot(x = df_occupurgen.occupation, y = df_occupurgen.female, color='red',
            alpha = 0.5, label = 'Female')
plt.title('Gender Purchase Distrubation')
plt.xlabel('Occupation')
plt.ylabel('Gender Purchase')
plt.legend()
plt.show()
df.stay_in_current_city_years.replace('4+','4',inplace=True)
df.stay_in_current_city_years = df.stay_in_current_city_years.astype(int)
df.replace(['0-17','18-25','26-35','36-45','46-50','51-55','55+'],[0,1,2,3,4,5,6],inplace=True)
agelist = list(df.age.unique())
purchasesum=[]
for age in agelist:
    purchasesum.append(df[df.age == age].purchase.sum())
fig, ax =plt.subplots(figsize=(10,6))
sns.pointplot(x=agelist,y=purchasesum,color='blue',alpha=0.8, markers='s')
ax.set_xticklabels(['0-17','18-25','26-35','36-45','46-50','51-55','55+'])
plt.title('Age vs. Purchase')
plt.xlabel('Age')
plt.ylabel('Total Purchase')
plt.grid()
plt.show()
# Percentage of Gender
fig, ax =plt.subplots(figsize=(10,6))
labels = ['Male','Female']
colors = ['red','blue']
sizes = [df[df.gender=='M'].gender.count(),df[df.gender=='F'].gender.count()]
plt.pie(sizes, shadow=True, colors=colors, autopct='%1.1f%%',textprops={'fontsize': 16})
plt.legend(labels,prop={'size': 16})
plt.show()