import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv("../input/annotated_video_review.csv")
plt.style.use('default')
df1 = data.groupby(['Primary_Impact_Type'])['Primary_Impact_Type'].count().plot(kind='bar', color = ['#ff4136','#FFDC00','#01FF70','#39cccC','#dddddd', '#AAAAAA'], figsize=(8,6))
df1.set_ylabel('No. Concussions', fontsize=12)
df1.set_title("Type of Impact that Caused Injury", fontsize=20)
for item in df1.get_xticklabels():
    item.set_rotation(0)

data.head() 
#change the above line to data.head(38) to see all values

df2 = data.groupby(['Player_Activity_Derived', 'Head_Position'])['Player_Activity_Derived'].count().unstack('Head_Position')

plt.style.use('default')
nicechart = df2[['Tackle-Side','Front On','Away-Side','Behind','Unclear','Not Applicable']].plot(kind='bar', color = ['#ff4136','#FFDC00','#01FF70','#39cccC','#dddddd', '#AAAAAA'], stacked=True, figsize=(6,8))
nicechart.set_ylabel('No. Concussions', fontsize=12)
nicechart.set_title("Tackler/Blocker Head Position in Contact", fontsize=20)
for item in nicechart.get_xticklabels():
    item.set_rotation(0)


df3 = data.groupby(['Body_Height_to_Partner_Shoulder', 'Head_Position'])['Body_Height_to_Partner_Shoulder'].count().unstack('Body_Height_to_Partner_Shoulder')
newchart = df3[['High','Low']].plot(kind='bar', color = ['#ff4136','#39cccC'], stacked=True, figsize=(6,8))
newchart.set_ylabel('No. Concussions', fontsize=12)
newchart.set_title("Tackler/Blocker Head Position and Body Height", fontsize=20)
