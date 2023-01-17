#importing all important packages

import numpy as np #linear algebra

import pandas as pd #data processing

import matplotlib.pyplot as plt #data visualisation

import seaborn as sns #data visualisation

%matplotlib inline
data = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')

data.head(50)
data.tail(50)

data.isna().sum()
data = data.dropna(subset=['Style'])

print(data["Style"].isna().sum())
for s in data['Stars']:

    try:

        s = float(s)

    except:

        print(s)
data = data[data['Stars'] != 'Unrated']

print(data[data['Stars'] == 'Unrated']['Stars'].sum()) #make sure if there are no 'Unrated'
data['Style'].unique()
data['Style'].value_counts()
data['Country'].value_counts()
#ราเม็งมีกี่ยี่ห้อในข้อมูลของเรา

print(len(data['Brand'].unique()))
brands = list(data['Brand'].unique())

counter = [0.0]*355



brands_cnt = dict(zip(brands, counter)) #นับคะแนนทั้งหมดและบันทึกค่า



for brand in brands:

    brands_data = data[data['Brand'] == brand]

    for star in brands_data['Stars']:

        brands_cnt[brand] += float(star) #นับคะแนนทั้งหมด

    brands_cnt[brand] /= len(brands_data) #เฉลี่ย
top50ratings = [] #รายการสำหรับบันทึกชื่อแบรนด์และการจัดอันดับโดยเฉลี่ย

for key, values in brands_cnt.items():

    top50ratings.append([key,values])



# 50 อันดับสูงสุดของแบรนด์ราเม็ง

top50ratings = sorted(top50ratings, key = lambda x : x[1], reverse = True) #เรียงลำดับค่าจากมากไปหาน้อย

top50ratings

for i in range(50):

    print('#{:<3}{:25} {}'.format(i+1, top50ratings[i][0], round(top50ratings[i][1],2)))
sns.set(style = 'darkgrid')

f, ax = plt.subplots(1,1,figsize = (15,5))

sns.countplot(x = 'Country', data = data)

plt.xticks(rotation=90)



plt.show()
labels = 'Pack', 'Bowl', 'Cup' , 'Tray', 'Box' #We can't include 'Bar' and 'Can' because they only appear once in our data.

size = [1531, 481, 450, 108, 6]



f, ax = plt.subplots(1,1, figsize= (10,10))



ax.pie(size, labels = labels, autopct = '%1.2f%%', startangle = 180)

ax.axis('equal')

ax.set_title("Style", size = 20)



plt.show()