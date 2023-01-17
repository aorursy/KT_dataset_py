# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# การนำข้อมูลเข้า โดยใช้ไฟล์ top50.csv

filename='/kaggle/input/top50spotify2019/top50.csv'

spoti=pd.read_csv(filename,encoding='ISO-8859-1')

# การแสดงข้อมูลทั้งหมด 50 ข้อมูล ที่เป็นเพลงฮิต 50 เพลง แต่เป็นการขึ้นต้นอันดับด้วย 0 จนถึง 49

spoti.head(50) 
#จากการทำการตรวจสอบถ้าข้อมูลขึ้น false จะแสดงได้ว่าข้อมูลถูกเคลียร์เรียบร้อยแล้วสามารถนำข้อมูลมาใช้ได้ต่อ

spoti.replace([np.inf, -np.inf], np.nan)

spoti.isnull().any()
#แสดงข้อมูลของแนวเพลงที่ติดใน 50 อันดับ

print(type(spoti['Genre']))

popular_genre=spoti.groupby('Genre').size().unique

print(popular_genre)

genre_list=spoti['Genre'].values.tolist()
#แสดงค่า ความนิยมของประชากร ในแต่ละเพลง 50 เพลง

#ขนาดของกราฟ กว้าง 8 สูง 6

plt.figure(figsize=(8,6))

plt.scatter(range(spoti.shape[0]), np.sort(spoti.Popularity.values))

#แกน x= index หมายถึง ค่าดัชนีทั้งหมด

plt.xlabel('index', fontsize=12)

#แกน y  = popularity หมายถึง ความนิยมในหมู่ประชากร

plt.ylabel('Popularity', fontsize=12)

# พล้อตกราฟและแสดงออกมา

plt.show()
#การแสดงข้อมูลของศิลฟินที่มีเพลงติดใน 50 อันดับ

#โดยกราฟจะแสดงผลดังนี้

#แกน x : Artist.Name คือ ชื่อศิลปิน

#แกน y : count of songs คือ จำนวนเพลง

#กราฟขนาด 15*7

fig = plt.figure(figsize = (15,7))

spoti.groupby('Artist.Name')['Track.Name'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Artist.Name', fontsize = 20)

plt.ylabel('Count of songs', fontsize = 20)

plt.title('Artist Name vs Count of songs', fontsize = 30)
# แสดงค่าแนวเพลงที่เป็นที่นิยมของประชากรส่วนใหญ่ 

#โดยกราฟจะแสดงผลดังนี้

#แกน x : Genre Count คือ แนวเพลง

#แกน y : Popularity คือ ความนิยมของกลุ่มประชากร

# ขนาดของกราฟ กว้าง 12 ยาว 8

plt.figure(figsize=(12,8))

sns.boxplot(x="Genre", y="Popularity", data=spoti)

plt.ylabel('Popularity', fontsize=12)

plt.xlabel('Genre Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("How Popularity changes with Genre ?", fontsize=15)

plt.show()
#กราฟแสดงแนวเพลงที่ติดอันดับใน 50 เพลงมีกี่แนวได้แก่อะไรบ้างและติดมากน้อยเท่าไหร่ ซึ่งกราฟนี้เป็นสิ่งที่สำคัญสำหรับนักแต่งเพลงในการหาสถิติมากที่สุด เพื่อจับกลุ่มเป้าหมายได้อย่างถูกต้อง

sns.catplot(y = "Genre", kind = "count",

            palette = "pastel", edgecolor = ".6",

            data = spoti)
#กราฟความยาวของเพลง

plt.figure(figsize=(8,4))

sns.distplot(spoti['Length.'], kde=False, bins=15,color='m', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
#เพลงที่มีความยาวคลื่นเยอะที่สุด

maximum_Length = spoti[spoti['Length.'] == spoti['Length.'].max()]

maximum_Length[['Track.Name', 'Artist.Name', 'Genre', 'Length.']].reset_index().drop('index', axis=1)
# แกน y เป็นแนวเพลง

y = spoti['Genre']

y.head()
#แกน x ที่มีความเกี่ยวข้องกับแนวเพลงว่าจะมีอะไรเกี่ยวข้องบ้าง

X = spoti.drop(columns=['Genre','Unnamed: 0','Track.Name','Artist.Name', 'Loudness..dB..', 'Liveness' , 'Valence.' ,'Length.' , 'Popularity'])

X.head()
#ทำการ import เทรนและเทสโมเดล

from sklearn.model_selection import train_test_split
#การใส่ test = 20%  test 80 %

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# การ join ของข้อมูล

statsTrain = X_train.join(pd.DataFrame(y_train))

statsTrain.head()
#การ import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score
#กำหนดตัวแปร aa 

aa = RandomForestClassifier(n_estimators=5)

aa.fit(X_train,y_train)
#การทำนายข้อมูลที่นำมา Test

predictions = aa.predict(X_test)
#การแสดงผลเปรียบเทียบระหว่าง y_test กับ predictions

print(classification_report(y_test,predictions))