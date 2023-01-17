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
#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#csv formatında veriyi okuyalım ve head ile ilk bakış.
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
#veri setinin sizeını verir
print(df.shape)
df.info()
#Tamamlayıcı istatistikler
df.describe()
#kayıp değerimiz (missing value) var mı bunu öğreniriz.
df.isnull().sum()
# total score sütununu oluşturduk.
df["Total Score"] = df['math score']+df['reading score']+df['writing score']
df.head()
#Korelasyonu verir.
df.corr()
#Korelasyon değerleri için ısı haritası (heatmap)
f,ax=plt.subplots(figsize=(6,6))
sns.heatmap(df.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)
plt.show()
(df.head(20)
 .style
 .hide_index()
 .bar(color='#FF7F61', vmin=0, subset=['math score'])
 .bar(color='#B2C248', vmin=0, subset=['reading score'])
 .bar(color='#FEDC56', vmin=0, subset=['writing score'])
 .bar(color='#82CAFA', vmin=0, subset=['Total Score'])
 .set_caption(''))
#gender'a göre writing score değerleri (ilk 5)
print(df.loc[:,['gender','writing score']].head())
# parental level of education'a göre writing score değerleri (ilk 5)
print(df.loc[:,['parental level of education','Total Score']].head())
#Total score için mean değeri.
df["Total Score"].mean()
plt.hist(df["math score"], color = "skyblue")
plt.xlabel('score')
plt.ylabel('frequency')
plt.title('Math score frequencies')
plt.show
plt.hist(df["reading score"], color = "skyblue")
plt.xlabel('score')
plt.ylabel('frequency')
plt.title('Reading score frequencies')
plt.show
plt.hist(df["writing score"], color = "skyblue")
plt.xlabel('score')
plt.ylabel('frequency')
plt.title('Writing score frequencies')
plt.show
# Eğitim seviyeleri için countplot.
fig,ax=plt.subplots()
sns.countplot(x='parental level of education',data=df,palette='spring')
plt.tight_layout()
fig.autofmt_xdate()
#Eğitim seviyelerine göre matematik puanları.
fig,ax=plt.subplots()
sns.barplot(x=df['parental level of education'],y='math score', data=df,palette='spring')
fig.autofmt_xdate()
# Hazırlık kursu alma durumuna göre math, reading, writing ve total skorlarının karşılaştırılması.
plt.figure(figsize=(15,6))
plt.subplot(1, 4, 1)
sns.barplot(x='test preparation course',y='math score',data=df,hue='lunch',palette='Paired')
plt.title('Math Score')
plt.subplot(1, 4, 2)
sns.barplot(x='test preparation course',y='reading score',data=df,hue='lunch',palette='Paired')
plt.title('Reading Scores')
plt.subplot(1, 4, 3)
sns.barplot(x='test preparation course',y='writing score',data=df,hue='lunch',palette='Paired')
plt.title('Writing Scores')
plt.subplot(1, 4,4 )
sns.barplot(x='test preparation course',y='Total Score',data=df,hue='lunch',palette='Paired')
plt.title('Total Score')
plt.show()