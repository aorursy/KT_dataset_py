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
data=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
#ageage in years
#sex(1 = male; 0 = female)
#cpchest pain type
#trestbpsresting blood pressure (in mm Hg on admission to the hospital)
#cholserum cholestoral in mg/dl
#fbs(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#restecgresting electrocardiographic results
#thalachmaximum heart rate achieved
#exangexercise induced angina (1 = yes; 0 = no)
#oldpeakST depression induced by exercise relative to rest
#slopethe slope of the peak exercise ST segment
#canumber of major vessels (0-3) colored by flourosopy
#thal3 = normal; 6 = fixed defect; 7 = reversable defect
#target1 or 0
data.info()
data.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
print(data['sex'].value_counts())
plt.bar(['male','Female'],data['sex'].value_counts())
plt.hist(data['age'])
plt.figure(figsize=(10,5))
plt.scatter(data['age'],data['target'])
plt.xlabel('Age')
plt.ylabel('target')
plt.figure(figsize=(10,5))
plt.subplot(121)
sns.distplot(data["age"], kde=False, bins=20)
plt.subplot(122)
sns.distplot(data["age"], hist=False, bins=20)
sns.jointplot(data['age'],data['target'],kind='kde',color='r')
labels = 'type 0', 'type 1', 'type 2', 'type 3'
k=data['cp'].value_counts()
sizes = [(k[0]/303)*100,(k[1]/303)*100,(k[2]/303)*100,(k[3]/303)*100]
explode = (0, 0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.stripplot(x="cp", y="age", data=data,jitter=True,hue='target',palette='Set1')
plt.subplot(222)
sns.swarmplot(x="cp", y="age",hue='target',data=data, palette="Set2", split=True)
plt.subplot(223)
sns.swarmplot(x="cp", y="age",hue='sex',data=data, palette="Set1", split=True)
plt.subplot(224)
sns.swarmplot(x="target", y="age",hue='sex',data=data, palette="Set2", split=True)
plt.figure(figsize=(9,9))
sns.violinplot(x="target", y="age", data=data,hue='cp',palette='Set1')
sns.swarmplot(x='cp',y='thalach',hue='target',data=data,palette='Set1',split=True)
sns.jointplot(data['thalach'],data['target'],kind='hex',color='g')
sns.swarmplot(x='restecg',y='oldpeak',hue='target',data=data,palette='Set1',split=True)
