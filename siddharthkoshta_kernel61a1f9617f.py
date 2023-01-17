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



import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt



#Problem Feature:(Heart Disease)

#Data Set:

#age - age in years 

#sex - (1 = male; 0 = female) 

#cp - chest pain type 

#trestbps - resting blood pressure (in mm Hg on admission to the hospital) 

#chol - serum cholestoral in mg/dl 

#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 

#restecg - resting electrocardiographic results 

#exang - exercise induced angina (1 = yes; 0 = no) 

#oldpeak - ST depression induced by exercise relative to rest 

#slope - the slope of the peak exercise ST segment 

#ca - number of major vessels (0-3) colored by flourosopy 

#thal - 3 = normal; 6 = fixed defect; 7 = reversable defect 

#target - have disease or not (1=yes, 0=no)
import pandas as pd

df1 = pd.read_csv('../input/heart-disease-uci/heart.csv')

df1
#Checking size of data set

df1.size
#checking shape of data set

df1.shape
df1.info()
df1.describe() #summary of numeric data
df1.head()   #give first 5 rows
# Bar plot of showing Gender v/s Target

sns.countplot(x='sex',data=df1,hue='target')

plt.show()
#Violin Plot of target,age and fbs

sns.violinplot(x='target',y='age',data=df1,hue='fbs')

plt.show()
#box plot of sex and age

sns.boxplot(x='sex',y='age',data=df1)

plt.show()
#stacked bar/Cross Tab

pd.crosstab(index=df1['target'],columns=df1['sex']).plot(kind='bar',stacked=True)

plt.show()
#Scatter Plot

sns.scatterplot(x='age',y='oldpeak',data=df1)

plt.show()