# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
#Coverting the gender column into 0,1
df["gender"] = df["gender"].apply(lambda x: 0  if x=="female" else 1)
df.rename(columns= {"parental level of education":"parental_level_of_education",
                   "test preparation course":"test_preparation_course",
                   "math score":"math_score",
                   "reading score":"reading_score",
                   "writing score":"writing_score"},inplace=True)
print("shape is: \n", df.shape)
print("datatypes:\n",df.dtypes)
print("describe:\n",df.describe())
df.isnull().sum()
df.head()
passmark = 40 
df["math_PC"] = np.where(df["math_score"] < passmark,"F","P")
df["reading_PC"] = np.where(df["reading_score"] < passmark,"F","P")
df["writing_PC"] = np.where(df["writing_score"] < passmark,"F","P")
print("Pass & Fail count from maths \n", df.math_PC.value_counts())
print("Pass & Fail count from maths \n", df.reading_PC.value_counts())
print("Pass & Fail count from maths \n", df.writing_PC.value_counts())
sns.countplot(x="parental_level_of_education", data=df, hue="math_PC", palette="bright")
x= plt.gca().xaxis
for i in x.get_ticklabels():
    i.set_rotation(25)
plt.subplots_adjust(bottom=0.33)
plt.title("Parental_Education Vs Maths PassCount")
plt.show()

sns.countplot(x="parental_level_of_education", data=df, hue="reading_PC", palette="bright")
x= plt.gca().xaxis
for i in x.get_ticklabels():
    i.set_rotation(25)
plt.subplots_adjust(bottom=0.33)
plt.title("Parental_Education Vs writing PassCount")
plt.show()

sns.countplot(x="parental_level_of_education", data=df, hue="writing_PC", palette="bright")
x= plt.gca().xaxis
for i in x.get_ticklabels():
    i.set_rotation(25)
plt.subplots_adjust(bottom=0.33)
plt.title("Parental_Education Vs Reading PassCount")
plt.show()
sns.countplot(x="test_preparation_course", data=df, hue="math_PC", palette="bright")
plt.title("test_preparation_course Vs Maths PassCount")
plt.show()
sns.countplot(x="test_preparation_course", data=df, hue="writing_PC", palette="bright")
plt.title("test_preparation_course Vs Writing PassCount")
plt.show()
sns.countplot(x="test_preparation_course", data=df, hue="reading_PC", palette="bright")
plt.title("test_preparation_course Vs Reading PassCount")
plt.show()
df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['math_PC'] == 'F' or 
                                    x['reading_PC'] == 'F' or x['writing_PC'] == 'F' else 'P', axis =1)

df.OverAll_PassStatus.value_counts()
df['Total_Marks'] = df['math_score']+df['reading_score']+df['writing_score']
df['Percentage'] = df['Total_Marks']/3
def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

df.Grade.value_counts()
sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'],  palette="muted")
plt.show()
