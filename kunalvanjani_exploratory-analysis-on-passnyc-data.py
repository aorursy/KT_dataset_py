import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
filename = "../input/2016 School Explorer.csv"
data1 = pd.read_csv(filename)

df = pd.DataFrame(data1, columns=["Economic Need Index", "School Income Estimate"])
plt.figure(figsize=(10,10))
df = df.dropna()
df['School Income Estimate'] = df['School Income Estimate'].replace('[\$,]', '', regex=True).astype(float)
income_estimate= df['School Income Estimate']

max_val = max(income_estimate)

plt.yticks(np.arange(0, max_val, 3000))
plt.ylabel("Student Income Estimate")
plt.xlabel("Economic need index")

plt.scatter(df['Economic Need Index'],df['School Income Estimate'])
plt.show()
plt.figure(figsize=(10,10))
sns.countplot(y="City", data=data1)
plt.figure(figsize=(10,10))
grade_scale = data1[['SED Code', 'Grade Low', 'Grade High']].pivot_table(values= 'SED Code',
                            index = 'Grade Low', columns = 'Grade High' , aggfunc = np.size
                                                                         ,fill_value = 0)
sns.heatmap(grade_scale, annot = True ,fmt ='d' )
sns.kdeplot(data1['Average ELA Proficiency'] , shade = True ,color ='g')
plt.subplot(111)

sns.kdeplot(data1['Average Math Proficiency'] , shade = True ,color ='b')
plt.subplot(111)
plt.figure(figsize=(7,7))
df1 = pd.DataFrame(data1, columns=["Collaborative Teachers %", "Supportive Environment %","Trust %"])
df1 = df1.dropna()
df1
df1['Collaborative Teachers %']=df1['Collaborative Teachers %'].str.rstrip('%').astype(float)
df1['Trust %']=df1['Trust %'].str.rstrip('%').astype(float)
df1['Supportive Environment %']=df1['Supportive Environment %'].str.rstrip('%').astype(float)
sns.pairplot(df1)
plt.figure(figsize=(8,8))
plt.subplot(211)
sns.countplot(x= "Student Achievement Rating" , data =data1 ,palette= "Blues")
plt.subplot(212)
sns.countplot(x= "Trust Rating" , data =data1 ,palette= "Blues")
df2 = pd.DataFrame(data1, columns=["Percent Black","Percent of Students Chronically Absent","Percent Asian","Percent Hispanic","Percent White","Percent ELL"])
df2 = df2.dropna()
df2['Percent Black']=df2['Percent Black'].str.rstrip('%').astype(float)
df2['Percent Hispanic']=df2['Percent Hispanic'].str.rstrip('%').astype(float)
df2['Percent ELL']=df2['Percent ELL'].str.rstrip('%').astype(float)
df2['Percent White']=df2['Percent White'].str.rstrip('%').astype(float)
df2['Percent Asian']=df2['Percent Asian'].str.rstrip('%').astype(float)
df2['Percent of Students Chronically Absent']=df2['Percent of Students Chronically Absent'].str.rstrip('%').astype(float)

sns.jointplot(y="Percent of Students Chronically Absent", x="Percent Black", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent Asian", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent White", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent ELL", data=df2)
sns.jointplot(y="Percent of Students Chronically Absent", x="Percent Hispanic", data=df2)