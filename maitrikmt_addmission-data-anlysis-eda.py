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
import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt
data=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv",index_col='Serial No.')
data.head()
data.info()
data.describe()
data.rename(columns={"Chance of Admit ":"Chance of Admit","LOR ":"LOR"},inplace=True)
data.isnull().sum()
data.shape
print("Average of GRE Score is: ",data["GRE Score"].mean())

print("Average of TOEFL Score is: ",data["TOEFL Score"].mean())

print("Average of University Rating is: ",data["University Rating"].mean())

print("Average of SOP is: ",data["SOP"].mean())

print("Average of LOR is: ",data["LOR"].mean())

print("Average of Research is: ",data["Research"].mean())

print("Average of Chance of Admit Score is: ",data["Chance of Admit"].mean())
plt.figure(figsize=(20,10))

sns.distplot(data["GRE Score"],kde=False,bins=30)

plt.show()
plt.figure(figsize=(20,10))

sns.distplot(data["TOEFL Score"],kde=False,bins=30)



plt.show()
plt.figure(figsize=(20,10))

sns.distplot(data["Chance of Admit"],kde=False,bins=30)

plt.show()
plt.figure(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,)

plt.show()
plt.figure(figsize=(10,10))

sns.countplot(data["Research"])

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(fontsize=19,xlabel="LOR")

plt.ylabel(fontsize=19,ylabel="count")

sns.countplot(data["LOR"])

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.show()
plt.figure(figsize=(10,10))

plt.xlabel(fontsize=19,xlabel="SOP")

plt.ylabel(fontsize=19,ylabel="count")

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

sns.countplot(data["SOP"])

plt.show()
sns.pairplot(data)
fig=px.density_contour(data,x="CGPA",y="Chance of Admit")

fig.show()
plt.figure(figsize=(16,10))



sns.jointplot(data=data, x='TOEFL Score',y="Chance of Admit",color="Indigo", marker="+",height=6)





plt.show()
fig=px.scatter(data, x='GRE Score',y='Chance of Admit',marginal_y='box',marginal_x='histogram')

fig.show()