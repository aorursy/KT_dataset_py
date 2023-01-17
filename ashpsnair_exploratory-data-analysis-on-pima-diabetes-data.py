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
data=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

data.head()
#Finding general information about the dataset

print("Data shape="+ str(data.shape))

data.info()
#Look for any discrepancies

print("Null Values for this dataset")

print(data.isnull().sum())

# There's no data is missingprint("Number of duplicated data= "+ str(data.duplicated().sum()))
# There's no data is missing

#Description

data.describe()


remove=["Glucose","BMI","SkinThickness","BloodPressure","Age","Insulin"]



#creating a duplicate of data

new_data=data

for i in remove:

  new_data=new_data[new_data[i]!=0]
#Description after data cleaning

print("No. of rows removed = "+ str(len(data)-len(new_data)))
#It is not advised to remove such large amount of data as it might result in misleading insights

#Therefore, it is suggested to replace 0 with NaN

modifs= ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age'] #variables to be modified

data[modifs] = data[modifs].replace(0,np.NaN)

#Looking at the null values now
missing_values = (data.isnull().sum() / len(data) * 100).round(2)

print(missing_values)
data.hist(bins=25, figsize=(20, 15));
for column in data:

    plt.figure()

    data.boxplot([column])
#Bivariate Analysis 

#Visualization

sns.pairplot(data=data)

corr=data.corr(method="spearman")

corr

#Some of the pairs who show some relationship are

factors=["Age","Pregnancies","BMI","SkinThickness","Glucose","Insulin","Outcome"]



# Creating a matrix and plotting the correlation matrix

data[factors].corr()
# Plotting the correlation matrix

sns.heatmap(data[factors].corr(), annot=True, cmap = 'Reds')

plt.show()
sns.pairplot(data=data, vars=factors, hue="Outcome")