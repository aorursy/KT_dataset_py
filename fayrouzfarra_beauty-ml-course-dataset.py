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
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
my_filepath = ('../input/mlcourse/beauty.csv')

df=pd.read_csv(my_filepath)

df
df.shape
df.describe()
df.isnull().sum()
for col in df.columns:

    print(col, len(df[col].unique()))
sns.heatmap(data=df, annot=True)
plt.figure(figsize=(10,6))

sns.heatmap(data=df.corr(), annot=True)     #correlation matrix
sns.pairplot(data=df[['wage', 'female', 'educ','exper']]);
df['looks'].value_counts()
plt.figure(figsize=(10,6))

sns.countplot(x=df['looks'])            #df['looks'].value_counts().plot(kind='bar')

plt.xlabel('Looks')

plt.ylabel('Number of People')

plt.title('Distribution of Looks',fontsize = 18)
df['wage'].sort_values()
plt.figure(figsize=(10,6))

sns.distplot(df['wage'], kde=False)

plt.title('Histogram of wage')
plt.figure(figsize=(10,8))

sns.lineplot(x=df['educ'],y=df['wage'])

plt.title('Average Variation in Wage with Education Level')

plt.xlabel('Education Level')

plt.ylabel('Wage Level')
plt.figure(figsize=(10,8))

sns.barplot(x=df.educ,y=df.wage , hue=df.female);
sns.scatterplot(x = 'wage' , y = 'exper' , hue='female', data=df )

plt.xlabel('Wage'), 

plt.ylabel('Years of Expertise') 

plt.title('Level of Expertise compared to Wage with the distinction gender')

plt.legend(['Female','Male'])

plt.show()
women_wage = df[df['female']==1]['wage'].groupby(df['exper']).mean()
plt.figure(figsize = (10 , 6))

sns.lineplot(data= women_wage)

plt.xlabel('Years of Expertise'), 

plt.ylabel('Mean Wage') 

plt.title('Years of Expertise compared to Wage with the distinction gender for females')
plt.figure(figsize=(10,6))

sns.regplot(x="educ", y="exper", data=df)

plt.title('Relationship between Education level and Expertise year')
plt.figure(figsize=(10,8))

plt.plot(df.groupby('goodhlth')['wage'].mean())

plt.title('Relationship between Wage and Good health')

plt.xlabel('Good Health')

plt.ylabel('Wage Level')
df.married.value_counts()
sns.countplot(df.married ,hue=df.female);

plt.figure(figsize=(10,6))

sns.jointplot(x="looks",y="wage", data = df, kind="kde");
plt.figure(figsize=(10,6))

sns.regplot(x="looks",y="wage", data = df)