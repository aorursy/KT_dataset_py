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

%matplotlib inline

import seaborn as sb

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/rice-dataset-gonenjasmine/Rice-Gonen andJasmine.csv')

df.head()
#Lets discover all the missing values in our dataset



df.isnull().sum()
# Lets discover all the data types of the dataset's columns

df.dtypes
df.describe()
Result1 = df.groupby('Class').count()

Result1
plt.bar(Result1.index, Result1.id)

plt.title('Classification by RICE Types (Most Rice)')

plt.ylabel('Counts')

plt.xlabel('Rice Class')

plt.show()
Result2 = df.groupby('Class').sum()

Result2
plt.bar(Result2.index, Result2.Area)

plt.title('Classification by RICE Types and Area Covered')

plt.ylabel('Area')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.Area, bins=30)

plt.title('Area distribution')

plt.xlabel('Area')

plt.show()
plt.bar(Result2.index, Result2.Eccentricity)

plt.title('Classification by RICE Types and Eccentricity')

plt.ylabel('Eccentricity')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.Eccentricity, bins=30, range=(0.71, 0.98))

plt.title('Eccentricity Distribution')

plt.xlabel('Eccentricity')

plt.show()
plt.bar(Result2.index, Result2.MajorAxisLength)

plt.title('Classification by RICE Types and MajorAxis Length')

plt.ylabel('Major Axis Length')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.MajorAxisLength, bins=30)

plt.title('Distribution by MajorAxisLength')

plt.xlabel('MajorAxisLength')

plt.show()
plt.bar(Result2.index, Result2.MinorAxisLength)

plt.title('Classification by RICE Types and MinorAxis Length')

plt.ylabel('Minor Axis Length')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.MinorAxisLength, bins=30)

plt.title('Distribution by MinorAxisLength')

plt.xlabel('MinorAxisLength')

plt.show()
Result2
plt.bar(Result2.index, Result2.ConvexArea)

plt.title('Classification by RICE Types and ConvexArea')

plt.ylabel('ConvexArea')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.ConvexArea, bins=30)

plt.title('Distribution by ConvexArea')

plt.xlabel('ConvexArea')

plt.show()
plt.bar(Result2.index, Result2.Roundness)

plt.title('Classification by RICE Types and Roundness')

plt.ylabel('Roundness')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.Roundness, bins=30)

plt.title('Distribution by Roundness')

plt.xlabel('Roundness')

plt.show()
plt.bar(Result2.index, Result2.AspectRation)

plt.title('Classification by RICE Types and AspectRation')

plt.ylabel('AspectRation')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.AspectRation, bins=30)

plt.title('Distribution by AspectRation')

plt.xlabel('AspectRation')

plt.show()
plt.bar(Result2.index, Result2.EquivDiameter)

plt.title('Classification by RICE Types and EquivDiameter')

plt.ylabel('EquivDiameter')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.EquivDiameter, bins=30)

plt.title('Distribution by EquivDiameter')

plt.xlabel('EquivDiameter')

plt.show()
plt.bar(Result2.index, Result2.Extent)

plt.title('Classification by RICE Types and Extent')

plt.ylabel('Extent')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.Extent, bins=30)

plt.title('Distribution by Extent')

plt.xlabel('Extent')

plt.show()
plt.bar(Result2.index, Result2.Perimeter)

plt.title('Classification by RICE Types and Perimeter')

plt.ylabel('Perimeter')

plt.xlabel('Rice Class')

plt.show()
plt.hist(df.Perimeter, bins=30)

plt.title('Distribution by Perimeter')

plt.xlabel('Perimeter')

plt.show()