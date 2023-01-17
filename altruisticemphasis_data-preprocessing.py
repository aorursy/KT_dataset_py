# This is my first notebook dealing with Data Preprocessing

# As you all know the most amount of time a 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Reading the Mock Data to be used for Data Preprocessing

df = pd.read_csv("/kaggle/input/mock-data/MOCK_DATA.csv")
# Displaying the Dataset

df.head
# This is a very important step when it comes to understanding the data and the amount of rows

# and columns. It also gives you an overview of the probable number of missing data in the dataset

df.info()
# Now from the dataset its clear we have quite a number of data missing in our data set 

# So for that lets see the amount of missing data in out dataset

df.isna()

# We see quite a lot of True values. Let's find out the number of missing values.
# Let's check for each column for missing values

len(df.columns)

# Since the above gives us the number of columns , we shall run the code spanning all rows and all columns

# So here we run the commmand to get the total sum of missing values in the dataframe.

df_copy= df.copy()

df_copy.isnull().sum()

total_cells = np.product(df_copy.shape)

total_missing = df_copy.isnull().sum()



# Finding out percentage of missing data

percent_missing = (total_missing/total_cells) * 100

print(percent_missing)
df_copy
# Now the issue that we are gonna face is here couple of string columns are missing values 

# So its not possible to replace it with any value we want 

# Moreover we see that the amount of missing values is quite less compared to our total number of samples

# So one thing that's generally done in these cases is drop the complete rows containing NaN values.

# lets drop the NaN value rows

df_copy.dropna(inplace=True)

df_copy.isnull().sum()
df_copy
# Let's check how much info we lost

df_copy.info()

# Now as we see that around 50% of our data is lost to NaN values. Lets check if something can be done to increase the 

# amount of data after cleaning
df_copy= df.copy()

# As we know in general gender is binary. So that can be used as a label encoding to see if encoding it actually helps 

# preserving our data.

# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

df_copy['gender'].unique() 

# Now as we see the gender column bein categorical actually can be label encoded



# I am intentionally increasing the steps , or else one more way is to actually check for the catgorical variable and 

# right away label encoding it without doing these steps

df_copy['gender'] = df_copy['gender'].fillna("missing")

df_copy['gender']= label_encoder.fit_transform(df_copy['gender']) 

df_copy['gender'].unique() 

df_copy.info
#You can use it to fill missing values for each column (using its own most frequent value) like this

df_copy = df_copy.fillna(df.mode().iloc[0])

df_copy.info
df_copy.dtypes
df_copy.isnull().sum()
# After doing this gives us a dataset clean of any NaN values but at the same time preserving the data.

# This is my first attempt at doing this preprocessing where I am trying imputation to actually save the data. The 

# data set is public. Go ahead and use it in your projects











# Any sort of comments are deeply appreciated.