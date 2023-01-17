# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd #pandas library for data manipulation

import matplotlib.pyplot as plt #library to plot graphs

import seaborn as sns #seaborn library for visualization and EDA



#display graphs within the jupyter notebook

%matplotlib inline 
# Read the csv file and store it in a pandas data frame

credit_raw_df = pd.read_csv ("../input/creditcard.csv")
# Let us use dataframe.describe()

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.

credit_raw_df.describe()
# Total number of columns? What is the data type of each column?

# credit_raw_df.columns -- This will just print the column names

credit_raw_df.info()
# Are there any missing values?

credit_raw_df.isnull().any()
#Let us visualize the "Class" to know its distribution



plt.figure(figsize=(6,4))

sns.countplot(x='Class',hue='Class',  data=credit_raw_df)



plt.xlabel('Class')

plt.ylabel('Total Records')

plt.title('Total Fraud VS Non-Fraud')

plt.show()
# Calculate correlation matrix and use sns to plot the heatmap

# https://seaborn.pydata.org/examples/many_pairwise_correlations.html



# Drop the "Class" column

credit_df = credit_raw_df.drop(columns= ['Class'])



# Calculate correlation matrix and use sns to plot the heatmap

corr = credit_df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Basic RandomForest to start an iterative analysis approach

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

# from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(credit_df,credit_raw_df['Class'],test_size=0.33, random_state=42)



model = RandomForestClassifier(n_estimators=10)

model.fit(X_train,y_train)



y_pred= model.predict(X_test)

accuracy_score(y_test, y_pred)

# print(mean_absolute_error(y_test, predicted_class))
