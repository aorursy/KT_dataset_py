# This is an amateur's attempt to predict ad-click event based on Time spent on site, Age and Area income



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ad_df=pd.read_csv('../input/advertising.csv')
ad_df.head()
sns.heatmap(ad_df.isnull(), yticklabels=False, cbar=False, cmap='viridis') # missing data heatmap
sns.set_style('whitegrid')
sns.jointplot(x='Daily Time Spent on Site',y='Age',data=ad_df)
sns.countplot(x='Clicked on Ad', hue='Male', data=ad_df) # Clicked event based on Male/female
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_df,color='red',kind='kde');
ad_df['Age'].hist(bins=40, figsize=(10,4))
plt.figure(figsize=(10,7))

sns.boxplot(x='Country', y='Age', data=ad_df) # country wise data range by 
X=ad_df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]

y=ad_df['Clicked on Ad']
from sklearn.model_selection import train_test_split #Random split dataset in to train and test 
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=101) # Random split X & y datasets
X_test.head()
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
#onehot_encoded_X_train=pd.get_dummies(X_train) categorical value dummies was found unnecessary

#onehot_encoded_y_train=pd.get_dummies(y_train)
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))