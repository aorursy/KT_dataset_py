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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/simple-shopping-dataset/shopping_customer_data.csv')

df
df.isnull().sum()
sns.countplot(x='Genre',data=df,palette="mako_r")

plt.xlabel("Genre(Male = 0 , Female = 1)")

plt.show()
df['Genre'] = df['Genre'].map({'Male':0 , 'Female':1})

malecount = len(df[df.Genre == 0])

femalecount = len(df[df.Genre == 1])

print("percentage of male: {:.2f}%".format(malecount/len(df.Genre)*100))

print("percentage of female: {:.2f}%".format(femalecount/len(df.Genre)*100))
pd.crosstab(df.Age,df.Genre).plot(kind="bar",figsize=(20,6))

plt.title('Sex Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Genre')

plt.show()
df=df.rename(columns={"Annual Income (k$)": "AnnualIncome", "Spending Score (1-100)": "SpendingScore"})
df.head()
x = df.drop(labels='SpendingScore',axis=1)

y = df['SpendingScore']
# Normalize

#X = (x - np.min(x)) / (np.max(x) - np.min(x)).values

from sklearn.preprocessing import StandardScaler , LabelEncoder

label = LabelEncoder()

df['SpendingScore'] = label.fit_transform(df['SpendingScore'])

df['AnnualIncome'] = label.fit_transform(df['AnnualIncome'])

df['Age'] = label.fit_transform(df['Age'])

scaler = StandardScaler()

X_scaled=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=135)

from sklearn.ensemble import GradientBoostingRegressor , RandomForestClassifier

from sklearn.metrics import accuracy_score

rf= RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_class = rf.predict(X_test)

accuracy_score(y_pred_class,y_test)