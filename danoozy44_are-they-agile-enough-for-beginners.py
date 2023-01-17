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
df = pd.read_csv("../input/fifa19/data.csv")
df
#Remove all columns with strings



df = df.select_dtypes(exclude = ['object'])
df
df.isnull().sum()
#Replacing null values with means of respective columns



for colname in df.columns:

    col_mean = df[colname].mean()

    df[colname] = df[colname].fillna(col_mean)
#Now there are no null values!



df.isnull().sum()
#Let's check what columns we have.



df.columns
#Checks if Agility is over 70. If so, then put 1, else put 0. A new column called isAgile is created in the process.



df['isAgile'] = (df['Agility'] >= 70).astype(int)
#Colourful plot!



import seaborn as sns

sns.catplot(x="Overall", y="Agility", kind="swarm", data=df)
#A new column came up on the extreme right! 



df
#Masking random rows, to select train and test



msk = np.random.rand(len(df)) < 0.8 
train = df[msk]

test = df[~msk]
X = train.drop(columns = ['isAgile']).values

y = train['isAgile'].values
print(X.shape)

print(y.shape)
from sklearn.linear_model import LogisticRegression

regr = LogisticRegression()

regr.fit(X, y)
from sklearn import metrics

pred = regr.predict(X)

metrics.accuracy_score(pred, y)
test_pred = test.drop(columns = ['isAgile'])

prediction = regr.predict(test_pred)
correct_output = pd.DataFrame({"ID:":test.ID, "isAgile":test.isAgile})

predicted_output = pd.DataFrame({"ID":test.ID, "isAgile":prediction})
correct_predictions = 0



for i in range(len(correct_output)):

    if(correct_output['isAgile'].iloc[i] == predicted_output['isAgile'].iloc[i]):

        correct_predictions += 1
correct_percent = (correct_predictions/len(correct_output))*100

correct_percent