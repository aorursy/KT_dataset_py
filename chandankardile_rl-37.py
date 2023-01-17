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
import numpy as np

import pandas as pd

import os

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

Glass_data = pd.read_csv(('/kaggle/input/glass/glass.csv'))

Glass_data.head(5)
Glass_data.dtypes
Glass_data['Type'].unique()
sns.boxplot('Type', 'RI', data =Glass_data)
Glass_data.describe()
# glass 1, 2, 3 are Bad glass

# glass 5, 6, 7 are Good glass

Glass_data['Type of glass'] = Glass_data.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

Glass_data.head()
#We will check correlation of values using Feature Matrix

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']



mask = np.zeros_like(Glass_data[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Correlation Matrix',fontsize=25)

sns.heatmap(Glass_data[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='b',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
from sklearn.model_selection import train_test_split



#Independent variable

X = Glass_data[['Al','Na']]

#Dependent variable

y = Glass_data['Type of glass']

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)
from sklearn.linear_model import LogisticRegression



# Create instance (i.e. object) of LogisticRegression

model = LogisticRegression(class_weight='balanced')

output=model.fit(X_train, y_train)

output
pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:



    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)







Ypredict = pickle_model.predict(X_test)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

y_predict