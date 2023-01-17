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
pd.options.mode.chained_assignment = None
#Load the matches that have already occurred 

df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/ufc-master.csv")



#Load the upcoming matches

df_upcoming = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/upcoming-event.csv")



#Get the number of upcoming fights

num_upcoming_fights = len(df_upcoming)

print(f"We are going to predict the winner of {num_upcoming_fights} fights.")
#Combine the upcoming fights to the previous fights so we can clean it all at the same time.

df_combined = df_upcoming.append(df)



#Let's put all the labels into a dataframe

df_combined['label'] = ''



#We need to convert 'Red' and 'Blue' to 0 and 1

mask = df_combined['Winner'] == 'Red'

df_combined['label'][mask] = 0

mask = df_combined['Winner'] == 'Blue'

df_combined['label'][mask] = 1



#Make sure label is numeric

df_combined['label'] = pd.to_numeric(df_combined['label'], errors='coerce')



#Make sure the date column is datetime

df_combined['date'] = pd.to_datetime(df['date'])
#Copy the labels to their own dataframe

label_df = df_combined['label']



#Split the train set from the test set



df_train = df_combined[num_upcoming_fights:]

label_train = label_df[num_upcoming_fights:]



df_test = df_combined[:num_upcoming_fights]

label_test = label_df[:num_upcoming_fights]





#Make sure the sizes are the same

print(len(df_test))

print(len(label_test))



print(len(df_train))

print(len(label_train))
from sklearn.tree import DecisionTreeClassifier

#Pick a model

my_model = DecisionTreeClassifier(max_depth=5)



#Pick some features

#I would not recommend placing bets based off of these features...

my_features = ['R_odds', 'B_Stance']



#Let's grab the names of the fighters for the upcoming event

#This will be useful to print predictions at the end.

fighters_test = df_test[['R_fighter', 'B_fighter']]





#Make dataframes that only contain the relevant features

df_train_prepped = df_train[my_features].copy()

df_test_prepped = df_test[my_features].copy()



#If we need to dummify the datasets do it now.  We need to be careful that the test set has all of the features

#that the training set does



df_train_prepped = pd.get_dummies(df_train_prepped)

df_test_prepped = pd.get_dummies(df_test_prepped)



#Ensure both sets are dummified the same

df_train_prepped, df_test_prepped = df_train_prepped.align(df_test_prepped, join='left', axis=1)    



#The new test set may have new new features after the above join.  Fill them with zeroes

df_test_prepped = df_test_prepped.fillna(0)



#Since we may have dropped some rows we need to drop the matching rows in the labels

label_train_prepped = label_train[label_train.index.isin(df_train_prepped.index)]

label_test_prepped = label_test[label_test.index.isin(df_test_prepped.index)]

fighters_test_prepped = fighters_test[fighters_test.index.isin(df_test_prepped.index)]





#Quick test that lengths match.

print(len(label_train_prepped))

print(len(df_train_prepped))

print(len(label_test_prepped))

print(len(df_test_prepped))

print(len(fighters_test_prepped))
#Train the model



my_model.fit(df_train_prepped, label_train_prepped)
#Get predictions and print them

predictions = my_model.predict(df_test_prepped)



#Zip the fighters to predictions.  This will help with printing

fighters_array = fighters_test_prepped.to_numpy()

prediction_list = np.array(list(zip(fighters_array, predictions)))

for p in prediction_list:

    if p[1] == 0.0:

        print(f"RED fighter {p[0][0]} is predicted to win over {p[0][1]}")

    else:

        print(f"BLUE fighter {p[0][1]} is predicted to win over {p[0][0]}")