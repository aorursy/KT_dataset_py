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
# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
credit_data = pd.read_csv("../input/creditcard.csv")
credit_data.head()
print("Number of rows: " + str(credit_data.shape[0]))
print("Number of columns: " + str(credit_data.shape[1]))
# Listing the column name
print("Columns: " + ", ".join(credit_data.columns))
print(credit_data.info())
credit_data['Class'] = credit_data['Class'].astype('category')
print(credit_data[['Time', 'Amount', 'Class']].describe())
credit_data['Hour_of_transaction'] = credit_data['Time'].apply(lambda x: (x%(24*60*60))/(60*60))
credit_data[['Hour_of_transaction']].describe()
for idx, feature in enumerate(['Time', 'Amount', 'Hour_of_transaction']):
    fg = sns.FacetGrid(credit_data, hue='Class', size=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()
sns.lmplot(x='Amount', y='Hour_of_transaction', hue='Class', truncate=True, data=credit_data)
plt.show()
credit_data.groupby('Class')['Amount'].describe()
# Function to find the similarity between two trasactions 
#i.e. the angle between the vectors representing the transactions

def length(v1):
    return np.sqrt(np.dot(v1, v1))

def angle_between(v1, v2):
    return np.arccos(np.dot(v1, v2)/(length(v1)*length(v2)))
# Testing
angle_between([1,2,3], [2,3,4])
# Remove Time and Hour of transaction
col_list = list(credit_data.columns)[1:-1] 
# Add Hour of transaction in the beginning
col_list.insert(0, 'Hour_of_transaction')
# Rearranging data columns
credit_data = credit_data[col_list]
credit_data.columns
# Subsetting data with maintainging the same class distribution
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(credit_data.iloc[:,:-1], credit_data['Class'], \
                                                    test_size=0.03, random_state=0, stratify=credit_data['Class'])
# Verifying the test class distribution
from collections import Counter
y_cnt = dict(Counter(y_test))
print(y_cnt)

tot = sum(y_cnt.values())
for k, v in y_cnt.items():
    y_cnt[k] = v/float(tot)
print(y_cnt)
X_test = X_test.reset_index(drop=True)
X_test.head()
print(type(y_test))
y_test = y_test.reset_index(drop=True)
y_test.head()
# Find the IDs of transactions that are classfied as fraud
y_test[y_test == 1]
# Total number of samples
len(y_test)
#t_id = int(input("Enter any id from 0 to " + str(len(y_test)-1) + " : "))
t_id = 10
similarities = []
input_vector = X_test.iloc[t_id, :].values
#print(input_vector)
for idx, row in X_test.iterrows():
    #print(dict(row))
    if idx == t_id:
        similarities.append(-1)
    else:
        similarity = angle_between(input_vector, list(dict(row).values())) 
        similarities.append(similarity)
X_test['Similarity'] = similarities
X_test.sort_values('Similarity').head(10)
def find_similar_transactions(t_id):
    #t_id = int(input("Enter any id from 0 to " + str(len(y_test)-1) + " : "))
    print("Class: {}\n".format(y_test[t_id]))

    similarities = []
    input_vector = X_test.iloc[t_id, :].values
    #print(input_vector)
    for idx, row in X_test.iterrows():
        #print(dict(row))
        if idx == t_id:
            similarities.append(-1)
        else:
            similarity = angle_between(input_vector, list(dict(row).values())) 
            similarities.append(similarity)

    X_test['Similarity'] = similarities

    t_ids_similar = list(X_test.sort_values('Similarity').index)[1:11]
    angles_similar = list(X_test.sort_values('Similarity')['Similarity'])[1:11]
    classes = y_test[t_ids_similar]

    for x,y,z in zip(t_ids_similar, classes, angles_similar):
        print("ID: {},  Class: {},  Angle Between: {}\n".format(x,y,z)) 
# set any t_id from 0 to 8544
find_similar_transactions(101)
find_similar_transactions(276)