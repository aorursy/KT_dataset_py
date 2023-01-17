import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
d = {'color': ['Green','Yellow','Red','Red','Yellow'], 'diameter': [3,3,1,1,3],'label':['Apple','Apple','Grape','Grape','Lemon']}

data=pd.DataFrame(d)

data.head()
# to find the gini impurity value for a given dataset (dataframe)

# function below takes a dataframe and the feature on which it will calculate the homogenity of the sets. 

def gini_impurity(df,key):

    data1 = df.copy()

    values = data1[key].unique() # find all the unique labels

    n = len(data1)

    prob = [] # to store the occurence of each label in the sets

    for value in values:

        freq = len(data1[data1[key]==value])

        prob.append(freq/n)

    # calculate the gini impurity

    gini = 1- sum([x**2 for x in prob]) # calculate the gini impurity value

    return gini



# checking the gini impurity for the whole dataset

initial_gini = gini_impurity(data,key='label')

initial_gini 
features = data.columns[0:-1] # store each features

# find unique values in each features

x1 = data['color'].unique()

x2 = data['diameter'].unique()

# total questions 

'number of questions that can be asked ? =', len(x1)+len(x2)
# splitting the dataframes based on each question

for feature in features:

    values = data[feature].unique()

    for value in values:

        print("--------------split1--------------")

        print(data[data[feature]==value])

        print("--------------split2--------------")

        print(data[data[feature]!=value])

        

        print("=================================")

        print("")
# splitting the dataframes based on each question

N = len(data)

for feature in features:

    values = data[feature].unique()

    for value in values:

        split1 = data[data[feature]==value]

        split2 = data[data[feature]!=value]

        # evaluating the gini for each split

        gini1 = gini_impurity(split1,key='label')

        gini2 = gini_impurity(split2,key='label')

        # calculating the information gained for the split

        information_gained = initial_gini -  (gini1*len(split1)/N + gini2*len(split2)/N)

        print(feature,'=',value,'| information_gained = ',information_gained)