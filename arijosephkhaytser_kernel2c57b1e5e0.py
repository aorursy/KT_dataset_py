# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read data from csv file

data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

data.head(4)
#Basic stats

data.describe()
#Lets try to see which input variable has the greatest impact on response variable 



#First, I found chi2 values for each variable, to see which ones are most dependant. 

chi2vals = []

n = len(data[data['class'] == 'e'].index)

for var in data.columns.values:

    groups = data[var].value_counts().axes[0]

    chi2 = 0

    for group in groups:

        expected = n / (len(groups))

        actual = len(data[(data[var] == group) & (data['class'] == 'e')].index)

        

        chi2 += ((actual - expected)**2)/expected

    chi2vals.append(chi2)



cols = data.columns.values

pd.DataFrame(list(zip(cols, chi2vals)))
#From those results, odor has by far the highest chi2 value, so its likely the most determining variable

#Other variables that seem highly dependant are stalk color above and below ring, and veil color

#Lets how the distribution of mushrooms looks when split by odor.



bars_edible = []

bars_poisonous = []

width = 0.3



pyplot.figure(figsize=(10,6), dpi = 100)



for odor in data['odor'].value_counts().axes[0]:

    bars_edible.append(len(data[(data['odor'] == odor) & (data['class'] == 'e')].index))

    bars_poisonous.append(len(data[(data['odor'] == odor) & (data['class'] == 'p')].index))

    

labels = data['odor'].value_counts().axes[0]    

    

pyplot.bar(np.arange(len(bars_edible)), bars_edible, width=width)

pyplot.bar(np.arange(len(bars_poisonous))+ width, bars_poisonous, width=width)

pyplot.xticks(range(len(labels)), labels)

pyplot.xlabel('Odor')

pyplot.ylabel('Count')





pyplot.show()
#From the graph above, it seems like in almost all cases, edibility is can be determined solely by odor.

#However, in the case of no odor, while almost all mushrooms are edible, there is still a small chance of getting a poisonous one. 



#Finally, I will try to generate a decision tree based on the data to decide the class of a mushroom. 

#This tree is certainly not very reliable, since converting unorderable categorical data to numberical data can and will create strange results

#However, it might be somewhat useful...

#To prevent the tree from being too specific to this particular dataset (overfit) I set the max depth allowed to 5





from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

for col in data.columns:

    data[col] = encoder.fit_transform(data[col])



X=data.drop(['class'], axis=1)

Y=data['class']



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz



X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)



tree = DecisionTreeClassifier(max_depth = 5)

tree.fit(X_train, Y_train)

dot_data = export_graphviz(tree, out_file=None, feature_names=X.columns)  

graph = graphviz.Source(dot_data)  

graph
#In conclusion, it seems like Odor is the most dependant variable to determine if a mushroom is poisonous, and most odors can tell you exactly whether or not a mushroom is poisonous. 

#Other variables that seem highly dependant are stalk color above and below ring, and veil color.

#By converting our data to numerical, we can create a somewhat-reliable decision tree to find if a mushroom is poisonous or not. 

#However, the conversion from categrical to numerical data does mean that the tree is far from perfect, and some of the splits make little or no sense.