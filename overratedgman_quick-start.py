# Let's start by importing the data



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display



# Load the dataset

in_file = '../input/data.csv'

full_data = pd.read_csv(in_file)



# Print the first few entries of the data

display(full_data.head())
from tabulate import tabulate

main_data=full_data.drop('id',axis=1)

main_data=main_data.drop('diagnosis',axis=1)

main_data=main_data.drop('Unnamed: 32',axis=1)

display(main_data.head())

column_names = list(main_data.columns.values)

#print (column_names)

y=0

for x in main_data:

    #min value 

    s=np.amin(main_data[x])

    t=column_names[y]

    print('{} min: {:.2f}'.format(t,s))

    

    #max value

    p=np.amax(main_data[x])

    print('{} max: {:.2f}'.format(t,p))

    

    #range value

    s=p-s

    print('{} range: {:.2f}'.format(t,s))

    

    #mean value

    s=np.mean(main_data[x])

    print('{} mean: {:.2f}'.format(t,s))

    

    #q1 value

    s=np.percentile(main_data[x],25)

    print('{} q1: {:.2f}'.format(t,s))

    

    #q3 value

    s=np.percentile(main_data[x],75)

    print('{} q3: {:.2f}'.format(t,s))

    

    #std value

    s=np.std(main_data[x])

    print('{} std: {:.2f}'.format(t,s))

    

    y=y+1
from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation



x_train,x_test,y_train,y_test= cross_validation.train_test_split(main_data,full_data['diagnosis'],

                                                                 test_size=0.3,random_state=0)

#print(y_test)

classifier = DecisionTreeClassifier(random_state=0)

classifier.fit(x_train,y_train)

print (classifier.score(x_test,y_test))