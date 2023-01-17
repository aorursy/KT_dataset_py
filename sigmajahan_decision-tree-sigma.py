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
import pprint

import pandas as pd

import numpy as np

from numpy import log2 as log

eps = np.finfo(float).eps



dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],

       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],

       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],

       'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}





df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])



def find_entropy(df):

    Class = df.keys()[-1]   #To make the code generic, changing target variable class name

    entropy = 0

    values = df[Class].unique()

    for value in values:

        fraction = df[Class].value_counts()[value]/len(df[Class])

        entropy += -fraction*np.log2(fraction)

    return entropy

  

  

def find_entropy_attribute(df,attribute):

  Class = df.keys()[-1]   #To make the code generic, changing target variable class name

  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'

  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)

  entropy2 = 0

  for variable in variables:

      entropy = 0

      for target_variable in target_variables:

          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])

          den = len(df[attribute][df[attribute]==variable])

          fraction = num/(den+eps)

          entropy += -fraction*log(fraction+eps)

      fraction2 = den/len(df)

      entropy2 += -fraction2*entropy

  return abs(entropy2)





def find_winner(df):

    Entropy_att = []

    IG = []

    for key in df.keys()[:-1]:

#         Entropy_att.append(find_entropy_attribute(df,key))

        IG.append(find_entropy(df)-find_entropy_attribute(df,key))

    return df.keys()[:-1][np.argmax(IG)]

  

  

def get_subtable(df, node,value):

  return df[df[node] == value].reset_index(drop=True)





def buildTree(df,tree=None): 

    Class = df.keys()[-1]   #To make the code generic, changing target variable class name

    

    #Here we build our decision tree



    #Get attribute with maximum information gain

    node = find_winner(df)

    

    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values

    attValue = np.unique(df[node])

    

    #Create an empty dictionary to create tree    

    if tree is None:                    

        tree={}

        tree[node] = {}

    

   #We make loop to construct a tree by calling this function recursively. 

    #In this we check if the subset is pure and stops if it is pure. 



    for value in attValue:

        

        subtable = get_subtable(df,node,value)

        clValue,counts = np.unique(subtable['Eat'],return_counts=True)                        

        

        if len(counts)==1:#Checking purity of subset

            tree[node][value] = clValue[0]                                                    

        else:        

            tree[node][value] = buildTree(subtable) #Calling the function recursively 

                   

    return tree

  

  



# In[70]:

def predict(inst,tree):

    #This function is used to predict for any input variable 

    

    #Recursively we go through the tree that we built earlier



    for nodes in tree.keys():        

        

        value = inst[nodes]

        tree = tree[nodes][value]

        prediction = 0

            

        if type(tree) is dict:

            prediction = predict(inst, tree)

        else:

            prediction = tree

            break;                            

        

    return prediction



tree = buildTree(df)

pprint.pprint(tree)



d = {'Taste': 'Sweet', 'Temperature': 'Hot', 'Texture' : 'Soft'}

data = pd.Series(d)

print(predict(data , tree))





# In[ ]:








