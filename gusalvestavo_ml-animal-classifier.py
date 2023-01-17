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
zoo = pd.read_csv('/kaggle/input/zoo-animal-classification/zoo.csv')

zoo = zoo.drop(['class_type', 'hair', 'airborne', 'venomous', 'domestic', 'catsize'], axis=1)



zoo.insert(col_count, 'class', 'undefined')    
zoo.loc[(zoo["class"] == "undefined") & (zoo['toothed']==1) & (zoo['backbone']==1) & (zoo['breathes']==1) & (zoo['milk']==1) ,"class"] = 'mammal'    



zoo.loc[(zoo["class"] == "undefined") & (zoo['feathers']==1) & (zoo['eggs']==1) & (zoo['backbone']==1) & (zoo['legs'] == 2) ,"class"] = 'bird'

    

zoo.loc[(zoo["class"] == "undefined") & (zoo['eggs']==1) & (zoo['aquatic']==1) & (zoo['toothed']==1) & (zoo['backbone']==1) & (zoo['breathes']==1) & (zoo['fins'] == 0) & (zoo['legs'] == 4) ,"class"] = 'amphibian'

    

zoo.loc[(zoo["class"] == "undefined") & (zoo['eggs']==1) & (zoo['aquatic']==1) & (zoo['toothed']==1) & (zoo['backbone']==1) & (zoo['fins']==1) & (zoo['tail']==1) & (zoo['legs'] == 0) ,"class"] = 'fish'



zoo.loc[(zoo["class"] == "undefined") & (zoo['backbone']==1) & (zoo['tail']==1) & ((zoo['legs'] == 0) | (zoo['legs'] == 4)) ,"class"] = 'reptile'

  

zoo.loc[(zoo["class"] == "undefined") & (zoo['eggs']==1) & (zoo['breathes']==1) & (zoo['legs'] == 6) ,"class"] = 'bug'

    

zoo.loc[(zoo["class"] == "undefined") & (zoo['backbone'] == 0) ,"class"] = 'invertebrate'
zoo.head(50)