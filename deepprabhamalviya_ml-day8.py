import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/ipl2013/IPL2013.csv")# i have to go... for dinner ...# :D ok eat  I'll send you this kernel link, or you 

#can find it on your kaggle account from phone , okok

data
# Now you start from here
# Look into the basic info about the dataset - attributes & their types [DONE]

# Show the initial 10 columns for the first 5 rows of the dataset

# Remove the irrelevant features
# So for the first step, we will use a function that is present in the Pandas library
# And it is called info()
# There are many more function libraries that output the same result so feel free to use any of them.
data.info()
# This is the first step that Sir wanted to see
# Show the initial 10 columns for the first 5 rows of the dataset









# So, here there are two ways to do it. One is to manually delete using the drop command that people used in the morning.
#But, I have used iLoc that was there in Sayantan Sir's untaught ppt.
data.iloc[:5, :10]
# Remove the irrelevant features



# So, if you remembered, Sir had mentioned that in a dataset, when we try to calculate any objective, some features are less

# important. For example Roll Number. In this dataset, I guess it is Sl.no which is the first column attribute.
#name also ++++++ yes name also and role # ok, role we might require when we are going to prepare the heatmap.
#so let's try to remove the first column. And teach me how to remember which is row and which is column.
#SENTust remember... laterals r a separate column... n branch change r different column... u columns in the class

#jioh. i'll remember.
#so, getting back to the removal of attribute.
from sklearn import linear_model

from sklearn.model_selection import train_test_split
target = data.iloc[:,0]

print (target)
target
#see, i took a variable named target, in the same way i took the main database as "data" variable

#i am just printing an updated table



#just target == print(target) with a format function in python
data # THIS IS FORMATTED
print (data) # THIS OUTPUT IS UNFORMMATTED
#writing only a variable name that refers to a dataset applies a format functon.

#Since target is the single attribute value, it doesnt matter if we write print (target) or target only

data.drop(["Sl.NO."],axis = 1, inplace = True)  # You can add more garbage columns beside this SL.NO



data
#See, that column is eliminated ok

# This is step 1-4, upto this I had done in the morning.



#5 is the difficult task that sir wanted us to research on.

#6-10 are linked.



#I think if you show this notebook, it will be ok. Do you want me to remove the comments here ? :pno  ok

#i ho 'll go through ok'