 

"""

USE Best Model + Transformation:



Model on 20% data, test on 30% data.



(1)  Isolate Patients and Segmentation training algorithm (restrict number and proportion)

(2)  try 4 algorithms (try transformations + Resizing + Masking):



     INPUT > patients                 OUTPUT > patients+segment



(3)  piecewise model by segment.

(4)  compare logloss of model vs. piecewise segment model.



"""











# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# we loop over the parameters



for y in range(0, 4):

    for x in range(0, 4):

            print(str(x)+" and "+str(y))



            # fit the model 10, 20 30, 40, 50

            

            # compute the logloss(s)

            

            # append paramters and logloss to a pandas dataframe



            

            

            

# Do the same thing for SVM



# Same thing for RF



# same thing for transformations



# write ensemble tester code



# setup the mask loop (try to speed up)



# do the same checks on the masked data



# test ensembles



# select best model