# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

ds = pd.read_csv("../input/Kaggle.csv")

ds.shape

ds.corr()



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ds.sort_values('Human Development Index HDI-2014', ascending=False).head(20)
#Prints the highers corrolated column pairs

def PrintHighestCorrolation(dataset):

    keys = dataset.keys()

    keyslen = len(keys)



    for i in range(1,keyslen):

        key = keys[i]

        for j in range(1,keyslen):

            innerkey = keys[j]

            if innerkey != key:        

                corrolationVal = dataset[key].corr(dataset[innerkey])

                if abs(corrolationVal) > 0.9: #if the corrolation is higher than the trashhold

                    print(key + "<-->" + innerkey + "\n")

                    print(corrolationVal)







                    

PrintHighestCorrolation(ds)
