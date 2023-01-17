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
raw_iris = pd.read_csv("../input/Iris.csv")
rm = [x for x in range(25,50)] + [x for x in range(75,100)] + [x for x in range(125,150)]

iris = raw_iris.drop(rm)

rm_test = [x for x in range(0,25)] + [x for x in range(50,75)] + [x for x in range(100,125)]

iris_test = raw_iris.drop(rm_test)
X1 = iris.loc[iris["Species"]=="Iris-setosa"].ix[:,1:5]

X2 = iris.loc[iris["Species"]=="Iris-virginica"].ix[:,1:5]

X3 = iris.loc[iris["Species"]=="Iris-versicolor"].ix[:,1:5]
m1 = X1.mean().values

m2 = X2.mean().values

m3 = X3.mean().values



cov1 = X1.cov().values

cov2 = X2.cov().values

cov3 = X3.cov().values

icov1 = np.linalg.inv(cov1)

icov2 = np.linalg.inv(cov2)

icov3 = np.linalg.inv(cov3)
success = 0

failures = []



for row in rm:

    y = iris_test.ix[row,1:5].values

    

    dy1 = y-m1

    dy2 = y-m2

    dy3 = y-m3



    stat1 =  np.dot(np.dot(dy1,icov1),dy1)

    stat2 =  np.dot(np.dot(dy2,icov2),dy2)

    stat3 =  np.dot(np.dot(dy3,icov3),dy3)



    if stat1 < stat2 and stat1 < stat3:

        lab = "Iris-setosa"

    elif stat2 < stat1 and stat2 < stat3:

        lab = "Iris-virginica"

    elif stat3 < stat1 and stat3 < stat2:

        lab = "Iris-versicolor"

        

    if(lab == iris_test.ix[row,5]):

        success = success + 1

    else:

        failures.append(iris_test.ix[row,0])

        print(stat1,stat2,stat3)

        print(lab,iris_test.ix[row,5])

    

    
print(success)

print(failures)
