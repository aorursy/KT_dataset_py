# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/insurance/insurance.csv").drop(columns = ["region"])

dataset.head(5)
import pandas as pd

import numpy as np



dataset = pd.read_csv("/kaggle/input/insurance/insurance.csv").drop(columns = ["region"])

for i, row in dataset.iterrows():

    if(row["sex"] == "female"):

        dataset.loc[i, "sex"] = 0

    elif(row["sex"] == "male"):

        dataset.loc[i, "sex"] = 1

    if(row["smoker"] == "yes"):

        dataset.loc[i, "smoker"] = 1

    elif(row["smoker"] == "no"):

        dataset.loc[i, "smoker"] = 0

dataset.head(5)



def do_linear_regression(d, exclude = []):

    dataset = d

    y = dataset["charges"].to_numpy()

    X = dataset.drop(columns = ["charges"] + exclude)

    names = X.columns

    X = X.to_numpy(dtype = "float")

    X = np.hstack((X, np.ones((1338, 1))))

    

    a = np.matmul(np.transpose(X), X)

    a = np.matmul(np.linalg.inv(a), np.transpose(X))

    theta = np.matmul(a, y)

    

    y_mean = sum(y)/len(y)

    ss_total = sum([(y[i] - y_mean)**2 for i in range(len(y))])

    ss_res = sum([(y[i] - np.dot(theta, X[i]))**2 for i in range(len(y))])

    r2 = 1 - ss_res/ss_total

    

    string = ""

    for i in range(len(names)):

        string += "{}*{} + ".format("%.3f" % theta[i], names[i])

    string += "%.3f" % theta[-1]

    

    print(string) 

    print("R^2 is: {}".format(r2))

    print("Sum of residuals squared is {}".format(ss_res))

    return string, "R^2 is: {}".format(r2), "Sum of residuals squared is {}".format(ss_res)



print(dataset.columns)

do_linear_regression(dataset, exclude = [])