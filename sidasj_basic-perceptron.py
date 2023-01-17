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
# How can we build our own AND OR or XOR Perceptron
import numpy as np

# For building basic Or perceptron we require two things they are weight and data points

# The OR perceptron gives output 0 if and only if value of both weight as well as data is 0

# If any one of the value is 1 then the output is also 1.



weight = [0,1,0,1]

data = [0,0,1,1]

result = list(np.empty(len(weight)))

def perceptron(w,i):

    for j in range(len(w)):

        print("The current value of w is: ",w[j])

        print("The current value of data is: ",i[j])

        result[j] = w[j] or i[j]

        print("The output of the corresponding value of Weight and Data is: ",result[j])

        print("*"*20)



a = perceptron(weight,data)

a

print("The Result array is: ",result)
# For building basic AND perceptron we require two things they are weight and data points

# The OR perceptron gives output 1 if and only if value of both weight as well as data is 1

# If any one of the value is 0 then the output is also 0.



weight = [0,1,0,1]

data = [0,0,1,1]

result = list(np.empty(len(weight)))

def perceptron(w,i):

    for j in range(len(w)):

        print("The current value of w is: ",w[j])

        print("The current value of data is: ",i[j])

        result[j] = w[j] and i[j]

        print("The output of the corresponding value of Weight and Data is: ",result[j])

        print("*"*20)



a = perceptron(weight,data)

a
# XOR perceptron

weight = [0,1,0,1]

data = [0,0,1,1]

result = list(np.empty(len(weight)))

def perceptron(w,i):

    for j in range(len(w)):

        print("The current value of w is: ",w[j])

        print("The current value of data is: ",i[j])

        if w[j]==0 and i[j]==0:

            result[j] = 0

        elif w[j]==1 and i[j]==1:

            result[j] = 0

        elif w[j]==1 and i[j]==0:

            result[j] = 1

        elif w[j]==0 and i[j]==1:

            result[j] = 1

        print("The output of the corresponding value of Weight and Data is: ",result[j])

        print("*"*20)



a = perceptron(weight,data)

a