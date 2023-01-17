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
import math

df = pd.read_csv("/kaggle/input/test_scores.csv")

df.head()

def gradient_descent(X, y):

    

    learning_rate = 0.001

    m_current = b_current = 0

    iterations = 1000

    m = len(y)

    cost_array = []

    cost_previous = 0

    

    for i in range(iterations):

        y_predicted = m_current * X + b_current

        m_derivative = -(2/ m) * sum(X * (y - y_predicted))

        b_derivative = -(2/ m) * sum(y - y_predicted)

        m_current = m_current - learning_rate * m_derivative

        b_current = b_current - learning_rate * b_derivative

        cost = (1/ m) * sum([val ** 2 for val in (y - y_predicted)])

        if math.isclose(cost, cost_previous, rel_tol=1e-20):

            break

        cost_previous = cost

        print("m{}, b{}, cost{}, iteration{}".format(m_current, b_current, cost, i))



X = df['math']

y = df['cs']
gradient_descent(X, y)