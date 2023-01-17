# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv').convert_objects(convert_numeric=True).values
 
# y = mx + b
# Steps we need to define m,b 
m,b = 3,-0.5 
# Define error functions for them
def error(point,m,b):
    return (abs(point[0]*m+b - point[1])**2)


# Update them for certain number of iterations using function
def update(m,b,points,iterations = 4, learning_rate = 0.01):
    N = len(points)
    b_new, m_new = 0,0
    b_grad,m_grad = 0,0
    for i in range(iterations):
        for x,y in points:
            b_grad = (2/N) * (x*m+b - y)
            m_grad = (2/N) *x*(x*m+b - y)
            b_new = b - (learning_rate*b_grad)
            m_new = m - (learning_rate*m_grad)
            m,b = m_new,b_new
    return m_new,b_new
# Find the total error
def total_error(m,b,points):
    result = 0
    for point in points:
        result += error(point,m,b)
    return result                    
m,b = 3,1
print(total_error(m,b,data))
m_new,b_new = update(m,b,data,2000,0.01)
print(total_error(m_new,b_new,data))
