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
#
# Hi i am practising Data Science Concepts and below Code Snipets are some examples for finding
# various calcutations on the Marks of 5 subjects of 100 students 
#
import numpy as np
exam=np.random.randint(0,100,(100,5))
#//maths,physics,chemistry,social,science
exam
### find the details of students who has marks greaterthan 120 in social and science 
exam[exam[:,3:].sum(axis=1)>120]
### find the person that has hghest maths,physics and chemistry 
exam[exam[:,:3].sum(axis=1) == exam[:,:3].sum(axis=1).max()]
### find the student that has min science ,and social score
exam[exam[:,3:].sum(axis=1) == exam[:,3:].sum(axis=1).min()]
### subset only students maths/phisics ratio is between 0.8 to 1.2 
exam[(exam[:,0]/exam[:,1]>=.8) & (exam[:,0]/exam[:,1]<=1.2)]
### subset only students whose maths marks between mean+ std and mean -std
exam[(exam[:,0]>=exam[:,0].mean()-exam[:,0].std())&(exam[:,0]<=exam[:,0].mean()+exam[:,0].std())]
import pandas as pd
### find the average of maths,physics, checmistry and find the difference with average of social and science
pd.DataFrame(exam[:,:3].mean(axis=1)-exam[:,3:].mean(axis=1))