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
p_know = 0.6

p_incorrect_given_know = 0.15

p_correct_given_not_know = 0.2



#What is p_know_given_correct



#p_know_given_correct = p_correct_given_know * p_know / p_correct



p_correct_given_know = 1 - p_incorrect_given_know

p_correct = p_correct_given_know * p_know + p_correct_given_not_know * (1 - p_know)

p_know_given_correct = p_correct_given_know * p_know / p_correct

print(p_know_given_correct)
