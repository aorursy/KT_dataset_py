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
# Read in the training data



traindat = pd.read_csv("../input/train.csv")
# Import SequenceMatcher to compare the questions



from difflib import SequenceMatcher



def similar(a, b):

    return SequenceMatcher(None, a, b).ratio()
# Check the questions for sequence matching.

# This is slow, but it does what I want for now.



def questionRead(datrow):

    q1 = datrow["question1"]

    q2 = datrow["question2"]

    if (q1 == q1) & (q2 == q2):

        return(similar(q1, q2))

    else:

        return(0)

            

traindat["overlap"] = traindat.apply(questionRead, 1)
# Take a look at the first row and make sure this is what I want



traindat["question1"].loc[0]
traindat["question2"].loc[0]
traindat["overlap"].loc[0]