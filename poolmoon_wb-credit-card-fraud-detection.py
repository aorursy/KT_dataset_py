# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv("../input/creditcard.csv")



# Any results you write to the current directory are saved as output.
data.head()
data.shape
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
from sklearn.preprocessing import StandardScaler



data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))

data = data.drop(['Time','Amount'],axis=1)
# TODO: Over-sampling
# TODO: Under-sampling
# TODO: Choose your own classifier and make predictions, compare the results from over-sampling and under-sampling.

# Hint: you can refer to other people's notebooks: https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now
# TODO: Threshold-moving