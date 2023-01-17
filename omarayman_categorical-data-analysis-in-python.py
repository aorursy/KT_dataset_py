# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

Data = pd.read_csv("../input/anonymous-survey-responses.csv")

Data.head()
Counts = Data["Just for fun, do you prefer dogs or cat?"].value_counts() 

Names = Counts.index
postionofbars = list(range(len(Names)))

plt.bar(postionofbars,Counts)

plt.xticks(postionofbars,Names)

import seaborn as sns

sns.countplot(Data["Just for fun, do you prefer dogs or cat?"]).set_title("Dogs Vs Cats")