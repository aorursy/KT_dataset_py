# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output





print ("Input file check")

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/xAPI-Edu-Data.csv')



#print(df.NationalITy)



sns.countplot(x="NationalITy", data=df);

plt.show()

from matplotlib.pyplot import pie, axis, show

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



print("Trying to groupby nationalities")

df = pd.read_csv('../input/xAPI-Edu-Data.csv')



group_by_sum_of_nationalities = df.NationalITy.groupby(df.NationalITy).count()

group_by_sum_of_nationalities_header = group_by_sum_of_nationalities.keys()



#print(group_by_sum_of_nationalities_header)

pie(group_by_sum_of_nationalities,labels=group_by_sum_of_nationalities_header)
