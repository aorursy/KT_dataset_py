# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/creditcard.csv")

def myColor(result):

    if result == 1:

        return "red"

    else:

        return "blue"

theColor = [myColor(result) for result in df.Class]

df["diff_time"]=df.Time.diff(1)
df.fillna(0, inplace=True)

df.head()
plt.scatter(df.diff_time, df.V2, c=theColor, s=df.Amount, alpha=0.3)
plt.scatter(df.Time, df.V2, c=theColor, s=df.Amount, alpha=0.3)