# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from scipy.stats import chisquare

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/DigiDB_digimonlist.csv")

observed = pd.crosstab(index=df["Stage"], columns="count")

expected = [df.shape[0] / observed.shape[0]] * observed.shape[0]

chisquare(observed, expected)



# Any results you write to the current directory are saved as output.