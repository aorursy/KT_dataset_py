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

df = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")
# Any results you write to the current directory are saved as output.
print (" This shows the total income per coder with respect to country: ")
df1 = df[['CountryCitizen','Income']].groupby(df['CountryCitizen']).sum().fillna(0)
df_income = df[['CountryCitizen','Income']].groupby(df['CountryCitizen']).count()
df1["Count"]=df_income["Income"]
df1["AverageIncome"]=df1["Income"]/df1["Count"]
print (df1.sort("AverageIncome", ascending=False)[:10])
