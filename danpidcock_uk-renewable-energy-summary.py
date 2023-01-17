# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.https://www.kaggle.com/louissg/renewable-energy-statistics

# Units are in GWh

df = pd.read_csv("../input/cleaned_generation.csv")

df.head(16)
# The regional figures are totalled under "England" so remove them

df = df[df.Region.isin(['England', 'Northern Ireland', 'Scotland', 'Wales', 'Other Sites4'])]
def to_float(series):

    if series.dtype != 'O':

        return series

    series = series.str.replace('-', '0').str.replace(',', '')

    return pd.to_numeric(series, errors='coerce')

df2 = df.drop(['Unnamed: 0', 'Total', 'Region'], axis=1).apply(to_float)

grouped = df2.groupby('Year')
grouped.sum()