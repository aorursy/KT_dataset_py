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
path = os.path.join(dirname, filename)

df = pd.read_csv(path)

df
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt
df1 = df[['ardt','titre']].groupby(['ardt'], as_index=False).count() 

df2 = df1.drop(df1[df1.ardt > 75020].index)

df2
import seaborn as sns



result = df2.groupby(["ardt"])['titre'].aggregate(np.median).reset_index().sort_values('titre', ascending=0)



sns.barplot( x="titre", y="ardt",data=df2,

            label="Number of films per district",orient='h', order=result['ardt'])