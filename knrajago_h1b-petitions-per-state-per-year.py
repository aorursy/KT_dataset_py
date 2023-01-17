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

h1b_data=pd.read_csv('../input/h1b_kaggle.csv')

h1b_data['STATE']=h1b_data['WORKSITE'].str.split(', ').str[1]

state_data=h1b_data.groupby(['STATE', 'YEAR']).size()

state_year_data=state_data.unstack()

state_year_data['TOTAL'] = state_year_data.sum(axis=1)

state_year_data
import matplotlib.pyplot as plt



#print(str(state_year_data.loc['ALABAMA',]))

state_year_data.shape[0]

state_year_data.index

for oneidx in state_year_data.index:

    state_data = state_year_data.loc[oneidx,]

    del state_data['TOTAL']

    state_data.plot(kind="line", grid=True, title=oneidx)

    plt.show()

   

    