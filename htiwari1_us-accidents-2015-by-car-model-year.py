# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.plotly as py

 

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



vehicle_df = pd.read_csv('../input/vehicle.csv', usecols=[0, 19, 100, 101])

vehicle_df = vehicle_df[(vehicle_df['DEATHS']==1)&(vehicle_df['DR_DRINK']==0)]

vehicle_df = vehicle_df.groupby(['MOD_YEAR']).count().reset_index()

vehicle_df = vehicle_df[(vehicle_df.MOD_YEAR != 9999)&(vehicle_df.MOD_YEAR > 1990)]

vehicle_df.plot(kind="bar", x='MOD_YEAR', y='DEATHS', label='Fatal Accidents')



# Any results you write to the current directory are saved as output.