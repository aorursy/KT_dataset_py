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
import numpy as np

import pandas as pd

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



terror_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1',

                          usecols=[0, 1, 2, 3, 8, 11, 13, 14, 29, 35, 84, 100, 103])

terror_data = terror_data.rename(

    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',

             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',

             'weaptype1_txt':'weapon', 'attacktype1_txt':'attack',

             'nkill':'fatalities', 'nwound':'injuries'})

terror_data['fatalities'] = terror_data['fatalities'].fillna(0).astype(int)

terror_data['injuries'] = terror_data['injuries'].fillna(0).astype(int)

pd.options.mode.chained_assignment = None
