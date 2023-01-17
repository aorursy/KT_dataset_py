# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from plotly import __version__

import plotly.plotly as py

import cufflinks as cf

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df=pd.read_csv('../input/Border crossing deaths.csv')



df.info()
df.head()
countries={1:'Spain',2:'Gibraltar',3:'Greece',4:'Italy',5:'Malta'}
df=df.replace({'Registration_country':countries})

#replaced coutnries code with their respective names

sex={1:'Male',2:'Female',np.nan:'UnKnown'}

df=df.replace({'Sex':sex})

#replaced gender with their code

origin={1:'North Africa',2:'Sub-Saharan Africa & Horn of Africa',3:'Middle East',4:'Balkans',5:'Asia',np.nan:'UnKnown'}

df=df.replace({'Origin':origin})

#replaced origing code with values
cause={1:'Drowning',2:'Dehydration',3:'Cardiorespiratory arrest',4:'Hypothermia',5:'Injuries & Violence',6:'suffocation',7:'Unknown'}

df=df.replace({'Cause':cause})

#replaced case code with valuese
df.groupby('Year')['Origin'].count().iplot(kind='bar')

#deaths per year from 1990-2013
df.groupby(by=['Cause'])['Cause'].count().iplot(kind='bar')

#over all deaths by reasons 

df.groupby(by=['Registration_country'])['Registration_country'].count().iplot(kind='bar')
df.groupby('Origin')['Origin'].count().iplot(kind='barh')

#origin of dead peoples