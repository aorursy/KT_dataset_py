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
import plotly.plotly as py

import plotly.offline as ploff



from ggplot import *

from subprocess import check_output

from plotly.offline import init_notebook_mode, plot



ploff.init_notebook_mode()



headers = ['BusinessYear', 'StateCode', 'Age', 

           'IndividualRate', 'Couple']



# read in chuncks for memory efficiency

filePath = '../input/Rate.csv'

chunks = pd.read_csv(filePath, iterator=True, chunksize=1000,

                    usecols=headers)

rates = pd.concat(chunk for chunk in chunks)



randomRows = rates.sample(n=6)

randomRows
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print (rates['Couple'].describe())
print (rates['IndividualRate'].describe())
ratesInd9000 = rates[rates.IndividualRate < 9000]

print (ratesInd9000['IndividualRate'].describe())
columns = ['BusinessYear', 'StateCode', 'IndividualRate']

indRates = pd.DataFrame(ratesInd9000, columns=columns)

indRates2016 = indRates[indRates.BusinessYear == 2016]

indRates2016 = indRates2016.dropna(subset=['IndividualRate'])

randomRows2016 = indRates2016.sample(n=6)

randomRows2016
indRates2016['IndividualRate'].describe()
indMean2016 = indRates2016.groupby('StateCode', as_index=False).mean()

indMean2016