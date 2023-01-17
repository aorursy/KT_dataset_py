# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as ss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

url = '../input/cereal.csv'

dataset = pd.read_csv(url)

#dataset['mfr']



chisquare_stat = ss.chisquare (dataset['type'].value_counts())

chisquare_statMfr = ss.chisquare(dataset['mfr'].value_counts())

#print(chisquare_stat)

#print(chisquare_statMfr)



# chisquare for two variables

contigencyTable = pd.crosstab(dataset['type'],dataset['mfr'])

#print(contigencyTable)

chisquarestat2 = ss.chi2_contingency(contigencyTable)

print(chisquarestat2)



# Any results you write to the current directory are saved as output.