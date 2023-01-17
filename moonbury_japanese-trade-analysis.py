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
country = pd.read_csv('../input/country_eng.csv')

custom = pd.read_csv('../input/custom.csv')

#exp_custom_latest = pd.read_csv('../input/exp_custom_latest_ym.csv')

hs2 = pd.read_csv('../input/hs2_eng.csv')

hs4 = pd.read_csv('../input/hs4_eng.csv')

hs6 = pd.read_csv('../input/hs6_eng.csv')

hs9 = pd.read_csv('../input/hs9_eng.csv')

year_latest = pd.read_csv('../input/year_1988_2015.csv')

ym_latest = pd.read_csv('../input/ym_latest.csv')
pd.set_option('precision',0)

pd.set_option('display.width',120)



#print(country.head(4))

#print(custom.head(4))

#print(exp_custom_latest.head(4))

#print(hs2.head(4))

#print(hs4.head(4))

#print(hs6.head(4))

#print(hs9.head(4))

print(year_latest.head(4))

print(ym_latest.head(4))
#boxplots = exp_custom_latest.plot(kind='scatter',x=0,y=3)