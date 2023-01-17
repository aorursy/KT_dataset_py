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
sub1 = pd.read_csv('/kaggle/input/cv19-w3-bt-sub1/submission.csv')

sub1 = sub1.rename({'ConfirmedCases': 'ConfirmedCases_1', 'Fatalities': 'Fatalities_1'}, axis=1)

sub2 = pd.read_csv('/kaggle/input/covid-19-w3-a-few-charts-and-a-simple-baseline/submission.csv')

sub2 = sub2.rename({'ConfirmedCases': 'ConfirmedCases_2', 'Fatalities': 'Fatalities_2'}, axis=1)
sub = sub1.copy()

sub = sub.merge(sub2, on='ForecastId', how='left')

sub.head()
sub['ConfirmedCases'] = sub[[c for c in sub.columns if c.startswith('ConfirmedCases_')]].mean(axis=1)

sub['Fatalities'] = sub[[c for c in sub.columns if c.startswith('Fatalities_')]].mean(axis=1)

sub.head()
sub[['ForecastId', 'ConfirmedCases', 'Fatalities']].round().to_csv('submission.csv', index=False)