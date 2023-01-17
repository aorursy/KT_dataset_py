import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
sub1 = pd.read_csv('../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv')

sub2 = pd.read_csv('../input/keras-neural-net-for-champs/workingsubmission-test.csv')

print( sub1['scalar_coupling_constant'].describe() )

print( sub2['scalar_coupling_constant'].describe() )
#Mean absolute difference

( sub1['scalar_coupling_constant'] - sub2['scalar_coupling_constant']).abs().mean()
# I used 0.6 weight for LGB just because it performed a little bit better in Public LB.

sub1['scalar_coupling_constant'] = 0.6*sub1['scalar_coupling_constant'] + 0.4*sub2['scalar_coupling_constant']

sub1.to_csv('weighted-avg-blend-lgb-keras-1.csv', index=False )

sub1['scalar_coupling_constant'].describe()