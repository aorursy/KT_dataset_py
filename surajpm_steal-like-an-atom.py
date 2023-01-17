#loading packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
#reading all submission files

sub1 = pd.read_csv('../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv')

sub2 = pd.read_csv('../input/keras-neural-net-for-champs/workingsubmission-test.csv')

sub3 = pd.read_csv('../input/no-memory-reduction-workflow-for-each-type-lb-1-28/LGB_2019-07-11_-1.4378.csv')

sub4 = pd.read_csv('../input/giba-r-data-table-simplefeat-cyv-interaction/submission-2.csv')

print( sub1['scalar_coupling_constant'].describe() )

print( sub2['scalar_coupling_constant'].describe() )

print( sub3['scalar_coupling_constant'].describe() )

print( sub4['scalar_coupling_constant'].describe() )
# Random weights to each submission by trying and experimenting

sub1['scalar_coupling_constant'] = 0.25*sub1['scalar_coupling_constant'] + 0.2*sub2['scalar_coupling_constant'] + 0.3*sub3['scalar_coupling_constant'] + 0.25*sub4['scalar_coupling_constant']

sub1.to_csv('submission.csv', index=False )
#plotting histogram

sub1['scalar_coupling_constant'].plot('hist', bins=100)