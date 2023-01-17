#loading packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics

import os

print(os.listdir("../input"))
#reading all submission files

#sub1 = pd.read_csv('../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv')

#sub2 = pd.read_csv('../input/keras-neural-net-for-champs/workingsubmission-test.csv')

#sub3 = pd.read_csv('../input/no-memory-reduction-workflow-for-each-type-lb-1-28/LGB_2019-07-18_-1.2243.csv')

sub4 = pd.read_csv('../input/from-sub-lgb/sub_lgb_model_individual.csv')

#sub5 = pd.read_csv('../input/stacking-try/stack_median.csv')

sub6 = pd.read_csv('../input/stealing/submission.csv')

sub7 = pd.read_csv('../input/stealing/submission1.csv')



#print( sub1['scalar_coupling_constant'].describe() )

#print( sub2['scalar_coupling_constant'].describe() )

#print( sub3['scalar_coupling_constant'].describe() )

print( sub4['scalar_coupling_constant'].describe() )

#print( sub5['scalar_coupling_constant'].describe() )

print( sub6['scalar_coupling_constant'].describe() )

print( sub7['scalar_coupling_constant'].describe() )

# Random weights to each submission by trying and experimenting

sub6['scalar_coupling_constant'] =  0.34*sub6['scalar_coupling_constant'] + 0.33*sub7['scalar_coupling_constant'] + 0.33*sub4['scalar_coupling_constant']

sub6.to_csv('submission.csv', index=False )
#plotting histogram

sub6['scalar_coupling_constant'].plot('hist', bins=100)