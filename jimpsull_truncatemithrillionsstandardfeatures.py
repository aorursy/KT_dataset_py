# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#this list was arrived at in the following way:
#ran cesium features + metadata through lgbm with trouble classes only
#csFeatsToAdd=importances[importances['fold']==5].nlargest(40,'mean_gain').loc[:,'feature'].unique()
#will try to remove redundancies
#it turns out that 32 are not redundant (although possible collinearities where same basic feature has different name)

cesiumFeaturesToConsider=['mjd_diff_det', 'distmod',
       'flux_by_flux_ratio_sq__longest_strike_above_mean',
       '__max_slope___2_', '__skew___4_',
       'flux__longest_strike_above_mean',
       '__median_absolute_deviation___2_', '__max_slope___3_',
       '__freq_varrat___3_', '__percent_amplitude___3_',
       '__percent_difference_flux_percentile___5_', '__std___5_',
       '__percent_amplitude___5_', '__median_absolute_deviation___1_',
       '__freq2_rel_phase2___2_', 'hostgal_photoz',
       '__freq_y_offset___0_', 'hostgal_photoz_certain',
       '__stetson_j___5_', '__freq_varrat___1_',
       '__qso_log_chi2_qsonu___0_', '__amplitude___2_',
       '__percent_difference_flux_percentile___2_', '__amplitude___0_',
       '__freq_varrat___5_', '__skew___5_', '__freq_varrat___2_',
       '__freq3_freq___2_', '__freq1_rel_phase4___5_',
       'flux__mean_change', '__flux_percentile_ratio_mid80___5_',
       '__percent_amplitude___2_', '__amplitude___5_',
       '__median_absolute_deviation___5_', '__freq3_freq___3_',
       '__qso_log_chi2_qsonu___5_', 'hostgal_photoz_err',
       '__freq1_rel_phase3___5_', '__freq2_rel_phase2___4_',
       '__freq2_rel_phase3___4_','object_id']
def convertNames(testName):
    trainName=""
    
    lenTest=len(testName)
    for charindex in range(lenTest):
        char=testName[charindex]
        #print(char)
        if char in [')','(', ' ',',',"'"]:
            trainName=trainName + '_'
            #print('changed')
        else:
            trainName+=char
            
    return trainName

            
testName="('percent_amplitude', 0)"
trainName=convertNames(testName)
print(trainName)
#for chunk in pd.read_csv(fn, chunksize=10)
fn='../input/plasticc-features/single_output_test_ts_features.csv'
for chunk in pd.read_csv(fn, chunksize=10):
    testCols=chunk.columns
    break
nameDict={}
colsToGrab=[]
for testCol in testCols:
    trainName=convertNames(testCol)
    if trainName in cesiumFeaturesToConsider:
        colsToGrab.append(testCol)
        nameDict[testCol]=trainName
        
#colsToGrab
    
testCesiumDf = pd.read_csv(fn, skipinitialspace=True, usecols=colsToGrab)
testCesiumDf.shape
#df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
testCesiumDf = testCesiumDf.rename(columns=nameDict)
testCesiumDf.to_csv('reducedCesiumTestRevB.csv', index=False)