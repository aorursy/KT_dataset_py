import subprocess
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import requests
import glob, os
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import mannwhitneyu
# Retreive Crime in Los Angeles data as chunked DF
process_chunksize = 100000
mainCSV = '../input/Crime_Data_2010_2017.csv'
mainDF = pd.read_csv(mainCSV, chunksize=process_chunksize)

# Retreive and clean tract census data
census_api_url='https://api.census.gov/data/2017/acs/acs5?key=65c608668f59dc2272076d077091ba8cd98a6286&get=NAME,B00001_001E,B19019_001E,B19083_001E&for=TRACT:*&in=state:06+COUNTY:037'
responseCensus = requests.get(census_api_url, headers={'Content-Type': 'application/json'})
textCensus=json.loads(responseCensus.content)
tractDF = pd.DataFrame(textCensus)
print('Shape of census tract file: ' + str(tractDF.shape))

tractDF.rename(columns = tractDF.iloc[0], inplace=True)
tractDF = tractDF.iloc[1:]
tractDF.rename(index=str,columns={'B00001_001E':'Pop','B19019_001E':'Income','B19083_001E':'Gini','tract':'Tract'}, inplace=True)
tractDF.set_index('Tract', inplace=True)
tractDF.drop(['NAME', 'state', 'county'],inplace=True, axis=1)
tractDF=tractDF[['Pop','Income','Gini']].apply(pd.to_numeric, errors='coerce')
print(tractDF.info())
print(tractDF.head(8))
#Define tract finding function
def getTract(lonlat):
    tract = -999
    api_url='https://geo.fcc.gov/api/census/area?lat='+str(lonlat[0])+'&lon='+str(lonlat[1])+'&format=json'
    headers = {'Content-Type': 'application/json'}
    response = requests.get(api_url, headers=headers)
    try:
        tract=json.loads(response.content.decode('utf-8'))['results'][0]['block_fips'][5:11]
    except:
        print('No Tract found from FCC geo API')
    return tract
df = pd.DataFrame()
itld=0
itlu=1
for mainDF_chunk in mainDF:
    nLinesMainCSVd=itld*process_chunksize
    nLinesMainCSVu=itlu*process_chunksize
    print('New group from main DF, lines '+str(nLinesMainCSVd) + ' to ' + str(nLinesMainCSVu) )
    
    # Pre Filtering to save memory
    mainDF_chunk.dropna(subset=['Crime Code Description'],inplace=True)
    mainDF_chunk= mainDF_chunk[mainDF_chunk['Crime Code Description'].str.contains('ROBBERY|THEFT|STOLEN|BURGLARY|COUNTERFEIT')]
    mainDF_chunk= mainDF_chunk[mainDF_chunk['Date Occurred'] > '12/31/2012']

    # Cleaning main
    mainDF_chunk.rename(columns=lambda x: x.replace(' ',''), inplace=True)
    mainDF_chunk.dropna(subset=['Location'],inplace=True)
    mainDF_chunk['Location']=mainDF_chunk['Location'].map(lambda x: eval(str(x)))
    mainDF_chunk[['DateReported','DateOccurred','TimeOccurred']].apply(pd.to_datetime,errors='coerce')
    mainDF_chunk[['DRNumber','AreaID','ReportingDistrict','CrimeCode','VictimAge','PremiseCode','WeaponUsedCode','CrimeCode1']].apply(pd.to_numeric)
    mainDF_chunk.drop(['CrimeCode2','CrimeCode3','CrimeCode4'],inplace=True, axis=1)
    print('Dimension of main DF group:' + str(mainDF_chunk.shape))
    
    # Add Tract from FCC geo AND violent bool
    mainDF_chunk['Violent']=mainDF_chunk['CrimeCodeDescription'].str.contains('ROBBERY')
    mainDF_chunk['Tract']=mainDF_chunk.Location.map(lambda x: getTract(x))

    df_group = pd.merge(mainDF_chunk,tractDF, how='left', on='Tract', sort=True)
    #print(df_group.head(5))
    
    df = df.append(df_group,sort=True)
    
    #print(df.head(3))
    
    itlu+=1
    itld+=1
df=df[df['Income']>0.0]
df=df[df['Gini']>0.0]
df.reset_index(inplace=True)

print(df.shape)
print(df.info())

viSetGini = np.array(df[df['Violent']==1]['Gini'])
nonviSetGini = np.array(df[df['Violent']!=1]['Gini'])

viSetIncome = np.array(df[df['Violent']==1]['Income'])
nonviSetIncome = np.array(df[df['Violent']!=1]['Income'])
#Plots
fGini, fGiniPlots = plt.subplots(3,sharex=True)
fGiniPlots[0].hist([viSetGini,nonviSetGini], bins=20, range=[0.2,0.8], stacked=True, color=['r','b'])
fGiniPlots[0].set_ylabel('Total Incidents')
fGiniPlots[0].legend(('Robbery', 'Nonviolent Theft'),loc='best')

fGiniPlots[1].hist(viSetGini, bins=20, range=[0.2,0.8], color='r')
fGiniPlots[1].set_ylabel('Robbery')
fGiniPlots[2].hist(nonviSetGini, bins=20, range=[0.2,0.8], color='b')
fGiniPlots[2].set_xlabel('Gini Coefficient at Crime Location Census Tract')
fGiniPlots[2].set_ylabel('Non-violent Theft')

fIncome, fIncomePlots = plt.subplots(3,sharex=True)
fIncomePlots[0].hist([viSetIncome,nonviSetIncome], bins=15, range=[0.,200000.], stacked=True, color=['r','b'])
fIncomePlots[0].set_ylabel('Total Incidents')
fIncomePlots[1].hist(viSetIncome, bins=15, range=[0.,200000.], color='r')
fIncomePlots[1].set_ylabel('Robbery')

fIncomePlots[2].hist(nonviSetIncome, bins=15, range=[0.,200000.], color='b')
fIncomePlots[2].set_xlabel('Median Household Income (12 mo) at Crime Location Census Tract')
fIncomePlots[2].set_ylabel('Non-violent Theft')

#ecdf comparison
ecdf, ecdfPlots = plt.subplots(2)
ecdf.subplots_adjust(hspace=0.45)

# Gini
ecdfViGini = ECDF(viSetGini)
ecdfNonviGini = ECDF(nonviSetGini)
ecdfPlots[0].plot(ecdfViGini.x,ecdfViGini.y,marker='.', linestyle='none',
                 color='red', alpha=0.5)
ecdfPlots[0].plot(ecdfNonviGini.x,ecdfNonviGini.y,marker='.', linestyle='none',
                 color='blue', alpha=0.5)
ecdfPlots[0].set_xlabel('Tract Gini Coefficient')
ecdfPlots[0].legend(('Robbery', 'Nonviolent Theft'),
           loc='best')

# Income
ecdfViIncome = ECDF(viSetIncome)
ecdfNonviIncome = ECDF(nonviSetIncome)
ecdfPlots[1].plot(ecdfViIncome.x,ecdfViIncome.y,marker='.', linestyle='none',
                 color='red', alpha=0.5)
ecdfPlots[1].plot(ecdfNonviIncome.x,ecdfNonviIncome.y,marker='.', linestyle='none',
                 color='blue', alpha=0.5)
ecdfPlots[1].set_xlabel('Tract Median Income')
ecdfPlots[1].legend(('Robbery', 'Nonviolent Theft'), loc='best')

def getMWU(values1,values2):
    return mannwhitneyu(values1,values2,alternative='less').statistic
def getMWP(values1,values2):
    return mannwhitneyu(values1,values2,alternative='less').pvalue

# Mann-Whitney Test (non-parametric test)
nbnGini = len(viSetGini)*len(nonviSetGini)
nbnIncome = len(viSetIncome)*len(nonviSetIncome)
MWGiniU = getMWU(nonviSetGini,viSetGini)
MWGiniP = getMWP(nonviSetGini,viSetGini)
MWIncomeU = getMWU(viSetIncome,nonviSetIncome)
MWIncomeP = getMWP(viSetIncome,nonviSetIncome)

print('length violent: ' + str(len(viSetGini)))
print('length nonviolent: ' + str(len(nonviSetGini)))
print("Mann-Whitney test for Gini, statistic U: " + str(MWGiniU) + ",  p-value: " + str(MWGiniP) + ",  n1 x n2: " + str(nbnGini))
print("Mann-Whitney test for Income, statistic U: " + str(MWIncomeU) + ",  p-value: " + str(MWIncomeP) + ",  n1 x n2: " + str(nbnIncome))
# Permutation test with test statistic
def PermutationTestStat(values1,values2,nTrials,testStatFunc):
    trials = np.empty(nTrials)
    concatValues = np.concatenate((values1,values2))
    for i in range(0,nTrials):
        permutedValues = np.random.permutation(concatValues)
        sampledValues1 = permutedValues[:len(values1)]
        sampledValues2 = permutedValues[len(values1):]
        trials[i] = testStatFunc(sampledValues1,sampledValues2)
    return trials

#p value mean diff comparison
testStat, testStatPlots = plt.subplots(2)
testStat.subplots_adjust(hspace=0.3)

trialsGini = PermutationTestStat(nonviSetGini,viSetGini,5000,getMWU)
measuredTestStatGini = getMWU(nonviSetGini,viSetGini)
pValGini = (trialsGini < measuredTestStatGini).sum()/len(trialsGini)
testStatPlots[0].hist(trialsGini,bins=30)
testStatPlots[0].axvline(measuredTestStatGini, color='red', linewidth=1)
testStatPlots[0].text(.07,.7,'p-value: '+ str(pValGini), {'color': 'r', 'fontsize': 12}, transform=testStatPlots[0].transAxes)

trialsIncome = PermutationTestStat(viSetIncome,nonviSetIncome,5000,getMWU)
measuredTestStatIncome = getMWU(viSetIncome,nonviSetIncome)
pValIncome = (trialsIncome < measuredTestStatIncome).sum()/len(trialsIncome)
testStatPlots[1].hist(trialsIncome,bins=30)
testStatPlots[1].axvline(measuredTestStatIncome, color='red', linewidth=1)
testStatPlots[1].text(.07,.7,'p-value: '+ str(pValIncome), {'color': 'r', 'fontsize': 12}, transform=testStatPlots[1].transAxes)
testStatPlots[1].set_xlabel('Mann-Whitney U Distribution')
plt.show()
