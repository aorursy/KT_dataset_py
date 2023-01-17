import numpy as np 
import pandas as pd
import os
fullgar = pd.read_csv("../input/gar15-full/gar15_full.csv")
#calculate the total urban population in each 5x5
fullgar['u2rratio'] = fullgar['TOT_PU']/(fullgar['TOT_PU'] + fullgar['TOT_PR'])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(fullgar[['u2rratio','TOT_POB']])
y_kmeans = kmeans.predict(fullgar[['u2rratio','TOT_POB']])
fullgar['clusters'] = kmeans.labels_ 
fullgar[['u2rratio','TOT_POB','clusters']].head(25)
fullgar.clusters.unique()
# Health Capital Stock Per Person
fullgar['healthpp'] = ((fullgar['BED_PRV_CU'] + fullgar['BED_PUB_CU'] + fullgar['BED_PUB_CR'] + fullgar['BED_PUB_PR'])/(fullgar['BED_PRV_PU'] + fullgar['BED_PUB_PU'] + fullgar['BED_PUB_PR'] + fullgar['BED_PUB_PR'])).replace(np.nan, 0)
# Education Capital Stock Per Person
fullgar['edupp'] = ((fullgar['EDU_PRV_CU'] + fullgar['EDU_PUB_CU'] + fullgar['EDU_PUB_CR'] + fullgar['EDU_PUB_PR'])/(fullgar['EDU_PRV_PU'] + fullgar['EDU_PUB_PU'] + fullgar['EDU_PUB_PR'] + fullgar['EDU_PUB_PR'])).replace(np.nan, 0)
# Employment: Agriculture Capital Stock Per Person
fullgar['empagripp'] = ((fullgar['EMP_AGR_CU'] + fullgar['EMP_AGR_CR'])/ (fullgar['EMP_AGR_PU'] + fullgar['EMP_AGR_PR'])).replace(np.nan, 0)
# Employment: Governmant Capital Stock Per Person
fullgar['empgovpp'] = ((fullgar['EMP_GOV_CU'] + fullgar['EMP_GOV_CR'])/ (fullgar['EMP_GOV_PU'] + fullgar['EMP_GOV_PR'])).replace(np.nan, 0)
# Employment: Industry Capital Stock Per Person
fullgar['empindpp'] = ((fullgar['EMP_IND_CU'] + fullgar['EMP_IND_CR'])/ (fullgar['EMP_IND_PU'] + fullgar['EMP_IND_PR'])).replace(np.nan, 0)
# Employment: ServiceCapital Stock Per Person
fullgar['empserpp'] = ((fullgar['EMP_SER_CU'] + fullgar['EMP_SER_CR'])/ (fullgar['EMP_SER_PU'] + fullgar['EMP_SER_PR'])).replace(np.nan, 0)
# The percentage of the population that is middle class
fullgar['middleclass'] = ((fullgar['IC_MHG_PR'] + fullgar['IC_MHG_PU'] + fullgar['IC_MLW_PR'] + fullgar['IC_MLW_PU']) / (fullgar['IC_MHG_PR'] + fullgar['IC_MHG_PU'] + fullgar['IC_LOW_PU'] + fullgar['IC_LOW_PR'] + fullgar['IC_HIGH_PU'] + fullgar['IC_HIGH_PR'] + fullgar['IC_MLW_PR'] + fullgar['IC_MLW_PU'])).replace(np.nan, 0)
def loop(df):
    '''
    looping through and summing the columns
    '''
    df_list = []
    for i in np.arange(df['clusters'].max()):
        cluster_df = df.loc[df['clusters'] == i]      
        df['healthppind'] = ((df['healthpp'] - df['healthpp'].min())/(df['healthpp'].max() - df['healthpp'].min())).replace(np.nan,0)
        df['eduppind'] = ((df['edupp'] - df['edupp'].min())/(df['edupp'].max() - df['edupp'].min())).replace(np.nan,0)
        df['empagripp'] = ((df['empagripp'] - df['empagripp'].min())/(df['empagripp'].max() - df['empagripp'].min())).replace(np.nan,0)
        df['empgovppind'] = ((df['empgovpp'] - df['empgovpp'].min())/(df['empgovpp'].max() - df['empgovpp'].min())).replace(np.nan,0)
        df['empindppind'] = ((df['empindpp'] - df['empindpp'].min())/(df['empindpp'].max() - df['empindpp'].min()) ).replace(np.nan,0)
        df['empserppind'] = ((df['empserpp'] - df['empserpp'].min())/(df['empserpp'].max() - df['empserpp'].min())).replace(np.nan,0)
        df['middleclassind'] = ((df['middleclass'] - df['middleclass'].min())/(df['middleclass'].max() - df['middleclass'].min())).replace(np.nan,0)
    
        #index created off arithmetic mean not geometric mean
        df['lmpi'] = ((df['healthppind']+df['eduppind']+df['empagripp']+df['empgovppind']+df['empindppind']+df['empserppind']+df['middleclassind'])/7).replace(np.nan,0)
        
        df_list.append(cluster_df)
        
    '''
    Concatenating the frames
    '''
    final_df = pd.concat(df_list)
    
    return(final_df)

output = loop(fullgar)
output.head()
final = output[['ID_5X', 'ISO3','lmpi']].to_csv('lmpi.csv')