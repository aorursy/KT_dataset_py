import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
bands = {

    

    'bandsNames': 

        ['Light', 'Medium', 'Heavy', 'X-Heavy', 'XX-Heavy', 'Pro-Light', 'Pro-Medium', 'Pro-Heavy', 'green15', 'blue35'], 

    'bandsResistance': 

        [2.1,       5.5,      6.2,      8.8,      9.2,          15.2,       11.6,         19,         12.4,     20]

    

        }
bandsDF = pd.DataFrame(data=bands)
bandsDF
bandsDF.sum(axis=0) 
import itertools
#combining and creating a DF - bandName

bandCombination = pd.DataFrame(itertools.permutations(bandsDF['bandsNames'],2))
bandCombination.head(10)
bandCombination


#creating the desired column - bandName

bandCombination['bandsNames'] = bandCombination[0] + ' + ' + bandCombination[1] #+  ' + '  + bandCombination[2]
bandCombination.head(10)
#combining and creating a DF - bandResistance

bandCombinationResistance = pd.DataFrame(itertools.permutations(bandsDF['bandsResistance'],2))
#creating the desired column - bandName

bandCombination['bandsResistance'] = bandCombinationResistance[0] + bandCombinationResistance[1] #+  ' + '  + bandCombination[2]
bandCombination
bandCombinationClean = bandCombination.drop(columns=[0,1])
bandCombinationClean.head(10)
bandCombinationClean = bandCombinationClean.append(bandsDF)
bandCombinationClean.sort_values(by=['bandsResistance'], inplace=True)
bandCombinationClean.head(10)
bandsCombi = bandCombinationClean.drop_duplicates(subset=['bandsResistance'])
bandsCombi.head(10)
bandsCombi.to_csv('bandsCombinations.csv', index=False, float_format='%.1f')