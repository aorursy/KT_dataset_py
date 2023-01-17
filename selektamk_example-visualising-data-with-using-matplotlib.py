# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')

print('Start Loading CSV files:')

trainData = pd.read_csv('../input/act_train.csv', parse_dates=['date'], date_parser=dateparse)

print('act_train.csv loaded')




    

colors = ['r','b']

def plotFeatureCount(x, featureColumnName, targetColumnName):

    uniqueTargets = x[targetColumnName].unique()

    tbl = x[[featureColumnName, targetColumnName]]

    width = 0.35

    for v in uniqueTargets:

        tblV = tbl[tbl[targetColumnName] == v]

        group = tblV.groupby([featureColumnName]).agg(['count'])

        bottom = np.zeros(len(group))

        for key,gr in group:

            plt.bar(group[key].iloc[:,0].axes[0],group[key].iloc[:,0].values, width, 

                    color=colors[v], bottom=bottom, label=v)

            bottom = bottom + group[key].iloc[:,0].values



    

    plt.ylabel('Counts')

    plt.xlabel(featureColumnName)

    plt.title('Counts by feature ' + featureColumnName)

    #plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))

    #plt.yticks(np.arange(0, 81, 10))

    plt.legend()



    plt.show()
trainData['char_1'] = trainData.apply(lambda row : int(str(row['char_1']).replace('nan', '-1').replace('type ', '')), axis=1)
plotFeatureCount(trainData, 'char_1','outcome')