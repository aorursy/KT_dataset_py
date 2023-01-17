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
#Import all the required libraries Pravin - np, pd done above

#no claim to this being 100% original - read thru others and took inspiration

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



employeesData = pd.read_csv('../input/HR_comma_sep.csv')





#correlation of leaving with other factors - pairwise correlation: comment on key ones there



allDataCorrelation = employeesData.corr()

plt.figure(figsize=(10,10))

sns.heatmap(allDataCorrelation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation for all employees and all aspects')







#setb = Significantly exceeding the bar

#etb = Exceeding the bar

#lp = Low Performancce

#mb = Meeting the bar

#Average as mb, one SDev as etb and so on..

perfEvalMean = employeesData['last_evaluation'].mean()

perfStdDev = np.std(employeesData['last_evaluation'])

employeesData['performance(standard units)'] = (employeesData['last_evaluation']- perfEvalMean)/perfStdDev





def perfRating(row):

    noOfStDevs = row['performance(standard units)']

    if noOfStDevs >0.7 and noOfStDevs <1.1:

        rating = 'etb'

    else:

        if noOfStDevs >= 1.1:

            rating = 'setb'

        else:

                if noOfStDevs < 0:

                    rating = 'lp'

                else:

                    rating = 'mb' 

    return(rating)



employeesData['performance rating'] = employeesData.apply(perfRating, axis = 1)

employeesData['left(as_string)'] = (employeesData['left'].astype(str))



#Separate data for O, SO and do pairwise correlation on those...likewise for AM/NI - do some visualization of these

highPerformingEmployees = employeesData.ix[(employeesData['performance rating']=='etb') | (employeesData['performance rating']=='setb')]



highPerfCorrelation = highPerformingEmployees.corr()

plt.figure(figsize=(10,10))

sns.heatmap(highPerfCorrelation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation for HP employees and all aspects')
employeesData.groupby('sales').mean()['satisfaction_level'].plot(kind='bar',color='b')