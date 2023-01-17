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



#Import the required libraries:

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



print('Libraries Imported')

import os

print(os.listdir('../input'))
#Import the data set:

haberman = pd.read_csv('../input/haberman.csv')
#Describe the overall data set:



haberman.info()
haberman.columns
haberman.columns = ['PatientAge','OperationYear','PositiveAxillaryNodes','SurvivalStatus']
haberman.describe().transpose()
#Lets segragate the data between patients who survived more than 5 years and who could not.



surviveMoreThan5Years = haberman[haberman["SurvivalStatus"] == 1]

surviveLessThan5Years = haberman[haberman["SurvivalStatus"] == 2]
#Describe "surviveMoreThan5Years":



surviveMoreThan5Years.describe().transpose()
np.percentile(surviveMoreThan5Years["PositiveAxillaryNodes"],90)
#Describe "surviveLessThan5Years":



surviveLessThan5Years.describe().transpose()
np.percentile(surviveLessThan5Years["PositiveAxillaryNodes"],90)
#Lets run a pair plot to see if there are any combination of features that can standout as an influencer of the survival probabilty.



sns.pairplot(haberman,hue="SurvivalStatus")
#Since the feature Positive Axillary Nodes seems to be an influencer of the survival probability, lets segragate whole data into 3 parts. 

#Patients in the 1st data set will have Positive Axillary Nodes upto 3.

#Patients in the 2nd data set will have Positive Axillary Nodes more than 3 but less than 8.

#Patients in the 3rd data set will have Positive Axillary Nodes more than 8.



FirstDS = haberman[haberman["PositiveAxillaryNodes"] <= 3]

SecondDS = haberman[(haberman["PositiveAxillaryNodes"] > 3) & (haberman["PositiveAxillaryNodes"] <= 8)]

ThirdDS = haberman[haberman["PositiveAxillaryNodes"] > 8]
#From the newly created 3 data sets, lets find out the probability of the patient surviving more than 5 years. 

#The probabilty will be calculated using the CDF and PDF charts



plt.figure(figsize=(20,10))



count,bin_edges = np.histogram(FirstDS["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



count,bin_edges = np.histogram(SecondDS["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



count,bin_edges = np.histogram(ThirdDS["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

#We have seen, from the overall data, that Patient Age does not affect the survival probability. 

#However, lets check if Patient Age does have an impact when all the patients are segragated into the 3 data set created above.

#For each newly created data set, we classify the patient to be either more than 50 years of age or not.



FirstDS_AgeLessThan50 = FirstDS[(FirstDS["PatientAge"] <= 50)]

FirstDS_AgeMoreThan50 = FirstDS[(FirstDS["PatientAge"] > 50)]



SecondDS_AgeLessThan50 = SecondDS[(SecondDS["PatientAge"] <= 50)]

SecondDS_AgeMoreThan50 = SecondDS[(SecondDS["PatientAge"] > 50)]



ThirdDS_AgeLessThan50 = ThirdDS[(ThirdDS["PatientAge"] <= 50)]

ThirdDS_AgeMoreThan50 = ThirdDS[(ThirdDS["PatientAge"] > 50)]
#For the 1st data set, when patients has less than 3 Positive Axillary Nodes, does age plays a factor?

#The probabilty will be calculated using the CDF and PDF charts



plt.figure(figsize=(20,10))

count,bin_edges = np.histogram(FirstDS_AgeLessThan50["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



count,bin_edges = np.histogram(FirstDS_AgeMoreThan50["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

#For the 2nd data set, when patients has Positive Axillary Nodes between 3 and 8, does age plays a factor?

#The probabilty will be calculated using the CDF and PDF charts



plt.figure(figsize=(20,10))

count,bin_edges = np.histogram(SecondDS_AgeLessThan50["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



count,bin_edges = np.histogram(SecondDS_AgeMoreThan50["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

#For the 3rd data set, when patients has Positive Axillary Nodes more than 8, does age plays a factor?

#The probabilty will be calculated using the CDF and PDF charts



plt.figure(figsize=(20,10))

count,bin_edges = np.histogram(ThirdDS_AgeLessThan50["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)



count,bin_edges = np.histogram(ThirdDS_AgeMoreThan50["SurvivalStatus"] == 1,bins=10,density=True)

pdf = count/sum(count)

cdf = np.cumsum(pdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)