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
survey_schema = pd.read_csv('../input/survey_results_schema.csv')



survey_schema.head()
survey_results = pd.read_csv('../input/survey_results_public.csv')



survey_results.head()
nans = survey_results.isna().sum(axis=1)

nans.shape
survey_results.drop(nans.index[nans >= 30], inplace=True)



survey_results.shape
df = survey_results[survey_results['Student'] == 'No'][['DevType', 'JobSatisfaction']]

df.shape
unique_devs = pd.unique(df['DevType'])

unique_devs
unique_dev_types = []



for dev_type in unique_devs.astype('str'):

    dev_types = dev_type.split(';')

    unique_dev_types.extend(dev_types)



unique_dev_types = list(set(unique_dev_types))

len(unique_dev_types), len(unique_devs)
unique_dev_types
unique_dev_types = [dev_type for dev_type in unique_dev_types if dev_type != 'nan']

unique_dev_types
unique_job_satisfactions = pd.unique(df['JobSatisfaction'])



unique_job_satisfactions
unique_job_satisfactions = unique_job_satisfactions[:-1]



unique_job_satisfactions
replace_dict = {

    'Extremely satisfied': 6,

    'Moderately satisfied': 5,

    'Slightly satisfied': 4,

    'Neither satisfied nor dissatisfied': 3,

    'Slightly dissatisfied':2,

    'Moderately dissatisfied': 1,

    'Extremely dissatisfied': 0

}



df = df[df['JobSatisfaction'] != np.nan].replace(replace_dict)

df.head()
df = df[df['DevType'] != np.nan]

df['DevType'] = df['DevType'].astype('str')
import matplotlib.pyplot as plt



%matplotlib inline
y_ticks = {i: val for val, i in replace_dict.items()}



for i, dev_type in enumerate(unique_dev_types):

    if i % 4 == 0:

        plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, i % 4 + 1)

    plt.title('DevType: {} Job Satisfaction'.format(dev_type))

    tmp = df[df['DevType'].str.contains(dev_type)]['JobSatisfaction']

    tmp.hist(label=dev_type, orientation='horizontal')

    plt.yticks(np.arange(7), [y_ticks[i] for i in range(7)])

    if i % 4 == 3:

        plt.tight_layout()

        #plt.legend()

        plt.show()

#plt.legend()