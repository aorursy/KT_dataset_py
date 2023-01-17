# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from scipy.stats import skew



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/kc_house_data.csv')

df.info()
df.describe() 

# sqft_living seems to be a very 'histogrammable' feature
fig,ax = plt.subplots(figsize=(12,8))

ax.hist(df['sqft_living'],bins=25,color='b',alpha=0.75)

plt.title('Sqft_living distribution');

# Note: The histogram is skewed
skew(df['sqft_living'])

# Note the skew, since the extremums of the distribution are very large, compared to the mean. 

# The max value of 13540 is ~ 12 standard deviations from the mean
# For a positive skew, usually a log transformation works

df['sqft_living'] = np.log1p(df['sqft_living'])

fig,ax = plt.subplots(figsize=(12,8))

ax.hist(df['sqft_living'],bins=25,color='b',alpha=0.75)

plt.title('Sqft_living distribution (Log)');
skew(df['sqft_living'])

# So we see that the skew can be much reduced by using a log transformation of the variable