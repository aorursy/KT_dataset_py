!df -h

!cat /proc/cpuinfo
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



tournament = pd.read_csv('../input/numerai_datasets/numerai_tournament_data.csv')

training = pd.read_csv('../input/numerai_datasets/numerai_training_data.csv')



print(tournament.shape,training.shape)

# Any results you write to the current directory are saved as output.
print(tournament.shape,training.shape)

training.describe()
import matplotlib.pyplot as plt

%matplotlib inline
feat_list = [x for x in training.columns if x.startswith('feature')]

for n,feature in enumerate(feat_list):

    _=plt.hist(training.get(feature),bins=100,histtype='step',label=feature,alpha=0.3)

plt.title('features histogram')