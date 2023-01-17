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
nyc_checkins = pd.read_csv('../input/dataset_TSMC2014_NYC.csv')

tky_checkins = pd.read_csv('../input/dataset_TSMC2014_TKY.csv')



nyc_checkins.info()

nyc_checkins.describe()
nyc_unique_lat_long = nyc_checkins.filter(['latitude', 'longitude']).groupby(['latitude', 'longitude']).count().reset_index()

tky_unique_lat_long = tky_checkins.filter(['latitude', 'longitude']).groupby(['latitude', 'longitude']).count().reset_index()

nyc_unique_lat_long.to_csv('nyc_unique_lat_long.csv', header=False, index=False)

tky_unique_lat_long.to_csv('tky_unique_lat_long.csv', header=False, index=False)
mask_nyc = np.random.rand(len(nyc_checkins)) < 0.8

sample_train_nyc = nyc_checkins[mask_nyc]

sample_test_nyc = nyc_checkins[~mask_nyc]



mask_tky = np.random.rand(len(tky_checkins)) < 0.8

sample_train_tky = tky_checkins[mask_tky]

sample_test_tky = tky_checkins[~mask_tky]



sample_train_nyc.to_csv('nyc_train.csv', index=False)

sample_test_nyc.to_csv('nyc_test.csv', index=False)

sample_train_tky.to_csv('tky_train.csv', index=False)

sample_test_tky.to_csv('tky_test.csv', index=False)