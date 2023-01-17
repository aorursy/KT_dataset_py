# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
all_journeys = []
for i in range(500) :
    random_journey = [0]
    for x in range(50) :
        station = random_journey[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            station = max(0, station - 1)
        elif dice <= 5:
            station =  station + 1
        else:
            station =  station + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            station = 0
        random_journey.append(station)
    all_journeys.append(random_journey)
# Create and plot np_aw_t
#path of all journeys
np_aw_t = np.transpose(np.array(all_journeys))
plt.plot(np_aw_t)
# Select last row from np_aw_t: ends
ends = np_aw_t[-1,:]
# Plot histogram of ends, display plot
plt.hist(ends,bins=10)
plt.show()
print("The probability of crossing minimum 30 stations :",str(np.mean(ends>=30)))
print("Chances that you'll win this bet :",str(np.mean(ends>=30)))