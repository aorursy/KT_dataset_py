import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import deque 
datasets=deque()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #1st test of datasets containing health in filenames
        if "health" in filename:
#             print(os.path.join(dirname, filename))
            datasets.append(pd.read_csv(os.path.join(dirname, filename)))
datasets[1].columns
#No. of address present in city ALAMOSA
#If we can get the data of age groups may be we can add more info for the populations which are at higher risk!
# test_data.groupby('city').count()[['state','address']]
#It seems this data contains the health services present in cities of CO State
#Let's tag this data and move to second data set 
test_data_eda=datasets[1]


max_operating_services=test_data_eda.groupby('city').count()['operating'].reset_index()
max_operating_services[max_operating_services['operating']==max_operating_services['operating'].max()]

plt.plot(max_operating_services.sort_values(by='operating').tail()['city'],max_operating_services.sort_values(by='operating').tail()['operating'])
plt.xticks(rotation='vertical')
plt.show()

