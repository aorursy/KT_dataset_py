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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime
import pandas as pd

case = pd.read_csv("../input/coronavirusdataset/case.csv")

patient = pd.read_csv("../input/coronavirusdataset/patient.csv")

route = pd.read_csv("../input/coronavirusdataset/route.csv")

time = pd.read_csv("../input/coronavirusdataset/time.csv")

trend = pd.read_csv("../input/coronavirusdataset/trend.csv")
patient.head()
patient.rename(columns = {'state': 'current_condition'}, inplace = True)

patient['current_condition'] = patient.current_condition.str.upper()



patient.rename(columns = {'sex': 'gender'}, inplace = True)

patient['gender'] = patient.gender.str.upper()







patient.head()
date = patient.groupby('confirmed_date').count()

date.head()
date_confirmed = date['patient_id']

date_confirmed.head()
import matplotlib

matplotlib.rcParams['figure.figsize'] = (16,8)

date_confirmed.plot()
date_confirmed.plot(marker = '.')

plt.grid(which = 'both')
patient.groupby('gender').count()
patient1 = patient.groupby('country').count()

patient1
exp_vals = patient1["patient_id"].values.tolist()

exp_vals
exp_labels =["china", "korea", "mongolia"]

plt.axis("equal")

plt.pie(exp_vals,labels= exp_labels, radius = 1.0, autopct = '%0.1f%%', explode = [1,0.1,3], startangle = 180)

plt.show
patient.infection_reason.unique()
transmission_reason = patient.groupby('infection_reason').count()

transmission_reason
reason = transmission_reason['patient_id']

reason
reason.plot.bar()
affected_region = patient.groupby('region').count()

affected_region
affected_region1 = affected_region['patient_id']

affected_region1
affected_region1.plot.bar()
condition = patient.groupby("current_condition").count()

condition 
current_condition =condition['patient_id']

current_condition
current_condition.plot.bar()