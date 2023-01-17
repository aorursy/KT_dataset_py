import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
patientInfo = pd.read_csv("/kaggle/input/coronavirusdataset/PatientInfo.csv")

patientInfo.head()
patientInfo.info()
patientInfo.describe()
covidData = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covidData.head()
covidData.info()