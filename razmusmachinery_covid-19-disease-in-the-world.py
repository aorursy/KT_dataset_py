# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
Spain= pd.read_csv("../input/covid19-in-spain/nacional_covid19.csv")
USA=pd.read_csv("../input/covid19-in-usa/us_covid19_daily.csv")
Turkey=pd.read_csv("../input/covid19-in-turkey/covid_19_data_tr.csv")
italy=pd.read_csv("../input/coronavirusdataset-france/contagioitalia.csv")
Spain["day"]= np.arange(1,46)
Usa= USA[::-1]
Usa["day"]= np.arange(1,44)
Turkey["day"]= np.arange(1,33)
italy["day"]= np.arange(1,49)
Usa.head()

plt.plot(Spain.day,Spain.casos,color="Yellow",label="Spain")
plt.plot(Usa.day,Usa.positive,color="Blue",label="USA")
plt.plot(Turkey.day,Turkey.Confirmed,color="Red",label="Turkey")
plt.plot(italy.day,italy.TotalPositiveCases,color="Green",label="Italy")
plt.legend()
plt.xlabel("Days begin first case")
plt.ylabel("Total cases")
plt.show()

#Spain 47 million people
#USA 328 million people
#Turkey 82 million people
#İtaly 60 million people

Spain["ReducedCase"]=(Spain["casos"]/47)*100
Usa["ReducedCase"] = (Usa["positive"]/328)*100
Turkey["ReducedCase"] = (Turkey["Confirmed"]/82)*100
italy["ReducedCase"]= (italy["TotalPositiveCases"]/60)*100

plt.plot(Spain.day,Spain.ReducedCase,color="Yellow",label="Spain")
plt.plot(Usa.day,Usa.ReducedCase,color="Blue",label="USA")
plt.plot(Turkey.day,Turkey.ReducedCase,color="Red",label="Turkey")
plt.plot(italy.day,italy.ReducedCase,color="Green",label="Italy")
plt.legend()
plt.xlabel("Days begin first case")
plt.ylabel("Total Reduced cases")
plt.show()

