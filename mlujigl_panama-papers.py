import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline

entities = pd.read_csv('../input/Entities.csv')

addresses = pd.read_csv('../input/Addresses.csv')

intermediaries = pd.read_csv('../input/Intermediaries.csv')

officers = pd.read_csv('../input/Officers.csv')

edges = pd.read_csv('../input/all_edges.csv')
entities.head(3)
intermediaries.head(3)
officers.head(3)
edges.head(3)
status = intermediaries['status'].value_counts()

status
stat = 'ACTIVE'

stat_country= 'Brazil'



mask1 = intermediaries['status'].str.contains(stat)

mask2 = intermediaries['countries'].str.contains(stat_country)



stage = intermediaries[mask1 & mask2]
#stage
status_filtered = intermediaries[intermediaries['status'] == 'DELINQUENT']

status_filtered.head(3)
entities.columns
entities.shape
C = entities['countries'].value_counts()

C[:10]
C[:10].plot(kind='bar')
provider = entities['service_provider'].value_counts()
provider[:10].plot(kind='pie',autopct='%1.0f%%')
Bra = entities.loc[entities['countries']=='Brazil'].copy()

Bra['service_provider'].value_counts().plot(kind='pie',autopct='%1.0f%%')