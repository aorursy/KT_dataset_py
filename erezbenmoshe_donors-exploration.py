# Reading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plots
import matplotlib.pyplot as plt
%matplotlib inline
color = sns.color_palette()

import os
print(os.listdir("../input"))
Donors = pd.read_csv('../input/Donors.csv', nrows=100000)
Donations = pd.read_csv('../input/Donations.csv', nrows=100000)
Projects = pd.read_csv('../input/Projects.csv', nrows=100000)
Schools = pd.read_csv('../input/Schools.csv', nrows=100000)
Teachers = pd.read_csv('../input/Teachers.csv', nrows=100000)
Resources = pd.read_csv('../input/Resources.csv', nrows=100000)
Resources.head()
Aggregations = {'Project ID':"count",
                'Donation ID':"count",
                'Donation Amount':sum}
DonorAgg = Donations.groupby('Donor ID').agg(Aggregations)
DonorAgg.head()
# sns.distplot(DonorAgg['Project ID'], kde=False, rug=True);