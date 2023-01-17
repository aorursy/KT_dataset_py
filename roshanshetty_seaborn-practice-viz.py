import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
import pandas as pd

US_Accidents_May19 = pd.read_csv("../input/us-accidents/US_Accidents_May19.csv")
US_Accidents_May19.head()
for i in range(US_Accidents_May19['Severity'].min(),US_Accidents_May19['Severity'].max()+1):

    sns.distplot(a=US_Accidents_May19.loc[US_Accidents_May19['Severity'] == i, 'Wind_Chill(F)'].dropna(), label=('Severity' + str(i)), kde=False)

plt.legend()