import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
crimedata = pd.read_csv("../input/database.csv",low_memory=False)

crimedata = crimedata.loc[(crimedata['Victim Race'] != 'Unknown') &

             (crimedata['Perpetrator Race'] != 'Unknown')]
Perpetrator_Race = crimedata["Perpetrator Race"]

Victim_Race = crimedata["Victim Race"]

plt.figure(figsize=(5,5))

sns.heatmap(pd.crosstab(Victim_Race,Perpetrator_Race), annot=True, fmt="d",

           cmap="Reds")
Victim_Race_Count = pd.DataFrame(crimedata['Victim Race'].value_counts())

Race = Victim_Race_Count*100/np.sum(Victim_Race_Count)



Perpetrator_Race_Count = pd.DataFrame(crimedata['Perpetrator Race'].value_counts())

Race.loc[:,'Perpetrator Race'] = Perpetrator_Race_Count*100/np.sum(Perpetrator_Race_Count)

Race
df = crimedata.loc[(crimedata['Victim Race'] != 'Unknown') &

             (crimedata['Perpetrator Race'] == 'Black')]

X=pd.crosstab(df['Victim Race'],df['Perpetrator Race'])*100/sum(Perpetrator_Race_Count.loc['Black',:])



df = crimedata.loc[(crimedata['Victim Race'] != 'Unknown') &

             (crimedata['Perpetrator Race'] == 'White')]

X.loc[:,'White']=(pd.crosstab(df['Victim Race'],df['Perpetrator Race'])*100

                  /sum(Perpetrator_Race_Count.loc['White',:]))



df = crimedata.loc[(crimedata['Victim Race'] != 'Unknown') &

             (crimedata['Perpetrator Race'] == 'Asian/Pacific Islander')]

X.loc[:,'Asian/Pacific Islander']=(pd.crosstab(df['Victim Race'],df['Perpetrator Race'])*100

                  /sum(Perpetrator_Race_Count.loc['Asian/Pacific Islander',:]))



df = crimedata.loc[(crimedata['Victim Race'] != 'Unknown') &

             (crimedata['Perpetrator Race'] == 'Native American/Alaska Native')]

X.loc[:,'Native American/Alaska Native']=(pd.crosstab(df['Victim Race'],df['Perpetrator Race'])*100

                  /sum(Perpetrator_Race_Count.loc['Native American/Alaska Native',:]))

sns.heatmap(X, annot = True, cmap = "Reds")

X