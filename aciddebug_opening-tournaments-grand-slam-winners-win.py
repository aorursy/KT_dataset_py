import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/ATP Dataset_2012-01 to 2017-07_Int_V4.csv')
opentours=("BrisbaneInternational","ChennaiOpen","QatarExxonMobilOpen")
dataopen=data[data['Tournament'].isin(opentours)]
dataopenwin=dataopen[dataopen['Round']=="TheFinal"]
dataopenwin.Winner
slamtours=("AustralianOpen","FrenchOpen","Wimbledon","USOpen")
dataslam=data[data['Tournament'].isin(slamtours)]
dataslamwin=dataslam[dataslam['Round']=="TheFinal"]
dataslamwin.Winner
dataopenslam=dataopenwin[dataopenwin.Winner.isin(dataslamwin.Winner)]
sns.countplot(dataopenslam.Winner)
sns.countplot(dataopenslam.Tournament)