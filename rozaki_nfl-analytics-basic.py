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
inj=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
inj.head(3)
inj['DM_SUM']=inj['DM_M1']+inj['DM_M7']*6+inj['DM_M28']*21+inj['DM_M42']*14

inj.drop(columns=['DM_M1','DM_M7','DM_M28','DM_M42'])
inj.groupby(['Surface'])['DM_SUM'].describe()
inj.groupby(['Surface','BodyPart'])['DM_SUM'].describe()
wall=inj.groupby(['Surface','BodyPart'])['DM_SUM'].describe()

# Cut the parameters which number of samples is lower than 5. 

wall.loc[wall['count']>5]
ply=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')
ply.head(3)
inj_ply=pd.merge(inj,ply)
inj_ply.groupby(['Surface','BodyPart','PlayType'])['DM_SUM'].describe()
pos=inj_ply.groupby(['Surface','PositionGroup'])['DM_SUM'].describe()

pos.loc[pos['count']>5]
trc=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
trc.head(2)