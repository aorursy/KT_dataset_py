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



data = pd.read_csv('../input/fifa19/data.csv')



data
data1 = data[['Name' , 'Age' , 'Position' , 'Overall' , 'Potential' , 'International Reputation' , 'Wage' , 'Photo' ]]
data1
GK = data1[data1['Position'] == 'GK']

best_GK = pd.DataFrame.copy(GK.sort_values(by = 'Overall' , ascending=False).head(10))

best_GK
LB = data1[data1['Position'] == 'LB']

best_LB = pd.DataFrame.copy(LB.sort_values(by = 'Overall' , ascending=False).head(10))

best_LB
RB = data1[data1['Position'] == 'RB']

best_RB = pd.DataFrame.copy(RB.sort_values(by = 'Overall' , ascending=False).head(10))

best_RB
CB = data1[data1['Position'] == 'CB']

best_CB = pd.DataFrame.copy(CB.sort_values(by = 'Overall' , ascending=False).head(10))

best_CB
CDM = data1[data1['Position'] == 'CDM']

best_CDM = pd.DataFrame.copy(CDM.sort_values(by = 'Overall' , ascending=False).head(10))

best_CDM
RCM = data1[data1['Position'] == 'RCM']

best_RCM = pd.DataFrame.copy(RCM.sort_values(by = 'Overall' , ascending=False).head(10))

best_RCM
LCM = data1[data1['Position'] == 'LCM']

best_LCM = pd.DataFrame.copy(RCM.sort_values(by = 'Overall' , ascending=False).head(10))

best_LCM
RW = data1[data1['Position'] == 'RW']

best_RW = pd.DataFrame.copy(RW.sort_values(by = 'Overall' , ascending=False).head(10))

best_RW
LW = data1[data1['Position'] == 'LW']

best_LW = pd.DataFrame.copy(LW.sort_values(by = 'Overall' , ascending=False).head(10))

best_LW
ST = data1[data1['Position'] == 'ST']

best_ST = pd.DataFrame.copy(ST.sort_values(by = 'Overall' , ascending=False).head(10))

best_ST
CF = data1[data1['Position'] == 'CF']

best_CF = pd.DataFrame.copy(CF.sort_values(by = 'Overall' , ascending=False).head(10))

best_CF
RF = data1[data1['Position'] == 'RF']

best_RF = pd.DataFrame.copy(RF.sort_values(by = 'Overall' , ascending=False).head(10))

best_RF
LF = data1[data1['Position'] == 'LF']

best_LF = pd.DataFrame.copy(LF.sort_values(by = 'Overall' , ascending=False).head(10))

best_LF
world = [best_GK.head(1) , best_CB.head(1) , best_RB.head(1) , best_LB.head(1) , best_CDM.head(1) , best_RCM.head(2) , best_LCM.head(1) , best_RW.head(1) , best_LW.head(1) , best_ST.head(1)]
world_XI=pd.concat(world).reset_index(drop=True)
world_XI