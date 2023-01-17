# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train0=pd.read_csv("../input/Match.csv", na_values="NA")

X0=train0.copy()

train1=pd.read_csv("../input/Player.csv", na_values="NA")

X1=train1.copy()

train2=pd.read_csv("../input/Player_Match.csv", na_values="NA")

X2=train2.copy()

train3=pd.read_csv("../input/Team.csv", na_values="NA")

X3=train3.copy()

train4=pd.read_csv("../input/Season.csv", na_values="NA")

X4=train4.copy()

train5=pd.read_csv("../input/Ball_by_Ball.csv", na_values="NA")

X5=train5.copy()
print(X0.shape)

print(X1.shape)

print(X2.shape)

print(X3.shape)

print(X4.shape)

print(X5.shape)

X6=X5.merge(X2,how='left', left_on='Match_Id', right_on='Match_Id')

X6.shape

X6=X6.merge(X0,how='left', left_on='Match_Id', right_on='Match_Id')

X6.shape
X6=X6.merge(X1,how='left', left_on='Player_Id', right_on='Player_Id')

X6.shape
X6=X6.merge(X4,how='left', left_on='Season_Id', right_on='Season_Id')

X6.shape
X6.columns.values
X6.drop(['Host_Country', 'Player_Name','Season_Year','Unnamed: 7'],axis=1,inplace=True)
X6.drop(['Team_Name_Id', 'Opponent_Team_Id'],axis=1,inplace=True)
X6.columns.values