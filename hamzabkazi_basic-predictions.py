# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# get train data file as Dataframe
train_df = pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
a=train_df.loc[ train_df["Pclass"]==1 , ["Survived"] ]
#view
All_survivor=np.array( [train_df["Survived"] == 1 ] )#.value_counts(normalize=True)
first_class=train_df.loc[  train_df["Pclass"]==1 ]

all_males=train_df.loc[ train_df['Sex']=='male' ]

F_C_M=first_class.loc[first_class['Sex']=='male', ['Survived','Sex']]

F_C_F=first_class.loc[first_class['Sex']=='female', ['Survived','Sex']]
F_C_F["Survived"].value_counts()#So 91/95 F_C_F survived
first_class_child=first_class.loc[first_class['Age'] < 18 ,['Survived','Sex']   ]
first_class_child['Survived'].value_counts() #shows 11/12  survived
second_class=train_df.loc[ train_df['Pclass']==2, ['Survived','Sex','Age']  ]
second_class['Survived'].value_counts()# 87/184 survived
second_class_female=second_class.loc[ second_class['Sex']=='female', ['Survived','Sex','Age']  ]
second_class_female['Survived'].value_counts()# so 70/76 survived
p_s=train_df.copy()

p_s['Survived']=0 #initialize with 0
p_s.loc[ (p_s['Pclass']<3) & (p_s['Sex']=='female') ,'Survived']=1 # set 1st and 2nd class female survive to 1
p_s.loc[ (p_s['Pclass']==1) & (p_s['Age'] < 18) ,'Survived']=1 # set 1st children survive to 1



Train_pred= p_s[['Survived']]

score=Train_pred==train_df[['Survived']]
score['Survived'].value_counts(normalize=True)
predict_test=test_df.copy()

predict_test['Survived']=0 #initialize with 0
predict_test.loc[ (predict_test['Pclass']<3) & (predict_test['Sex']=='female') ,'Survived']=1 # set 1st and 2nd class female survive to 1
predict_test.loc[ (predict_test['Pclass']==1) & (predict_test['Age'] < 18) ,'Survived']=1 # set 1st children survive to 1



Y_pred= predict_test['Survived'] # just store a list not data frame


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)