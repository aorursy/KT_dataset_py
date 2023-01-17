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



import numpy as np



from sklearn.ensemble import RandomForestClassifier
train_df=pd.read_csv("../input/train.csv")
train_df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
train_df.head()
train_df['Embarked']= train_df.groupby('Embarked').ngroup()
train_df['Sex']=train_df.groupby('Sex').ngroup()
train_df.head()
train_df.fillna(0,inplace=True)
train_df_o=pd.DataFrame(train_df['Survived'])

# train_df['Survived']=train_df_o

train_df_i=train_df.drop(columns=['Survived'],inplace=False)
model = RandomForestClassifier()
model.fit(train_df_i,train_df_o)
test_df=pd.read_csv("../input/test.csv")
p_id=pd.DataFrame(test_df['PassengerId'])
test_df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
test_df['Embarked']=test_df.groupby('Embarked').ngroup()

test_df['Sex']=test_df.groupby('Embarked').ngroup()
test_df.head()
test_df.fillna(0,inplace=True)
p=model.predict(test_df)
submission = pd.DataFrame({'PassengerId':p_id['PassengerId'],'Survived':p})
submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)