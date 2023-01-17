# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/train.csv')

df.head()



df.Sex.replace(to_replace='male',value=1,inplace=True)

df.Sex.replace(to_replace='female',value=0,inplace=True)





#df_features = df[['Pclass','Sex','Embarked']]



df.shape



df = df[pd.notnull(df['Embarked'])]

df.shape



df.Embarked.replace(to_replace='C',value=0,inplace=True)

df.Embarked.replace(to_replace='Q',value=1,inplace=True)

df.Embarked.replace(to_replace='S',value=2,inplace=True)



df.head()



df_features = df[['Pclass','Sex','Embarked']]

df_features



df_target = df['Survived']

df_target



from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()



RF.fit(df_features.values,df_target.values)



testdf = pd.read_csv('../input/test.csv')



testdf



testdf.isnull().sum()



testdf.Sex.replace(to_replace='male',value=1,inplace=True)

testdf.Sex.replace(to_replace='female',value=0,inplace=True)



testdf.Embarked.replace(to_replace='C',value=0,inplace=True)

testdf.Embarked.replace(to_replace='Q',value=1,inplace=True)

testdf.Embarked.replace(to_replace='S',value=2,inplace=True)



testdf.head()

testdf_features = testdf[['Pclass','Sex','Embarked']]

testdf_features

Prediction = RF.predict(testdf_features.values)



pred_df = pd.DataFrame(data = Prediction)

pred_df

submission_df = pd.concat( [testdf.PassengerId , pred_df], axis=1  )

submission_df.columns = ['PassengerId','Survived']

submission_df

submission_df.to_csv('submission.csv',index=False)
