# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.listdir("/kaggle/input/titanic")
pd.read_csv("/kaggle/input/titanic/gender_submission.csv").shape

train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.head()
train_data=train_data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

xtest=test_data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
train_data.isna().sum()
train_data.head()
test_data.isna().sum()
train_data['Age']=train_data['Age'].fillna(np.median(train_data['Age'].dropna()))

xtest['Age']= xtest['Age'].fillna(np.median(train_data['Age'].dropna()))
xtest.isna().sum()
xtest['Fare']=xtest['Fare'].fillna(np.median(train_data['Fare']))
xtest.isna().sum()
train_data.isna().sum()
unique_embarked_values=list(set(train_data['Embarked']) or set(xtest['Embarked']))

train_data['Embarked']=train_data['Embarked'].replace(unique_embarked_values,range(len(unique_embarked_values)))

xtest['Embarked']=xtest['Embarked'].replace(unique_embarked_values,range(len(unique_embarked_values)))

#train_data['Embarked']=to_categorical(train_data['Embarked'])

#test_data['Embarked']=to_categorical(test_data['Embarked'])

unique_sex=list(set(train_data['Sex']) or set(xtest['Sex']))

train_data['Sex']=train_data['Sex'].replace(unique_sex,range(len(unique_sex)))

xtest['Sex']=xtest['Sex'].replace(unique_sex,range(len(unique_sex)))
x=train_data.drop('Survived',axis=1).values

y=train_data['Survived'].values

y=y.reshape(len(y),1)



xtest=xtest.values
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x=sc.fit_transform(x)

xtest=sc.transform(xtest)

from keras import Sequential

from keras.layers import Dense, LeakyReLU

from keras.optimizers import Adam
classifier = Sequential()



classifier.add(Dense(output_dim = 12, init = 'uniform',activation = LeakyReLU(alpha=.051), input_dim = 7))



classifier.add(Dense(output_dim = 6, init = 'uniform', activation = LeakyReLU(alpha=.051)))



classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



classifier.compile(optimizer = Adam(0.006), loss = 'mse', metrics = ['accuracy'])



classifier.fit(x, y, batch_size = 200, nb_epoch = 5000,shuffle=True)
submission=pd.DataFrame()
predicted= classifier.predict(xtest)
predicted[predicted>0.5]=1

predicted[predicted<=0.5]=0
submission['PassengerId']=test_data['PassengerId']

submission['Survived']=np.array(predicted).astype('int')
submission
filename="submission1.csv"

csv=submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
from IPython.display import HTML



def create_download_link(filename,title = "Download CSV file"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename)