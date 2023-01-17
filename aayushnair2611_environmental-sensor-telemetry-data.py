# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('../input/environmental-sensor-data-132k/iot_telemetry_data.csv')

data.head(15)
data['device'].unique()
labelencoder = LabelEncoder()

devices = labelencoder.fit_transform(data['device'])

lights = labelencoder.fit_transform(data['light'])

devices
onehotencoder = OneHotEncoder()
data['device'] = devices

data['light'] = lights
data.drop(['ts','motion'],axis=1,inplace=True)
data.info()
plt.figure(figsize=(20,15))

sns.barplot( 'co','humidity', data= data)
sns.countplot('temp',data=data)
fig, ax =plt.subplots(1,2,figsize=(24, 6))

sns.barplot('lpg','smoke',ax=ax[0],data=data.sort_values(by='lpg',ascending=False).head(10)).set_title('Ratio of LPG and Smoke')

sns.barplot('smoke','temp',ax=ax[1],data=data.sort_values(by='smoke',ascending=False).head(10)).set_title('Ratio of temprature and smaoke')
activities=['co','humidity','lpg','smoke','temp']

slice=[3,7,8,6,2]

color=['r', 'g', 'm', 'b','c']

plt.pie(slice, labels=activities, colors=color, startangle=90,shadow=True, 

       explode=(0.2,0,0,0,0),autopct='%1.2f%%')

plt.legend(bbox_to_anchor =(0.85, 1.20), ncol = 2)

plt.show()
fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot_trisurf(data['temp'], data['co'], data['smoke'], cmap = plt.cm.twilight_shifted)

plt.title('Relation between Carbon di oxide levels, Smoke and Temperature.')

plt.xlabel('co')

plt.ylabel('smoke')

plt.show()
data.head(12)
logi = LogisticRegression()
X = data.drop(['light'],axis=1)

y = data[['light']].values
X
y
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=14)
X_train
X_test
y_train
y_test
Rudy=logi.fit(X_train,y_train) 
predict=logi.predict(X_test)
predict
score = accuracy_score(y_test, predict)



score
cm = confusion_matrix(y_test, predict)

sns.heatmap(cm, annot=True, cmap="winter" ,fmt='g')

plt.tight_layout()

plt.title('Confusion matrix')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')