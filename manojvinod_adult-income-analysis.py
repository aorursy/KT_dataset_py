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

data = pd.read_csv("../input/adult.csv")
#To remove space  in columns name 

data.columns= data.columns.str.replace(' ','',regex=True)



data.mask(data==" ?",np.nan)
data["country"]="Non-US"

data["country"][data["native-country"]==" United-States"]="US"
data['income_flag'] = data['income'].map({' <=50K':0, ' >50K':1})
data["marital"]="married"

data["marital"][data["marital-status"]==" Never-married"]="single"
data["workcl"]="private"

data["workcl"][data["workclass"]==" Local-gov"]="government"

data["workcl"][data["workclass"]==" State-gov"]="government"

data["workcl"][data["workclass"]==" Federal-gov"]="government"
data["occ"]="service"

data["occ"][data["occupation"].isin([" Protective-serv"," Priv-house-serv"," Handlers-cleaners"," Transport-moving"," Other-service"," Craft-repair"," Machine-op-inspct"])]="non service"        

data["occ"][data["occupation"]==" Armed-Forces"]=" Armed-Forces"

data["educt"]="school Grade"

data["educt"][data["education"].isin([" Some-college"," Bachelors"])]="graduate"

data["educt"][data["education"].isin([" Masters"," Doctorate"])]="post graduate"



x = data[['sex','marital','occ','workcl','educt','race','country','hours-per-week','capital-loss','capital-gain']]

y = data['income_flag']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label = LabelEncoder()

data['sex']=label.fit_transform(data['sex'])

data['occ'] =pd.get_dummies(data['occ'])

data['workcl'] =pd.get_dummies(data['workcl'])

data['educt'] =pd.get_dummies(data['educt'])

data['marital']=label.fit_transform(data['marital'])

data['race'] =pd.get_dummies(data['race'])

data['country']= label.fit_transform(data['country'])
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LogisticRegression

s = LogisticRegression()

s.fit(x_train,y_train)
print("The test score is ",s.score(x_test,y_test))

print("The train score is ",s.score(x_train,y_train))
from sklearn.ensemble import RandomForestClassifier

f = RandomForestClassifier(criterion='entropy')

f.fit(x_train,y_train)
print("The test score is ",f.score(x_test,y_test))

print("The train score is ",f.score(x_train,y_train))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.svm import SVC

d = SVC(kernel='linear')

d.fit(x_train,y_train)
print("The test score is ",d.score(x_test,y_test))

print("The train score is ",d.score(x_train,y_train))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(x_train, y_train)

print("The test score is ",classifier.score(x_test,y_test))

print("The train score is ",classifier.score(x_train,y_train))