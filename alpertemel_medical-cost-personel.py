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

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")
data.head()
data.isnull().sum()/len(data)
data.describe().T
corr=data.corr()

sns.heatmap(corr)
sns.pairplot(data,hue="smoker")
sns.scatterplot(x=data.bmi,y=data.charges,hue=data.smoker)
a=0

bmi_grup=[]



while a<1339:

    

    if (data.bmi[a]<=18.5):

        bmi_grup.append("under_weight")

    elif (data.bmi[a]>18.5 and data.bmi[a]<=25):

        bmi_grup.append("normal")

    elif (data.bmi[a]>25 and data.bmi[a]<=30):

        bmi_grup.append("over_weight")

    elif (data.bmi[a]>30 and data.bmi[a]<=35):

        bmi_grup.append("moderately_obese")

    elif (data.bmi[a]>35 and data.bmi[a]<=40):

        bmi_grup.append("several_obese")

    else:

        bmi_grup.append("very_obese")

        

    a+=1

    

bmi_grup=pd.DataFrame(bmi_grup,columns=["bmi_grup"])

data["bmi_grup"]=bmi_grup
data.head()
sns.scatterplot(x=data.bmi_grup,y=data.charges,hue=data.smoker)
i=0

obese_smoker=[]



while i<1339:

    

    if(data.bmi_grup[i]=="very_obese" or data.bmi_grup[i]=="several_obese" or data.bmi_grup[i]=="moderately_obese") and (data.smoker[i]=="yes"):

        obese_smoker.append(1)

    else:

        obese_smoker.append(0)

    

    i+=1
obese_smoker=pd.DataFrame(obese_smoker,columns=["obese_smoker?"])

data["obese_smoker?"]=obese_smoker
data.head()
sns.scatterplot(x=data.charges,y=data["obese_smoker?"])
u=0

age_segment=[]



while u<1339:

    

    if(data.age[u]<=20):

        age_segment.append("young")

    elif(data.age[u]>20 and data.age[u]<=30):

        age_segment.append("young_adult")

    elif(data.age[u]>30 and data.age[u]<=40):

        age_segment.append("adult")

    elif(data.age[u]>40 and data.age[u]<=50):

        age_segment.append("old")

    else:

        age_segment.append("very_old")

        

    u+=1
age_segment=pd.DataFrame(age_segment,columns=["age_segment"])

data["age_segment"]=age_segment

data.head()
sns.countplot(data.age_segment)
sns.scatterplot(x=data.bmi,y=data.charges,hue=data.smoker)

sns.scatterplot(x=data.bmi,y=data.charges,hue=data.age_segment)
p=0

obese_smoker_oldies=[]



while p<1339:

    

    if(data.bmi_grup[p]=="moderately_obese" or data.bmi_grup[p]=="several_obese" or data.bmi_grup[p]=="very_obese") and (data.smoker[p]=="yes") and (data.age_segment[p]=="old" or data.age_segment[p]=="very_old"):

        obese_smoker_oldies.append(1)

    else:

        obese_smoker_oldies.append(0)

    

    p+=1
obese_smoker_oldies=pd.DataFrame(obese_smoker_oldies,columns=["obese_smoker_oldies"])
data["obese_smoker_oldies"]=obese_smoker_oldies

data.head()
sns.scatterplot(x=data.obese_smoker_oldies,y=data.charges)
data.dtypes
cols=["sex","smoker","region","bmi_grup","age_segment"]



from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()



for i in cols:

    

    data[i]=le.fit_transform(data[i])

    

data.head()
x=data.iloc[:,0:6]

x1=data.iloc[:,7:11]

x=pd.concat([x,x1],axis=1)

y=data.iloc[:,6:7]

x.head()

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)





from xgboost import XGBRegressor



xgb=XGBRegressor()

xgb.fit(x_train,y_train)

xgb_tahmin=xgb.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score





print("R^2 skoru",r2_score(y_test,xgb_tahmin),

     "Hata Kareler Ortalaması", np.sqrt(mean_squared_error(y_test,xgb_tahmin)))
data.describe()
from sklearn.model_selection import GridSearchCV



xgb_params={

    

    "colsample_bytree": np.arange(0.2,1,0.1),

    "max_depth": np.arange(3,10,1),

    "learning_rate": np.arange(0.1,1,0.1)

    

}





xgb_tune=XGBRegressor()

xgb_cv=GridSearchCV(xgb_tune,xgb_params,cv=10,n_jobs=-1,verbose=2)



xgb_cv.fit(x_train,y_train)
xgb_cv.best_params_