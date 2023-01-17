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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")



data = data[data.Outcome != 0]



data.head()
data.shape
data.skew()
sns.set(font_scale=1.5)

data.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));
# How many missing zeros are in each feature

feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin', 'DiabetesPedigreeFunction', 'Age']

for column in feature_columns:

    print("============================================")

    print(f"{column} ==> Missing zeros : {len(data.loc[data[column] == 0])}")
data[feature_columns]=data[feature_columns].replace(0, np.nan)

data.head()
data['BMI'] = data['BMI'].fillna(data['BMI'].median())



feat = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']

data[feat] = data[feat].fillna(data[feat].mean())



data.head()
#features = ['BMI','Age','BloodPressure']

features = ['BloodPressure','BMI','Age']

extra_feat = ['Pregnancies', 'BMI', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']



x = data[extra_feat]   #independent variables

y = data.Glucose     #dependent variables

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)



x_train.head()
regr = LinearRegression()

regr.fit(x_train,y_train)



print(regr.score(x_test,y_test))
y_pred=regr.predict(x_test)



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
regr.coef_
regr.intercept_
a = 72

b = 33.6

c = 50



y = a*0.2193 + b*0.23 + c*0.15 + 112.318

y
a = 64

b = 23.3

c = 32



y = a*0.2193 + b*0.23 + c*0.15 + 112.318

y