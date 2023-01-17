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
#ignore warnings

import warnings

warnings.filterwarnings('ignore')
FILEPATH = '/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(FILEPATH)
df.sample(5)
import missingno as miss
miss.bar(df)
df = df.drop(['Serial No.'], axis = 1)
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True)

plt.show()
import plotly.express as px
df_visual = df
fig = px.histogram(df_visual, x = 'TOEFL Score', color = 'Research')

fig.show()
fig = px.histogram(df_visual, x = 'GRE Score', color = 'Research')

fig.show()
df_visual['LOR_level'] = df_visual['LOR '].apply(lambda x:1 if x > 3.0 else 0)
df_visual.head()
fig = px.histogram(df, x = 'SOP', color = 'LOR_level')

fig.show()
X = df.drop(columns = ['Chance of Admit '])

y = df['Chance of Admit '] #Target Variable
# # Or use this

# X = df.iloc[:, :-1]

# y = df.iloc[:, -1]
X.columns
#Converting x & y into NumPy Arrays



X = np.array(X)

y = np.array(y)
y = y.reshape(-1,1)

y.shape
#Scaling the Data



from sklearn.preprocessing import StandardScaler, MinMaxScaler



scaler = StandardScaler()

minmax = MinMaxScaler()



X = scaler.fit_transform(X)

y = scaler.fit_transform(y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.23)





print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import HuberRegressor
best_model_accuracy = 0

best_model = None



models = [

    LinearRegression(),

    RandomForestRegressor(n_estimators= 100, max_depth=25), #Instantiate an object

    linear_model.BayesianRidge(),

    AdaBoostRegressor(),

    GradientBoostingRegressor(),

    HuberRegressor()

]



for model in models:

    

    model_name = model.__class__.__name__

    

    # fit 

    model.fit(X_train, y_train)

    

    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)

    

    print("-" * 30)

    print(model_name + ": " )

    

    if(accuracy > best_model_accuracy):

        best_model_accuracy = accuracy

        best_model = model_name

    

    print("Accuracy: {:.2%}".format(accuracy))
print("Best Model : {}".format(best_model))

print("Best Model Accuracy : {:.2%}".format(best_model_accuracy))