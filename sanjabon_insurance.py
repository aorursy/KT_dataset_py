import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

df_test=pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")
df.describe()
df.columns
Y=df["Response"]
df["Vehicle_Age"].unique()
dictt={"> 2 Years":2, "1-2 Year":1, "< 1 Year":0}
df["Gender"].replace({"Male":1, "Female":0}, inplace=True)
df["Vehicle_age"]=df["Vehicle_Age"].replace(dictt)

df
df["Vehicle_Damage"].replace({"Yes":1, "No":0}, inplace=True)
df
X=df[["Driving_License", "Previously_Insured", "Vehicle_age", "Vehicle_Damage", "Policy_Sales_Channel", "Vintage"]]
#train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 1)
T=RandomForestClassifier(n_estimators=100, n_jobs=-1)
df_test["Gender"].replace({"Male":1, "Female":0}, inplace=True)
df_test["Vehicle_age"]=df_test["Vehicle_Age"].replace(dictt)
df_test["Vehicle_Damage"].replace({"Yes":1, "No":0}, inplace=True)
T.fit(X,Y)
from sklearn.metrics import mean_absolute_error
predictions = T.predict(df_test[["Driving_License", "Previously_Insured", "Vehicle_age", "Vehicle_Damage", "Policy_Sales_Channel", "Vintage"]])
final=pd.DataFrame({"id":df_test.id, "Response":predictions})
final
"""

for x in [50,100,200,300,400]:

    freg=RandomForestClassifier(n_estimators=x, random_state=2, n_jobs=-1)

    freg.fit(train_X, train_y)

    print(mean_absolute_error(freg.predict(val_X), val_y))"""
#freg.predict(val_X.head(20))
#print(val_y.head(20))