import numpy as np
import pandas as pd

df = pd.read_csv("../input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv")
df.head()
df.isnull().sum()
series = pd.isnull(df['cigsPerDay'])
df[series]
data = df.drop(['currentSmoker','education'], axis = 'columns')
data.head()
cigs = data['cigsPerDay']
cigs.head()
cig = cigs.mean()
import math
integer_value = math.floor(cig)
integer_value
cigs.fillna(integer_value, inplace = True)
data.isnull().sum()
data.dropna( axis = 0, inplace = True)
data.isnull().sum()
data.shape
Heart_Attack = data[data.TenYearCHD == 1]
Heart_Attack.head()
No_Heart_Attack = data[data.TenYearCHD == 0]
No_Heart_Attack.head()
data.groupby('TenYearCHD').mean()
final = data.drop(['diaBP','BMI','heartRate'], axis = 'columns')
No_Heart_Attack = final[final.TenYearCHD == 0]
No_Heart_Attack.head()
Heart_Attack = final[final.TenYearCHD == 1]
Heart_Attack.head()
final.groupby('TenYearCHD').mean()
X = final[['male','age','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','glucose']]
y = final['TenYearCHD']
X
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 99)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test,y_test)


