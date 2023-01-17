import numpy as np
import pandas as pd
V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])
df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3})

df = df.astype(float)
df
from sklearn import preprocessing
preprocessing.scale(df)
preprocessing.normalize(df)
import seaborn as sns
tips= sns.load_dataset('tips')
tips.head()
from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder().fit_transform(tips["sex"])
lbe
tips["new_sex"] = LabelEncoder().fit_transform(tips["sex"])
tips.head()
tips["new_day"]=LabelEncoder().fit_transform(tips["day"])
tips.head()
from sklearn.preprocessing import OneHotEncoder
tips_one_hot = pd.get_dummies(tips, columns = ["sex"], prefix = ["sex"])
tips_one_hot.head()
tips_one_hot=pd.get_dummies(tips,columns=["day"],prefix=["day"])
tips_one_hot.head()