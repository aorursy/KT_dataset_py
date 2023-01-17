# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/train.csv")

test_df = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/test.csv")

test_df.head()

train_df.columns
train_df.head()
train_df["ISLEM_TURU"].unique()
train_df["Record_Count"].unique()
train_df["CUSTOMER"].unique()
train_df = train_df.drop(["Record_Count"], axis=1)
train_df.info()
def bar_plot(variable):

    var = train_df[variable]

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values, rotation = "vertical")

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    
category1 = ["ISLEM_TURU","SEKTOR"]

for c in category1:

    bar_plot(c)
train_df["ISLEM_ADEDI"] = train_df["ISLEM_ADEDI"].astype(int)

train_df["ISLEM_TUTARI"].max()
train_df["ISLEM_ADEDI"].max()


plt.figure(figsize = (9,3))

plt.hist(train_df["ISLEM_TUTARI"],range=[0,1000], bins = 50)

plt.xlabel("ISLEM_TUTARI")

plt.ylabel("Frequency")

plt.title("{} histogram dağılımı".format("ISLEM_TUTARI"))

plt.show()
plt.figure(figsize = (9,3))

plt.hist(train_df["ISLEM_ADEDI"],range=[0,50], bins = 10)

plt.xlabel("ISLEM_ADEDI")

plt.ylabel("Frequency")

plt.title("{} histogram dağılımı".format("ISLEM_ADEDI"))

plt.show()
# ISLEM_TURU vs ISLEM_TUTARI

train_df[["ISLEM_TURU","ISLEM_TUTARI"]].groupby(["ISLEM_TURU"], as_index = False).mean().sort_values(by="ISLEM_TUTARI",ascending = False)
# SEKTOR vs ISLEM_TUTARI

train_df[["SEKTOR","ISLEM_TUTARI"]].groupby(["SEKTOR"], as_index = False).mean().sort_values(by="ISLEM_TUTARI",ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["ISLEM_ADEDI", "ISLEM_TUTARI"])]
train_df.columns[train_df.isnull().any()]

train_df.isnull().sum()
test_df.columns[test_df.isnull().any()]

test_df.isnull().sum()
train_df["YIL_AY"] = train_df["YIL_AY"].astype(str)
year = []

month = []

for s in train_df["YIL_AY"]:

    y = s[0:4]

    m = s[4:]

    year.append(y)

    month.append(m)

train_df["YEAR"] = year

train_df["MONTH"] = month

train_df["DAY"] = 15

pd.to_datetime(train_df[["YEAR", "MONTH", "DAY"]])
train_df.drop("YIL_AY", axis=1, inplace = True)
train_df.head()
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]

train_df.isnull().sum()

train_df.drop(["YEAR", "MONTH","DAY","ID","YIL_AY","Record_Count"], axis=1, inplace=True)
train_df["ISLEM_TURU"] = [0 if i == "PESIN" else 1 for i in train_df["ISLEM_TURU"]]
train_df = pd.get_dummies(train_df,columns=["ISLEM_TURU"])

train_df.head()
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

train_df["SEKTOR"] = le.fit_transform(train_df["SEKTOR"])
train_df = pd.get_dummies(train_df,columns=["SEKTOR"])

train_df.head()
from sklearn.model_selection import train_test_split
test = train_df[train_df_len:]

test.drop(labels = ["ISLEM_TUTARI"],axis = 1, inplace = True)
train = train_df[:train_df_len]

X_train = train.drop(labels = "ISLEM_TUTARI", axis = 1)

y_train = train["ISLEM_TUTARI"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
test_Id = test_df["ID"]
lin_reg_mod = LinearRegression()

a = lin_reg_mod.fit(X_train, y_train)

test_sonuç = pd.Series(a.predict(test), name = 'Predicted').astype(int)

results = pd.concat([test_Id, test_sonuç],axis = 1)

results.to_csv("sampleSubmission.csv", index = False)