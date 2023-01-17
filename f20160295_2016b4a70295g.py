import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



np.random.seed(126)
!ls ../input
df = pd.read_csv("../input/eval-lab-1-f464-v2/train.csv", index_col="id")

X_test = pd.read_csv("../input/eval-lab-1-f464-v2/test.csv", index_col="id")

X_test = X_test.fillna(df.mean())

df= df.dropna()

X = df.iloc[:,:-1]

y = df.iloc[:,-1]

# X = X[["feature6","feature8","feature11","feature4","feature2",]]

# X_test = X_test[["feature6","feature8","feature11","feature4","feature2",]]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn import preprocessing

from sklearn.pipeline import Pipeline

pipeline = Pipeline([('norm', preprocessing.RobustScaler())])

le = preprocessing.LabelEncoder()



le.fit(X_train.type)

X_train.loc[:,"type"] = le.transform(X_train["type"])

X_val.loc[:,"type"] = le.transform(X_val["type"])

X_test.loc[:,"type"] = le.transform(X_test["type"])

X.loc[:,"type"] = le.transform(X["type"])



# X_train = pipeline.fit_transform(X_train)

# X_val = pipeline.transform(X_val)

# X_test = pipeline.transform(X_test)

# X = pipeline.transform(X)



from sklearn.ensemble import ExtraTreesRegressor

best_regr = ExtraTreesRegressor(n_estimators=2250)



best_regr.fit(X, y)
def float_to_int(preds):

    ints = []

    for i in preds:

        if i<0.5:

            ints.append(0);

        elif i>=0.5 and i<1.5:

            ints.append(1);

        elif i>=1.5 and i<2.5:

            ints.append(2);

        elif i>=2.5 and i<3.5:

            ints.append(3);

        elif i>=3.5 and i<4.5:

            ints.append(4);

        elif i>=4.5 and i<5.5:

            ints.append(5);

        elif i>=5.5:

            ints.append(6);

    return np.array(ints)

        
y_sub = float_to_int(best_regr.predict(X_test))

df_sub = pd.DataFrame(y_sub, index=pd.read_csv("../input/eval-lab-1-f464-v2/test.csv").id, columns=["rating"])

df_sub.to_csv("goboi.csv")