
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data=pd.read_csv("../input/machinelearning/Data.csv")
X=data.iloc[: , :-1].values
Y=data.iloc[: , 3].values
data.head()
from sklearn.impute import SimpleImputer
imputer=  SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer =SimpleImputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_X=LabelEncoder()  # to convert categorical values into numerical values.
X[:, 0]=label_X.fit_transform(X[:, 0])

from sklearn.compose import ColumnTransformer
#onehotencoder = OneHotEncoder(categorical_features=[0])
#x=onehotencoder.fit_transform(X).toarray()
label_Y=LabelEncoder()
y=label_Y.fit_transform(Y)

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
#labelencoder_X = LabelEncoder()
#X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler(with_mean=False)
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)