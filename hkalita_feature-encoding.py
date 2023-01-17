import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



%matplotlib inline

plt.style.use("ggplot")

warnings.filterwarnings("ignore")

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")



train = train_data.copy()
train_data.shape, test_data.shape
train.head(2)
def dataDetails(df):

    res = pd.DataFrame()

    res["Columns"] = list(df.columns)

    res["Missing_Values"] = list(df.isna().sum())

    res["Data_Type"] = list(df.dtypes)

    res["Unique_Values"] = [df[x].nunique() for x in df.columns]

    

    return res



details = dataDetails(train)

details
X, y = train.drop("target",axis=1), train.target
def model(X,y):

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import precision_score,recall_score,f1_score

    

    lr = LogisticRegression()

    cv = StratifiedKFold(n_splits = 10, random_state = 3)

    scores = cross_val_score(lr,X,y,cv=cv)

    

    print("Mean Score : {:.3f}".format(np.mean(scores)))
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)



ohe_columns = ["bin_3","bin_4","nom_0","nom_1","nom_2","nom_3","nom_4"]

ohe_result = ohe.fit_transform(train[ohe_columns])



# To help visualize the outputs of OneHotEncoder we will use Pandas get_dummies() method.

# But from implementation-point-of-view, One-Hot Encoder from sklearn is better, as it could be directly used

# to fit into the test data.



ohe_output = pd.get_dummies(train[ohe_columns])

ohe_output.head(2)
train.ord_2.value_counts()
categories = [["Freezing","Cold","Warm","Hot","Boiling Hot","Lava Hot"]]

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=categories, dtype = int)

out = oe.fit_transform(np.array(train.ord_2).reshape(-1,1))
for i in range(10):

    print(str(train.loc[i,"ord_2"])+" --> "+str(out[i][0]))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le_out = le.fit_transform(np.array(train.nom_5).flatten())
for i in range(10):

    print(str(train.loc[i,"nom_5"])+" --> "+str(le_out[i]))
le_out_ohe = ohe.fit_transform(np.array(le_out).reshape(-1,1))
print("Shape: {}".format(le_out_ohe.shape))
# if category_encoders isn't installed, use !pip install category_encoders

be_col = train[["nom_5","nom_6","nom_7","nom_8","nom_9"]]

import category_encoders as ce

be = ce.binary.BinaryEncoder()

be_out = be.fit_transform(be_col,y)
be_out.shape
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(input_type='string')

X_hash = X.ord_2.astype(str)

fh_out = fh.fit_transform(X_hash.values)

fh_out.shape
fh = FeatureHasher(input_type='string',n_features=20) #n_features = 20

X_hash = X.ord_2.astype(str)

fh_out = fh.fit_transform(X_hash.values)

fh_out.shape
# One-Hot Encoding

ohe_columns = ["bin_0","bin_1","bin_2","bin_3","bin_4"]

ohe_out = pd.get_dummies(train[ohe_columns],drop_first=True)



# Nominal Feature Encoding : Binary Encoding or (Ordinal Encoding + Label Binarizer)

import category_encoders as ce

nom_columns = ["nom_0","nom_1","nom_2","nom_3","nom_4","nom_5","nom_6","nom_7","nom_8","nom_9"]

be = ce.BinaryEncoder(return_df = True)

be_out = be.fit_transform(train[nom_columns],y)



# Ordinal Feature Encoding : Ordinal Encoding

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(dtype=int)

ord_columns = ["ord_1","ord_2","ord_3","ord_4","ord_5"]

ord_out = oe.fit_transform(train[ord_columns])

ord_out = pd.DataFrame(ord_out,columns=ord_columns)



new_train = pd.concat([ohe_out,be_out],axis=1)

new_train["ord_0"] = train["ord_0"]

new_train = pd.concat([new_train,ord_out],axis=1)
model(new_train,y) #fitting and calculating model performance