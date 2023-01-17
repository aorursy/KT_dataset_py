import numpy as np

import pandas as pd
import pandas as pd

df = pd.read_csv("../input/churn-predictions-personal/Churn_Predictions.csv", index_col=0)

df.head()
print(df.shape)

df.tail()
df.info()
df.describe()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import FeatureUnion, Pipeline

import warnings

warnings.filterwarnings('ignore')
def Normalize(df):

    cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

    return (df[cols] - df[cols].mean()) / df[cols].std()



def Categorize(df):

    cols = ['Geography', 'Gender']

    return df[cols].astype('category')

    

def OneHot(df):

#     cols = ['NumOfProducts']

    cols = ['Geography', 'Gender', 'NumOfProducts']

    return pd.get_dummies(df[cols], drop_first=True)
FunctionTransformer(OneHot, validate=False).fit_transform(df).head()
preprocess = FeatureUnion([

    ('normalize', FunctionTransformer(Normalize, validate=False)),

#     ('categorize', FunctionTransformer(Categorize, validate=False)),

    ('onehot', FunctionTransformer(OneHot, validate=False)),

])



pipe = Pipeline([

    ('union', preprocess),

    ('clf', LogisticRegression())

])



X_train, X_test, y_train, y_test = train_test_split(df, df[['Exited']], test_size=0.3)

pipe.fit(X_train, y_train)

accuracy = pipe.score(X_test, y_test)

print('acc: {:.2f}'.format(accuracy))