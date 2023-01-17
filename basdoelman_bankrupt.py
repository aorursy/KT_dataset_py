import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# for preprocessing the data
from sklearn.preprocessing import StandardScaler

# the model
from sklearn.linear_model import LogisticRegression

# for combining the preprocess with model training
from sklearn.pipeline import Pipeline


from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
print("reading the data...")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.info()
train_df.describe()
# preprocessing and modifying features
#average evaluation scores (except C_score)
scores = ['A_score', 'B_score', 'D_score']
train_df['av_score']=train_df[scores].mean(axis=1)
test_df['av_score']=test_df[scores].mean(axis=1)
# new features delta revenue relative to revenue:
train_df['delta_revenue1']=(train_df['revenue2016']-train_df['revenue2015'])/train_df['revenue2015']
train_df['delta_revenue2']=(train_df['revenue2015']-train_df['revenue2014'])/train_df['revenue2014']
test_df['delta_revenue1']=(test_df['revenue2016']-test_df['revenue2015'])/test_df['revenue2015']
test_df['delta_revenue2']=(test_df['revenue2015']-test_df['revenue2014'])/test_df['revenue2014']

# new feature revenue / employees^2
train_df['rev_emp']=np.log(train_df['revenue2016']/train_df['num_employees']**2)
test_df['rev_emp']=np.log(test_df['revenue2016']/test_df['num_employees']**2)

# logscales of employees and revenue features

train_df['num_employees'] = np.log(train_df['num_employees'])
train_df['revenue2014'] = np.log(train_df['revenue2014'])
train_df['revenue2015'] = np.log(train_df['revenue2015'])
train_df['revenue2016'] = np.log(train_df['revenue2016'])
test_df['num_employees'] = np.log(test_df['num_employees'])
test_df['revenue2014'] = np.log(test_df['revenue2014'])
test_df['revenue2015'] = np.log(test_df['revenue2015'])
test_df['revenue2016'] = np.log(test_df['revenue2016'])
target = "bankrupt"

print("preprocessing...")
train_df = pd.get_dummies(train_df.fillna(0), columns=["country"])
test_df = pd.get_dummies(test_df.fillna(0), columns=["country"])
print("Columns:", list(train_df.columns))
train_df['country_CN']=train_df['country_CN'].astype('category')
test_df['country_CN']=test_df['country_CN'].astype('category')
train_df['country_EN']=train_df['country_EN'].astype('category')
test_df['country_EN']=test_df['country_EN'].astype('category')
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.2, 
                                    stratify=train_df[target], shuffle=True, random_state=0)
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
numeric_features = ["num_employees",
                    "country_CN", "country_EN", "country_NL", "country_US", "revenue2015",
                    "revenue2016","delta_revenue1", "delta_revenue2","rev_emp", 'av_score']

text_features = "industry_desc"

all_features = ["num_employees",
                    "country_CN", "country_EN", "country_NL", "country_US", "revenue2015",
                    "revenue2016", "delta_revenue1", "delta_revenue2","rev_emp", "industry_desc", 'av_score']

get_text_data = FunctionTransformer(lambda x: x[text_features], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[numeric_features], validate=False)

countvec = TfidfVectorizer(min_df=1, ngram_range=(1,2))

numeric_pipeline = Pipeline([('selector', get_numeric_data),('scale', StandardScaler())])

text_pipeline = Pipeline([('selector', get_text_data), ('vectorizer', countvec)])

pipe = Pipeline([('union', FeatureUnion([('numeric', numeric_pipeline),('text',text_pipeline)])),('logistic', LogisticRegression(solver="liblinear"))]) 
pipe.fit(train_df, train_df[target])
from sklearn.metrics import roc_auc_score

val_df["pred"] = pipe.predict_proba(val_df[all_features])[:, 1]

print("Validation score:", roc_auc_score(val_df[target], val_df["pred"]))
test_df[target] = pipe.predict_proba(test_df[all_features])[:, 1]

print("creating submission...")
test_df[["id", target]].to_csv("lr_submission.csv", index=False)