import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler,OneHotEncoder

import seaborn as sns

from sklearn.linear_model import LinearRegression



#import libaries to transform our features 

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer



# Read the data into csvs and see if there's null values within the data

TRAIN_DATA="/kaggle/input/cais-exec-team-in-house/train.csv"

SUBMISSIONS_DATA="/kaggle/input/cais-exec-team-in-house/sampleSubmission.csv"

TEST_DATA="/kaggle/input/cais-exec-team-in-house/test.csv"

df=pd.read_csv(TRAIN_DATA,index_col='id')

test_df=pd.read_csv(TEST_DATA,index_col='id')

sub_df=pd.read_csv(SUBMISSIONS_DATA,index_col='id')

df.info()
# See descriptive statistics of the data

df.describe()

# look at the distribution of features and their correlations with respect to the target value

%matplotlib inline

df.hist(bins=20 , figsize=(20,15))

plt.show()

# heat map of correlation of features

correlation_matrix = df.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_attribs=list(df.select_dtypes(include=numerics))

print(num_attribs)

num_attribs.remove("grade")

cat_attribs=list(df.select_dtypes(exclude=numerics))

num_pipline=make_pipeline(StandardScaler())

full_pipeline=make_column_transformer(

(num_pipline,num_attribs),

(OneHotEncoder(),cat_attribs))





X=df.drop(columns="grade")

full_pipeline=full_pipeline.fit(X)

X=full_pipeline.transform(X)

y=df.grade

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

models=[

    LinearRegression(),

    DecisionTreeRegressor(),

    RandomForestRegressor() 

]

for model in models:

    scores= cross_val_score(model,X,y,scoring="neg_mean_squared_error",cv=5)

    real_scores=np.sqrt(-scores)

    print(f"The scores for {model.__class__.__name__} were {real_scores} and the average was {np.average(real_scores)}")

    print("-------------------------------------------------")



bestModel=RandomForestRegressor()

bestModel.fit(X,y)



test_X=full_pipeline.transform(test_df)

predictions=bestModel.predict(test_X)
sub_df.grade=predictions

sub_df.to_csv("predictions.csv",index=True)