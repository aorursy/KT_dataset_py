import numpy as np

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelBinarizer, RobustScaler, Binarizer, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import time
# Update 2017-03-11 - Kaggle added sklearn_pandas support

from sklearn_pandas import DataFrameMapper
# Update 2017-03-11: Kaggle added sklearn_pandas to kernels, yeah !

from sklearn_pandas import DataFrameMapper
df_train = pd.read_json(open("../input/train.json", "r"))

df_train.head()
df_test = pd.read_json(open("../input/test.json", "r"))

df_test.head()
# This transformer extracts the number of photos

class PP_NumPhotTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        df = pd.DataFrame(X) #Python thinks X is a list input by default instead of a Dataframe

        return df.assign(

            NumPhotos = df['photos'].str.len()

            )

    

# This transformer extracts the number of features

class PP_NumFeatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        df = pd.DataFrame(X) #Python thinks X is a list input by default instead of a Dataframe

        return df.assign(

            NumFeat = df['features'].str.len()

            )

    

# This transformer extracts the number of words in the description

class PP_NumDescWordsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        df = pd.DataFrame(X) #Python thinks X is a list input by default instead of a Dataframe

        return df.assign(

            NumDescWords = df["description"].apply(lambda x: len(x.split(" ")))

            )

    

# This transformer extracts the date/month/year and timestamp in a neat package

class PP_DateTimeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        df = pd.DataFrame(X) #Python thinks X is a list input by default instead of a Dataframe

        df = df.assign(

            Created_TS = pd.to_datetime(df["created"])

        )

        return df.assign(

            Created_Year = df["Created_TS"].dt.year,

            Created_Month = df["Created_TS"].dt.month,

            Created_Day = df["Created_TS"].dt.day

            )



####### Debug Transformer ###########

# Use this transformer anywhere in your Pipeline to dump your dataframe to CSV

class DebugTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        X.to_csv('./debug.csv')

        return X
# Impact sklearn_pandas which Pandas DataFrame compatibility with Scikit's classifiers and Pipeline

from sklearn_pandas import DataFrameMapper
mapper = DataFrameMapper([

    (["bathrooms"],RobustScaler()), #Some bathrooms number are 1.5, Some outliers are 112 or 20    (["bedrooms"],OneHotEncoder()),

    (["latitude"],None),

    (["longitude"],None),

    (["price"],RobustScaler()),

    # (["NumDescWords"],None),

    (["NumFeat"],StandardScaler()),

    (["Created_Year"],None),

    (["Created_Month"],None),

    (["Created_Day"],None)

])
pipe = Pipeline([

    ("extract_numphot", PP_NumPhotTransformer()),

    ("extract_numfeat", PP_NumFeatTransformer()),

    ("extract_numdesc", PP_NumDescWordsTransformer()),

    ("extract_datetime", PP_DateTimeTransformer()),

    # ("DEBUG", DebugTransformer()), #Uncomment to debug

    ("featurize", mapper),

    ("xgb",XGBClassifier(

        n_estimators=1000,

        seed=42,

        objective='multi:softprob',

        subsample=0.8,

        colsample_bytree=0.8,

    ))

])
##### Cross Validation #######

def crossval():

    cv = cross_val_score(pipe, X_train, y_train, cv=5)

    print("Cross Validation Scores are: ", cv.round(3))

    print("Mean CrossVal score is: ", round(cv.mean(),3))

    print("Std Dev CrossVal score is: ", round(cv.std(),3))
####### Get top features and noise #######

def top_feat():

    dummy, model = pipe.steps[-1]



    feature_list = []

    for feature in pipe.named_steps['featurize'].features:

        if isinstance(feature[1], OneHotEncoder):

            for feature_value in feature[1].active_features_:

                feature_list.append(feature[0][0]+'_'+str(feature_value))

        else:

            try:

                for feature_value in feature[1].classes_:

                    feature_list.append(feature[0]+'_'+feature_value)

            except:

                feature_list.append(feature[0])





    top_features = pd.DataFrame({'feature':feature_list,'importance':np.round(model.feature_importances_,3)})

    top_features = top_features.sort_values('importance',ascending=False).set_index('feature')

    top_features.to_csv('./top_features.csv')

    top_features.plot.bar()
####### Predict and format output #######

def output():

    predictions = pipe.predict_proba(df_test)

    

    #debug

    print(pipe.classes_)

    print(predictions)

    

    result = pd.DataFrame({

        'listing_id': df_test['listing_id'],

        pipe.classes_[0]: [row[0] for row in predictions], 

        pipe.classes_[1]: [row[1] for row in predictions],

        pipe.classes_[2]: [row[2] for row in predictions]

        })

    result.to_csv(time.strftime("%Y-%m-%d_%H%M-")+'baseline.csv', index=False)
################ Training ################################



X_train = df_train

y_train = df_train['interest_level']
################ Cross Validation ################################

crossval()
##### Fit ######

pipe.fit(X_train, y_train, xgb__eval_metric='mlogloss')
######### Most influential features ########

top_feat()
######## Predict ########

output()