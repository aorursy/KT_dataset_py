import numpy as np

import pandas as pd
df = pd.read_csv('../input/pokemon_alopez247.csv')
df.head()
# change types of columns from int to float to avoid future warnings

for col in df.columns:

    if df[col].dtype == int:

        df[col] = df[col].astype(float)
# we want to predict if pokemon is legendary or not

df['isLegendary'].value_counts()
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
def get_arrays(df):

    X = np.array(df[['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def']])

    y = np.array(df['isLegendary'])

    

    return X, y



X_train, y_train = get_arrays(df_train)

X_test, y_test = get_arrays(df_test)



X_train.shape, y_train.shape
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression()

)
model = pipeline.fit(X_train, y_train)

model.predict(X_train)[:5]
from sklearn.metrics import confusion_matrix



confusion_matrix(y_train, model.predict(X_train))
from sklearn.metrics import roc_auc_score



roc_auc_score(y_train, model.predict(X_train))
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer



def cross_validate_auc(pipeline, X_train, y_train):

    results = cross_val_score(

        pipeline,

        X_train,

        y_train,

        scoring=make_scorer(roc_auc_score),

        cv=10,

    )



    return np.mean(results)

    

cross_validate_auc(pipeline, X_train, y_train)
from sklearn.base import BaseEstimator, TransformerMixin



class PandasSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, selected_columns):

        self.selected_columns = selected_columns

    

    def fit(self, df, *args):

        return self



    def transform(self, df):

        return np.array(df[self.selected_columns])
pipeline = make_pipeline(

    PandasSelector(['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def']),

    StandardScaler(),

    LogisticRegression()

)



model = pipeline.fit(df_train, y_train)

model.predict(df_train)[:5]
from sklearn.pipeline import make_union



pipeline_stats = make_pipeline(

    PandasSelector(['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def']),

    StandardScaler(),

)



pipeline_hasGender = make_pipeline(

    PandasSelector(['hasGender']),

)



pipeline = make_pipeline(

    make_union(

        pipeline_stats,

        pipeline_hasGender,

    ),

    LogisticRegression(),

)



cross_validate_auc(pipeline, df_train, y_train)
# before we use OnHotEncoder we need to convert strings to ints



class StringConverter(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        self.map = {} # column : string : int

    

    def fit(self, X, *args):

        for col in range(X.shape[1]):

            self.map[col] = {}

            idx = 1

            for row in range(X.shape[0]):

                s = X[row, col]

                if s not in self.map[col]:

                    self.map[col][s] = idx

                    idx += 1

        return self



    def transform(self, X):

        X_int = np.zeros(shape=X.shape)

        for col in range(X.shape[1]):

            for row in range(X.shape[0]):

                s = X[row, col]

                X_int[row, col] = self.map[col].get(s, 0)



        return X_int
from sklearn.preprocessing import OneHotEncoder



pipeline_color = make_pipeline(

    PandasSelector(['Color']),

    StringConverter(),

    OneHotEncoder(),

)



pipeline = make_pipeline(

    make_union(

        pipeline_stats,

        pipeline_hasGender,

        pipeline_color,

    ),

    LogisticRegression(),

)



cross_validate_auc(pipeline, df_train, y_train)
np.mean(df_train['Pr_Male'].isnull())
from sklearn.preprocessing import Imputer



pipeline_PrMale = make_pipeline(

    PandasSelector(['Pr_Male']),

    Imputer(),

)



pipeline = make_pipeline(

    make_union(

        pipeline_stats,

        pipeline_hasGender,

        pipeline_color,

        pipeline_PrMale,

    ),

    LogisticRegression(),

)



cross_validate_auc(pipeline, df_train, y_train)
from sklearn.feature_extraction.text import TfidfVectorizer



pipeline_name = make_pipeline(

    PandasSelector('Name'),

    TfidfVectorizer(

        analyzer='char',

        ngram_range=(1, 5),

        min_df=10,

    ),

)



pipeline = make_pipeline(

    make_union(

        pipeline_stats,

        pipeline_hasGender,

        pipeline_color,

        pipeline_PrMale,

        pipeline_name,

    ),

    LogisticRegression(),

)



cross_validate_auc(pipeline, df_train, y_train)
from sklearn.model_selection import GridSearchCV



parameters = {

    'logisticregression__C': [0.01, 0.1, 1, 10, 100],

    'logisticregression__class_weight': [None, 'balanced'],

}



grid = GridSearchCV(

    pipeline,

    parameters,

    scoring=make_scorer(roc_auc_score),

).fit(df_train, y_train)



print('Best params: {}'.format(grid.best_params_))

print('Best AUC: {:.3f}'.format(grid.best_score_))



final_model = grid.best_estimator_
roc_auc_score(y_test, final_model.predict(df_test))
from sklearn.externals import joblib



joblib.dump(final_model, 'final_model.pkl');
!ls
loaded_model = joblib.load('final_model.pkl')

roc_auc_score(y_test, final_model.predict(df_test))