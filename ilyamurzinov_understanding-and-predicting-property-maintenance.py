import numpy as np

import pandas as pd



base_dir = '../input/'

train_only_cols = ['payment_amount','balance_due', 'payment_date', 'payment_status', 'collection_status', 'grafitti_status', 'compliance_detail']

blight = pd.read_csv('{}train.csv'.format(base_dir),

                     encoding='iso-8859-1',

                     dtype={11:str,12:str,31:str}) #.sample(1000, random_state=42)

X_test = pd.read_csv('{}test.csv'.format(base_dir), dtype={11:str,12:str,31:str})

addresses = pd.read_csv('{}addresses.csv'.format(base_dir))

latlons = pd.read_csv('{}latlons.csv'.format(base_dir))



blight.drop(train_only_cols, axis='columns', inplace=True)

blight.set_index(['ticket_id'], inplace=True)

addresses.set_index(['ticket_id'], inplace=True)

latlons.set_index(['address'], inplace=True)

X_test.set_index(['ticket_id'], inplace=True)



def join_latlons(df):

    result = df.join(addresses)

    result = result.join(latlons, on='address')

    result.drop('address', axis='columns', inplace=True)

    return result



blight = join_latlons(blight)

X_test = join_latlons(X_test)



blight.dropna(subset=['compliance'], axis='rows', inplace=True)

blight.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.heatmap(blight.isnull(), cbar=False)



plt.show()
paid=blight.loc[blight['compliance'] == 1.0]

unpaid=blight.loc[blight['compliance'] == 0.0]



alpha=0.01



plt.scatter(paid['lat'], paid['lon'], c='g', alpha=alpha)

plt.show()



plt.scatter(unpaid['lat'], unpaid['lon'], c='r', alpha=alpha)

plt.show()



plt.scatter(X_test['lat'], X_test['lon'], alpha=alpha)

plt.show()
blight = blight[(blight['lat'] < 42.50) & (blight['lon'] > -83.4) & (blight['lon'] < 83.75)]
corrmat = blight.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, 

            vmax=.8, 

            square=True,

            fmt='.2f',

            annot_kws={'size': 10},

            annot=True,);
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer



def process(df):

    result = df.copy()

    

    numeric_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='median', fill_value=0)),

        ('scaler', MinMaxScaler())])

    

    result.drop([

        'violation_zip_code', 

        'violation_street_number', 

        'mailing_address_str_number', 

        'judgment_amount',

        'admin_fee',

        'state_fee',

        'clean_up_cost'

    ], axis='columns', inplace=True)

    

    numeric_features = result.select_dtypes(include=['int64', 'float64']).columns

    result = result[numeric_features]

    

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numeric_transformer, numeric_features)])

    

    transformed = Pipeline(steps=[('preprocessor', preprocessor)]).fit_transform(result)



    return pd.DataFrame(data=transformed, index=result.index, columns=result.columns)



X = blight.drop('compliance', axis='columns')

y = blight['compliance']



X = process(X)

X_test = process(X_test)



X.head()
X.describe()
X_test.describe()
corrmat = X.join(y).corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, 

            vmax=.8, 

            square=True,

            fmt='.2f',

            annot_kws={'size': 10},

            annot=True,);
from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



clf = GradientBoostingClassifier()



crossval_scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=3)



crossval_scores
import scikitplot as skplt

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train , y_valid = train_test_split(X, y, random_state=42)



clf.fit(X_train, y_train)



skplt.metrics.plot_roc(y_valid.values, clf.predict_proba(X_valid))

plt.show()
plt.figure()

pd.Series(data=y, index=y.index).hist()

plt.show()
clf.fit(X, y)

preds = pd.Series(data=clf.predict_proba(X_test)[:,1], index=X_test.index)
plt.figure()

preds.hist()

plt.show()