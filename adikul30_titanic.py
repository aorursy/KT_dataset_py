%load_ext autoreload
%autoreload 2

%matplotlib inline
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

!apt update && apt install -y libsm6 libxext6
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import train_test_split


df_raw = pd.read_csv('../input/train.csv', low_memory=False)
test_raw = pd.read_csv('../input/test.csv', low_memory=False)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df_raw.tail().T)
df_raw.head()
df_raw.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
test_raw.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
train_cats(df_raw)
apply_cats(test_raw, df_raw)
df_raw.Sex.cat.categories
df_raw.Sex = df_raw.Sex.cat.codes
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/titanic-raw')
df_raw = pd.read_feather('tmp/titanic-raw')
df, y, nas = proc_df(df_raw, 'Survived')
test, _, nas = proc_df(test_raw, na_dict=nas)
m = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=60)
m.fit(df,y)
m.score(df,y)

test_prediction = m.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': test_prediction})
submission.to_csv('submission.csv', index=False)