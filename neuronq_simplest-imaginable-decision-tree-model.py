# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
!ls -lh '../input'

# Any results you write to the current directory are saved as output.
INPUT_DIR = '../input/'
OUTPUT_DIR = './'
%load_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
# load
df_raw = pd.read_csv(f'{INPUT_DIR}train.csv', low_memory=False)
# quick sanity check that it loaded the right thing
display_all(df_raw.tail().T)
# types
df_raw.dtypes
# missing values
display_all(df_raw.isnull().sum().sort_index() / len(df_raw))
for n, c in df_raw.items():
    if pd.api.types.is_numeric_dtype(c):
        if pd.isnull(c).sum():
            print("Column %s has %d missing values" % (
                n, pd.isnull(c).sum()))
df_raw.columns.sort_values()
def convert_cats(df, extra_cats):
    """Convert string values + what we know is category, to categorical vars"""
    for n, c in df.items():
        if pd.api.types.is_string_dtype(c) or n in extra_cats:
            df[n] = c.astype('category').cat.as_ordered()
convert_cats(df_raw, extra_cats={'Pclass'})
df_raw.dtypes
display_all(df_raw.describe(include='all').T)
df = df_raw.copy()
def fix_missing(df):
    for n, c in df.items():
        if pd.api.types.is_numeric_dtype(c):
            if pd.isnull(c).sum():
                df[n] = c.fillna(c.median())
fix_missing(df)
def numericalize(df):
    """Numericalize categories and get rid of -1's for NaNs"""
    for n, c in df.items():
        if not pd.api.types.is_numeric_dtype(c):
            df[n] = df[n].cat.codes + 1  # +1: NaNs -1 -> 0
numericalize(df)
df.dtypes
y = df.Survived.values
df.drop('Survived', axis=1, inplace=True)
VAL_FR = 0.2
trn_sz = int(len(df) * (1 - VAL_FR))
x_trn = df.iloc[:trn_sz]
y_trn = y[:trn_sz]
x_val = df.iloc[trn_sz:]
y_val = y[trn_sz:]
def proc_df(df):
    convert_cats(df, extra_cats={'Pclass'})
    fix_missing(df)
    numericalize(df)
m = DecisionTreeClassifier(max_depth=5)
m.fit(x_trn, y_trn)
print("Training score: %.2f%%" % (m.score(x_trn, y_trn) * 100))
dot_data = export_graphviz(
    m,
    out_file=None,
    feature_names=df.columns,
    class_names=['died', 'survived'],
    filled=True,
    rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)
graph
# y_pred = m.predict(x_val)
# accuracy_score(y_val, y_pred)
print("Validation score: %.2f%%" % (m.score(x_val, y_val) * 100))
from sklearn.model_selection import cross_val_score
m_cv = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(m_cv, df, y, cv=10)
print("Scores:", scores)
print("Accuracy: %0.2f%% (+/- %0.2f%%)" % (scores.mean() * 100, scores.std() * 2 * 100))
M = DecisionTreeClassifier(max_depth=5)
M.fit(df, y)
print("Final training score: %.2f%%" % (M.score(df, y) * 100))
df_test = pd.read_csv(f'{INPUT_DIR}test.csv')
proc_df(df_test)
print(df_test.dtypes)
df_test.head()
y_pred_final = M.predict(df_test)
result = pd.DataFrame({'PassengerId': df_test.PassengerId,
                       'Survived': y_pred_final})
result.head()
result.to_csv(f'{OUTPUT_DIR}results.csv', index=False)