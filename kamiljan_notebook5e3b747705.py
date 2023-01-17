# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.float_format', lambda x: '%.3f' % x)
products_df = pd.read_csv("/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
categories_df = pd.read_csv("/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv")
products_df.head()
def plot_missing_data(df):
    columns_with_null = df.columns[df.isna().sum() > 0]
    null_pct = (df[columns_with_null].isna().sum() / df.shape[0]).sort_values(ascending=False) * 100
    plt.figure(figsize=(8,6));
    sns.barplot(y = null_pct.index, x = null_pct, orient='h')
    plt.title('% Na values in dataframe by columns');
plot_missing_data(products_df)
products_df.drop("merchant_profile_picture", axis=1, inplace=True)
print("Unique values: ", products_df["urgency_text"].unique(), "\n")
print("Value counts:\n", products_df['urgency_text'].value_counts())
products_df['urgency_text'] = products_df['urgency_text'].replace({
        'Quantité limitée !': 'QuantityLimited',
        'Réduction sur les achats en gros': 'WholesaleDiscount',
        np.nan: 'noText'
})
print(products_df["urgency_text"].value_counts())
print("Unique values: ", products_df["has_urgency_banner"].unique(), "\n")
print("Value counts:\n", products_df["has_urgency_banner"].value_counts())
products_df['has_urgency_banner'] = products_df['has_urgency_banner'].replace(np.nan,0)
print(products_df["has_urgency_banner"].value_counts())
rating_columns = ['rating_one_count','rating_two_count','rating_three_count','rating_four_count','rating_five_count']
products_df[rating_columns] = products_df[rating_columns].fillna(value=-1)
products_df.loc[products_df['rating_five_count'] == -1, 'rating_count'].value_counts()
products_df[rating_columns] = products_df[rating_columns].replace(-1, 0)
nan_cat_cols = ['origin_country','product_color','product_variation_size_id','merchant_name','merchant_info_subtitle']
products_df[nan_cat_cols] = products_df[nan_cat_cols].replace(np.nan, 'Unknown')
products_df.isnull().sum()[products_df.isnull().sum() > 0]
products_df.drop([
                  "product_picture",
                  "title",
                  "title_orig",
                  'merchant_title',
                  'merchant_name',
                  'merchant_info_subtitle',
                  "product_url",
                  "tags",
                  'rating_five_count',
                  'rating_four_count',
                  'rating_three_count',
                  'rating_two_count',
                  'rating_one_count',
                  "crawl_month"], axis=1, inplace=True)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def get_score(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    return mean_absolute_error(y_test, model.predict(X_test))

def test_on_targets(df):
    target = "units_sold"
    X_u = df.drop(target, axis=1)
    y_u = df[target]
    scaler = StandardScaler()
    X_u = pd.DataFrame(scaler.fit_transform(X_u), index=X_u.index, columns=X_u.columns)
    units_mae = get_score(X_u, y_u)

    target = "rating"
    X_r = df.drop(target, axis=1)
    y_r = df[target]
    scaler = StandardScaler()
    X_r = pd.DataFrame(scaler.fit_transform(X_r), index=X_r.index, columns=X_r.columns)
    rating_mae = get_score(X_r, y_r)
    return units_mae, rating_mae
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders.count import CountEncoder
from category_encoders import CatBoostEncoder
from category_encoders.target_encoder import TargetEncoder

df = products_df.copy()
cat_cols = df.select_dtypes(include=["O"]).columns
oe = OrdinalEncoder()
df[cat_cols] = oe.fit_transform(df[cat_cols])
oe_units_mae, oe_rating_mae = test_on_targets(df)
oe_units_mae, oe_rating_mae
df = products_df.copy()
le = LabelEncoder()
for cat_col in cat_cols:
    df[cat_col] = le.fit_transform(df[cat_col])
    
le_units_mae, le_rating_mae = test_on_targets(df)
le_units_mae, le_rating_mae
df = products_df.copy()
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_cols = pd.DataFrame(ohe.fit_transform(df[cat_cols]))
oh_cols.index = df.index
num_cols = df.drop(cat_cols, axis=1)
df = pd.concat([num_cols, oh_cols], axis=1)

ohe_units_mae, ohe_rating_mae = test_on_targets(df)
ohe_units_mae, ohe_rating_mae
df = products_df.copy()
ce = CountEncoder()
df[cat_cols] = ce.fit_transform(df[cat_cols])
ce_units_mae, ce_rating_mae = test_on_targets(df)
ce_units_mae, ce_rating_mae
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

categorical_transformer = Pipeline(steps=[('cbe', CatBoostEncoder())])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_cols),
    ])

model = LinearRegression()
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', model)
                      ])

target = "units_sold"
df = products_df.copy()
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pipe.fit(X_train, y_train)
cbe_units_mae = mean_absolute_error(y_test, pipe.predict(X_test))

target = "rating"
df = products_df.copy()
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pipe.fit(X_train, y_train)
cbe_rating_mae = mean_absolute_error(y_test, pipe.predict(X_test))
cbe_units_mae, cbe_rating_mae
categorical_transformer = Pipeline(steps=[('te', TargetEncoder())])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_cols),
    ])

model = LinearRegression()
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', model)
                      ])

target = "units_sold"
df = products_df.copy()
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pipe.fit(X_train, y_train)
te_units_mae = mean_absolute_error(y_test, pipe.predict(X_test))

target = "rating"
df = products_df.copy()
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pipe.fit(X_train, y_train)
te_rating_mae = mean_absolute_error(y_test, pipe.predict(X_test))
te_units_mae, te_rating_mae
cat_encoders = pd.DataFrame([[ohe_units_mae, ohe_rating_mae], [ce_units_mae, ce_rating_mae], [te_units_mae, te_rating_mae], [cbe_units_mae, cbe_rating_mae], [oe_units_mae, oe_rating_mae], [le_units_mae, le_rating_mae]],
                            columns=["Units_sold", "Rating"], index=["One Hot Encoder", "Count Encoder", "Target Encoder", "Cat Boost Encoder", "Ordinal Encoder", "Label Encoder"])
cat_encoders.sort_values(by="Units_sold")
cat_encoders.sort_values(by="Rating")
oe = OrdinalEncoder()
df = products_df.copy()
df[cat_cols] = oe.fit_transform(df[cat_cols])
import statsmodels.api as sm

def get_reg_summary(X, y):
    X_sm = sm.add_constant(X)
    est = sm.OLS(y, X_sm)
    est2 = est.fit()
    return est2.params
target = "units_sold"
X = df.drop(target, axis=1)
y = df[target]
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
coeffs = get_reg_summary(X, y).sort_values().drop("const")
sns.barplot(x=coeffs.head(10).values, y=coeffs.head(10).index)
plt.show()
sns.barplot(x=coeffs.tail(10).values, y=coeffs.tail(10).index)
plt.show()
target = "rating"
X = df.drop(target, axis=1)
y = df[target]
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
coeffs = get_reg_summary(X, y).sort_values().drop("const")
sns.barplot(x=coeffs.head(10).values, y=coeffs.head(10).index)
plt.show()
sns.barplot(x=coeffs.tail(10).values, y=coeffs.tail(10).index)
plt.show()