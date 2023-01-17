import numpy as np # linear algebra

import pandas as pd # data processing
# load data

df = pd.read_csv('/kaggle/input/individual-company-sales-data/sales_data.csv')

df.head()
# shape of dataframe

df.shape
# dataframe dtypes for each feature

df.dtypes
for cat in df.columns:

    print(cat, df[cat].unique())
df['gender'] = df.gender.replace('U', np.NaN)

df['age'] = df.age.replace('1_Unk', np.NaN)

df['child'] = df.child.replace('U', np.NaN)

df['child'] = df.child.replace('0', np.NaN)
df.isnull().sum()
# relative

df.isnull().sum() / df.shape[0] * 100
def category_stackedbar(df, category):

    '''Returns stacked bar plot'''

    return pd.DataFrame(

        df.groupby(category).count()['flag'] / df.groupby(category).count()['flag'].sum() * 100).rename(columns={"flag": "%"}).T.plot(

            kind='bar', 

            stacked=True

    );
category_stackedbar(df, 'house_owner');
df['house_owner'] = df['house_owner'].fillna(df.mode()['house_owner'][0])
category_stackedbar(df, 'age');
df = df.dropna(subset=['age'])
category_stackedbar(df, 'child');
# percentage of null values in *child*

(df.isnull().sum() / df.shape[0] * 100)['child']
df = df.drop('child', axis=1)
category_stackedbar(df, 'marriage');
df['marriage'] = df['marriage'].fillna(df.mode()['marriage'][0])
df = df.dropna(subset=['gender', 'education'])
# checking data cleaning

df.isnull().sum()
df.dtypes
df['flag'] = df['flag'].apply(lambda value: 1 if value == 'Y' else 0)

df['online'] = df['online'].apply(lambda value: 1 if value == 'Y' else 0)
df.dtypes
# explore categories of features with hierarchy

for cat in ['education', 'age', 'mortgage', 'fam_income']:

    print(cat, df[cat].unique())
# education to integer

df['education'] = df['education'].apply(lambda value: int(value[0]) + 1)
# age to integer

df['age'] = df['age'].apply(lambda value: int(value[0]) - 1)
# mortgage to integer

df['mortgage'] = df['mortgage'].apply(lambda value: int(value[0]))
#fam_income label dictionary

dict_fam_income_label = {}

for i, char in enumerate(sorted(df['fam_income'].unique().tolist())):

    dict_fam_income_label[char] = i + 1
df['fam_income'] = df['fam_income'].apply(lambda value: dict_fam_income_label[value])
dummy_features = ['gender', 'customer_psy', 'occupation', 'house_owner', 'region', 'marriage']
# explore categories of dummy features

for cat in dummy_features:

    print(cat, df[cat].unique())
def apply_dummy(df, cat, drop_first=True):

    return pd.concat([df, pd.get_dummies(df[cat], prefix=cat, drop_first=drop_first)], axis=1).drop(cat, axis=1)
for cat in dummy_features:

    df = apply_dummy(df, cat)
# dataframe with just numbers

df.head()
from xgboost import XGBClassifier
X = df.drop('flag', axis=1)

y = df['flag']
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# scale to handle imbalanced dataset

scale = y_train[y_train == 0].count() / y_train[y_train == 1].count()
xgbmodel = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=1000, scale_pos_weight=scale)
xgbmodel.fit(X_train, y_train)
y_pred_test = xgbmodel.predict(X_test)

y_pred_train = xgbmodel.predict(X_train)
print('Train')

print('Precision: {:.2f}% \tRecall: {:.2f}% \t\tF1 Score: {:.2f}%'.format(precision_score(y_train, y_pred_train)*100, recall_score(y_train, y_pred_train)*100, f1_score(y_train, y_pred_train)*100))
print('Test')

print('Precision: {:.2f}% \tRecall: {:.2f}% \t\tF1 Score: {:.2f}%'.format(precision_score(y_test, y_pred_test)*100, recall_score(y_test, y_pred_test)*100, f1_score(y_test, y_pred_test)*100))
import shap
# load JS visualization code to notebook

shap.initjs()
# explain the model's predictions using SHAP values

# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

explainer = shap.TreeExplainer(xgbmodel)

shap_values = explainer.shap_values(X_train)
# summarize the effects of all the features

shap.summary_plot(shap_values, X_train)
shap.summary_plot(shap_values, X, plot_type="bar")
shap.dependence_plot("age", shap_values, X_train)
shap.dependence_plot("education", shap_values, X_train)