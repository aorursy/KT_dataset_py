# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_raw =  pd.read_csv("/kaggle/input/freddie-mac-singlefamily-loanlevel-dataset/loan_level_500k.csv",
                  index_col=19,
                  low_memory=False)
df = df_raw.copy()
df.head()
df.info()
df.isna().sum()
df.describe()
df.dropna().describe()
df.ORIGINAL_DEBT_TO_INCOME_RATIO                                                        
def get_plot_and_stat(series):
    series.plot.box(vert=False)
    print(f"Med:{series.median()}, mean{series.mean()}")   

get_plot_and_stat(df.CREDIT_SCORE)
df["Missing_CREDIT_SCORE"] = df.CREDIT_SCORE.isna()
df.CREDIT_SCORE.fillna(df.CREDIT_SCORE.median(),inplace=True)
get_plot_and_stat(df.CREDIT_SCORE)
get_plot_and_stat(df.MORTGAGE_INSURANCE_PERCENTAGE)
df["Missing_MORTGAGE_INSURANCE_PERCENTAGE"] = df.MORTGAGE_INSURANCE_PERCENTAGE.isna()
df.MORTGAGE_INSURANCE_PERCENTAGE.fillna(df.MORTGAGE_INSURANCE_PERCENTAGE.mean(),inplace=True)
get_plot_and_stat(df.MORTGAGE_INSURANCE_PERCENTAGE)
get_plot_and_stat(df.ORIGINAL_COMBINED_LOAN_TO_VALUE)
df["Missing_ORIGINAL_COMBINED_LOAN_TO_VALUE"] = df.ORIGINAL_COMBINED_LOAN_TO_VALUE.isna()
df.ORIGINAL_COMBINED_LOAN_TO_VALUE.fillna(df.ORIGINAL_COMBINED_LOAN_TO_VALUE.mean(),inplace=True)
get_plot_and_stat(df.ORIGINAL_COMBINED_LOAN_TO_VALUE)
get_plot_and_stat(df.ORIGINAL_DEBT_TO_INCOME_RATIO)
df["Missing_ORIGINAL_DEBT_TO_INCOME_RATIO"] = df.ORIGINAL_DEBT_TO_INCOME_RATIO.isna()
df.ORIGINAL_DEBT_TO_INCOME_RATIO.fillna(df.ORIGINAL_DEBT_TO_INCOME_RATIO.mean(),inplace=True)
get_plot_and_stat(df.ORIGINAL_DEBT_TO_INCOME_RATIO)


df["Missing_ORIGINAL_LOAN_TO_VALUE"] = df.ORIGINAL_LOAN_TO_VALUE.isna()
df.ORIGINAL_LOAN_TO_VALUE.fillna(df.ORIGINAL_LOAN_TO_VALUE.mean(),inplace=True)
get_plot_and_stat(df.ORIGINAL_LOAN_TO_VALUE)

get_plot_and_stat(df.NUMBER_OF_BORROWERS)
df["Missing_NUMBER_OF_BORROWERS"] = df.NUMBER_OF_BORROWERS.isna()
df.NUMBER_OF_BORROWERS.fillna(df.NUMBER_OF_BORROWERS.mode()[0],inplace=True)
get_plot_and_stat(df.NUMBER_OF_BORROWERS)

df["Missing_NUMBER_OF_UNITS"] = df.NUMBER_OF_UNITS.isna()
df.NUMBER_OF_UNITS.fillna(df.NUMBER_OF_UNITS.mode()[0],inplace=True)
get_plot_and_stat(df.NUMBER_OF_UNITS)
# get_plot_and_stat(df.FIRST_TIME_HOMEBUYER_FLAG)
df["Missing_FIRST_TIME_HOMEBUYER_FLAG"] = df.FIRST_TIME_HOMEBUYER_FLAG.isna()
df.FIRST_TIME_HOMEBUYER_FLAG.fillna(df.FIRST_TIME_HOMEBUYER_FLAG.mode()[0],inplace=True)
# get_plot_and_stat(df.FIRST_TIME_HOMEBUYER_FLAG)
get_plot_and_stat(df.METROPOLITAN_STATISTICAL_AREA)
df["Missing_METROPOLITAN_STATISTICAL_AREA"] = df.METROPOLITAN_STATISTICAL_AREA.isna()
df.METROPOLITAN_STATISTICAL_AREA.fillna(df.METROPOLITAN_STATISTICAL_AREA.mode()[0],inplace=True)
# get_plot_and_stat(df.METROPOLITAN_STATISTICAL_AREA)
# get_plot_and_stat(df.PREPAYMENT_PENALTY_MORTGAGE_FLAG)
df["Missing_PREPAYMENT_PENALTY_MORTGAGE_FLAG"] = df.PREPAYMENT_PENALTY_MORTGAGE_FLAG.isna()
df.PREPAYMENT_PENALTY_MORTGAGE_FLAG.fillna(df.PREPAYMENT_PENALTY_MORTGAGE_FLAG.mode()[0],inplace=True)
# get_plot_and_stat(df.PREPAYMENT_PENALTY_MORTGAGE_FLAG)

df["Missing_POSTAL_CODE"] = df.POSTAL_CODE.isna()
df.POSTAL_CODE.fillna(df.POSTAL_CODE.mode()[0],inplace=True)
# get_plot_and_stat(df.PREPAYMENT_PENALTY_MORTGAGE_FLAG)

df["Missing_PROPERTY_TYPE"] = df.PROPERTY_TYPE.isna()
df.PROPERTY_TYPE.fillna(df.PROPERTY_TYPE.mode()[0],inplace=True)
# get_plot_and_stat(df.PREPAYMENT_PENALTY_MORTGAGE_FLAG)
df.isna().sum()
filled_numeric_df = (df.select_dtypes(["number","bool"])).apply(pd.to_numeric)
filled_numeric_df.head()
filled_cat_df = df.select_dtypes(["category","object"])
filled_cat_df.head()
from sklearn.preprocessing import *

end = OneHotEncoder()
end.fit(filled_cat_df.to_numpy())
dummy_vars = end.fit_transform(filled_cat_df).toarray()

dummy_df = pd.DataFrame(dummy_vars, index=filled_numeric_df.index,columns=end.get_feature_names())
dummy_df.head()
joined_dum_cat_num_df = filled_numeric_df.join(dummy_df)
joined_dum_cat_num_df.head()
y=  joined_dum_cat_num_df.DELINQUENT
x = joined_dum_cat_num_df.drop(["DELINQUENT","PREPAID"],axis=1)
x.head()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(oob_score=True,min_samples_leaf=int(175 **.5))
clf.fit(x,y)
def plot_feature_importance_for_model(data, feature_names, kind="bar", scale=False, keep_n=None, plot_kwargs=None, ):
    plot_kwargs = plot_kwargs or {}
    kind = kind.lower()
    keep_n = keep_n or len(feature_names)
    df = get_feature_importance_sorted(data, feature_names).head(keep_n)

    if scale:
        df.Importance /= df.Importance.sum()
    if kind == "pie":

        if not plot_kwargs:
            plot_kwargs = {"autopct": "%1.2f%%",
                           "legend": False,
                           "startangle": 15,
                           "shadow": True}

        plot_kwargs["y"] = "Importance"
        plot_kwargs["labels"] = df.Features
        df.plot.pie(**plot_kwargs)

    elif kind == "line":
        df.plot.line(**plot_kwargs)
    else:
        df.plot.bar(**plot_kwargs)


def get_feature_importance_sorted(data, feature_names):
    return pd.DataFrame(zip(data, feature_names), index=range(
        len(data)),
        columns=("Importance", "Features")
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

clf.oob_score_

plot_feature_importance_for_model(clf.feature_importances_,x.columns,kind="pie",
                                  keep_n=10,
                                  scale=True,
                                )
new_x = x [get_feature_importance_sorted(clf.feature_importances_,x.columns,).Features]
new_x
clf.fit(new_x,y)
plot_feature_importance_for_model(clf.feature_importances_,new_x.columns,kind="pie",
                                  keep_n=10,
                                  scale=True,
                                )
clf.oob_score_

# def get_pd_dataframe_vis(data, file_name=None, online=False):
#     profile = pandas_profiling.ProfileReport(data)
#     profile.to_file(f"{file_name}_profile.html")
# #     data.iplot(filename = f'{file_name}_cuff',asUrl=True,online=online)
# # get_pd_dataframe_vis(df,"pd")
# # %%html
# # <a href=" pd_profile.html">profile_link</a>