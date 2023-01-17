# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, SelectPercentile

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier



from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
pd.options.display.max_columns = 30
d_data = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
d_data.head()
d_data.shape
d_data.dtypes
def to_float(x):

    try:

        return float(x)

    except:

        return None
d_data["TotalCharges"] = d_data.TotalCharges.apply(to_float)
d_data.TotalCharges.fillna(d_data.TotalCharges.mean(skipna=True), inplace=True)
d_data["TotalCharges"] = d_data.TotalCharges.astype(float)
d_data.dtypes
d_data.columns
category_feats = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 

                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']



numeric_feats = ['tenure', 'MonthlyCharges', 'TotalCharges']



target = ['Churn']
for feat in category_feats:

    print(feat, len(d_data[feat].unique()))
d_data['Churn_le'] = d_data.Churn.map({'Yes': 1, 'No': 0})
def label_encoder(d_data):

    

    le_dict = {}

    for feat in category_feats:

        le_dict[feat] = LabelEncoder()

        d_data[feat + '_le'] = le_dict[feat].fit_transform(d_data[feat])

        

    return d_data
def onehot_encoder(d_data):

    ohe_dict = {}

    for feat in category_feats:

        ohe_dict[feat] = OneHotEncoder()

        ohe = ohe_dict[feat].fit_transform(d_data[feat].values.reshape(-1, 1))

        columns_ohe = [feat + '_' + str(i) for i in range(ohe.shape[1])]

        d_data = pd.concat([d_data, pd.DataFrame(data = ohe.toarray(), columns = columns_ohe)], axis = 1)

        

    return d_data
le = False

if le == True:

    d_data = label_encoder(d_data)

else:

    d_data = onehot_encoder(d_data)
ss_dict = {}

for feat in numeric_feats:

    ss_dict[feat] = StandardScaler()

    d_data[feat + '_ss'] = ss_dict[feat].fit_transform(d_data[feat].values.reshape(-1, 1))
churn_count = d_data.Churn.value_counts().plot.bar(y = 'Churn', color = ['red', 'green'])
fig, axes = plt.subplots(4,4, figsize = (15, 15))

row = 0

col = 0

for feat in category_feats:

    fig.tight_layout()

    d_count = d_data[feat].value_counts().reset_index()

    sns.barplot(x = "index", y = feat, data = d_count, ax = axes[row, col])

    

    if col == 3:

        row += 1

        col = 0

    else:

        col += 1
fig, axes = plt.subplots(1, 3, figsize=(8, 5))

for idx, feat in enumerate(numeric_feats):

    fig.tight_layout()

    sns.boxplot(y = feat, data = d_data, ax = axes[idx])
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for idx, feat in enumerate(numeric_feats):

    fig.tight_layout()

    sns.boxplot(x = 'Churn', y = feat, data = d_data, ax = axes[idx])
def remove_outlier(d_data):

    

    ## Tenure

    tenure_yes = d_data[d_data["Churn"] == "Yes"].tenure.describe()



    IQR = tenure_yes["75%"] - tenure_yes["25%"]

    ceiling = tenure_yes["75%"] + 1.5 * IQR



    # replace outlier with ceiling

    d_data.loc[(d_data["Churn"] == "Yes") & (d_data["tenure"]  > ceiling), "tenure"] = ceiling

    

    ## Total Charges

    tc_yes = d_data[d_data["Churn"] == "Yes"].TotalCharges.describe()



    IQR = tc_yes["75%"] - tc_yes["25%"]

    ceiling = tc_yes["75%"] + 1.5 * IQR



    # replace outlier with ceiling

    d_data.loc[(d_data["Churn"] == "Yes") & (d_data["TotalCharges"]  > ceiling), "TotalCharges"] = ceiling

    

    return d_data
d_data = remove_outlier(d_data)
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for idx, feat in enumerate(numeric_feats):

    fig.tight_layout()

    sns.boxplot(x = 'Churn', y = feat, data = d_data, ax = axes[idx])
g = sns.FacetGrid(d_data, col="Churn")

g.map(plt.hist, "MonthlyCharges");
g = sns.FacetGrid(d_data, col="Churn")

g.map(plt.scatter, "tenure", "MonthlyCharges", alpha=.7)

g.add_legend();
def hexbin(x, y, color, **kwargs):

    cmap = sns.light_palette(color, as_cmap=True)

    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)



with sns.axes_style("dark"):

    g = sns.FacetGrid(d_data, hue="Churn", col="Churn", height=4)

g.map(hexbin, "tenure", "MonthlyCharges", extent=[0, 120, 0, 80]);
d_data.columns
if le == True:

    cat_cols = ['gender_le', 'SeniorCitizen_le', 'Partner_le',

           'Dependents_le', 'PhoneService_le', 'MultipleLines_le',

           'InternetService_le', 'OnlineSecurity_le', 'OnlineBackup_le',

           'DeviceProtection_le', 'TechSupport_le', 'StreamingTV_le',

           'StreamingMovies_le', 'Contract_le', 'PaperlessBilling_le',

           'PaymentMethod_le']

else:

    cat_cols = ['gender_0', 'gender_1', 'SeniorCitizen_0',

       'SeniorCitizen_1', 'Partner_0', 'Partner_1', 'Dependents_0',

       'Dependents_1', 'PhoneService_0', 'PhoneService_1', 'MultipleLines_0',

       'MultipleLines_1', 'MultipleLines_2', 'InternetService_0',

       'InternetService_1', 'InternetService_2', 'OnlineSecurity_0',

       'OnlineSecurity_1', 'OnlineSecurity_2', 'OnlineBackup_0',

       'OnlineBackup_1', 'OnlineBackup_2', 'DeviceProtection_0',

       'DeviceProtection_1', 'DeviceProtection_2', 'TechSupport_0',

       'TechSupport_1', 'TechSupport_2', 'StreamingTV_0', 'StreamingTV_1',

       'StreamingTV_2', 'StreamingMovies_0', 'StreamingMovies_1',

       'StreamingMovies_2', 'Contract_0', 'Contract_1', 'Contract_2',

       'PaperlessBilling_0', 'PaperlessBilling_1', 'PaymentMethod_0',

       'PaymentMethod_1', 'PaymentMethod_2', 'PaymentMethod_3']

    

num_cols = ['MonthlyCharges_ss', 'TotalCharges_ss']



target = ['Churn_le']
X = d_data[cat_cols + num_cols]

y = d_data.Churn_le 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
y_train.value_counts(), y_test.value_counts()
def scoring(y_test, y_pred):

    

    print("accuracy:", accuracy_score(y_test, y_pred))

    print("recall:", recall_score(y_test, y_pred))

    print("precision:", precision_score(y_test, y_pred))

    print("f1 score:", f1_score(y_test, y_pred))
def rfc_model(X_train, X_test, y_train, y_test):

    

    rfc = RandomForestClassifier(n_estimators = 200, max_depth = 20, max_features = None)

    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    scoring(y_test, y_pred)
def rfc_fs_model(X_train, X_test, y_train, y_test):

    

    mi = SelectPercentile(mutual_info_classif, percentile = 50)

    X_train_fs = mi.fit_transform(X_train, y_train)

    X_test_fs = mi.transform(X_test)

    rfc = RandomForestClassifier(n_estimators = 200, max_depth = 20, max_features = None)

    rfc.fit(X_train_fs, y_train)

    y_pred = rfc.predict(X_test_fs)

    scoring(y_test, y_pred)
def xgb_model(X_train, X_test, y_train, y_test):

    

    xgb = XGBClassifier()

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    scoring(y_test, y_pred)
def xgb_fs_model(X_train, X_test, y_train, y_test):

    

    mi = SelectPercentile(mutual_info_classif, percentile = 50)

    X_train_fs = mi.fit_transform(X_train, y_train)

    X_test_fs = mi.transform(X_test)

    xgb = XGBClassifier()

    xgb.fit(X_train_fs, y_train)

    y_pred = xgb.predict(X_test_fs)

    scoring(y_test, y_pred)
def cat_model(X_train, X_test, y_train, y_test):

    

    cat = CatBoostClassifier(verbose=False)

    cat.fit(X_train, y_train)

    y_pred = cat.predict(X_test)

    scoring(y_test, y_pred)
def cat_fs_model(X_train, X_test, y_train, y_test, percent):

    

    mi = SelectPercentile(mutual_info_classif, percentile = percent)

    X_train_fs = mi.fit_transform(X_train, y_train)

    X_test_fs = mi.transform(X_test)

    cat = CatBoostClassifier(verbose=False)

    cat.fit(X_train_fs, y_train)

    y_pred = cat.predict(X_test_fs)

    scoring(y_test, y_pred)
def etc_model(X_train, X_test, y_train, y_test):

    

    etc = ExtraTreesClassifier()

    etc.fit(X_train, y_train)

    y_pred = etc.predict(X_test)

#     y_pred = np.where(y_pred > 0.5, 1, 0)

    scoring(y_test, y_pred)
rfc_model(X_train, X_test, y_train, y_test)
rfc_fs_model(X_train, X_test, y_train, y_test)
xgb_model(X_train, X_test, y_train, y_test)
xgb_fs_model(X_train, X_test, y_train, y_test)
cat_model(X_train, X_test, y_train, y_test)
cat_fs_model(X_train, X_test, y_train, y_test, 20)
models = [

 ('RandomForestClassifier', RandomForestClassifier()),

 ('XGBoost', XGBClassifier()),

 ('Catboost', CatBoostClassifier(verbose=False)),

 ('ExtraTreesClassifier', ExtraTreesClassifier()),

 ('AdaBoostClassifier', AdaBoostClassifier(RandomForestClassifier())), 

 ('BaggingClassifier', BaggingClassifier(RandomForestClassifier())), 

 ('GradientBoostingClassifier', GradientBoostingClassifier())

]
for name, algo in models:

    f1score = cross_val_score(algo, X, y, scoring= 'f1', cv = 5)

    print("Algorithm:", name)

    print("f1 score:", f1score.mean(), "std:", f1score.std())

    print("\n")