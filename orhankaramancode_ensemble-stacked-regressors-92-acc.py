# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import warnings



warnings.simplefilter(action="ignore", category=FutureWarning)

import missingno as msno

from sklearn.metrics import mean_squared_error

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option("display.max_rows", 2000)

pd.set_option("display.max_columns", 500)

pd.set_option("display.width", 100)

pd.set_option("display.max_colwidth", 500)

pd.set_option("display.float_format", lambda x: "%.2f" % x)



from sklearn.inspection import permutation_importance

from IPython.display import display, HTML, display_html



display(HTML("<style>.container { width:100% !important; }</style>"))

from matplotlib.ticker import MaxNLocator

import matplotlib.ticker as mticker

from lightgbm import LGBMRegressor

import optuna

from mlxtend.regressor import StackingCVRegressor

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer

from sklearn.model_selection import cross_val_score, KFold

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer

import math

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



cwd = os.getcwd()

print(cwd)





import gc



gc.enable()



from bokeh.io import output_notebook, show

from bokeh.models import (

    BasicTicker,

    ColorBar,

    ColumnDataSource,

    LinearColorMapper,

    PrintfTickFormatter,

)

from bokeh.plotting import figure

from bokeh.transform import transform





random_state = 55



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

%%javascript

IPython.OutputArea.auto_scroll_threshold = 9999;
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print(

    f"Train set contains {train.shape[0]} rows,{train.shape[1]} columns. \nTest set contains {test.shape[0]} rows, {test.shape[1]} columns.\n"

)

print(

    f"{set(train.columns) - set(test.columns)} are the fields that are IN TRAIN and NOT IN TEST.\n {set(test.columns) - set(train.columns)} are the fields that are IN TEST and NOT IN TRAIN. "

)

train.info()
display(

    train.describe().iloc[:, 0:18].applymap("{:,g}".format),

    train.describe().iloc[:, 18:].applymap("{:,g}".format),

)

sample_count = 5



display(

    train.sample(sample_count, random_state=random_state)

    .iloc[:, :30]

    .style.hide_index(),

    train.sample(sample_count, random_state=random_state)

    .iloc[:, 30:60]

    .style.hide_index(),

    train.sample(sample_count, random_state=random_state)

    .iloc[:, 60:]

    .style.hide_index(),

)

Id = "Id"



submission_ID = test.loc[:, Id]



train.drop(Id, axis=1, inplace=True)

test.drop(Id, axis=1, inplace=True)



# For identification purposes

train.loc[:, "Train"] = 1

test.loc[:, "Train"] = 0



test["SalePrice"] = 0



stacked_DF = pd.concat([train, test], ignore_index=True)

plt.style.use('ggplot')

params = {

    "axes.labelsize": 18,

    "axes.titlesize": 20,

    "xtick.labelsize": 16,

    "ytick.labelsize": 16,

}

plt.rcParams.update(params)



(fig, ax) = plt.subplots(nrows=2, ncols=1, figsize=[12, 12], sharex=True)



ax[0].set_xlabel("SalePrice")

ax[0].set_ylabel("Count")



ax[1].set_xlabel("SalePrice")

ax[1].set_ylabel("Count")



plot_X = stacked_DF.loc[stacked_DF["Train"] == 1]["SalePrice"]



plot = ax[0].hist(plot_X, bins=75, log=False, color="darkslategrey")

plot = ax[1].hist(plot_X, bins=75, log=True, color="darkslategrey")



ax[0].set_title("Distribution of SalePrice")

ax[1].set_title("Distribution of SalePrice (Log Transformed)")

params = {"axes.labelsize": 20, 

          "xtick.labelsize": 14, 

          "ytick.labelsize": 14}

plt.rcParams.update(params)



features_to_viz = [

    "GrLivArea",

    "YearBuilt",

    "WoodDeckSF",

    "LotArea",

    "GarageArea",

    "1stFlrSF",

    "2ndFlrSF",

    "TotalBsmtSF",

    "LotFrontage",

    "GarageYrBlt",

]



# Because there are a lot of variables to vizualize,

# sorting them helps me keep track of which variable is where



features_to_viz = sorted(features_to_viz)



ncols = 1

nrows = math.ceil(len(features_to_viz) / ncols)

unused = nrows * ncols - len(features_to_viz)



figw = ncols * 10

figh = nrows * 8



(fig, ax) = plt.subplots(nrows, ncols, sharey=True, figsize=(figw, figh))

fig.subplots_adjust(hspace=0.3)

ax = ax.flatten()



for i in range(unused, 0, -1):

    fig.delaxes(ax[-i])



for (n, col) in enumerate(features_to_viz):

    if n % 2 != 0:

        ax[n].yaxis.label.set_visible(False)

    ax[n].set_xlabel(col)

    ax[n].set_ylabel("SalePrice")

    sns.scatterplot(

        x=col,

        y="SalePrice",

        data=stacked_DF.loc[stacked_DF["Train"] == 1],

        hue="SalePrice",

        palette="gist_earth",

        s=75,

        legend=False,

        ax=ax[n],

    )



plt.show()

features_to_viz = [

    "Neighborhood",

    "BsmtQual",

    "ExterQual",

    "FireplaceQu",

    "ExterCond",

    "KitchenQual",

    "LotShape",

    "OverallQual",

    "FullBath",

    "HalfBath",

    "TotRmsAbvGrd",

    "Fireplaces",

    "KitchenAbvGr",

]



# Because there are a lot of variables to vizualize,

# sorting them helps me keep track of which var is where



features_to_viz = sorted(features_to_viz)



ncols = 1

nrows = math.ceil(len(features_to_viz) / ncols)

unused = nrows * ncols - len(features_to_viz)



(figw, figh) = (ncols * 10, nrows * 8)



(fig, ax) = plt.subplots(nrows, ncols, figsize=(figw, figh))

fig.subplots_adjust(hspace=0.2, wspace=0.2)



# ax = ax.flatten()

# for i in range(unused, 0, -1):

#     fig.delaxes(ax[-i])



for (n, col) in enumerate(features_to_viz):

    ordering = (

        stacked_DF.loc[stacked_DF["Train"] == 1]

        .groupby(by=col)["SalePrice"]

        .median()

        .sort_values()

        .index

    )

    sns.boxplot(

        x="SalePrice",

        y=col,

        data=stacked_DF.loc[stacked_DF["Train"] == 1],

        order=ordering,

        ax=ax[n],

        orient="h",

    )



plt.show()
print("Missing Value Counts in Train DF")

stacked_DF[stacked_DF["Train"] == 1].isna().sum()[

    stacked_DF[stacked_DF["Train"] == 1].isna().sum() > 0

].sort_values(ascending=False)
print("Missing Values in Test DF")

stacked_DF[stacked_DF["Train"] == 0].isna().sum()[

    stacked_DF[stacked_DF["Train"] == 0].isna().sum() > 0

].sort_values(ascending=False)

# Check missing records in train set

na_cols = (stacked_DF.isna().sum()[stacked_DF.isna().sum() > 0]).index

mat = msno.matrix(

    stacked_DF.loc[:, na_cols], labels=True, figsize=(16, 14), fontsize=16, inline=False

)
# Assuming Neighborhood and MSZoning are related.

lookup = (

    stacked_DF.loc[stacked_DF["Train"] == 1]

    .groupby(by="Neighborhood")["MSZoning"]

    .agg(pd.Series.mode)

)

stacked_DF["MSZoning"] = stacked_DF["MSZoning"].fillna(

    stacked_DF["Neighborhood"].map(lookup)

)



# Assuming KitchenQual and OverallQual are related.

lookup = (

    stacked_DF.loc[stacked_DF["Train"] == 1]

    .groupby(by="OverallQual")["KitchenQual"]

    .agg(pd.Series.mode)

)

stacked_DF["KitchenQual"] = stacked_DF["KitchenQual"].fillna(

    stacked_DF["OverallQual"].map(lookup)

)



# For these features I replace nan with a string indicator: "missing"

cols_na_to_missing = {

    "Alley",

    "BsmtCond",

    "BsmtExposure",

    "BsmtFinType1",

    "BsmtFullBath",

    "BsmtQual",

    "Fence",

    "FireplaceQu",

    "GarageCond",

    "GarageFinish",

    "GarageQual",

    "GarageType",

    "MasVnrType",

    "MiscFeature",

    "PoolQC",

    "BsmtFinType2",

}



# For these features I replace nan with the integer 0

cols_na_to_zero = {

    # 'BsmtUnfSF',

    "GarageArea",

    "GarageCars",

    "TotalBsmtSF",

    "MasVnrArea",

    "BsmtFinSF1",

    "BsmtFinSF2",

    "BsmtFullBath",

    "BsmtHalfBath",

    "GarageYrBlt",

}



# For these features I replace nan with the mode of the feature the record is missing.

cols_na_to_mode = {

    "Functional",

    "Electrical",

    "Utilities",

    "Exterior1st",

    "Exterior2nd",

    "SaleType",

}



for col in cols_na_to_missing:

    stacked_DF[col] = stacked_DF[col].astype(object).fillna("Missing")



for col in cols_na_to_zero:

    stacked_DF[col] = stacked_DF[col].astype(object).fillna(0)



for col in cols_na_to_mode:

    stacked_DF[col] = (

        stacked_DF[col]

        .astype(object)

        .fillna(stacked_DF.loc[stacked_DF["Train"] == 1, col].mode()[0])

    )



# Imputing remaining missing values with the help of iterative imputer.

num_features = stacked_DF.drop(columns=["Train"]).select_dtypes("number").columns

imputer = IterativeImputer(

    RandomForestRegressor(max_depth=8),

    n_nearest_features=10,

    max_iter=10,

    random_state=random_state,

)

stacked_DF.loc[stacked_DF["Train"] == 1, num_features] = imputer.fit_transform(

    stacked_DF.loc[stacked_DF["Train"] == 1, num_features].values

)

stacked_DF.loc[stacked_DF["Train"] == 0, num_features] = imputer.transform(

    stacked_DF.loc[stacked_DF["Train"] == 0, num_features].values

)

stacked_DF["WarmSeason"] = np.where(

    stacked_DF["MoSold"].isin([10, 11, 12, 1, 2, 3]), 0, 1

)

stacked_DF["SqFtPerRoom"] = stacked_DF["GrLivArea"] / (

    stacked_DF["TotRmsAbvGrd"]

    + stacked_DF["FullBath"]

    + stacked_DF["HalfBath"]

    + stacked_DF["KitchenAbvGr"]

)



# Converting MSSubClass to categorical

stacked_DF["MSSubClass"] = stacked_DF["MSSubClass"].astype(str)





# Below I establish ranking among categories within a handful of features,

# Ranking is based on the median SalePrice they show for each category.

cateogies_to_rank = [

    "BsmtQual",

    "ExterQual",

    "ExterCond",

    "Exterior1st",

    "FireplaceQu",

    "GarageCond",

    "GarageQual",

    "Heating",

    "Fence",

    "HeatingQC",

    "OverallQual",

    "OverallCond",

    "HouseStyle",

    "KitchenQual",

    "LotShape",

    "BsmtCond",

    "MSSubClass",

    "Neighborhood",

    "SaleCondition",

    "SaleType",

    "MasVnrType",

    "ExterQual",

]



for col in cateogies_to_rank:

    rank = np.array(

        range(0, len(stacked_DF.loc[stacked_DF["Train"] == 1, col].unique()))

    )

    field_val = (

        stacked_DF.loc[stacked_DF["Train"] == 1]

        .groupby(by=col)["SalePrice"]

        .median()

        .sort_values()

        .index

    )

    rankval_mapping = dict(zip(field_val, rank))

    stacked_DF[col + "_ranking"] = stacked_DF[col].map(

        rankval_mapping, na_action="ignore"

    )



# There is an MSSubClass category in test set but not in train. This creates an nan record during ranking.

# I impute that nan record with the mode of the field (based on the train set)

stacked_DF.loc[stacked_DF["MSSubClass_ranking"].isna(), "MSSubClass_ranking"] = 10





# I combine underrepresented categories under one umbrella and/or with another category in the same field

ext2_map = {"AsphShn": "Oth1", "CBlock": "Oth1", "CmentBd": "Oth2", "Other": "Oth2"}

roofmatl_map = {

    "Roll": "Oth1",

    "ClyTile": "Oth1",

    "Metal": "Oth1",

    "CompShg": "Oth1",

    "Membran": "Oth2",

    "WdShake": "Oth2",

}



cond2_map = {"PosA": "Pos", "PosN": "Pos", "RRAe": "Norm", "RRAn": "Norm"}





stacked_DF["Exterior2nd"] = (

    stacked_DF["Exterior2nd"].map(ext2_map).fillna(stacked_DF["Exterior2nd"])

)

stacked_DF["RoofMatl"] = (

    stacked_DF["RoofMatl"].map(roofmatl_map).fillna(stacked_DF["RoofMatl"])

)

stacked_DF["Condition2"] = (

    stacked_DF["Condition2"].map(cond2_map).fillna(stacked_DF["Condition2"])

)



# Below I establish new features, mainly via feature crossing

stacked_DF["QualCond"] = (

    stacked_DF["OverallQual_ranking"] * stacked_DF["OverallCond_ranking"]

)

stacked_DF["HighQualSF"] = stacked_DF["1stFlrSF"] + stacked_DF["2ndFlrSF"]

stacked_DF["HoodNExtCond"] = (

    stacked_DF["Neighborhood_ranking"] * stacked_DF["ExterCond_ranking"]

)

stacked_DF["HoodNPrivacy"] = (

    stacked_DF["Neighborhood_ranking"] * stacked_DF["Fence_ranking"]

)

stacked_DF["AreaOverallQualCond"] = (

    stacked_DF["HighQualSF"]

    * stacked_DF["OverallQual_ranking"]

    * stacked_DF["OverallCond_ranking"]

)

stacked_DF["KitchenQCHighQualSF"] = (

    stacked_DF["HighQualSF"] * stacked_DF["KitchenQual_ranking"]

)

stacked_DF["HoodNOverallQual"] = (

    stacked_DF["Neighborhood_ranking"] * stacked_DF["OverallQual_ranking"]

)

stacked_DF["HoodNMasVnrType"] = (

    stacked_DF["Neighborhood_ranking"] * stacked_DF["MasVnrType_ranking"]

)

stacked_DF["HoodNKitchenQual"] = (

    stacked_DF["Neighborhood_ranking"] * stacked_DF["KitchenQual_ranking"]

)

stacked_DF["HoodNCond1"] = stacked_DF["Neighborhood_ranking"] * stacked_DF["Condition1"]

stacked_DF["HoodNCond2"] = stacked_DF["Neighborhood_ranking"] * stacked_DF["Condition2"]

stacked_DF["HoodNPorch"] = stacked_DF["Neighborhood_ranking"] * (

    stacked_DF["3SsnPorch"] + stacked_DF["EnclosedPorch"] + stacked_DF["OpenPorchSF"]

)



stacked_DF["Age_YrBuilt"] = stacked_DF["YrSold"] - stacked_DF["YearBuilt"]

stacked_DF["Age_YrRemod"] = stacked_DF["YrSold"] - stacked_DF["YearRemodAdd"]

stacked_DF["Age_Garage"] = stacked_DF["YrSold"] - stacked_DF["GarageYrBlt"]

stacked_DF["Remodeled"] = stacked_DF["YearBuilt"] != stacked_DF["YearRemodAdd"]

stacked_DF["Age_YrBuilt"] = stacked_DF["Age_YrBuilt"].apply(lambda x: 0 if x < 0 else x)

stacked_DF["Age_YrRemod"] = stacked_DF["Age_YrRemod"].apply(lambda x: 0 if x < 0 else x)

stacked_DF["Age_Garage"] = stacked_DF["Age_Garage"].apply(lambda x: 0 if x < 0 else x)



sqft_price_table = (

    stacked_DF.loc[stacked_DF["Train"] == 1]

    .groupby(by="Neighborhood")["SalePrice", "GrLivArea"]

    .agg(pd.Series.sum)

)

sqft_price_table["AvgPricePerSqFt"] = (

    sqft_price_table["SalePrice"] / sqft_price_table["GrLivArea"]

)

sqft_price_table.drop(columns=["SalePrice", "GrLivArea"], inplace=True)

stacked_DF["AvgPricePerSqFtPerHood"] = stacked_DF["Neighborhood"].map(

    sqft_price_table.to_dict()["AvgPricePerSqFt"]

)



# Dropping a handful of features as there are other variables that are perfectly correlated to these

# I did trial and error here based on the impact of removing features on RMSE.

stacked_DF.drop(columns=["GarageYrBlt", "Utilities"], inplace=True)



# Would like to visualize some of the newly established features to see their relationship with target.

# Hoping to see some correlation

cols_to_viz = [

    "HighQualSF",

    "KitchenQCHighQualSF",

    "AreaOverallQualCond",

    "HoodNOverallQual",

    "HoodNMasVnrType",

    "HoodNKitchenQual",

]

ncols = 1

nrows = math.ceil(len(cols_to_viz) / ncols)

unused = (nrows * ncols) - len(cols_to_viz)





figw, figh = ncols * 10, nrows * 8



fig, ax = plt.subplots(nrows, ncols, sharey=True, figsize=(figw, figh))

fig.subplots_adjust(hspace=0.2, wspace=0.2)

ax = ax.flatten()

for i in range(unused, 0, -1):

    fig.delaxes(ax[-i])





for n, col in enumerate(cols_to_viz):

    if n % 2 != 0:

        ax[n].yaxis.label.set_visible(False)

    ax[n].set_xlabel(col)

    ax[n].set_ylabel("SalePrice")

    sns.scatterplot(

        x=col,

        y="SalePrice",

        data=stacked_DF.loc[stacked_DF["Train"] == 1],

        hue="SalePrice",

        palette='gist_earth',

        s=75,

        legend=False,

        ax=ax[n],

    )



plt.show()

output_notebook()



df_to_viz = stacked_DF[stacked_DF["Train"] == 1].drop(columns="Train")



xcorr = abs(df_to_viz.corr())

xcorr.index.name = "Feature1"

xcorr.columns.name = "Feature2"



df = pd.DataFrame(xcorr.stack(), columns=["Corr"]).reset_index()



source = ColumnDataSource(df)



colors = [

    "#75968f",

    "#a5bab7",

    "#c9d9d3",

    "#e2e2e2",

    "#dfccce",

    "#ddb7b1",

    "#cc7878",

    "#933b41",

    "#550b1d",

]



mapper = LinearColorMapper(palette=colors, low=df.Corr.min(), high=df.Corr.max())



f1 = figure(

    plot_width=1000,

    plot_height=1000,

    title="Correlation Heat Map",

    x_range=list(sorted(xcorr.index)),

    y_range=list(reversed(sorted(xcorr.columns))),

    toolbar_location=None,

    tools="hover",

    x_axis_location="above",

)



f1.rect(

    x="Feature2",

    y="Feature1",

    width=1,

    height=1,

    source=source,

    line_color=None,

    fill_color=transform("Corr", mapper),

)



color_bar = ColorBar(

    color_mapper=mapper,

    location=(0, 0),

    ticker=BasicTicker(desired_num_ticks=len(colors)),

    formatter=PrintfTickFormatter(format="%d%%"),

)

f1.add_layout(color_bar, "right")



f1.hover.tooltips = [

    ("Feature1", "@{Feature1}"),

    ("Feature2", "@{Feature2}"),

    ("Corr", "@{Corr}{1.1111}"),

]



f1.axis.axis_line_color = None

f1.axis.major_tick_line_color = None

f1.axis.major_label_text_font_size = "10px"

f1.axis.major_label_standoff = 2

f1.xaxis.major_label_orientation = 1.0



show(f1)

stacked_DF[stacked_DF["Train"] == 1].skew()[

    abs(stacked_DF[stacked_DF["Train"] == 1].skew()) > 5

]

highly_skewed_cols = (

    stacked_DF[stacked_DF["Train"] == 1]

    .skew()[abs(stacked_DF[stacked_DF["Train"] == 1].skew()) > 5]

    .index.to_list()

)



ptransformer = PowerTransformer(standardize=False)



stacked_DF.loc[

    stacked_DF["Train"] == 1, highly_skewed_cols

] = ptransformer.fit_transform(

    stacked_DF.loc[stacked_DF["Train"] == 1, highly_skewed_cols]

)

stacked_DF.loc[stacked_DF["Train"] == 0, highly_skewed_cols] = ptransformer.transform(

    stacked_DF.loc[stacked_DF["Train"] == 0, highly_skewed_cols]

)
# Obtaining a list of categorical, numerical, and boolean - like features.

bool_features = [

    col

    for col in stacked_DF.select_dtypes(include=["number"]).columns

    if np.array_equal(

        np.sort(stacked_DF[col].unique(), axis=0), np.sort([0, 1], axis=0)

    )

]



cat_features = [col for col in stacked_DF.select_dtypes(exclude=["number"]).columns]

num_features = [

    col

    for col in stacked_DF.select_dtypes(include=["number"]).columns

    if col not in (bool_features) and col != "SalePrice"

]



# Holding these two DF 's on the side.

# Will need to concatenate later with the preprocessed(scaled and oh encoded) DF.

bool_features.remove("Train")

X_train_bool = stacked_DF.loc[stacked_DF["Train"] == 1, bool_features]

X_test_bool = stacked_DF.loc[stacked_DF["Train"] == 0, bool_features]



# This list contains features that has the same set of values between train - test

ohe_cols_a = []



# This list contains features that has different set of values between train - test

ohe_cols_b = []



for col in cat_features:

    if set(stacked_DF.loc[stacked_DF["Train"] == 1, col].unique()) != set(

        stacked_DF.loc[stacked_DF["Train"] == 0, col].unique()

    ):

        ohe_cols_b.append(col)



ohe_cols_a = list(set(cat_features) - set(ohe_cols_b))
X_train = stacked_DF.loc[stacked_DF["Train"] == 1].drop(

    labels=["SalePrice", "Train"], axis=1

)

# Applying log transformation to the target variable

y_train = stacked_DF.loc[stacked_DF["Train"] == 1, "SalePrice"].apply(np.log)

X_test = stacked_DF.loc[stacked_DF["Train"] == 0].drop(

    labels=["SalePrice", "Train"], axis=1

)





preprocessor = ColumnTransformer(

    transformers=[

        ("onehota", OneHotEncoder(sparse=False, drop="first"), ohe_cols_a),

        ("onehotb", OneHotEncoder(sparse=False, handle_unknown="ignore"), ohe_cols_b),

        ("scaler", StandardScaler(), num_features),

    ],

    remainder="drop",

)





pipeline = Pipeline(

    [("Preprocessor", preprocessor), ("VarThreshold", VarianceThreshold(0.01))]

)



X_train_preprocessed = pipeline.fit_transform(X_train)

X_test_preprocessed = pipeline.transform(X_test)



# Get the list of one hot encoded columns and combine them

oh_encoded_a = list(

    preprocessor.named_transformers_.onehota.get_feature_names(ohe_cols_a)

)

oh_encoded_b = list(

    preprocessor.named_transformers_.onehotb.get_feature_names(ohe_cols_b)

)

oh_encoded_cols = oh_encoded_a + oh_encoded_b



feature_names = np.array(oh_encoded_cols + num_features, order="K")



# Filtering out the features dropped by variance threshold

feature_names = feature_names[pipeline.named_steps.VarThreshold.get_support()]



# Putting back the column names to help with analysis

X_train_preprocessed = pd.DataFrame(data=X_train_preprocessed, columns=feature_names)

X_test_preprocessed = pd.DataFrame(

    data=X_test_preprocessed, columns=feature_names, index=X_test_bool.index

)



# Combine the DF's back together

X_train = pd.concat([X_train_bool, X_train_preprocessed], axis=1)

X_test = pd.concat([X_test_bool, X_test_preprocessed], axis=1)

model = Lasso(alpha=0.01)

model.fit(X_train, y_train)





feature_imp = permutation_importance(

    model, X_train, y_train, n_repeats=10, n_jobs=-1, random_state=random_state

)



perm_ft_imp_df = pd.DataFrame(

    data=feature_imp.importances_mean, columns=["FeatureImp"], index=X_train.columns

).sort_values(by="FeatureImp", ascending=False)

model_ft_imp_df = pd.DataFrame(

    data=model.coef_, columns=["FeatureImp"], index=X_train.columns

).sort_values(by="FeatureImp", ascending=False)



fig, ax = plt.subplots(2, 1, figsize=(12, 22))



perm_ft_imp_df_nonzero = perm_ft_imp_df[perm_ft_imp_df["FeatureImp"] != 0]

model_ft_imp_df_nonzero = model_ft_imp_df[model_ft_imp_df["FeatureImp"] != 0]



sns.barplot(

    x=perm_ft_imp_df_nonzero["FeatureImp"],

    y=perm_ft_imp_df_nonzero.index,

    ax=ax[0],

    palette="vlag",

)

sns.barplot(

    x=model_ft_imp_df_nonzero["FeatureImp"],

    y=model_ft_imp_df_nonzero.index,

    ax=ax[1],

    palette=sns.diverging_palette(10, 220, sep=2, n=80),

)



ax[0].set_title("Permutation Feature Importance")

ax[1].set_title("Lasso Feature Importance")



plt.show()
# def objective(trial):

#     _C = trial.suggest_float("C", 0.1, 0.5)

#     _epsilon = trial.suggest_float("epsilon", 0.01, 0.1)

#     _coef = trial.suggest_float("coef0", 0.5, 1)



#     svr = SVR(cache_size=5000, kernel="poly", C=_C, epsilon=_epsilon, coef0=_coef)



#     score = cross_val_score(

#         svr, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"

#     ).mean()

#     return score





# optuna.logging.set_verbosity(0)



# study = optuna.create_study(direction="maximize")

# study.optimize(objective, n_trials=100)



# svr_params = study.best_params

# svr_best_score = study.best_value

# print(f"Best score:{svr_best_score} \nOptimized parameters: {svr_params}")
# def objective(trial):



#     _alpha = trial.suggest_float("alpha", 0.5, 1)

#     _tol = trial.suggest_float("tol", 0.5, 0.9)



#     ridge = Ridge(alpha=_alpha, tol=_tol, random_state=random_state)



#     score = cross_val_score(

#         ridge, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"

#     ).mean()

#     return score





# optuna.logging.set_verbosity(0)



# study = optuna.create_study(direction="maximize")

# study.optimize(objective, n_trials=100)



# ridge_params = study.best_params

# ridge_best_score = study.best_value

# print(f"Best score:{ridge_best_score} \nOptimized parameters: {ridge_params}")

# def objective(trial):



#     _alpha = trial.suggest_float("alpha", 0.0001, 0.01)



#     lasso = Lasso(alpha=_alpha, random_state=random_state)



#     score = cross_val_score(

#         lasso, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"

#     ).mean()

#     return score





# optuna.logging.set_verbosity(0)



# study = optuna.create_study(direction="maximize")

# study.optimize(objective, n_trials=100)



# lasso_params = study.best_params

# lasso_best_score = study.best_value

# print(f"Best score:{lasso_best_score} \nOptimized parameters: {lasso_params}")

# def objective(trial):

#     _n_estimators = trial.suggest_int("n_estimators", 50, 200)

#     _max_depth = trial.suggest_int("max_depth", 5, 12)

#     _min_samp_split = trial.suggest_int("min_samples_split", 2, 8)

#     _min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 6)

#     _max_features = trial.suggest_int("max_features", 10, 50)



#     rf = RandomForestRegressor(

#         max_depth=_max_depth,

#         min_samples_split=_min_samp_split,

#         ccp_alpha=_ccp_alpha,

#         min_samples_leaf=_min_samples_leaf,

#         max_features=_max_features,

#         n_estimators=_n_estimators,

#         n_jobs=-1,

#         random_state=random_state,

#     )



#     score = cross_val_score(

#         rf, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"

#     ).mean()

#     return score





# optuna.logging.set_verbosity(0)



# study = optuna.create_study(direction="maximize")

# study.optimize(objective, n_trials=100)



# rf_params = study.best_params

# rf_best_score = study.best_value

# print(f"Best score:{rf_best_score} \nOptimized parameters: {rf_params}")

# def objective(trial):

#     _num_leaves = trial.suggest_int("num_leaves", 5, 20)

#     _max_depth = trial.suggest_int("max_depth", 3, 8)

#     _learning_rate = trial.suggest_float("learning_rate", 0.1, 0.4)

#     _n_estimators = trial.suggest_int("n_estimators", 50, 150)

#     _min_child_weight = trial.suggest_float("min_child_weight", 0.2, 0.6)



#     lgbm = LGBMRegressor(

#         num_leaves=_num_leaves,

#         max_depth=_max_depth,

#         learning_rate=_learning_rate,

#         n_estimators=_n_estimators,

#         min_child_weight=_min_child_weight,

#     )



#     score = cross_val_score(

#         lgbm, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"

#     ).mean()

#     return score





# optuna.logging.set_verbosity(0)



# study = optuna.create_study(direction="maximize")

# study.optimize(objective, n_trials=100)



# lgbm_params = study.best_params

# lgbm_best_score = study.best_value

# print(f"Best score:{lgbm_best_score} \nOptimized parameters: {lgbm_params}")

rf_params = {"max_depth": 8, "max_features": 40, "n_estimators": 132}

svr_params = {

    "kernel": "poly",

    "C": 0.053677105521141605,

    "epsilon": 0.03925943476562099,

    "coef0": 0.9486751042886584,

}

ridge_params = {

    "alpha": 0.9999189637151178,

    "tol": 0.8668539399622242,

    "solver": "cholesky",

}

lasso_params = {"alpha": 0.0004342843645993161, "selection": "random"}

lgbm_params = {

    "num_leaves": 16,

    "max_depth": 6,

    "learning_rate": 0.16060612646519587,

    "n_estimators": 64,

    "min_child_weight": 0.4453842422224686,

}
cv = KFold(n_splits=4, random_state=random_state)



svr = SVR(**svr_params)

ridge = Ridge(**ridge_params, random_state=random_state)

lasso = Lasso(**lasso_params, random_state=random_state)

lgbm = LGBMRegressor(**lgbm_params, random_state=random_state)

rf = RandomForestRegressor(**rf_params, random_state=random_state)

stack = StackingCVRegressor(

    regressors=[svr, ridge, lasso, lgbm, rf],

    meta_regressor=LinearRegression(n_jobs=-1),

    random_state=random_state,

    cv=cv,

    n_jobs=-1,

)



svr_scores = cross_val_score(

    svr, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"

)

ridge_scores = cross_val_score(

    ridge, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"

)

lasso_scores = cross_val_score(

    lasso, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"

)

lgbm_scores = cross_val_score(

    lgbm, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"

)

rf_scores = cross_val_score(

    rf, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"

)

stack_scores = cross_val_score(

    stack, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"

)



scores = [svr_scores, ridge_scores, lasso_scores, lgbm_scores, rf_scores, stack_scores]

models = ["SVR", "RIDGE", "LASSO", "LGBM", "RF", "STACK"]

score_medians = [

    round(np.median([mean for mean in modelscore]), 5) for modelscore in scores

]

fig, ax = plt.subplots(figsize=(14, 8))



vertical_offset = 0.001



ax.set_title("Model Score Comparison")

bp = sns.boxplot(x=models, y=scores, ax=ax)





for xtick in bp.get_xticks():

    bp.text(

        xtick,

        score_medians[xtick] - vertical_offset,

        score_medians[xtick],

        horizontalalignment="center",

        size=18,

        color="w",

        weight="semibold",

    )



plt.show()

stack.fit(X_train.values, y_train.values)



predictions = stack.predict(X_test.values)

predictions = np.exp(predictions)



submission = pd.DataFrame({"Id": submission_ID, "SalePrice": predictions})

submission.to_csv("submission.csv", index=False)