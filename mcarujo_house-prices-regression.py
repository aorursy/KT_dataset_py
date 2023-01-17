import warnings

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import math

import numpy as np



warnings.filterwarnings("ignore")
!pip install nb_black -q
%load_ext nb_black
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv(

    "/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv"

)

train = pd.read_csv(

    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

)
def plot_hists(df, labels):

    row = 1

    col = 1

    num_graphs = len(labels)

    rows = math.ceil(num_graphs / 2)

    fig = make_subplots(rows=rows, cols=2, subplot_titles=labels)



    index = []

    for row in range(1, rows + 1):

        for col in range(1, 3):

            index.append({"row": row, "col": col})



    graphs = []

    pos_g = 0

    for label in labels:

        local_data = df[label].value_counts()

        x = list(local_data.index)

        y = list(local_data)

        fig.add_trace(

            go.Histogram(x=df[label]), row=index[pos_g]["row"], col=index[pos_g]["col"],

        )

        pos_g = pos_g + 1



    fig.update_layout(

        autosize=False,

        width=1200,

        height=300 * rows,

        margin=dict(l=50, r=50, b=100, t=100, pad=4),

        #         paper_bgcolor="LightSteelBlue",

    )



    fig.show()
def plot_dists(df, labels):

    row = 1

    col = 1

    num_graphs = len(labels)

    rows = math.ceil(num_graphs / 3)

    fig = make_subplots(rows=rows, cols=3, subplot_titles=labels)



    index = []

    for row in range(1, rows + 1):

        for col in range(1, 4):

            index.append({"row": row, "col": col})



    graphs = []

    pos_g = 0

    for label in labels:

        local_data = df[label].value_counts()

        x = list(local_data.index)

        y = list(local_data)

        fig.add_trace(

            go.Bar(x=x, y=y, text=y, textposition="auto",),

            row=index[pos_g]["row"],

            col=index[pos_g]["col"],

        )

        pos_g = pos_g + 1

    

    fig.update_layout(

        autosize=False,

        width=1200,

        height=300*rows,

        margin=dict(

            l=50,

            r=50,

            b=100,

            t=100,

            pad=4

        ),

#         paper_bgcolor="LightSteelBlue",

    )



    fig.show()

train.info()
cat_col = [

    "MSSubClass",

    "MSZoning",

    "Street",

    "Alley",

    "LotShape",

    "LandContour",

    "Utilities",

    "LotConfig",

    "LandSlope",

    "Neighborhood",

    "Condition1",

    "Condition2",

    "BldgType",

    "HouseStyle",

    "OverallQual",

    "OverallCond",

    "YearBuilt",

    "YearRemodAdd",

    "RoofStyle",

    "RoofMatl",

    "Exterior1st",

    "Exterior2nd",

    "MasVnrType",

    "ExterQual",

    "ExterCond",

    "Foundation",

    "BsmtQual",

    "BsmtCond",

    "BsmtExposure",

    "BsmtFinType1",

    "BsmtFinType2",

    "Heating",

    "HeatingQC",

    "CentralAir",

    "Electrical",

    "BsmtFullBath",

    "BsmtHalfBath",

    "FullBath",

    "HalfBath",

    "BedroomAbvGr",

    "KitchenAbvGr",

    "KitchenQual",

    "TotRmsAbvGrd",

    "Functional",

    "Fireplaces",

    "FireplaceQu",

    "GarageType",

    "GarageYrBlt",

    "GarageFinish",

    "GarageCars",

    "GarageQual",

    "GarageCond",

    "PavedDrive",

    "PoolQC",

    "Fence",

    "MiscFeature",

    "MoSold",

    "YrSold",

    "SaleType",

    "SaleCondition",

]

plot_dists(

    train, cat_col,

)
set_cat_col = set(cat_col)

set_all_col = set(train.columns)

set_num_col = set_all_col - set_cat_col - {"Id"}

plot_hists(train, list(set_num_col))
with sns.axes_style("white"):

    table = train.corr()

    mask = np.zeros_like(table)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(20, 20))

    sns.heatmap(

        table, cmap="Reds", mask=mask, center=0, linewidths=0.5, annot_kws={"size": 15},

    )
# Creating correlation matrix

corr = pd.DataFrame(train.corr()["SalePrice"])

# Set feature as index

corr["Feature"] = corr.index

corr.sort_values("SalePrice", inplace=True, ascending=False)

# Plot our feature lest in dataset

fig = px.bar(corr, x="Feature", y="SalePrice", color="SalePrice", height=400)

fig.show()
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()





def fromat(dataset):

    # Defing not a numbers in all columns

    dataset["MSSubClass"].fillna(0, inplace=True)

    dataset["MSZoning"].fillna("Not Know", inplace=True)

    dataset["Street"].fillna("Not Know", inplace=True)

    dataset["Alley"].fillna("Not Know", inplace=True)

    dataset["LotShape"].fillna("Not Know", inplace=True)

    dataset["LandContour"].fillna("Not Know", inplace=True)

    dataset["Utilities"].fillna("Not Know", inplace=True)

    dataset["LotConfig"].fillna("Not Know", inplace=True)

    dataset["LandSlope"].fillna("Not Know", inplace=True)

    dataset["Neighborhood"].fillna("Not Know", inplace=True)

    dataset["Condition1"].fillna("Not Know", inplace=True)

    dataset["Condition2"].fillna("Not Know", inplace=True)

    dataset["BldgType"].fillna("Not Know", inplace=True)

    dataset["HouseStyle"].fillna("Not Know", inplace=True)

    dataset["OverallQual"].fillna(0, inplace=True)

    dataset["OverallCond"].fillna(0, inplace=True)

    dataset["YearBuilt"].fillna(0, inplace=True)

    dataset["YearRemodAdd"].fillna(0, inplace=True)

    dataset["RoofStyle"].fillna("Not Know", inplace=True)

    dataset["RoofMatl"].fillna("Not Know", inplace=True)

    dataset["Exterior1st"].fillna("Not Know", inplace=True)

    dataset["Exterior2nd"].fillna("Not Know", inplace=True)

    dataset["MasVnrType"].fillna("Not Know", inplace=True)

    dataset["ExterQual"].fillna("Not Know", inplace=True)

    dataset["ExterCond"].fillna("Not Know", inplace=True)

    dataset["Foundation"].fillna("Not Know", inplace=True)

    dataset["BsmtQual"].fillna("Not Know", inplace=True)

    dataset["BsmtCond"].fillna("Not Know", inplace=True)

    dataset["BsmtExposure"].fillna("Not Know", inplace=True)

    dataset["BsmtFinType1"].fillna("Not Know", inplace=True)

    dataset["BsmtFinType2"].fillna("Not Know", inplace=True)

    dataset["Heating"].fillna("Not Know", inplace=True)

    dataset["HeatingQC"].fillna("Not Know", inplace=True)

    dataset["CentralAir"].fillna("Not Know", inplace=True)

    dataset["Electrical"].fillna("Not Know", inplace=True)

    dataset["BsmtFullBath"].fillna(0, inplace=True)

    dataset["BsmtHalfBath"].fillna(0, inplace=True)

    dataset["FullBath"].fillna(0, inplace=True)

    dataset["HalfBath"].fillna(0, inplace=True)

    dataset["BedroomAbvGr"].fillna(0, inplace=True)

    dataset["KitchenAbvGr"].fillna(0, inplace=True)

    dataset["KitchenQual"].fillna("Not Know", inplace=True)

    dataset["TotRmsAbvGrd"].fillna(0, inplace=True)

    dataset["Functional"].fillna("Not Know", inplace=True)

    dataset["Fireplaces"].fillna(0, inplace=True)

    dataset["FireplaceQu"].fillna("Not Know", inplace=True)

    dataset["GarageType"].fillna("Not Know", inplace=True)

    dataset["GarageYrBlt"].fillna(0, inplace=True)

    dataset["GarageFinish"].fillna("Not Know", inplace=True)

    dataset["GarageCars"].fillna(0, inplace=True)

    dataset["GarageQual"].fillna("Not Know", inplace=True)

    dataset["GarageCond"].fillna("Not Know", inplace=True)

    dataset["PavedDrive"].fillna("Not Know", inplace=True)

    dataset["PoolQC"].fillna("Not Know", inplace=True)

    dataset["Fence"].fillna("Not Know", inplace=True)

    dataset["MiscFeature"].fillna("Not Know", inplace=True)

    dataset["MoSold"].fillna(0, inplace=True)

    dataset["YrSold"].fillna(0, inplace=True)

    dataset["SaleType"].fillna("Not Know", inplace=True)

    dataset["SaleCondition"].fillna("Not Know", inplace=True)

    dataset['BsmtFinSF2'].fillna(0.0, inplace=True)

    dataset['GarageArea'].fillna(0.0, inplace=True)

    dataset['PoolArea'].fillna(0.0, inplace=True)

    dataset['ScreenPorch'].fillna(0.0, inplace=True)

    dataset['LotArea'].fillna(0.0, inplace=True)

    dataset['1stFlrSF'].fillna(0.0, inplace=True)

    dataset['LotFrontage'].fillna(0.0, inplace=True)

    dataset['OpenPorchSF'].fillna(0.0, inplace=True)

    dataset['GrLivArea'].fillna(0.0, inplace=True)

    dataset['MiscVal'].fillna(0.0, inplace=True)

    dataset['BsmtUnfSF'].fillna(0.0, inplace=True)

    dataset['EnclosedPorch'].fillna(0.0, inplace=True)

    dataset['WoodDeckSF'].fillna(0.0, inplace=True)

    dataset['LowQualFinSF'].fillna(0.0, inplace=True)

    dataset['MasVnrArea'].fillna(0.0, inplace=True)

    dataset['BsmtFinSF1'].fillna(0.0, inplace=True)

    dataset['2ndFlrSF'].fillna(0.0, inplace=True)

    dataset['3SsnPorch'].fillna(0.0, inplace=True)

    dataset['TotalBsmtSF'].fillna(0.0, inplace=True)   

    

    # Defining type for categories

    dataset["MSSubClass"].astype(int)

    dataset["MSZoning"].astype(str)

    dataset["Street"].astype(str)

    dataset["Alley"].astype(str)

    dataset["LotShape"].astype(str)

    dataset["LandContour"].astype(str)

    dataset["Utilities"].astype(str)

    dataset["LotConfig"].astype(str)

    dataset["LandSlope"].astype(str)

    dataset["Neighborhood"].astype(str)

    dataset["Condition1"].astype(str)

    dataset["Condition2"].astype(str)

    dataset["BldgType"].astype(str)

    dataset["HouseStyle"].astype(str)

    dataset["OverallQual"].astype(int)

    dataset["OverallCond"].astype(int)

    dataset["YearBuilt"].astype(int)

    dataset["YearRemodAdd"].astype(int)

    dataset["RoofStyle"].astype(str)

    dataset["RoofMatl"].astype(str)

    dataset["Exterior1st"].astype(str)

    dataset["Exterior2nd"].astype(str)

    dataset["MasVnrType"].astype(str)

    dataset["ExterQual"].astype(str)

    dataset["ExterCond"].astype(str)

    dataset["Foundation"].astype(str)

    dataset["BsmtQual"].astype(str)

    dataset["BsmtCond"].astype(str)

    dataset["BsmtExposure"].astype(str)

    dataset["BsmtFinType1"].astype(str)

    dataset["BsmtFinType2"].astype(str)

    dataset["Heating"].astype(str)

    dataset["HeatingQC"].astype(str)

    dataset["CentralAir"].astype(str)

    dataset["Electrical"].astype(str)

    dataset["BsmtFullBath"].astype(int)

    dataset["BsmtHalfBath"].astype(int)

    dataset["FullBath"].astype(int)

    dataset["HalfBath"].astype(int)

    dataset["BedroomAbvGr"].astype(int)

    dataset["KitchenAbvGr"].astype(int)

    dataset["KitchenQual"].astype(str)

    dataset["TotRmsAbvGrd"].astype(int)

    dataset["Functional"].astype(str)

    dataset["Fireplaces"].astype(int)

    dataset["FireplaceQu"].astype(str)

    dataset["GarageType"].astype(str)

    dataset["GarageYrBlt"].astype(int)

    dataset["GarageFinish"].astype(str)

    dataset["GarageCars"].astype(int)

    dataset["GarageQual"].astype(str)

    dataset["GarageCond"].astype(str)

    dataset["PavedDrive"].astype(str)

    dataset["PoolQC"].astype(str)

    dataset["Fence"].astype(str)

    dataset["MiscFeature"].astype(str)

    dataset["MoSold"].astype(int)

    dataset["YrSold"].astype(int)

    dataset["SaleType"].astype(str)

    dataset["SaleCondition"].astype(str)

    

    # Defining type for numeric

    dataset['BsmtFinSF2'].astype(float)

    dataset['GarageArea'].astype(float)

    dataset['PoolArea'].astype(float)

    dataset['ScreenPorch'].astype(float)

    dataset['LotArea'].astype(float)

    dataset['1stFlrSF'].astype(float)

    dataset['LotFrontage'].astype(float)

    dataset['OpenPorchSF'].astype(float)

    dataset['GrLivArea'].astype(float)

    dataset['MiscVal'].astype(float)

    dataset['BsmtUnfSF'].astype(float)

    dataset['EnclosedPorch'].astype(float)

    dataset['WoodDeckSF'].astype(float)

    dataset['LowQualFinSF'].astype(float)

    dataset['MasVnrArea'].astype(float)

    dataset['BsmtFinSF1'].astype(float)

    dataset['2ndFlrSF'].astype(float)

    dataset['3SsnPorch'].astype(float)

    dataset['TotalBsmtSF'].astype(float)



    

    types = pd.DataFrame(

        dataset.dtypes, columns=["type"]

    )  # prepare the categorical columns

    columns = list(

        types[types.type == "object"].index

    )  # making a list to the 'for' loop





    lb_make = LabelEncoder()

    for column in columns:

        dataset[column] = lb_make.fit_transform(dataset[column])



    return dataset

# Train dataset formatted

train_fmt = fromat(train)

train_fmt = train_fmt.fillna(0)

train_fmt.set_index("Id", inplace=True)



# Test dataset formatted

test_fmt = fromat(test)

test_fmt.set_index("Id", inplace=True)

test_fmt = test_fmt.fillna(0)
def verify_outline(df, quants):

    for quant in quants:

        if df[quant["col"]] > quant["max_val"]:

            return "Yes"

    return "No"





def remove_outline(df, plot=True, quant=0.90, set_num_col=set_num_col):

    Id = list(df.index)

    quants = []

    for col in set_num_col:

        quants.append({"col": col, "max_val": df[col].quantile(quant)})



    # Verifying the dataset

    df["isOutline"] = [verify_outline(df[1], quants) for df in train_fmt.iterrows()]



    # plot if flag is true

    if plot:

        fig = px.scatter(df, x="GrLivArea", y="SalePrice", color="isOutline")

        fig.show()

    # removing outlines from dataset

    df = df[df.isOutline == "No"]

    df.drop("isOutline", axis=1)

    # plot if flag is true

    if plot:

        print("Starting with {} lines".format(len(Id)))

        print("Finishing with {} lines".format(len(df)))

    # returing the dataset without outlines

    return df
%%time

train_no_out = remove_outline(train_fmt, True, 0.80, ['SalePrice'])
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score





def split(data, plot=False):

    # seed

    # train_test_split

    train_x, test_x, train_y, test_y = train_test_split(

        data.drop("SalePrice", axis=1),

        data["SalePrice"],

        test_size=0.2,

        random_state=42367,

    )

    if plot:

        print(

            "sizes: train (x,y) and test (x,y)",

            train_x.shape,

            train_y.shape,

            test_x.shape,

            test_y.shape,

        )

    return train_x, test_x, train_y, test_y





def run_reg_linear(train_x, test_x, train_y, test_y, model, plot=False):

    #     model = LinearRegression()

    scaler = StandardScaler()

    train_x_s = scaler.fit_transform(train_x)

    model.fit(train_x_s, train_y)

    test_pred = model.predict(scaler.transform(test_x))



    mse = mean_squared_error(test_y, test_pred)

    mae = mean_absolute_error(test_y, test_pred)

    r2 = r2_score(test_y, test_pred)



    if plot:

        print("*" * 80)

        print("r2 score", r2)

        print("mse", mse)

        print("mae", mae)

        print("*" * 80)



    return model, r2





def r2_outline_qantile(q, model):

    train_no_out = remove_outline(train_fmt, False, q, ["SalePrice"])

    try:

        train_no_out.drop("isOutline", axis=1, inplace=True)

    except:

        pass

    train_x, test_x, train_y, test_y = split(train_no_out)

    model, r2 = run_reg_linear(train_x, test_x, train_y, test_y, model)

    return r2
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neighbors import NearestCentroid

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor





from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    Lasso,

    BayesianRidge,

)



valores = []

models = [

    ("LinearRegresion", LinearRegression()),

    ("BayesianRidge", BayesianRidge()),

    ("Ridge", Ridge(alpha=0.5)),

    ("Lasso", Lasso(alpha=0.1)),

    ("DecisionTreeRegressor", DecisionTreeRegressor()),

    ("NearestCentroid", NearestCentroid()),

    ("RandomForestRegressor", RandomForestRegressor()),

    ("SVR", SVR(C=1000000)),

]

for model in models:

    print(model[0])

    for q in range(95, 101):

        quantile = q / 100.0

        valores.append((quantile, r2_outline_qantile(quantile, model[1]), model[0]))



df_hist = pd.DataFrame(valores, columns=["quantile", "r2", "model"])

fig = px.line(df_hist, x="quantile", y="r2", color="model", title="R2 Curve / Quantile")

fig.show()
from yellowbrick.regressor import PredictionError



train_no_out = remove_outline(train_fmt, False, 1.0, ["SalePrice"])

try:

    train_no_out.drop("isOutline", axis=1, inplace=True)

except:

    pass



train_x, test_x, train_y, test_y = split(train_no_out)

model, r2 = run_reg_linear(train_x, test_x, train_y, test_y, BayesianRidge())



print(model, "\n", r2)





def visualiza_erros(train_x, train_y, test_x, test_y, model):

    scaler = StandardScaler()

    train_x_s = scaler.fit_transform(train_x)

    test_x_s = scaler.fit_transform(test_x)

    visualizer = PredictionError(model)

    visualizer.fit(train_x_s, train_y)

    visualizer.score(test_x_s, test_y)

    visualizer.poof()





visualiza_erros(train_x, train_y, test_x, test_y, model)
from sklearn.model_selection import RandomizedSearchCV

import numpy as np



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split

max_features = ["auto", "sqrt"]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num=11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {

    "n_estimators": n_estimators,

    "max_features": max_features,

    "max_depth": max_depth,

    "min_samples_split": min_samples_split,

    "min_samples_leaf": min_samples_leaf,

    "bootstrap": bootstrap,

}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation,

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(

    estimator=rf,

    param_distributions=random_grid,

    n_iter=100,

    cv=3,

    verbose=2,

    random_state=42,

    n_jobs=-1,

)

# Fit the random search model

rf_random.fit(

    train_fmt.drop(["SalePrice", "isOutline"], axis=1), train_fmt["SalePrice"]

)
rf_random.best_params_
# Plot-outputs

model = RandomForestRegressor(

    n_estimators=600,

    min_samples_split=5,

    min_samples_leaf=1,

    max_features="sqrt",

    max_depth=60,

    bootstrap=False,

)

model.fit(train_fmt.drop(["SalePrice", "isOutline"], axis=1), train_fmt["SalePrice"])

y_predict = model.predict(train_fmt.drop(["SalePrice", "isOutline"], axis=1))

print("MSE: ", mean_squared_error(train_fmt["SalePrice"], y_predict))

print("R2: ", r2_score(train_fmt["SalePrice"], y_predict))

# Fig-size

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=train_fmt.drop(["SalePrice", "isOutline"], axis=1).index,

        y=y_predict,

        mode="lines",

        name="Predict",

    )

)

fig.add_trace(

    go.Scatter(

        x=train_fmt.drop(["SalePrice", "isOutline"], axis=1).index,

        y=train_fmt["SalePrice"],

        mode="markers",

        name="Real",

    )

)

fig.update_layout(title="Real X Pedricted", xaxis_title="Sample", yaxis_title="Price")

fig.show()
model
# creating our submission csv

model.fit(train_no_out.drop("SalePrice", axis=1), train_no_out.SalePrice)

y_test = model.predict(test_fmt)

sample["SalePrice"] = y_test

sample.to_csv("submission_competion_RandomForestRegressor.csv", index=False)

sample