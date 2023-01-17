import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization
import seaborn as sb # for visualization
%matplotlib inline

from math import ceil
from scipy.stats import skew # for statistics
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
def check_null_info(df, display=True):
    """
    This function will display and return columns with percentage of null values present in it.
    Columns having no null values will be ignored.  
    """
    
    null_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    if len(null_cols) > 0:
        df_null = pd.DataFrame(np.round(df[null_cols].isnull().mean(), 3)*100, columns=["null_percent"])
        df_null.sort_values(by="null_percent", axis=0, ascending=False, inplace=True)
        if display:
            sb.barplot(df_null.index, df_null["null_percent"])
            plt.xticks(rotation="90")
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Percentage of missing values", fontsize=12)
            plt.title("Percent missing data by feature", fontsize=15)
            plt.show()
        return df_null
    else:
        print("No null values present")
        return None
def update_date_cols(df, show_result=False):
    """
    This function will update the date columns with difference in year sold and the input year. Ex: knowing
    how old the house is and how long it has been since it was last remodeled is more informative then just 
    knowing the exact year when it was done.
    """
    
    year_cols = [col for col in df.columns if "Yr" in col or "Year" in col]
    for col in year_cols:
        if col != "YrSold":
            df[col] = df["YrSold"] - df[col]
            
    df.rename(columns={"YearBuilt": "YearsOld", "YearRemodAdd": "RemodAge", "GarageYrBlt": "GarageAge"}, 
              inplace=True)
    
    if show_result:
        print(df.filter(regex="YearsOld|RemodAge|GarageAge", axis=1).head())
def check_correlation(df, fig_size=(12, 9)):
    """
    This function will show correlation among the features.
    """
    
    corr_matrix = df.corr()
    plt.subplots(figsize=fig_size)
    sb.heatmap(corr_matrix, square=True)
def get_numeric_cols(df):
    """
    This function will return all numeric type columns.
    """
    
    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    return [col for col in df.columns if df[col].dtype in numeric_dtypes]
def get_categorical_cols(df):
    """
    This function will return all categorical columns.
    """
    
    categorical_types = ["object"]
    return [col for col in df.columns if df[col].dtype in categorical_types]
def show_target_vs_continuous_graph(df, target, threshold=50, display=True):
    """
    This function will display relationship between target values and all the continuous input features.
    It will display 2 graphs in a single row.
    """
    
    numeric_cols = get_numeric_cols(df)
    continuous_numeric_cols = [col for col in numeric_cols if len(df[col].unique()) >= threshold]
    if display and continuous_numeric_cols:
        n_graphs = len(continuous_numeric_cols)
        n_rows = ceil(n_graphs/2)
        for index, col in enumerate(continuous_numeric_cols):
            plt.figure(figsize=(12,24))
            plt.subplot(n_rows, 2, index+1)
            sb.scatterplot(x=df[col], y=target)
            plt.xlabel("{}".format(col))
            plt.ylabel("{}".format(target.name))
            plt.title("{} VS {}".format(target.name, col))
            plt.show()
    return continuous_numeric_cols
def show_skewness(df, threshold=0.5, display=True):
    """
    This function will return and show skewness in the numeric features. Only those columns will be
    shown whose skewness is more than or less than the given threshold.
    """
    
    numeric_cols = get_numeric_cols(df)
    if numeric_cols:
        skewness_df = df[numeric_cols].apply(lambda col: skew(col))
        dropped_cols = []
        for col in numeric_cols:
            if abs(skewness_df[col]) <= threshold:
                dropped_cols.append(col)

        if dropped_cols:
            skewness_df.drop(dropped_cols, inplace=True)

        if not skewness_df.empty:
            skewness_df.sort_values(inplace=True, ascending=False)
            if display:
                sb.barplot(skewness_df.index, skewness_df)
                plt.xticks(rotation="90")
                plt.xlabel("Features", fontsize=12)
                plt.ylabel("Skewness", fontsize=12)
                plt.title("Skewness in data by feature", fontsize=15)
                plt.yticks(np.arange(skewness_df[skewness_df.size-1], skewness_df[0]))
                plt.show()
            return skewness_df
        else:
            print("No columns have skewness more than the given threshold")
            return None
    else:
        print("No numeric columns in the dataframe")
        return None
def perform_grid_search(train_x, train_y, model, parameters, scoring_type=None, kfold_obj=None, concurrency=-1):
    """
    This function will perform grid search to get the optimal paramters for the input model.
    It will return the grid search object.
    """
    
    best_model = GridSearchCV(
                            estimator=model,
                            param_grid=parameters,
                            scoring=scoring_type,
                            cv=kfold_obj,
                            n_jobs= concurrency
                            )
    best_model.fit(train_x, train_y)
    return best_model
# Preparing training and testing data
df_train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

# Saving ID values of test data for submission
test_IDs = df_test_data["Id"]
# Removing Outliers
df_train_data.drop(df_train_data[(df_train_data['GrLivArea']>4500) & (df_train_data['SalePrice']<300000)].index, inplace=True)

# Separating label column from training data
train_y = df_train_data["SalePrice"]

# Transforming target values into normal distribution as it's skewed towards right
train_y.update(np.log1p(train_y))

# Getting dataset sizes
n_train_data = df_train_data.shape[0]
n_test_data = df_test_data.shape[0]

df_train_data.drop("SalePrice", axis=1, inplace=True)

# Combining both dataset into one
combined_data = pd.concat([df_train_data, df_test_data], ignore_index=True)
# Removing unnecessary columns
cols_dropped = ["PoolQC", "Alley", "Id", "Utilities"]
combined_data.drop(cols_dropped, axis=1, inplace=True)

# Updating some features
update_date_cols(combined_data)
# Here NA means no basement
for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
    combined_data[col].fillna("None", inplace=True)
    
# No basement means no basement bathrooms and other things
for col in ["BsmtHalfBath", "BsmtFullBath", "BsmtUnfSF", "TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2"]:
    combined_data[col].fillna(0, inplace=True)
    
# Here NA means no garage
for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
    combined_data[col].fillna("None", inplace=True)
    
# No garage means no cars and no garage area
for col in ["GarageAge", "GarageCars", "GarageArea"]:
    combined_data[col].fillna(0, inplace=True)
    
# Here NA means absent of the feature
for col in ["Fence", "MiscFeature", "FireplaceQu", "MasVnrType"]:
    combined_data[col].fillna("None", inplace=True)
    
# No misc feature means no misc value
combined_data["MiscVal"].fillna(0, inplace=True)

# No firequality means no fireplace
combined_data["Fireplaces"].fillna(0, inplace=True)

# No masonry means no masonry area
combined_data["MasVnrArea"].fillna(0, inplace=True)

# Neighborhood share similar lot frontage
combined_data["LotFrontage"] = combined_data.groupby("Neighborhood")["LotFrontage"] \
        .transform(lambda x: x.fillna(x.median()))
    
# Filling remaining missing values with the median
combined_data["Electrical"].fillna(combined_data["Electrical"].mode()[0], inplace=True)
combined_data["MSZoning"].fillna(combined_data["MSZoning"].mode()[0], inplace=True)
combined_data["KitchenQual"].fillna(combined_data["KitchenQual"].mode()[0], inplace=True)
combined_data["Exterior1st"].fillna(combined_data["Exterior1st"].mode()[0], inplace=True)
combined_data["Exterior2nd"].fillna(combined_data["Exterior2nd"].mode()[0], inplace=True)
combined_data["SaleType"].fillna(combined_data["SaleType"].mode()[0], inplace=True)
combined_data["Functional"].fillna("Typ", inplace=True)
skewed_df = show_skewness(combined_data, threshold=0.7, display=False)
skewed_cols = skewed_df.index
lmbda = 0.15 
for col in skewed_cols:
    combined_data[col] = boxcox1p(combined_data[col], lmbda)
# Applying label encoding to our categorical features

label_encoder = LabelEncoder()
for col in get_categorical_cols(combined_data):
    combined_data[col] = label_encoder.fit_transform(combined_data[col])
# Getting our train and test data back
train_x = combined_data[:n_train_data]
test_x = combined_data[n_train_data:]

# Will perform cross-validation on 12 folds
kfold_obj = KFold(n_splits=12, shuffle=True, random_state=28)
def cross_validation_score(model, X, labels, kfold_obj, scoring_type):
    """
    This function will provide RMSE cross-validation score on 12 folds.
    """
    
    score = abs(cross_val_score(model, X, labels, cv=kfold_obj, scoring=scoring_type))
    return np.sqrt(score)
# RF model
rf_model = RandomForestRegressor(
                                n_estimators=500,
                                max_depth = 8,
                                min_samples_split = 10,
                                min_samples_leaf = 10,
                                random_state = 28
                                )
score = cross_validation_score(rf_model, train_x, train_y, kfold_obj, "neg_mean_squared_error")
print("Random Forest: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# GB Model
gb_model = GradientBoostingRegressor(
                                    loss= "huber",
                                    learning_rate= 0.05,
                                    n_estimators= 1000,
                                    min_samples_split= 10,
                                    min_samples_leaf= 10,
                                    max_depth= 5
                                    )
score = cross_validation_score(gb_model, train_x, train_y, kfold_obj, "neg_mean_squared_error")
print("Random Forest: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# Training our final best model
gb_model.fit(train_x, train_y)
print("Final score on entire testing data:{:.4f}".format(gb_model.score(train_x, train_y)))
sub_df = pd.DataFrame() 
sub_df["SalePrice"] = np.expm1(gb_model.predict(test_x))
sub_df["Id"] = test_IDs
sub_df.to_csv('submission.csv',index=False)
# For finding best parameters
parameters = {"n_estimators": [1000],
             "max_depth": [5, 6, 8],
             "min_samples_split": [5, 10],
             "min_samples_leaf": [5, 10],
             "random_state": [28]}
best_model = perform_grid_search(train_x,
                                 train_y,
                                 RandomForestRegressor(),
                                 parameters,
                                 "neg_mean_squared_error",
                                 kfold_obj
                                )
print("Best Random Forest Model: \n\tScore:{} \n\tParams{}"
      .format(best_model.best_score_, best_params_))