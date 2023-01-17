# Importing all the dependencies required for this Notebook

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

df_test = pd.DataFrame(test_data)

df_train = pd.DataFrame(train_data)

df_train.head(n=5)
# The size of the train and test data:



print("Size of the train data: ", train_data.shape)

print("Size of the test data: ", test_data.shape)
df_train.columns
df_train.describe()
# Let us see the columns that contain NaNs

def missing_values():

    temp_dict = dict()

    for i in df_train.columns:

        if df_train[i].isnull().sum() > 0: 

            temp_dict[i] = df_train[i].isnull().sum()

    return temp_dict

            

# missing features and number of NaN in the features 



#missing_values()



# Few columns/decorations have more than 50% of the data points missing, we will discard those columns.

# uncomment the missing_values() functions called above to look them in detail.
# Deleting the columns which has more than 50% of the missing values



def delete_columns(col):

    if df_train[col].isnull().sum() > df_train[col].count()/2:

        del df_train[col]



for col in df_train.columns:

    delete_columns(col)



# let us now look at the columns that still have missing values/NaNs

missing_values()
# This function predicts the LotFrontage of the missing data values using a Linear Regression model 

# Build a linear regression model with known LotArea and LotFrontage and predicts the LotFrontage for 

# the data with missing values



def fillna_lotfrontage(X_train, y_train):



    reg = linear_model.LinearRegression()

    reg.fit(X_train, y_train)

    

    return reg.coef_.tolist()[0], reg.intercept_ 
X_train_list = df_train['LotArea'].dropna()[:1000]

X_train = [[X_train_list[i]] for i in range(len(X_train_list))]

y_train = df_train['LotFrontage'].dropna()[:1000]



w, intercept = fillna_lotfrontage(X_train, y_train)



for i in range(len(df_train["LotFrontage"])):

    if pd.isnull(df_train.loc[i, "LotFrontage"]):

        df_train.loc[i, "LotFrontage"] = df_train.loc[i, "LotArea"]*w + intercept
# No NaN values in the LotFrontage column left

df_train["LotFrontage"].isnull().sum()
# Let us see the distribution of "MasVnrType" in the data

df_train["MasVnrType"].describe()
for i in range(len(df_train["MasVnrType"])):  

    if pd.isnull(df_train.loc[i, "MasVnrType"]) and pd.isnull(df_train.loc[i, "MasVnrArea"]):

        df_train.loc[i, "MasVnrType"] = "None"

        df_train.loc[i, "MasVnrArea"] = 0

        

# NaNs values in both MasVnrType and MasVnrArea are now removed

df_train["MasVnrType"].isnull().sum(), df_train["MasVnrArea"].isnull().sum()
# Let us see the type distribution.

df_train["BsmtQual"].describe()

# it seems "TA" repeats most for alomost fifty percent of the data

#  there are four possible options
plt.figure(figsize=(10, 6))

sns.boxplot(x="BsmtQual", y="SalePrice", data=df_train)
# List of the 'BsmtQual' types (four types from the above boxplot)



pd.DataFrame(df_train["BsmtQual"].unique()).dropna()[0].tolist()
# Replacing the NaNs based on the above boxplot: SalePrice and BsmtQual distribution



for i in range(len(df_train["BsmtQual"])):  

    if pd.isnull(df_train.loc[i, "BsmtQual"]):

        

        sale_price = df_train.loc[i, "SalePrice"]

        

        if 0 < sale_price < 125000:

            df_train.loc[i, "BsmtQual"] = 'Fa'

        

        elif 125000 < sale_price < 170000:

            df_train.loc[i, "BsmtQual"] = 'TA'

        

        elif 170000 < sale_price < 255000:

            df_train.loc[i, "BsmtQual"] = 'Gd'

            

        elif sale_price > 255000:

            df_train.loc[i, "BsmtQual"] = 'Ex'

        

# NaNs values in both BsmtQual and MasVnrArea are now removed

df_train["BsmtQual"].isnull().sum()
# The other columns related with BsmtQual don't have much clear impact on the SalePrice

# Taking the most important variable on the category into consideration, we drop the other

# related columns: 'BsmtExposure',  'BsmtFinType1', 'BsmtFinType2', 'BsmtCond'.



delete_columns = ['BsmtExposure',  'BsmtFinType1', 'BsmtFinType2', 'BsmtCond']

for col in delete_columns:

    del df_train[col]
garage_columns = ['GarageType', 'GarageYrBlt', 'GarageFinish','GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', \

                  'SalePrice']

# Let us plot heatplot diagrams to see the correlation of the above variables on the SalePrice

sns.heatmap(df_train[garage_columns].corr(), annot=True, cmap="RdBu")
# Since GarageCars doesn't have any missing values(NaN), we don't have to replace any NaN values here.

# We simply delete the columns that are not important and will not consider ahead for further study.



garage_columns.remove('GarageCars')

garage_columns.remove('SalePrice')

for col in garage_columns:

    # deleting the columns except those two in the garage_columns list we consider 

    del df_train[col]
# We have one value NaN in the column 'Electrical'

# Let us see its importance: relation with SalePrice



sns.boxplot(x='Electrical', y='SalePrice', data=df_train)
df_train['Electrical'].unique()
for i in range(len(df_train['Electrical'])):

    if pd.isnull(df_train.loc[i, 'Electrical']):

        # print(df_train.loc[i, 'SalePrice'])

        # Checked the value of SalePrice corresponding to the missing Electrical value

        # and the missing value is imputed looking at the above graph

        df_train.loc[i, 'Electrical'] = 'SBrkr'

# No NaN left in 'Electrical' column now

df_train['Electrical'].isnull().sum()
df_train.columns, df_train.columns.shape[0]
# separating categorical and numerical variables



variables_list = df_train.columns.tolist()

numerical_vars = []

for col in variables_list:

    try:

        if float(df_train[col][1]).is_integer():

            numerical_vars.append(col)

    except ValueError: pass



catergorical_vars = set(variables_list) - set(numerical_vars)

#catergorical_vars



# Let us see number of each type of columns/decorations

len(catergorical_vars), len(numerical_vars)
interesting_cols = ["OverallCond", "GrLivArea", "GarageCars", "YearBuilt", "LotArea", "SalePrice"]



plt.figure(figsize=(14,10))

sns.pairplot(df_train[interesting_cols], dropna=True)

plt.show()

del interesting_cols
correlation_matrix = df_train.drop(["Id"], axis=1).corr()

#plt.figure(figsize=(12,8))

#sns.heatmap(corr_mat, square=False)

#del corr_mat
# Let us find out the variable names that have high correlation with SalePrice

# We call these variables important variables.



variables = correlation_matrix.columns

important_variables = []

for row in variables:

    corrl = correlation_matrix.loc[row, 'SalePrice']

    if corrl > 0.1:

        important_variables.append(row)



hp = correlation_matrix.loc[important_variables, important_variables]

plt.figure("Heatmap-Important Variables", figsize=(12,8))

sns.heatmap(hp, annot=True, cbar=True)
# deleting these columns from the dataframe

for col in ['TotRmsAbvGrd', 'BsmtFinSF1', 'BsmtUnfSF', 'ScreenPorch', 'HalfBath']:

    del df_train[col]
num_variables = set(numerical_vars) - set(['Id', 'TotRmsAbvGrd', 'BsmtFinSF1', 'BsmtUnfSF', 'ScreenPorch', 'HalfBath'])
# Boxplot



plt.figure(figsize=(10,6))

sns.boxplot(x="OverallQual", y="SalePrice", data=df_train)
# Data Normalaization

# Normalizing the right skewed SalePrice



fig, ax =plt.subplots(1,2)

sns.distplot(df_train['SalePrice'], ax=ax[0])

sns.distplot(np.log10(df_train['SalePrice']), ax=ax[1])

plt.xlabel('log10(SalePrice)')

fig.show()
df_train.loc[:, 'SalePrice'] = np.round(np.log10(df_train['SalePrice']), 3)
# The house price is in logarithmic scale now

df_train['SalePrice'].head()
# Plottng the LotArea - SalePrice graph



plt.scatter(df_train["LotArea"], df_train["SalePrice"])

plt.xlabel("Lot Area")

plt.ylabel("Sale Price")

plt.show()
df_train['LotArea'].shape[0]
# the four point's indices (we drop four rows with index names in following list)

drop_index_list = df_train[df_train['LotArea'] > 100000].index.tolist()

df_train.drop(drop_index_list, inplace=True)

print("avoiding the above dataframe printing")
# Again plottng the LotArea - SalePrice graph after the outlier removal



plt.scatter(df_train["LotArea"], df_train["SalePrice"])

plt.title("After Outliers Removal")

plt.xlabel("Lot Area")

plt.ylabel("log10(Sale Price)")

plt.xlim(0, 225000)

plt.show()
corr_with_SalePrice = df_train.drop(["Id"], axis=1).corr()

plot_data = corr_with_SalePrice["SalePrice"].sort_values(ascending=True)

plt.figure(figsize=(12,6))

plot_data.plot.bar()

plt.title("Correlations with the Sale Price")

plt.show()

del plot_data
# creating list of the final numerical columns we are going to consider for the prediction model

num_vars_list = []

for var in num_variables:

    num_vars_list.append(var)



print("Total numerical columns we finally have is: ", len(num_vars_list))
categorical_list = list()

for i in catergorical_vars:

    categorical_list.append(i)

# Let us see all the possible categories in all of the categorical variables

for col in categorical_list:

    print(col, ": ", df_train[col].unique())
# violinplot: for all columns/decorations in the categorical column list



few_cat_variables = ['KitchenQual', 'BsmtQual', 'Heating', 'ExterQual', 'LandSlope', 'HeatingQC', 'Foundation', 'Electrical', \

                     'LandContour', 'LotShape', 'CentralAir', 'SaleType']

# categorical_list => plotted all the variables in this list before showing only few of them in the above list

for i in range(len(few_cat_variables)):

    sns.violinplot(x=few_cat_variables[i], y='SalePrice', data=df_train)

    plt.show()
important_categorical_vars = ['KitchenQual', 'BsmtQual', 'Heating', 'ExterQual']



# Let us once again categories of these important variables

for col in important_categorical_vars:

    print(col, ": ", df_train[col].unique())
# dataframe now contains only the numerical variables

all_vars_now = num_vars_list + important_categorical_vars

# The df_train ahead contains only these - all_vars_now variables

df_train = df_train[all_vars_now]
# This function takes in the categorical variables and introduces the numerical variable columns for 

# each of the categories for that categorical variables/columns/decorations in the original data

# For each of original categorical columns - will be added separate columns in df_frame for each of the categories

def numerical_columns(cat_var):

    # new DataFrame for the categorical variable cat_var

    df = pd.get_dummies(df_train[cat_var], dummy_na=None)

    df_vars = df.columns

    modified_vars = [(cat_var + "-" + i) for i in df_vars]

    # new columns added in df_frame for each of the categories in cat_var

    for i in range(len(modified_vars)):

        df_train[modified_vars[i]] = pd.Series(df[df_vars[i]]).tolist()

    # making the modified_vars list empty for next cat_var

    modified_vars = []
# Calling numerical_columns function to intoduce new categorical columns as numerical values



for cat_var in important_categorical_vars:

    numerical_columns(cat_var)

# Run only once
# Now deleting the original categorical form of the variables converted into the numerical form

for var in important_categorical_vars:

    del df_train[var]
# The dataframe we now have is ready to use in the regression model 

# we finally have 48 columns in the dataframe

df_train.tail()

# Ready to build regression model now
# Importing sklearn methods

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn import svm

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV
# reassigning the final DataFrame from analysis above as df

df = df_train

df.columns
# Separating 'SalePrice' column



X_columns = df.columns.tolist()

X_columns.remove('SalePrice')
# The input data for the regression models are:



X = df[X_columns]

y = df['SalePrice']
# The input data: dividing into train (75%) and validation data (25%)



x_train_len = round(X.shape[0]*0.75)

x_train = X[:x_train_len]

x_validation = X[x_train_len:]

y_train = y[:x_train_len]

y_validation = y[x_train_len:]



# print Test and Validation data lenght

print("Train data points: ", x_train_len)

print("Validation data points: ", x_validation.shape[0])
x_train.head()
# A class that will define all the regression models as methods



class Models(object):

    

    global seed 

    seed = 34234

    

    # Initialization 

    def __init__(self, x_train, x_validation, y_train, y_validation):

        # changing input as dataframe to list

        self.x_train = [x_train.iloc[i].tolist() for i in range(len(x_train))]

        self.x_validation = [x_validation.iloc[i].tolist() for i in range(len(x_validation))]

        self.y_train = y_train.tolist()

        self.y_validation = y_validation.tolist()

    

    

    @staticmethod

    def print_info(cross_val_scores, mse):

        print("Cross Validation Scores: ", cross_val_scores)

        print("Mean Squared Error: ", mse)

        

        

    # Linear Regression 

    def linear_regression(self, x_train, x_validation,  y_train, y_validation):

        reg = linear_model.LinearRegression()

        # X = np.array(X).reshape([-1, 1])

        reg.fit(self.x_train, self.y_train)

        y_pred_list = reg.predict(self.x_validation)

        mse = mean_squared_error(self.y_validation, y_pred_list)

        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

        cross_val_scores = cross_val_score(reg, self.x_train, self.y_train, cv=kfold)

        print("\nLinear Regression Model")

        self.print_info(cross_val_scores, mse)

        return cross_val_scores, mse

        

    # Random Forest Regression model 

    def random_forest(self, x_train, x_validation,  y_train, y_validation):

        rfr = RandomForestRegressor(n_estimators=8, max_depth=8, random_state=12, verbose=0)

        # X = np.array(X).reshape([-1, 1])

        rfr.fit(self.x_train, self.y_train)

        y_pred_list = rfr.predict(self.x_validation)

        mse = mean_squared_error(self.y_validation, y_pred_list)

        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

        cross_val_scores = cross_val_score(rfr, self.x_train, self.y_train, cv=kfold)

        print("\nRandom Forest Regressor")

        self.print_info(cross_val_scores, mse)

        return cross_val_scores, mse

            

    # Lasso method 

    def lasso(self, x_train, x_validation,  y_train, y_validation):

        reg = linear_model.Lasso(alpha = 0.1)

        # X = np.array(X).reshape([-1, 1])

        reg.fit(self.x_train, self.y_train)

        y_pred_list = reg.predict(self.x_validation)

        mse = mean_squared_error(self.y_validation, y_pred_list)

        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

        cross_val_scores = cross_val_score(reg, self.x_train, self.y_train, cv=kfold)

        print("\nLasso Regression Model")

        self.print_info(cross_val_scores, mse)

        return cross_val_scores, mse

    

    # Gradient Boosing Regressor

    def GBR(self, x_train, x_validation,  y_train, y_validation):

        gbr = GradientBoostingRegressor(n_estimators=175, learning_rate=0.08, max_depth=3, random_state=1232, loss='ls')

        gbr.fit(self.x_train, self.y_train)

        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

        cross_val_scores = cross_val_score(gbr, self.x_train, self.y_train, cv=kfold)

        mse = mean_squared_error(self.y_validation, gbr.predict(self.x_validation))

        print('\nGradient Boosting Regressor')

        self.print_info(cross_val_scores, mse)

        return cross_val_scores, mse
# We use GridSearchCV to find out the best set of parameters for GBR and use it for the 

# regression model analysis and prediction

parameters = {

    'n_estimators' : [170, 175, 180],

    'learning_rate' : [0.075, 0.08, 0.1],

    'max_depth' : [2, 3, 4]

}



gbr = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=5, random_state=232, loss='ls')

gs_cv = GridSearchCV(gbr, parameters).fit(x_train, y_train)

gs_cv.best_params_
from types import FunctionType



methods = [x for x, y in Models.__dict__.items() if type(y) == FunctionType]

methods.remove('__init__')

# Now calling the all regression methods

cross_scores_list, mse_list = [], []

for model in methods:

    reg = Models(x_train, x_validation, y_train, y_validation)

    cross_val_scores, mse = getattr(reg, model)(x_train, x_validation, y_train, y_validation)

    cross_scores_list.append(cross_val_scores)

    mse_list.append(mse)
plot_df = pd.DataFrame()

for i in range(len(methods)):

    plot_df[methods[i]] = cross_scores_list[i]
plt.figure(figsize=(10,6))

plt.title('Comparison of Algorithms')

sns.boxplot(plot_df)

plt.ylim(0.7, 1.0)

plt.ylabel('Cross Val Score')

plt.show()
# Plot Mean Squared Error



plt.plot(mse_list, c='b')

plt.title('Comparision of Algorithms')

plt.ylabel('Mean Squared Error')

plt.ylim(0.0008, 0.015)

x = np.array([0,1,2,3])

plt.scatter(x, mse_list, c='r', marker="s")

plt.xticks(x, methods)

plt.show()