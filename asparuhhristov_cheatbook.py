# First import the required packages
from IPython.display import HTML
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'
get_ipython().magic('matplotlib inline')
MAX_LEVELS = 10
TOP_X = 10
TARGET_VARIABLE = "SalePrice"
def read_data():
    # Read the files
    test_data = pd.read_csv('../input/test.csv')
    train_data_full = pd.read_csv('../input/train.csv')
    
    # Seperate the target variable
    y_train = train_data_full[TARGET_VARIABLE]
    train_data = train_data_full.drop([TARGET_VARIABLE], 
                                 axis = 1, inplace = False)
    ###train_data = train_data.iloc[:,:-1]
    return train_data, test_data, y_train, train_data_full

train_data, test_data, y_train, train_data_full = read_data()

# Get a brief view of the data
print("There are", train_data.shape[0], "train samples")
print("There are", test_data.shape[0], "test samples")
print("\nThe target variable is:" ,TARGET_VARIABLE)
print("and the top 5 values are:")
print(y_train.nlargest(5).to_string(index=False))
###print(y_train.value_counts().index[:5].to_string(index=False))
print("In total there are", test_data.shape[1], "features")
print ("\nThose features are:\n", train_data.columns.tolist())
print ("\nThe data looks like this:\n", train_data.head())
print (train_data.describe())
def assert_statements(train_data, test_data, y_train):
    assert train_data.shape[1] == test_data.shape[1], "Test and train data have different number of features!"
    assert train_data.Id.nunique()==train_data.shape[0], "IDs are not unique"
    assert y_train.isnull().sum()==0, "There are missing values in the target variable"

assert_statements(train_data, test_data, y_train)
def prepare_preprocess(train_data, test_data):
    # Combine the training and test data for preprocessing purposes
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    complete_data = pd.concat([train_data, test_data],
                              axis=0, ignore_index=True)
    return complete_data

complete_data = prepare_preprocess(train_data, test_data)
if y_train.dtypes == object:
    y_train = y_train.astype('category')
    # Get the frequency of the categorical levels
    print(y_train.value_counts(dropna = False))
    ###print(y_train.value_counts().index[:10].to_string(index=False)
    
    # Convert Target Variable from text to 0 and 1
    ##y_train = y_train.map({'positive': 1, 'negative': 0})
    ##print(y_train.head())
    
else:
    print ("Skew is:", y_train.skew())
    plt.subplot(1,2,1)
    plt.hist(y_train, color='blue')
    plt.title("Histogram of the %s" %(TARGET_VARIABLE))

    plt.subplot(1,2,2)
    plt.boxplot(y_train)
    plt.title("Boxplot of the %s" %(TARGET_VARIABLE))

    plt.tight_layout()
    plt.show()
    
    if y_train.skew()>1:
        y_train_unskewed = np.log1p(y_train)
        print ("Skew is:", y_train_unskewed.skew())
        plt.subplot(1,2,1)
        plt.hist(y_train_unskewed, color='blue')
        plt.title("Histogram of the skewed %s" %(TARGET_VARIABLE))

        plt.subplot(1,2,2)
        plt.boxplot(y_train_unskewed)
        plt.title("Boxplot of the skewed %s" %(TARGET_VARIABLE))

        plt.tight_layout()
        plt.show()
def to_categorical(dataset):
    # Check the type of features and convert objects to categorical
    categorical_features = dataset.select_dtypes(exclude=[np.number])
    categorical_columns = categorical_features.columns.tolist()
    
    for col in categorical_columns:
        dataset[col] = dataset[col].astype('category')

    return(categorical_features,categorical_columns)

def to_numerical(dataset):
    numeric_features = train_data.select_dtypes(include=[np.number])
    numeric_columns = numeric_features.columns.tolist()
    return(numeric_features, numeric_columns)

categorical_features,categorical_columns = to_categorical(complete_data)
numeric_features, numeric_columns = to_numerical(complete_data)

print("There are:",len(categorical_columns),"categorical features.\n ")
print(categorical_features.describe())
print("\n There are:",len(numeric_columns),"numeric features.\n ")
print(numeric_features.describe())
# Examine the correlation between numerical features
corrmat = train_data_full.corr()
print (corrmat[TARGET_VARIABLE].sort_values(ascending=False)[:TOP_X], '\n')
print (corrmat[TARGET_VARIABLE].sort_values(ascending=False)[-TOP_X:])

#TARGET_VARIABLE correlation matrix
cols = corrmat.nlargest(TOP_X, TARGET_VARIABLE)[TARGET_VARIABLE].index
cm = np.corrcoef(train_data_full[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': TOP_X}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()
def plot_corr_numeric_cat_bar(dataset,feature):
    quality_pivot = dataset.pivot_table(index=feature, values=TARGET_VARIABLE, aggfunc=np.median)
    quality_pivot.plot(kind='bar', color='blue')
    plt.title("Plot of the %s according to %s" %(TARGET_VARIABLE,feature))
    plt.xlabel('%s' %(feature))
    plt.ylabel('Median %s' %(TARGET_VARIABLE))
    plt.xticks(rotation=0)
    plt.show()

def plot_corr_numeric_cat_box(dataset,feature):
    dataset.boxplot(column=TARGET_VARIABLE, by=feature)
    plt.title("Boxplot of %s according to %s" %(TARGET_VARIABLE,feature))
    plt.suptitle("")
    plt.ylabel('%s' %(TARGET_VARIABLE))
    plt.xlabel('%s' %(feature))
    plt.show()
    
def plot_corr_numeric_feat(dataset,feature):
    plt.scatter(data=dataset, x=feature, y=TARGET_VARIABLE)
    plt.title("Correlation between %s and %s" %(TARGET_VARIABLE,feature))
    plt.ylabel('%s' %(TARGET_VARIABLE))
    plt.xlabel('%s' %(feature))
    plt.show()
    
plot_corr_numeric_cat_bar(train_data_full,'OverallQual')
plot_corr_numeric_cat_box(train_data_full,'SaleCondition')
plot_corr_numeric_feat(train_data_full,'GrLivArea')
# Simple outlier removal, need more work
print("The highest five prices are:")
print(y_train.sort_values(ascending=False)[:5])
print(", whereas the 0.95 quantile:", y_train.quantile(0.95))

print("\nThe highest five GrLivArea are:")
print(complete_data[complete_data['is_train']==1]['GrLivArea'].sort_values(ascending=False)[:5])
print(", whereas the 0.95 quantile:", complete_data[complete_data['is_train']==1]['GrLivArea'].quantile(0.95))

indxs = (complete_data['GrLivArea'] > 4000) & (complete_data.is_train==1)
complete_data = complete_data[~indxs]
y_train = y_train[~indxs]
y_train_unskewed = y_train_unskewed[~indxs]

print("\nAfter removing, the highest five GrLivArea are:")
print(complete_data[complete_data['is_train']==1]['GrLivArea'].sort_values(ascending=False)[:5])
print(", whereas the 0.95 quantile:", complete_data[complete_data['is_train']==1]['GrLivArea'].quantile(0.95))
# Examine the TOP_X columns with most null values
def get_nulls(dataset):
    nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:TOP_X])
    nulls_perc = (dataset.isnull().sum() / len(dataset)) * 100
    nulls.columns = ['Null Count']
    nulls['Null Percentage'] = nulls_perc
    nulls.index.name = 'Feature'
    return nulls

def plot_null_values(nulls):
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x = nulls['Null Percentage'].sort_values(ascending=False)[:TOP_X].index, 
                y = nulls['Null Percentage'].sort_values(ascending=False)[:TOP_X])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

nulls = get_nulls(complete_data)
print(nulls)
plot_null_values(nulls)
def average_imputing(dataset,impute_averages):
    for feature in impute_averages:
        if dataset[feature].dtype.name == "category":
            dataset[feature].fillna(value=dataset[feature].value_counts().index[0],
                   inplace=True)
        else:
            dataset[feature].fillna(value=dataset[feature].mean(),
                   inplace=True)

def zero_imputing(dataset,impute_zero):    
    for feature in impute_zero:
        if dataset[feature].dtype.name == "category":
            if "Missing" not in dataset[feature].cat.categories:
                dataset[feature].cat.add_categories(["Missing"], inplace=True)
            dataset[feature].fillna("Missing", inplace=True)
        else:
            dataset[feature].fillna(value = 0, inplace=True)
            dataset[feature].replace(".", 0, inplace=True)

# impute automatically (median or most-frequent)
impute_averages = ["MasVnrArea","Utilities","BsmtQual"]
impute_zero = complete_data.columns.tolist()

average_imputing(complete_data,impute_averages)
zero_imputing(complete_data,impute_zero)
# Group all minorly represented levels in one big level "Others"
# one can include the missing values in this "Others" level or not
def cutting(dataset, level_cut, max_level = MAX_LEVELS, incl_miss = 1):
    for feature in level_cut:
        if "Others" not in dataset[feature].cat.categories:
            dataset[feature].cat.add_categories(["Others"], inplace=True)
        if incl_miss == 1:
            pop_levels = dataset[feature].value_counts().index[:max_level-1]
            keep_indxs = dataset[feature].isin(pop_levels)
            dataset.loc[~keep_indxs, feature] = "Others"            
        else:
            pop_levels = dataset[feature].value_counts().loc[lambda x: x.index != 'Missing'].index[:max_level-1]
            keep_indxs = (dataset[feature].isin(pop_levels)) | (dataset[feature]=='Missing')
            dataset.loc[~keep_indxs, feature] = "Others"

# Create a feature with only two levels - the important one and the "others"
def max_cutting(dataset, feature, level):
    new_feature = 'EngFeat_2lvl_'+feature
    eng_data[new_feature] = 1 - ~(eng_data[feature] == level)
    print("A new column", new_feature, 
          "was created from", feature, 
          "indicating if the value is", level,
          "or not.")

# create a new column indicating that it is missing
def create_miss(dataset, indicators_feat):
    for feature in indicators_feat:
        new_feature='EngFeat_has_'+feature
        dataset[new_feature] = ~(dataset[feature]=="Missing")
        print("A new column", new_feature, 
              "was created from", feature, 
              "indicating if there is a", feature, 
              "or not.\n")
        
# Bin numerical variables (like age or just grouping ordered numericals)
def binning(dataset, features, levels):
    for feature in features:
        dataset[feature] = pd.qcut(dataset[feature], 
                                   q=levels, labels=False ,
                                   duplicates="drop")
        print(feature, "levels after grouping:")
        print(dataset[feature].value_counts(dropna=False))
        
# First create a new dataframe which will include the new features.
eng_data = complete_data.copy()    

# Manually modify a column
new_feature = "EngFeat_old"
basis_feature = "YearBuilt"
eng_data[new_feature]=eng_data[basis_feature].apply(lambda n : 2018-n)
print("A new column", new_feature,"was created from", 
      basis_feature, "indicating how old a house is.\n")

eng_data['TotalSF'] = eng_data['TotalBsmtSF'] + eng_data['1stFlrSF'] + eng_data['2ndFlrSF']

# Extract Title from Name, store in column and plot barplot
##eng_data['Title'] = eng_data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

cutting(eng_data, ["GarageCond"])

create_miss(eng_data, ["GarageCond"])

max_cutting(eng_data, "SaleCondition", 'Partial')

binning(eng_data,['LotArea'],4)


# One hot encoding features for all categorical, no need to fill miss values, no need to encode them
list_ohe = [col for col in eng_data if eng_data[col].dtypes.name == "category"]
print ("Number of categorical features considered to be one-hot encoded:", len(list_ohe))
list_ohe = ["Street","Alley"]
if len(list_ohe)<5:
    for feature in list_ohe:
        ohe_feature = pd.get_dummies(eng_data[feature],
                                     prefix='EngFeat_' + 
                                     feature, drop_first=True, 
                                     dummy_na=True)
        eng_data = pd.concat([eng_data, ohe_feature], 
                             axis=1);
    print("The following categorical features were successfully one-hot encoded: \n",list_ohe)
else:
    print("The categorical features are way too many for one hot encoding, consider other solutions")
    
# Label Encoder for the rest
list_le = [col for col in eng_data if eng_data[col].dtypes.name == "category"]
encoder = LabelEncoder()
for feature in list_le:
    eng_data[feature] = encoder.fit_transform(eng_data[feature])
print("The following categorical features were successfully encoded: \n",list_le)

# Standardize the other numerical variables
listToScale = [col for col in eng_data if 
               (eng_data[col].dtypes.name != "uint8") &
               (eng_data[col].dtypes.name != 'category') &
               (col!='Id') & (col!="is_train")]
statsCol=zip(eng_data[listToScale],
             eng_data[listToScale].mean(),
             eng_data[listToScale].std())
print("The mean and std of the numerical features for scaling considerations")
for stats in statsCol:
    print (stats)
#listToScale = ['MSSubClass','LotFrontage']   
scaler = StandardScaler()                              
eng_data[listToScale]=scaler.fit_transform(eng_data[listToScale])
#Drop unnecesary stuff
listDrop = [colName for colName in eng_data.columns if (colName.find('Garage')>-1) & (colName.find('EngFeat')==-1)]
eng_data.drop(listDrop, axis=1, inplace = True)
eng_data.drop(list_ohe, axis=1, inplace = True)
# First get back to the training and test data
X_train = eng_data[eng_data.is_train==1]
X_test = eng_data[eng_data.is_train==0]
X_train.drop(["is_train","Id"], axis=1, inplace = True)
X_test.drop(["is_train","Id"], axis=1, inplace = True)

# Create a train/test set
TEST_SIZE=.3
data_train, data_test, pred_train, pred_test = train_test_split(X_train, y_train_unskewed, random_state=1, test_size=TEST_SIZE)
print("The train / test size ration is:",1-TEST_SIZE,"/",TEST_SIZE)
def model_builder(model, params, cv = 5):
    grid_model = GridSearchCV(estimator=model, 
                              param_grid=params, 
                              scoring='neg_mean_squared_error', 
                              cv=cv, 
                              verbose=0)
    grid_model.fit(data_train, pred_train)
    return grid_model

def print_results(grid_model, params, name, cv = 5):
    plt.figure(figsize=[13,8])
    sizeParam = len(params.keys())
    param1 = list(params.keys())[0]
    
    if sizeParam==1:
        plt.plot(params.get(param1), 
                 np.sqrt(np.abs(grid_model.cv_results_['mean_test_score'])))
    
    if sizeParam>1:
        param2 = list(params.keys())[1]
        sizeParam2 = len(params.get(param2))
        for i in range(sizeParam2):
            plt.plot(params.get(param1), 
                     np.sqrt(np.abs(grid_model.cv_results_['mean_test_score'][i::sizeParam2])),
                     label = str('CV Score for '+ 
                                 param2 +' = ' 
                                 + str(params.get(param2)[i])))
            plt.legend()
    plt.title(name + " Grid Search " + str(cv) + "-fold CV")
    plt.xlabel('{}'.format(list(params.keys())[0]))
    plt.ylabel('RMSE')
    plt.xticks(params.get(param1),params.get(param1))
    plt.show()
    print("Best parameters found: ", grid_model.best_params_)
    print("RMSE on train: ", np.round(np.sqrt(np.abs(
        grid_model.best_score_)),3))
    print("RMSE on test:", np.round(np.sqrt(np.abs(
        grid_model.score(data_test, pred_test))),3))

def print_feature_ranking(grid_model, name, top_x = 10, with_std = 0 , coef = 0):
    # Print the feature ranking
    if coef == 1:
        importances = grid_model.best_estimator_.coef_
        print(name, "picked" , str(sum(importances != 0)) , 
              "variables and eliminated the other ",
              str(sum(importances == 0)), 
              "variables")

    else:    
        importances = grid_model.best_estimator_.feature_importances_
        
    indices = np.argsort(importances)[-top_x:]
    print("\nFeature ranking for", name)
    for f in np.arange(top_x-1,-1,-1):
        print("%d. %s (%f)" % (top_x-f, data_train.iloc[:,indices[f]].name, 
                               importances[indices[f]]))
    # Plot the feature importances of the model
    plt.figure()
    plt.title("Feature importances for " + name)
    if with_std == 1:
        std = np.std([tree.feature_importances_ for tree in 
              grid_model.best_estimator_.estimators_],
             axis = 0)
        plt.bar(range(top_x), 
                importances[indices], 
                color="r",
                yerr=std[indices],
                align="center")
    else:
        plt.bar(range(top_x), 
                importances[indices], 
                color="r",
                align="center")
    plt.xticks(range(top_x), 
               data_train.iloc[:,indices].columns.tolist(),
               rotation=75)
    plt.xlim([-1, top_x])
    plt.show()
    
def print_predicted(grid_model, name):
    model_pred = grid_model.predict(data_test)

    plt.scatter(np.expm1(model_pred), 
                np.expm1(pred_test), 
                alpha=.75, color='b') #alpha helps to show overlapping data
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title(name + " with = {}".format(grid_model.best_params_))
    overlay = 'RMSE is: {}'.format(
                        np.round(np.sqrt(mean_squared_error(pred_test, model_pred)),3))
    plt.annotate(s=overlay,xy=(0.65, 0.15),xycoords='figure fraction',size='large')
    plt.show()
# Fitting a Lasso Linear Regression
lasso_param_grid = {'alpha': [0.001,0.005,0.01,0.05]}
lasso = linear_model.Lasso()
lasso_model = model_builder(lasso, lasso_param_grid)

print_results(lasso_model, lasso_param_grid, "Lasso Linear Regression")
print_feature_ranking(lasso_model, "Lasso Linear Regression", coef = 1)
print_predicted(lasso_model, "Lasso Linear Regression")
# Fitting a K-Neighbors Regressor
knn_param_grid = {'n_neighbors': list(np.arange(1, 30))}
knn = KNeighborsRegressor()
knn_model = model_builder(knn, knn_param_grid)

print_results(knn_model, knn_param_grid, "K-Neighbors Regressor")
print_predicted(knn_model, "K-Neighbors Regressor")
# Fitting a Random Forest
rf_param_grid = {'n_estimators': [150, 200, 250], 
                  'max_depth': [40, 50]}
rf = RandomForestRegressor()
rf_model = model_builder(rf, rf_param_grid)

print_results(rf_model, rf_param_grid, "Random Forest")
print_feature_ranking(rf_model, "Random Forest", with_std = 1)
print_predicted(rf_model, "Random Forest")
# Fitting a Gradient Boosting Regressor
gbm_param_grid = {'learning_rate': [0.05,0.1,0.15,0.2], 
                  'n_estimators': [300,400,500]}
gbm = GradientBoostingRegressor()
gbm_model = model_builder(gbm, gbm_param_grid)

print_results(gbm_model, gbm_param_grid, "Gradient Boosting Regression")
print_feature_ranking(gbm_model, "Gradient Boosting Regression")
print_predicted(gbm_model, "Gradient Boosting Regression")
lasso_model.fit(X_train,y_train_unskewed)
knn_model.fit(X_train,y_train_unskewed)
rf_model.fit(X_train,y_train_unskewed)
gbm_model.fit(X_train,y_train_unskewed)

predictions_lasso = lasso_model.predict(X_test)
predictions_knn = knn_model.predict(X_test)
predictions_rf = rf_model.predict(X_test)
predictions_gbm = gbm_model.predict(X_test)

submission = pd.DataFrame()
submission['Id'] = test_data.Id
final_predictions = 0.4*np.expm1(predictions_lasso) + 0.2*(
    np.expm1(predictions_gbm) 
    + np.expm1(predictions_knn) 
    + np.expm1(predictions_rf))
submission['SalePrice'] = final_predictions
print(submission.head())
submission.to_csv('submission.csv', index=False)
