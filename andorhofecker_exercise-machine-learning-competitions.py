from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/J561xrzA0dg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
# For simple vectorized calculations
import numpy as np

# Mainly data handling and representation
import pandas as pd

# For statisctics
import scipy
from scipy import stats

# Models
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Data preparation
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Model validation, scoreing
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score

# Math helper
from math import isnan

# Plotting and display
from IPython.display import display
from matplotlib import pyplot as plt
import seaborn as sns
# Path of the file to read.
train_file_path = '../input/train.csv'

# Read the file
home_data = pd.read_csv(train_file_path)

# The shape of the data
home_data.shape
# Create target object and call it y_orig, because we will format this value later
y_orig = pd.DataFrame(home_data.SalePrice)
# Let's see the distribution for y
sns.distplot(y_orig)
# Calculate log(y)
y_log = np.log(y_orig)

# Calculate y_power by PowerTransformer
y_power_scaler = PowerTransformer()
y_power_orig = y_power_scaler.fit_transform(y_orig)

# Calculating skews
orig_skew = scipy.stats.skew(y_orig)[0]
log_skew = scipy.stats.skew(y_log)[0]
power_skew = scipy.stats.skew(y_power_orig)[0]

# Calculating kurtosises
original_kurtosis = scipy.stats.kurtosis(y_orig)[0]
log_kurtosis = scipy.stats.kurtosis(y_log)[0]
power_kurtosis = scipy.stats.kurtosis(y_power_orig)[0]

columns = ["skew", "kurtosis"]

kurtosis_skew_comparison = pd.DataFrame([[orig_skew, original_kurtosis],
                                        [log_skew, log_kurtosis],
                                        [power_skew, power_kurtosis],],
                                        index=["original", "logarithm", "power"],
                                        columns=columns)
kurtosis_skew_comparison
# Plotting distribution of log_y
fig = plt.figure(1)
fig.set_size_inches(18.5, 6)
plt.subplot(121)
plt.title('Log(y)')
plt.xlabel('log(y)')
plt.ylabel('distribution')

sns.distplot(y_log)

# Plotting distribution of power_y
plt.subplot(122)
plt.title('power_y')
plt.xlabel('power_y')
plt.ylabel('distribution')
sns.distplot(y_power_orig)

plt.show()
# Dropping y outliers - the model training showed that there is no need to drop any outliers
from scipy.stats import norm

plt.plot(np.sort(norm.pdf(y_power_orig), axis=0))
plt.title('Normal distribution probabilities ordered.')
plt.xlabel('Datapoints')
plt.ylabel('Probability')

plt.show()
threshold = 0.00
y_filter = [False if x < threshold else True for x in norm.pdf(y_power_orig)]

y_power_pd = pd.DataFrame(y_power_orig)
print("New shape of the y: {}".format(y_power_pd.shape))

y_power = y_power_pd.iloc[y_filter, 0].values.reshape(-1,1)
home_data.describe()
home_data.info()
# Let's copy the original data.
home_data_copy = home_data.copy()
if "Id" in  home_data_copy.columns:
    home_data_copy = home_data_copy.drop(["SalePrice", "Id"], axis=1)
# Separate the categorical values from the numerical values
def separate_X_numerical_categorical(X):
    
    X_numericals = X.copy()
    X_categoricals = X.copy()
    
    # Loop over the columns
    for column in X.columns[1:]:
        if str(X[column].dtype) == "object":
            X_numericals.drop(column, axis=1, inplace=True)
        else:
            X_categoricals.drop(column, axis=1, inplace=True)
             
    return X_numericals, X_categoricals
# Fill nan values with "nan" strings
def fill_nan_X_categoricals(X_categoricals):
    
    X_categoricals_filled = X_categoricals.fillna(value="nan")
    
    return X_categoricals_filled
# One-hot encode the categorical values
def one_hot_encode_categories(X_categoricals):
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    X_cat_one_hot = pd.DataFrame(encoder.fit_transform(X_categoricals), columns=encoder.get_feature_names())
        
    return X_cat_one_hot
def fill_nan_X_numericals(X_numericals, my_imputer=None):
    
    if my_imputer is None:
        # Imputation
        my_imputer = SimpleImputer()
    
        X_numericals_filled = my_imputer.fit_transform(X_numericals)
    else:
        X_numericals_filled = my_imputer.transform(X_numericals)
    
    return pd.DataFrame(X_numericals_filled, columns=X_numericals.columns), my_imputer
def X_numericals_transform(X_numericals, X_scaler=None):
    
    if X_scaler == None:
        # Create transformer
        X_scaler = PowerTransformer()
        
        # Fit and transform
        X_numericals_scaled = X_scaler.fit_transform(X_numericals)
        
    else:
        # Only transform
        X_numericals_scaled = X_scaler.transform(X_numericals)
    
    return pd.DataFrame(X_numericals_scaled, columns=X_numericals.columns), X_scaler
from sklearn.decomposition import PCA

def PCA_transform(X_numericals, PCA_transformer=None):
    if PCA_transformer == None:
        # Create transformer
        PCA_transformer = PCA()
        
        # Fit and transform
        X_numericals_scaled = PCA_transformer.fit_transform(X_numericals)
        
    else:
        # Only transform
        X_numericals_scaled = PCA_transformer.transform(X_numericals)
    
    return pd.DataFrame(X_numericals_scaled, columns=X_numericals.columns), PCA_transformer
def drop_unknown_categories(X_categoricals_test, X_categoricals):
    
    for column in X_categoricals_test.columns:
        if column not in X_categoricals.columns:
            X_categoricals_test.drop(column, axis=1, inplace=True)
            
    return X_categoricals_test
def add_known_categories(X_categoricals_test, X_categoricals):
    
    for column in X_categoricals.columns:
        if column not in X_categoricals_test.columns:
            X_categoricals_test = pd.concat([X_categoricals_test,
                                             pd.DataFrame(np.zeros((X_categoricals_test.shape[0])).reshape(-1,1), columns=[column])],
                                             axis=1).reset_index(drop=True)

    return X_categoricals_test
# Separate the categoricals from the numericals
X_numericals, X_categoricals = separate_X_numerical_categorical(home_data_copy.iloc[y_filter])

##############
# Fill X_categoricals nan values with "nan"
X_cat_filled = fill_nan_X_categoricals(X_categoricals)

# On hot encode the categoricals
X_cat_one_hot = one_hot_encode_categories(X_cat_filled)


##############
# Fill X_numericals nans with imputing
X_numericals_filled, my_imputer = fill_nan_X_numericals(X_numericals)

# Transform also the X_numericals values by power transformer
X_numericals_scaled, X_scaler = X_numericals_transform(X_numericals_filled)

# The principal component analysis did not help the model
#X_numericals_PCA, X_PCA_transformer = PCA_transform(X_numericals_scaled)

#X_numericals_PCA.describe()
X_numericals_outlier = X_numericals_scaled
# Calculate the mean of X_numericals_scaled
X_mean = np.mean(X_numericals_outlier.values, axis=0, keepdims=True)[0]

# Calculate the covariance matrix of X_numericals_scaled
X_covariance_matrix = np.cov(X_numericals_outlier.values.T)

# Calculate the multivariate normal distribution, which is a matrix filled with measurement probabilities being not outlier.
MGD = stats.multivariate_normal.pdf(X_numericals_outlier, mean=X_mean, cov=X_covariance_matrix)

# Use the log for plotting so the outliers can be visualised better
plt.plot(np.log(np.sort(MGD)))
plt.title('Multivariate normal distribution probabilities ordered.')
plt.xlabel('Datapoints')
plt.ylabel('Probability')
plt.show()
# Select a threshold
threshold = -150

filter_array = [True if np.log(meas) > threshold else False for meas in MGD]

X_numericals_outlier_scaled, X_scaler2 = X_numericals_transform(X_numericals_outlier)

print("Original number of measurement points: {}".format(X_cat_one_hot.shape[0]))

# Remove the outliers from the numerical features
X_numericals_outlierless = X_numericals_outlier_scaled.iloc[filter_array]

# Remove the outliers from the categorical features
X_categoricals_outlierless = X_cat_one_hot.iloc[filter_array]

# Concatenate the numerical and categorical features
X_outlierless = pd.concat([X_numericals_outlierless, X_categoricals_outlierless], axis=1)

# Remove the outliers from the output
Y_outlierless = y_power[filter_array]

print("New number of measurement points: {}".format(Y_outlierless.shape[0]))
plt.plot(np.log(np.sort(MGD)))
plt.plot(np.ones(len(MGD))*threshold)
plt.title('Multivariate normal distribution probabilities ordered.')
plt.xlabel('Datapoints')
plt.ylabel('Probability')
plt.show()

# The outliers are cut below the orange line
# Concatenate the X_categoricals and X_numericals
X = X_outlierless
y = Y_outlierless
print("Final shape of the features: {}".format(X.shape))
print("Final shape of the features: {}".format(y.shape))
# Define the score metrics to be the mean absolute error
def mae(predict, actual):
    score = mean_absolute_error(predict, actual)
    return score

mae_score = make_scorer(mae)

# Define the cross validation function
def score(model, X, y):
    score = cross_val_score(model, X, (y.reshape(-1)), cv=5, scoring=mae_score).mean()
    return score
hyperparameters = pd.DataFrame()

models = []
scores = []

# grid search
for i in range(10):
    
    # The hyperparameters:
    
    # The learning rate between 0.03 and 0.01
    learning_rate = 0.03 - 0.02 * (np.random.rand())
    
    # The max depth between 5-6
    max_depth = np.random.randint(3, 6)
    
    # The number of estimators between 1000 and 3000
    n_estimators = np.random.randint(1000, 3000)
    
    # Base score between 0.8 and 0.4
    base_score = 0.8 - 0.4 * (np.random.rand())
    
    # Subsample between 0.8 and .04
    subsample = 0.8 - 0.4 * (np.random.rand())
    
    # Create the model
    model = XGBRegressor(max_depth=max_depth,
                     learning_rate=learning_rate,
                     n_estimators=n_estimators,
                     silent=True,
                     objective='reg:linear',
                     booster='gbtree',
                     subsample=subsample,
                     base_score=base_score,
                     random_state=0,
                     importance_type='gain')
    
    # Calculate the score
    mean_cross_validation = score(model, X, y)
    
    # Save the scores for later evaluation
    scores.append(mean_cross_validation)
    
    # Save the model for later evaluation
    models.append(model)
    
    # Append the hyperparameters for lates examination
    hyperparameters = hyperparameters.append(pd.DataFrame([[mean_cross_validation,
                                                            learning_rate,                                                    
                                                            max_depth,
                                                            n_estimators,
                                                            base_score,
                                                            subsample]],
                                                          columns=["mean_cross_validation",
                                                                   "learning_rate",
                                                                   "max_depth",
                                                                   "n_estimators",
                                                                   "base_score",
                                                                   "subsample"]))
    display(hyperparameters)
corr = hyperparameters.corr()
corr.columns = hyperparameters.columns

corr["index"] = hyperparameters.columns

corr.set_index("index", inplace=True)
print(corr)

sns.heatmap(data=corr, center=0)
plt.show()
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

for model in models:
    eval_set = [(train_X, train_y), (val_X, val_y)]
    eval_metric = ["mae"]
    
    # Fit the model
    %time history = model.fit(train_X, train_y, eval_metric=eval_metric, eval_set=eval_set, verbose=False, early_stopping_rounds=150)

    # Make the predictions on the training set
    train_predictions = model.predict(train_X)

    # Rescale the output to original
    train_predictions_scaled = y_power_scaler.inverse_transform(train_predictions.reshape(-1,1))
    train_y_scaled = y_power_scaler.inverse_transform(train_y)
    
    # Calculate the Mean Average Error of the training set
    train_mae = mean_absolute_error(train_predictions_scaled.astype("float64"), train_y_scaled.astype("float64"))
    print("Mean average error of the training set: {}".format(train_mae))

    # Make the predictions on the validation set
    val_predictions = model.predict(val_X)
    
    # Rescale the output to original
    val_predictions_scaled = y_power_scaler.inverse_transform(val_predictions.reshape(-1,1))
    val_y_scaled = y_power_scaler.inverse_transform(val_y)
    
    # Calculate the Mean Average Error of the validation set
    val_mae = mean_absolute_error(val_predictions_scaled.astype("float64"), val_y_scaled.astype("float64"))
    print("Mean average error of the validation set: {}".format(val_mae))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(history.evals_result()["validation_0"]["mae"], label='training set')
    ax.plot(history.evals_result()["validation_1"]["mae"], label='validation set')
    ax.legend()
    plt.title('Score during training.')
    plt.xlabel('Training step')
    plt.ylabel('Score (MAE)')
    plt.show()
scores
model = models[np.argmin(np.array(scores))]
print(model)
model.fit(X, y.reshape(-1,1))
def convert_test_data(test_data, X_cat_one_hot, my_imputer, X_scaler, X_scaler2):

    ##############
    # Separate the test set categoricals from the numericals
    X_numericals_test, X_categoricals_test = separate_X_numerical_categorical(test_data)

    ##############
    # Fill X_categoricals_test nans with "nan"
    X_categorical_filled_test = fill_nan_X_categoricals(X_categoricals_test)

    # One-hot encode X_categoricals_test
    X_cat_one_hot_test = one_hot_encode_categories(X_categorical_filled_test)

    # Add missing categories
    X_cat_one_hot_test_dropped = add_known_categories(X_cat_one_hot_test, X_cat_one_hot)

    # Remove unkown categories
    X_cat_one_hot_filled_test = drop_unknown_categories(X_cat_one_hot_test_dropped, X_cat_one_hot)

    ##############
    # Fill X_numericals_test nans with imputing
    X_numericals_filled_test, _ = fill_nan_X_numericals(X_numericals_test, my_imputer)

    # Transform also the X_numericals_test values by power transformer
    X_numericals_test_scaled, _ = X_numericals_transform(X_numericals_filled_test, X_scaler)

    X_numericals_test_scaled2, _ = X_numericals_transform(X_numericals_test_scaled, X_scaler2)

    #X_numericals_test_PCA, _ = PCA_transform(X_numericals_test_scaled, X_PCA_transformer)

    X_test = pd.concat([X_numericals_test_scaled2, X_cat_one_hot_filled_test], axis=1)
    print("Test data dimensions: {}".format(X_test.shape))
    
    return X_test
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data_orig = pd.read_csv(test_data_path)

# Drop the index column
if "Id" in test_data_orig.columns:
    test_data = test_data_orig.drop(["Id"], axis=1)

# Preprocess the test data
X_test = convert_test_data(test_data, X_cat_one_hot, my_imputer, X_scaler, X_scaler2)

# Sort the colums like the original in case of misalignements
X_test = X_test[X.columns]
# Make predictions which we will submit.
test_preds_unscaled = model.predict(X_test).reshape(-1, 1)

# Inverse transform the predictions to the original scale
test_preds = y_power_scaler.inverse_transform(test_preds_unscaled)[:,0]

# Save the predictions
output = pd.DataFrame({'Id': test_data_orig['Id'],
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pandas as pd

def get_data(url):
    raw_html = simple_get(url)
    html = BeautifulSoup(raw_html, 'html.parser')
    return parse_html(html)

def parse_html(html):
    SalePrice = 0 # Done
    YearSold = 2019 # Done
    FullBathroom = 0 # Done
    HalfBathroom = 0 # Done
    Bedroom = 0 # Done
    Utilities = "AllPub" # Done
    Garage = 0
    Fence = 0

    SalePrice = int(html.find_all('span', class_='price')[0].text[1:])
    brbaline = html.find_all('span', class_='shared-line-bubble')[0].text
    Bedroom = int(brbaline[:brbaline.index("BR")])
    bath = float(brbaline[brbaline.index("/")+1:brbaline.index("Ba")])
    FullBathroom = int(bath)
    HalfBathroom = int((bath-FullBathroom)>0)

    Garage = len([1 for x in html.select('p.attrgroup span') if "garage" in x.text.lower()])>0

    for i, sec in enumerate(html.select('section')):
        if sec.get("id") is not None:
            if "postingbody" in sec.get("id"):
                if "fence" in sec.text.lower():
                    Fence = 1
                if "garage" in sec.text.lower():
                    Garage = 1

    output = pd.DataFrame([[SalePrice,YearSold,FullBathroom,HalfBathroom,Bedroom,Utilities,Garage,Fence]],
                          columns=["SalePrice", "YrSold", "FullBath", "HalfBath", "BedroomAbvGr", "Utilities", "GarageCars", "Fence"])
    return output

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


weburl = 'https://ames.craigslist.org/reb/d/ames-stop-renting-when-you-can-own/6796373030.html'
urls = ["https://ames.craigslist.org/reb/d/ames-stop-renting-when-you-can-own/6796373030.html",
        "https://ames.craigslist.org/reo/d/kelloggjasperth-ave-kellogg-ia/6796762804.html",
        "https://ames.craigslist.org/reo/d/story-city-commercial-light-industrial/6799601680.html",
        "https://ames.craigslist.org/reb/d/ankeny-split-foyer/6800347988.html",
        "https://ames.craigslist.org/reo/d/polk-city-brand-new-villa-in-wolf-creek/6777063925.html",
        "https://ames.craigslist.org/reo/d/omaha-walk-out-ranch/6797562631.html",
        "https://ames.craigslist.org/reb/d/ames-open-sat-1-12-at-130-to-3-pm/6791622440.html",
        "https://ames.craigslist.org/reb/d/waukee-modern-living/6784701226.html",
        "https://ames.craigslist.org/reo/d/ames-ames-ia-why-pay-rent-buy-and-save/6767869407.html",
        "https://ames.craigslist.org/reo/d/collins-house-for-sale/6798248546.html"]

real_data_list = []

for url in urls:
    print(url)
    real_data_list.append(get_data(url))

real_data = get_data(weburl)
real_data = pd.concat(real_data_list)
real_data
scraped_columns = ["SalePrice", "YrSold", "FullBath", "HalfBath", "BedroomAbvGr", "Utilities", "GarageCars", "Fence"]
X_columns = ["YrSold", "FullBath", "HalfBath", "BedroomAbvGr", "Utilities", "GarageCars", "Fence"]
real_data = real_data[scraped_columns]
X_real_data = real_data[X_columns]
X_real_data.shape
def X_preprocess(home_data_copy):
    # Separate the categoricals from the numericals
    X_numericals, X_categoricals = separate_X_numerical_categorical(home_data_copy)

    ##############
    # Fill X_categoricals nan values with "nan"
    X_cat_filled = fill_nan_X_categoricals(X_categoricals)

    # On hot encode the categoricals
    X_cat_one_hot = one_hot_encode_categories(X_cat_filled)


    ##############
    # Fill X_numericals nans with imputing
    X_numericals_filled, my_imputer = fill_nan_X_numericals(X_numericals)

    # Transform also the X_numericals values by power transformer
    X_numericals_scaled, X_scaler = X_numericals_transform(X_numericals_filled)

    #X_numericals_PCA, X_PCA_transformer = PCA_transform(X_numericals_scaled)

    #X_numericals_PCA.describe()
    X_numericals_outlier = X_numericals_scaled

    # Calculate the mean of X_numericals_scaled
    X_mean = np.mean(X_numericals_outlier.values, axis=0, keepdims=True)[0]

    # Calculate the covariance matrix of X_numericals_scaled
    X_covariance_matrix = np.cov(X_numericals_outlier.values.T)

    # Calculate the multivariate normal distribution, which is a matrix filled with measurement probabilities being not outlier.
    MGD = stats.multivariate_normal.pdf(X_numericals_outlier, mean=X_mean, cov=X_covariance_matrix)

    # Select a threshold
    threshold = -150

    # Calculate a filter array for removing the outliers from the dataset
    filter_array = [True if np.log(meas) > threshold else False for meas in MGD]

    X_numericals_outlier_scaled, X_scaler2 = X_numericals_transform(X_numericals_outlier)
    
    # Remove the outliers from the categoricals and numericals
    X_numericals_outlierless = X_numericals_outlier_scaled.iloc[filter_array]
    X_categoricals_outlierless = X_cat_one_hot.iloc[filter_array]

    # Concatenate the X_numericals_scaled and X_categoricals
    X_outlierless = pd.concat([X_numericals_outlierless, X_categoricals_outlierless], axis=1)
    print("The final shape of the features: {}".format(X_outlierless.shape))
    
    
    Y_outlierless = y_power[filter_array]
    print("The final shape of the output: {}".format(Y_outlierless.shape))

    plt.plot(np.log(np.sort(MGD)))
    plt.plot(np.ones(len(MGD)) * threshold)
    plt.title('Multivariate normal distribution probabilities ordered.')
    plt.xlabel('Datapoints')
    plt.ylabel('Probability')
    plt.show()
    
    return X_outlierless, Y_outlierless, X_cat_one_hot, my_imputer, X_scaler, X_scaler2


def grid_search(iterations):
    hyperparameters = pd.DataFrame()

    models = []

    # grid search
    for i in range(iterations):

        # The hyperparameters:

        # The learning rate between 0.333 and 0.00333
        learning_rate = 10 ** (-2 * np.random.rand()) / 3

        # The max_depth between 3-12
        max_depth = np.random.randint(3, 12)

        # The number of estimators between 1000 and 10000
        n_estimators = np.random.randint(1000, 10000)

        # Base score between 1 and 0.5
        base_score = 1 - 0.5 * (np.random.rand())

        # Subsample between 1 and 0.6
        subsample = 1 - 0.4 * (np.random.rand())

        model = XGBRegressor(max_depth=max_depth,
                     learning_rate=learning_rate,
                     n_estimators=n_estimators,
                     subsample=subsample,
                     base_score=base_score,
                     random_state=0)

        # Calculate the score
        mean_cross_validation = score(model, X, y)

        models.append(model)

        # Append the hyperparameters for lates examination
        hyperparameters = hyperparameters.append(pd.DataFrame([[mean_cross_validation,
                                                                learning_rate,                                                    
                                                                max_depth,
                                                                n_estimators,
                                                                base_score,
                                                                subsample]],
                                                              columns=[
                                                                  "mean_cross_validation",
                                                                  "learning_rate",
                                                                  "max_depth",
                                                                  "n_estimators",
                                                                  "base_score",
                                                                  "subsample"
                                                              ]))

    return models, hyperparameters
X_real_data.info()
home_data[scraped_columns].info()
home_data_fenced = home_data_copy.copy()

# The fence feature of the scraped data is only categorical so concentrate the train data
home_data_fenced["Fence"] = home_data_copy["Fence"].apply(lambda x: 1 if isinstance(x, str) else 0)
X, y, X_cat_one_hot, my_imputer, X_scaler, X_scaler2 = X_preprocess(home_data_fenced[X_columns])
X.info()
# Teach previous model on new columns
models, hyperparameters = grid_search(20)
scores = hyperparameters["mean_cross_validation"]
display(hyperparameters)
# Try the default GradientBoostingRegressor
model = GradientBoostingRegressor()
scores.append(models.append(score(model, X, y)))
models.append(model)
# Try the default XGBRegressor
model = XGBRegressor()
scores.append(models.append(score(model, X, y)))
models.append(model)
model = models[np.argmin(scores.values)]
print(model)
model.fit(X, y)
X_test = convert_test_data(X_real_data, X_cat_one_hot, my_imputer, X_scaler, X_scaler2)
X_test = X_test[X.columns]
X_test.shape, X_real_data.shape
X_test
assert 0 == np.sum(~np.isfinite(X_test.values)), "There are infinite values in the X_test."
assert 0 == np.sum(np.isnan(X_test.values)), "There are nan values in the X_test."

# Make predictions which we will submit.
test_preds_unscaled = model.predict(X_test).reshape(-1, 1)
# Inverse transform the predictions to the original scale
predicted_prices = list(y_power_scaler.inverse_transform(test_preds_unscaled)[:,0].astype('int'))
actual_prices = list(real_data["SalePrice"])

pd.DataFrame(np.array([predicted_prices, actual_prices]).T, columns=["Predicted prices (USD)",
                                                                       "Actual prices (USD)"]).transpose()
prices_comparison = pd.DataFrame([predicted_prices, actual_prices]).transpose().sort_values(1)
prices_comparison.columns = ["predicted", "actual"]

ax = prices_comparison.plot(kind="bar", title="Comparison real vs predicted house prices.")

ax.set_xlabel('House Id')
ax.set_ylabel('Price (USD)')
prices_comparison.corr()
print("The final MAE score of the real life data: {}".format(mean_absolute_error(prices_comparison["predicted"],
                                                                                 prices_comparison["actual"])))