# load packages
import pandas as pd; import numpy as np

# set Jupyter's max column width to 50
pd.set_option('display.max_columns', 50)

# display warnings only the first time
import warnings
warnings.filterwarnings('ignore')
# traffic station characteristics
traffic_station_df = pd.read_csv('../input/us-traffic-2015/dot_traffic_stations_2015.txt.gz',
                                 header=0, sep=',', quotechar='"')

# traffic volume metrics associated to each traffic station
traffic_df = pd.read_csv('../input/us-traffic-2015/dot_traffic_2015.txt.gz',
                         header=0, sep=',', quotechar='"')

# rename terribly long feature names
traffic_station_df.rename(columns = {"number_of_lanes_in_direction_indicated": "lane_count"}, inplace = True)
# view top of station dataframe
print('Traffic Station data:')
traffic_station_df.head()
# view top of traffic volume dataframe
print('Traffic data:')
traffic_df.head()
# specify the features we'll want going forward
station_vars = ["direction_of_travel", "fips_county_code", "fips_state_code",
                "lane_of_travel", "lane_count", "latitude", "longitude", 
                "station_id", "station_location", "type_of_sensor", "year_of_data",
                "year_station_established"]

traffic_vars = ["date", "day_of_data", "day_of_week", "direction_of_travel",
                "fips_state_code", "lane_of_travel", "month_of_data", "record_type",
                "restrictions", "station_id", "traffic_volume_counted_after_0800_to_0900"]
# filter data to just columns of interest and MN based
traffic_station_df = traffic_station_df[station_vars][traffic_station_df.fips_state_code==27]
traffic_df = traffic_df[traffic_vars][traffic_df.fips_state_code==27]

# I don't want to carry that super long column name through the project, shorten it
traffic_df.rename(columns = {"traffic_volume_counted_after_0800_to_0900": "traffic_volume"}, inplace = True)
import folium
from IPython.display import HTML

map_osm = folium.Map(location=[44.9778, -93.2650], zoom_start=11)
# logitude points need to be negative to map
traffic_station_df["longitude_2"] = traffic_station_df["longitude"] * -1

traffic_station_df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude_2"]], 
                                              radius=10, popup=row['station_location'])
                                             .add_to(map_osm), axis=1)

# click on ring of cirlce to see station location name
map_osm
# check out datatypes for traffic station on I-394
traffic_df[traffic_df.station_id=='000326'].info()
# filter down dataset to only traffic station on I-394
traffic_df = traffic_df[traffic_df.station_id=='000326']
# examine numerical features in slimmed dataset
traffic_df.describe()
# drop columns we don't want - drop 'em
traffic_df.drop(['station_id', 'fips_state_code', 'lane_of_travel',
                 'restrictions', 'record_type'], axis=1, inplace=True)
# make sure we have evenly distributed data across the year
traffic_df['month_of_data'].value_counts()
# pulled weather data and saved locally as csv
weather = pd.read_csv('../input/mn-weather-for-2015-traffic/weather_data.csv')

# drop columns with no data
weather.drop(["Unnamed: 0", "fog", "hail"], axis=1, inplace=True)
# join weather data to the traffic volume dataset
traffic_df = pd.merge(traffic_df, weather, how='left', on='date')
# create histograms of each variable
%matplotlib inline
import matplotlib.pyplot as plt
traffic_df.hist(bins=50, figsize=(20,15))
plt.show()
# read in US bank holiday data. Source: https://gist.github.com/shivaas/4758439
holidays = pd.read_csv('../input/holidays-for-2015-traffic/holidays.csv', header=0, sep=',', quotechar='"')
holidays.head()
# join traffic data with holiday data on the date
traffic_df = pd.merge(traffic_df, holidays, how='left', on='date')
traffic_df[['holiday_name']] = traffic_df[['holiday_name']].fillna(value=0)
traffic_df['holiday_flag'] = [0 if x == 0 else 1 for x in traffic_df['holiday_name']]
traffic_df.drop(['id', 'holiday_name'], axis=1, inplace=True)
# how many days during this time period are US bank holidays?
traffic_df["holiday_flag"].value_counts()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(traffic_df, traffic_df["holiday_flag"]):
    strat_train_set = traffic_df.loc[train_index]
    strat_test_set = traffic_df.loc[test_index]
strat_test_set["holiday_flag"].value_counts() / len(strat_test_set)
traffic_df["holiday_flag"].value_counts() / len(traffic_df)
# slimming down traffic_df to only our training data
traffic_df = strat_train_set.copy()
# what columns do we have at our disposal?
traffic_df.columns
# seaborn for vis
# after running this a few times, became most interested in the impact of holidays on traffic volume
import seaborn as sns

# view traffic by DOW with impact of holiday flag
sns.pairplot(x_vars=["day_of_week"], y_vars=["traffic_volume"],
             data=traffic_df, hue="holiday_flag", size=5)
traffic_df[traffic_df.traffic_volume<100]
# remove unpredictable outliers
traffic_df = traffic_df[traffic_df.traffic_volume>50]
# checking to see how much data we have left... training set is getting small :(
traffic_df.shape
print("Median traffic volume is ", traffic_df["traffic_volume"].median())
# let's look at linear correlations
corr_matrix = traffic_df.corr()
corr_matrix["traffic_volume"].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix

attributes = ["traffic_volume", "direction_of_travel", "day_of_week",
              "holiday_flag", "month_of_data", "tempi", "snow", "visi"]

_ = scatter_matrix(traffic_df[attributes], figsize=(12, 8))
# this is the first example of something we'll make reproducible in the Sklean pipeline later
traffic_df["precip"] = traffic_df["snow"] + traffic_df["rain"]
# look at the correlation matrix again
corr_matrix = traffic_df.corr()
corr_matrix["traffic_volume"].sort_values(ascending=False)
traffic_benchmark_data = strat_train_set.copy()
traffic_df = strat_train_set.drop("traffic_volume", axis=1)
traffic_labels = strat_train_set["traffic_volume"].copy()
traffic_df.isnull().sum()
traffic_df[traffic_df.isnull().any(axis=1)]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# subset numerical columns only
traffic_num = traffic_df.drop(["date", "conds"], axis=1)

# fit the imputer to numerical data (aka for our case, find the median values in each column)
imputer.fit(traffic_num)

print("Median values:")
for i in range(len(imputer.statistics_)):
    print(traffic_num.columns[i], imputer.statistics_[i])
X = imputer.transform(traffic_num)
traffic_tr = pd.DataFrame(X, columns=traffic_num.columns)
traffic_tr.loc[(traffic_tr.month_of_data==8) & (traffic_tr.day_of_data==6)]
# finding the average weather condition in August
# TODO: this is NOT repeatable when conds=NA across months
aug_cond_mode = traffic_df[traffic_df['month_of_data']==8]['conds'].mode()

# subset only the 'conds' column
traffic_cat = traffic_df["conds"]
traffic_cat[traffic_cat.isnull()]
# fill the missing values
traffic_cat.fillna(value=aug_cond_mode[0], inplace=True)
# did we fill in the missing values? 
traffic_cat[441]
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Source: http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.CategoricalEncoder.html

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
# Quote from Hands-On:
# "Note that fit_transform() expects a 2D array, but housing_cat
# is a 1D array, so we need to reshape it" traffic_cat is my equivalent to housing_cat
traffic_cat_reshaped = traffic_cat.values.reshape(-1, 1)
cat_encoder = CategoricalEncoder(encoding="onehot-dense")
traffic_cat_1hot = cat_encoder.fit_transform(traffic_cat_reshaped)
cat_encoder.categories_
# Good idea to create custom transformer to create 'precip' variable done manually above
# Also creating the option to make new_weather_var by multiplying tempi by visi
# The output of this class' transform function drops the rain and snow features
# since those will get captured in the categorical features one-hot encoding pipeline

from sklearn.base import BaseEstimator, TransformerMixin

# these are the columns of these variables in traffic_num
rain_ix, snow_ix, tempi_ix, visi_ix = 4, 5, 6, 7

class CombinedWeather(BaseEstimator, TransformerMixin):
    def __init__(self, combine_weather_vars = True):
        self.combine_weather_vars = combine_weather_vars
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        precip = X[:, rain_ix] + X[:, snow_ix]
        if self.combine_weather_vars:
            new_weather_var = X[:, tempi_ix] * X[:, visi_ix]
            X = np.delete(X, np.s_[rain_ix:snow_ix], axis=1)
            return np.c_[X , precip, new_weather_var]
        else:
            X = np.delete(X, np.s_[rain_ix:snow_ix], axis=1)
            return np.c_[X, precip]
            
attr_adder = CombinedWeather(combine_weather_vars = True)
traffic_extra_attribs = attr_adder.transform(traffic_num.values)
# double check column assumption
for i in range(len(traffic_num.columns)):
    print(traffic_num.columns[i], i)
# the result is sort of a copy of traffic_num (except in matrix form
# instead of a pandas df) with the newly created features
# in traffic_num we have 8 columns, in traffic_extra_attribs we should have 10 (new precip and new_weather_var)
len(traffic_extra_attribs[0])
# here is traffic_num as it stands currently
# notice our newly created features aren't in the dataset - we'll get this in the Sklearn pipelines below
traffic_num.head()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedWeather(combine_weather_vars=True)),
        ('std_scaler', StandardScaler()),
    ])

traffic_num_tr = num_pipeline.fit_transform(traffic_num)
# we converted any missing values with the median for that column
# then created additional weather related variables
# then scaled all features using the standard scaler method (see note below on what this does)
# the pipeline takes a pandas dataframe and outputs a numpy array
print(len(traffic_num_tr))
traffic_num_tr[0]
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
num_attribs = list(traffic_num)
cat_attribs = ["conds"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedWeather(combine_weather_vars=False)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
traffic_prepared = full_pipeline.fit_transform(traffic_df)
traffic_df.head(3)
traffic_prepared[0:3]
dow_median = traffic_benchmark_data.groupby(['day_of_week', 'month_of_data'])['traffic_volume'].median().reset_index()
# join on median flows values for each row based on day of week and month
traffic_bench = pd.merge(traffic_benchmark_data, dow_median, how='left', on=['day_of_week','month_of_data'])
# subset the predictions (which is the median traffic volume by dow and month) and actual observed values
traffic_bench_preds = traffic_bench['traffic_volume_y']
traffic_bench_labels = traffic_bench['traffic_volume_x']
from sklearn.metrics import mean_squared_error
benchmark_mse = mean_squared_error(traffic_bench_labels, traffic_bench_preds)
benchmark_rmse = np.sqrt(benchmark_mse)
benchmark_rmse
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(traffic_prepared, traffic_labels)
# FYI: when we are ready to predict on the test set (or on new data) we'll need to run
# that data through our preparation pipeline. Below is an example of how to do that.
some_data = traffic_df.iloc[:5]
some_labels = traffic_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
# for the first 5 values of our traffic_df, predictions above and actual values below
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error
traffic_predictions = lin_reg.predict(traffic_prepared)
lin_mse = mean_squared_error(traffic_labels, traffic_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(traffic_prepared, traffic_labels)
traffic_predictions = tree_reg.predict(traffic_prepared)
tree_mse = mean_squared_error(traffic_labels, traffic_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, traffic_prepared, traffic_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(traffic_prepared, traffic_labels)
# from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, traffic_prepared, traffic_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
# from sklearn.externals import joblib
# joblib.dump(forest_reg, "forest_reg.pkl")
# and later...
# forest_reg = joblib.load("forest_reg.pkl")
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30, 60], 'max_features': [2, 4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(traffic_prepared, traffic_labels)
grid_search.best_params_
# from sklearn.model_selection import cross_val_score
scores = cross_val_score(grid_search.best_estimator_, traffic_prepared, traffic_labels,
                         scoring="neg_mean_squared_error", cv=10)
gr_forest_rmse_scores = np.sqrt(-scores)
display_scores(gr_forest_rmse_scores)
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["precip"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
num_attribs = [x for x in traffic_num if x not in ["rain"]] # bc removed in CombinedWeather
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("traffic_volume", axis=1)
y_test = strat_test_set["traffic_volume"].copy()
# this is a faux pas, but for sake of brevity I'm going with it
X_test['conds'].fillna(value=aug_cond_mode[0], inplace=True)
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
traffic_benchmark_data.groupby(['holiday_flag'])['traffic_volume'].median().reset_index()