# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)    # 랜덤함수 초기화 

# To plot pretty figures
%matplotlib inline     
import matplotlib
import matplotlib.pyplot as plt   # 시각화 패키지 
plt.rcParams['axes.labelsize'] = 14   # 시각화 관련 세팅 
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."   # 현재 디렉토리를 가르킴 
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):  # 그림 저장 함수 
    if not os.path.isdir(IMAGES_PATH):     
        os.makedirs(IMAGES_PATH)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import pandas as pd

data_path = "../input/housing.csv"
housing = pd.read_csv(data_path)

# see the basic info
housing.info()
np.sum(pd.isnull(housing)) 
housing["ocean_proximity"].value_counts()  # pandas 인덱싱의 한 예.  여기서는 ocean_proximity 열을 선택 
housing.describe(include='all')
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))   # pandas DataFrame이 간단한 시각화 함수를 갖고 있음 
save_fig("attribute_histogram_plots")
plt.show()
housing.info()
# to make this notebook's output identical at every run
np.random.seed(42)
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_set.shape, test_set.shape
test_set.head()
housing["median_income"].hist()
# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].value_counts()
housing["income_cat"].hist()
# StratifiedShuffleSplit 이용 
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housing["income_cat"].value_counts() / len(housing)
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props
corr_matrix = housing.corr()
corr_matrix
corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
# from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, c='blue')
plt.axis([0, 16, 0, 550000])   # x축 범위: 0~16,  y축 범위: 0~550000 
save_fig("income_vs_house_value_scatterplot")
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
housing.info()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
housing.describe()
housing.info()
from scipy import stats
from scipy.stats import norm, skew 
numeric_features = housing.dtypes[housing.dtypes != "object"].index
numeric_features
# Check the skew of all numerical features
skewed_feats = housing[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness
import seaborn as sns
sns.distplot(housing['rooms_per_household']);
housing["rooms_per_household"] = np.log1p(housing["rooms_per_household"])
sns.distplot(housing['rooms_per_household']);
sns.distplot(housing['population_per_household']);
housing["population_per_household"] = np.log1p(housing["population_per_household"])
sns.distplot(housing['population_per_household']);

housing["population_per_household"].where(housing["population_per_household"] < 2.75, 2.75, inplace=True)
sns.distplot(housing['population_per_household']);
#Check the skew of all numerical features
skewed_feats = housing[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness
sns.distplot(housing['population']);

housing["population"] = np.log1p(housing["population"])
sns.distplot(housing['population']);
housing["population"].where(housing["population"] > 2.5, 2.5, inplace=True)
sns.distplot(housing['population']);
sns.distplot(housing['total_rooms']);
housing["total_rooms"] = np.log1p(housing["total_rooms"])
sns.distplot(housing['total_rooms']);
housing["total_rooms"].where(housing["total_rooms"] > 4, 4, inplace=True)
sns.distplot(housing['total_rooms']);
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
housing.columns
features = "longitude+latitude+housing_median_age+total_rooms+total_bedrooms+population+households+median_income+ocean_proximity+income_cat+rooms_per_household+bedrooms_per_room+population_per_household"

# Break into left and right hand side; y and X
y, X = dmatrices("median_house_value ~" +features, data=housing, return_type="dataframe")

# For each Xi, calculate VIF
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Fit X to y
result = sm.OLS(y, X).fit()
print(result.summary())
housing.drop('total_bedrooms', axis=1, inplace=True)
features = "longitude+latitude+housing_median_age+total_rooms+population+households+median_income+ocean_proximity+income_cat+rooms_per_household+bedrooms_per_room+population_per_household"

#Break into left and right hand side; y and X
y, X = dmatrices("median_house_value ~" +features, data=housing, return_type="dataframe")
#For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
#you can drop column like this 
#housing.drop('households', axis=1, inplace=True)

#바뀐 데이터로 다시 train test 나눠준다 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels(median_house_value) for training set
housing_labels = strat_train_set["median_house_value"].copy()
housing.info()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows    # 모두 total_bedrooms 에 결측치가 있음  
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")  
housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
housing_num.median().values
X = imputer.transform(housing_num)
X   # X는 numpy의 ndarray
housing_tr = pd.DataFrame(X, columns=housing_num.columns,   # 결측치가 대치된 X를 pandas DataFrame으로 변환 
                          index = list(housing.index.values))  
sample_incomplete_rows.index.values   
housing_tr.loc[sample_incomplete_rows.index.values]   # 확인 
imputer.strategy
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.head()
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
housing.info()
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

"""
CombinedAttributesAdder :
앞에서 bedrooms, population attribute(column, 열) 들에 대해 행한 조작을 한 번에 처리하기 위한 클래스 
"""
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):  
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num.info()
housing_num_tr
housing.head()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
 

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

#oc~만 뺴고 housing_num에 담는다
housing_num = housing.drop("ocean_proximity",axis=1)
housing_num
#이를 
housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet

#credit to @hesenp 
class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizerPipelineFriendly()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape
pd.DataFrame(housing_prepared).info()
pd.DataFrame(housing_prepared).head()   # 모든 열들에 결측치 없고, standardize, 수치화 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()    # 선형 회귀 알고리즘을 실행할 수 있는 객체(lin_reg)를 만들어 
#lin_reg = LinearRegression(normalize=True, copy_X=True, n_jobs=1)

# lin_reg 에게 입력 training set인 housing_prepared 와 해당 레이블인 housing_labels을 주며 
# 학습(fit) 하라고 지시.  학습된 모델은 lin_reg 자신에게 있음 
lin_reg.fit(X=housing_prepared, y=housing_labels)    
# let's try the full pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))   # 이것이 학습한 모델이 예측한 값 
print("Labels:", list(some_labels))      # 이것이 실제 값 (정답)
some_data_prepared
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(y_true=housing_labels, y_pred=housing_predictions)    # MSE
lin_rmse = np.sqrt(lin_mse)    # RMSE 
lin_rmse
from sklearn.model_selection import cross_val_score   # K-Fold Cross Validation 

 
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
