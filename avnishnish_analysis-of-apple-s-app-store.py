import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def visualizer(x, y, plot_type, title, xlabel, ylabel, rotation=False, rotation_value=60, figsize=(15,8)):
    plt.figure(figsize=figsize)
    
    if plot_type == "bar":  
        sns.barplot(x=x, y=y)
    elif plot_type == "count":  
        sns.countplot(x)
    elif plot_type == "reg":  
        sns.regplot(x=x,y=y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.yticks(fontsize=13)
    if rotation == True:
        plt.xticks(fontsize=13,rotation=rotation_value)
    plt.show()
store_data = pd.read_csv('../input/AppleStore.csv')
app_desc = pd.read_csv('../input/appleStore_description.csv')
store_data.head()
store_data.info()
app_desc.head()
store_data_sorted = store_data.sort_values('rating_count_tot', ascending=False)
subset_store_data_sorted = store_data_sorted[:10]

visualizer(subset_store_data_sorted.track_name, subset_store_data_sorted.rating_count_tot, "bar", "TOP 10 APPS ON THE BASIS OF TOTAL RATINGS",
          "APP NAME", "RATING COUNT (TOTAL)", True, -60)
store_data_download = store_data.sort_values('size_bytes', ascending=False)
store_data_download.size_bytes /= 1024*1024 #Conversion from Bytes to MegaBytes
subset_store_data_download = store_data_download[:10]

visualizer(subset_store_data_download.track_name, subset_store_data_download.size_bytes, "bar", "TOP 10 APPS ON THE BASIS OF DOWNLOAD SIZE",
          "APP NAME", "DOWNLOAD SIZE (in MB)", True, -60)
store_data.currency.unique()
store_data_price = store_data.sort_values('price', ascending=False)
subset_store_data_price = store_data_price[:10]

visualizer(subset_store_data_price.price, subset_store_data_price.track_name, "bar", "TOP 10 APPS ON THE BASIS OF PRICE",
          "Price (in USD)", "APP NAME")
corr_store_data = store_data.corr()
corr_store_data["rating_count_tot"].sort_values(ascending=False)
plt.figure(figsize=(15,15))
plt.title("CORRELATION OF FEATURES", fontsize=20)
sns.heatmap(corr_store_data)
plt.xticks(rotation=(-60), fontsize=15)
plt.yticks(fontsize=15)
plt.show()
visualizer(store_data["lang.num"], store_data.rating_count_tot, "reg", 
          "CORRELATION OF NUMBER OF LANGUAGES AND RATING COUNT", "NUMBER OF LANGAUGES",
          "RATING COUNT (TOTAL)", False)
store_data['revenue'] = store_data.rating_count_tot * store_data.price
store_data_business = store_data.sort_values("revenue", ascending=False)
subset_store_data_business = store_data_business[:10]

visualizer(subset_store_data_business.track_name, subset_store_data_business['revenue'], "bar", "BEST IN BUSINESS",
         "APP NAME", "REVENUE", True, -60)
visualizer(store_data.user_rating, None, "count","RATINGS ON APP STORE",
         "RAITNGS", "NUMBER OF APPS RATED")
store_data["favourites_tot"] = store_data["rating_count_tot"] * store_data["user_rating"]
store_data["favourites_ver"] = store_data["rating_count_ver"] * store_data["user_rating_ver"]
favourite_app = store_data.sort_values("favourites_tot", ascending=False)
favourite_app_subset = favourite_app[:10]

visualizer(favourite_app_subset.track_name, favourite_app_subset.rating_count_tot, "bar", "FAVOURITES (ALL TIME)",
         "APP NAME",  "RATING COUNT(TOTAL)", True, -60)
favourite_app_ver = store_data.sort_values("favourites_ver", ascending=False)
favourite_app_ver_subset = favourite_app_ver[:10]

visualizer(favourite_app_ver_subset.rating_count_ver,favourite_app_ver_subset.track_name,
           "bar", "FAVOURITES (CURRENT VERSION)","RATING COUNT(CURRENT VERSION)","APP NAME", False)

visualizer(store_data.cont_rating, None, "count", "COTNENT RAITNG", "NUMBER OF APP RATED",
           "DISTRIBUTION OF APPS ON THE BASIS OF CONTENT RATING", False)
app_desc["desc_len"] = app_desc["app_desc"].apply(lambda x: len(x))
store_data["desc_len"] = app_desc["desc_len"]
store_data.head()
from sklearn.model_selection import train_test_split
store_train, store_test = train_test_split(store_data, test_size=0.2)

store_data = store_train
store_data.info()
from sklearn.base import BaseEstimator, TransformerMixin

#Drops unncessary columns
class dropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        pass
    def fit_transform(self, X, y=None):
        X = pd.DataFrame(X)
        return X.drop(["currency", "rating_count_tot", "rating_count_ver","track_name",
                                       "Unnamed: 0", "vpp_lic", "revenue", 
                                      "favourites_tot", "favourites_ver"], axis=1)

#Trims version number and changes to int   
def ver_cleaner(data):
    try:
        if "V3" in data: #To handle a single exception
                return str(3)
        else:   
             return int(data.split(".")[0])
    except:
        return int(0)

class version_trimmer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        pass
    def fit_transform(self, X, y=None):
        X["ver"] = X["ver"].apply(ver_cleaner)
        return X

#Helps with dataframes (from hands on ML)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

#Dual label encoder
class dual_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self):
        return self
    def transform(self):
        pass
    def fit_transform(self, X, y=None):
        self.encoder_cont = LabelEncoder()
        cont_encoded = self.encoder_cont.fit_transform(X['cont_rating'])
        
        self.encoder_prime_genre = LabelEncoder()
        genre_encoded = self.encoder_prime_genre.fit_transform(X['prime_genre'])
        
        X["cont_encoded"] = cont_encoded
        X["genre_encoded"] = genre_encoded
        
        return X.drop(["cont_rating", "prime_genre"], axis=1)
    
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler

category_attributes = ["cont_rating","prime_genre"]
numerical_attributes = store_data.drop(["cont_rating","prime_genre"], axis=1).columns

numline = Pipeline([("dataframe", DataFrameSelector(numerical_attributes)),
                    ("dropper", dropper()),
                    ("version-trimmer", version_trimmer()),
                   ("scaling",StandardScaler())])

encoder = dual_encoder()

catline = Pipeline([("dataframe", DataFrameSelector(category_attributes)),
                    ("cat-encoder", encoder)])

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", numline),
                                               ("cat_pipeline", catline)])

store_data_prepared = full_pipeline.fit_transform(store_data)

store_data_prepared    
#Encoders
cont_codes = encoder.encoder_cont.classes_
genre_codes = encoder.encoder_prime_genre.classes_
store_data_prepared.shape
y = np.c_[store_data["rating_count_tot"]] #labels
X = store_data_prepared #Attributes
from sklearn.model_selection import cross_val_score

#Scoring ML model(Using Negative root mean squared error) made easy
def model_scoring(model_name, model, X, y):
    
    #Cross Validation
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    
    #Scores
    rmse = np.sqrt(-scores)
    mean = rmse.mean()
    std = rmse.std()
    print(model_name)
    print()
    print("RMSE: {}".format(rmse))
    print("MEAN: {}".format(mean))
    print("STD: {}".format(std))
# Model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg = lin_reg.fit(X, y)

# Scores
model_scoring("Linear Regression", lin_reg, X, y)
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)

x_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()

poly_reg = poly_reg.fit(x_poly, y)

# Scores
model_scoring("Polynomial Regression", poly_reg, x_poly, y)
from sklearn.svm import SVR

svr = SVR(kernel="linear")

y_ravel = y.ravel()

svr = svr.fit(X, y_ravel)

# Scores
model_scoring("Support Vector Regression", svr, X, y_ravel)
# Model
from sklearn.tree import DecisionTreeRegressor

dec_tree = DecisionTreeRegressor()

dec_tree = dec_tree.fit(X, y)

# Scores
model_scoring("Decision Tree Regression", dec_tree, X, y)