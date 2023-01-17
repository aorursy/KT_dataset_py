import numpy as np

import pandas as pd

import os

import time

import warnings

import os

from six.moves import urllib

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')



import statsmodels.api  as sm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.utils import shuffle

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import LabelEncoder



from lightgbm import LGBMRegressor

from sklearn.feature_selection import RFECV

from sklearn.model_selection import KFold 
DataFile = pd.read_csv("../input/cost-prediction-for-logistic-company/train.csv")
DataFile.head(2)
DataFile.info()
DataFile.describe()
date_ac = np.vstack(DataFile.date.astype(str).apply(lambda x:list(map(int,x.split('-')))).values)

DataFile['Year'] = date_ac[:,0]

DataFile['Month']= date_ac[:,1]

DataFile['Day'] = date_ac[:,2]
DataFile.head(4)
obs = DataFile.isnull().sum().sort_values(ascending = False)

percent = round(DataFile.isnull().sum().sort_values(ascending = False)/len(DataFile)*100, 2)

pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])
DataFile = DataFile.drop(['packageType'],axis=1)
DataFile['type'].fillna("NA_type", inplace=True)

DataFile['exWeatherTag'].fillna("NA_Weather", inplace=True)
DataFile.head(5)
weight_Aggregated = pd.DataFrame(DataFile.groupby(["Month","originLocation","destinationLocation","carrier"],sort=True)["weight"].mean()).reset_index()
weight_Aggregated = weight_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

weight_Aggregated = pd.DataFrame(weight_Aggregated.groupby(["carrier","Month"],sort=True)["weight"].mean()).reset_index()



weight_Aggregated.head(3)
Distance_Aggregated = pd.DataFrame(DataFile.groupby(["Month","originLocation","destinationLocation","carrier"],sort=True)["distance"].sum()).reset_index()
Distance_Aggregated = Distance_Aggregated.drop(["originLocation","destinationLocation"],axis=1)



Distance_Aggregated = pd.DataFrame(Distance_Aggregated.groupby(["carrier","Month"],sort=True)["distance"].mean()).reset_index()



Distance_Aggregated.head(4)
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12, 20)



sns.pointplot(x= weight_Aggregated["Month"], y= weight_Aggregated["weight"],hue=weight_Aggregated["carrier"], data=weight_Aggregated, join=True,ax=ax1)

ax1.set(xlabel='Month of the year', ylabel='weight',title="Average weight carried by different carriers across months",label='big')



sns.pointplot(x= Distance_Aggregated["Month"], y= Distance_Aggregated["distance"],hue=Distance_Aggregated["carrier"], data=Distance_Aggregated, join=True,ax=ax2)

ax2.set(xlabel='Month of the year', ylabel='distance',title="Total distance travelled by different carriers across months",label='big')
weight_Aggregated = pd.DataFrame(DataFile.groupby(["Day","originLocation","destinationLocation","carrier"],sort=True)["weight"].mean()).reset_index()
weight_Aggregated = weight_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

weight_Aggregated = pd.DataFrame(weight_Aggregated.groupby(["carrier","Day"],sort=True)["weight"].mean()).reset_index()



weight_Aggregated.head(3)
Distance_Aggregated = pd.DataFrame(DataFile.groupby(["Day","originLocation","destinationLocation","carrier"],sort=True)["distance"].sum()).reset_index()
Distance_Aggregated = Distance_Aggregated.drop(["originLocation","destinationLocation"],axis=1)



Distance_Aggregated = pd.DataFrame(Distance_Aggregated.groupby(["carrier","Day"],sort=True)["distance"].sum()).reset_index()



Distance_Aggregated.head(4)
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12, 20)



sns.pointplot(x= weight_Aggregated["Day"], y= weight_Aggregated["weight"],hue=weight_Aggregated["carrier"], data=weight_Aggregated, join=True,ax=ax1)

ax1.set(xlabel='Day of Month', ylabel='weight',title="Average weight carried by different carriers across days",label='big')



sns.pointplot(x= Distance_Aggregated["Day"], y= Distance_Aggregated["distance"],hue=Distance_Aggregated["carrier"], data=Distance_Aggregated, join=True,ax=ax2)

ax2.set(xlabel='Day of Month', ylabel='distance',title="Total distance travelled by different carriers across days",label='big')
daypart_Weight_Aggregated = pd.DataFrame(DataFile.groupby(["Month","originLocation","destinationLocation","carrier","dayPart"],sort=True)["weight"].mean()).reset_index()
daypart_distance_Aggregated = pd.DataFrame(DataFile.groupby(["Month","originLocation","destinationLocation","carrier","dayPart"],sort=True)["distance"].sum()).reset_index()
daypart_Weight_Aggregated = daypart_Weight_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

daypart_Weight_Aggregated = pd.DataFrame(daypart_Weight_Aggregated.groupby(["dayPart","Month","carrier"],sort=True)["weight"].mean()).reset_index()



daypart_Weight_Aggregated.head(3)
daypart_distance_Aggregated = daypart_distance_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

daypart_distance_Aggregated = pd.DataFrame(daypart_distance_Aggregated.groupby(["dayPart","Month","carrier"],sort=True)["distance"].sum()).reset_index()



daypart_distance_Aggregated.head(3)
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12, 20)



sns.pointplot(x= daypart_Weight_Aggregated["Month"], y= daypart_Weight_Aggregated["weight"],hue=(daypart_Weight_Aggregated["dayPart"]+"-"+daypart_Weight_Aggregated["carrier"]), data=daypart_Weight_Aggregated, join=True,ax=ax1)

ax1.set(xlabel='Month of the year', ylabel='weight',title="Average weight carried by day / night",label='big')



sns.pointplot(x= daypart_distance_Aggregated["Month"], y= daypart_distance_Aggregated["distance"],hue=(daypart_distance_Aggregated["dayPart"]+"-"+daypart_distance_Aggregated["carrier"]), data=daypart_distance_Aggregated, join=True,ax=ax2)

ax2.set(xlabel='Month of the year', ylabel='distance',title="Total distance travelled by different carriers during day / night across months",label='big')
weather_Weight_Aggregated = pd.DataFrame(DataFile.groupby(["Day","originLocation","destinationLocation","carrier","dayPart","exWeatherTag"],sort=True)["weight"].mean()).reset_index()
weather_distance_Aggregated = pd.DataFrame(DataFile.groupby(["Day","originLocation","destinationLocation","carrier","dayPart","exWeatherTag"],sort=True)["distance"].sum()).reset_index()
#Filter out the blank rows

exWeatherTag = ['NA_Weather']

weather_Weight_Aggregated = weather_Weight_Aggregated[~weather_Weight_Aggregated.exWeatherTag.isin(exWeatherTag)].reset_index()
weather_Weight_Aggregated = weather_Weight_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

weather_Weight_Aggregated = pd.DataFrame(weather_Weight_Aggregated.groupby(["carrier","Day","exWeatherTag"],sort=True)["weight"].mean()).reset_index()



weather_Weight_Aggregated.head(3)
#Filter out the blank rows

exWeatherTag = ['NA_Weather']

weather_distance_Aggregated = weather_distance_Aggregated[~weather_distance_Aggregated.exWeatherTag.isin(exWeatherTag)].reset_index()
weather_distance_Aggregated = weather_distance_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

weather_distance_Aggregated = pd.DataFrame(weather_distance_Aggregated.groupby(["carrier","Day","exWeatherTag"],sort=True)["distance"].sum()).reset_index()



weather_distance_Aggregated.head(3)
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12, 20)



sns.pointplot(x= weather_Weight_Aggregated["Day"], y= weather_Weight_Aggregated["weight"],hue=(weather_Weight_Aggregated["carrier"]+"-"+weather_Weight_Aggregated["exWeatherTag"]), data=weather_Weight_Aggregated,join=False,ax=ax1)

ax1.set(xlabel='Day of the month', ylabel='weight',title="Average weight carried by day / night (with heat / snow)",label='big')



sns.pointplot(x= weather_distance_Aggregated["Day"], y= weather_distance_Aggregated["distance"],hue=(weather_Weight_Aggregated["carrier"]+"-"+weather_Weight_Aggregated["exWeatherTag"]), data=weather_distance_Aggregated,join=False,ax=ax2)

ax2.set(xlabel='Day of the month', ylabel='distance',title="Total distance travelled by different carriers during day / night (with heat/snow) across days",label='big')
weather_Weight_Aggregated = pd.DataFrame(DataFile.groupby(["Month","originLocation","destinationLocation","carrier","dayPart","exWeatherTag"],sort=True)["weight"].mean()).reset_index()
weather_distance_Aggregated = pd.DataFrame(DataFile.groupby(["Month","originLocation","destinationLocation","carrier","dayPart","exWeatherTag"],sort=True)["distance"].sum()).reset_index()
#Filter out the blank rows

exWeatherTag = ['NA_Weather']

weather_Weight_Aggregated = weather_Weight_Aggregated[~weather_Weight_Aggregated.exWeatherTag.isin(exWeatherTag)].reset_index()
weather_Weight_Aggregated = weather_Weight_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

weather_Weight_Aggregated = pd.DataFrame(weather_Weight_Aggregated.groupby(["carrier","Month","exWeatherTag"],sort=True)["weight"].mean()).reset_index()



weather_Weight_Aggregated.head(3)
#Filter out the blank rows

exWeatherTag = ['NA_Weather']

weather_distance_Aggregated = weather_distance_Aggregated[~weather_distance_Aggregated.exWeatherTag.isin(exWeatherTag)].reset_index()
weather_distance_Aggregated = weather_distance_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

weather_distance_Aggregated = pd.DataFrame(weather_distance_Aggregated.groupby(["carrier","Month","exWeatherTag"],sort=True)["distance"].sum()).reset_index()



weather_distance_Aggregated.head(3)
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12, 20)



sns.pointplot(x= weather_Weight_Aggregated["Month"], y= weather_Weight_Aggregated["weight"],hue=(weather_Weight_Aggregated["carrier"]+"-"+weather_Weight_Aggregated["exWeatherTag"]), data=weather_Weight_Aggregated,join=False,ax=ax1)

ax1.set(xlabel='Month of the year', ylabel='weight',title="Average weight carried by day / night (with heat / snow)",label='big')



sns.pointplot(x= weather_distance_Aggregated["Month"], y= weather_distance_Aggregated["distance"],hue=(weather_Weight_Aggregated["carrier"]+"-"+weather_Weight_Aggregated["exWeatherTag"]), data=weather_distance_Aggregated,join=False,ax=ax2)

ax2.set(xlabel='Month of the year', ylabel='distance',title="Total distance travelled by different carriers during day / night (with heat/snow) across months",label='big')
Type_Weight_Aggregated = pd.DataFrame(DataFile.groupby(["Day","originLocation","destinationLocation","carrier","type"],sort=True)["weight"].mean()).reset_index()
Type_distance_Aggregated = pd.DataFrame(DataFile.groupby(["Day","originLocation","destinationLocation","carrier","type"],sort=True)["distance"].sum()).reset_index()
#Filter out the blank rows

Type_Tag = ['NA_type']

Type_Weight_Aggregated = Type_Weight_Aggregated[~Type_Weight_Aggregated.type.isin(Type_Tag)].reset_index()
#Filter out the blank rows

Type_Tag = ['NA_type']

Type_distance_Aggregated = Type_distance_Aggregated[~Type_distance_Aggregated.type.isin(Type_Tag)].reset_index()
Type_Weight_Aggregated = Type_Weight_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

Type_Weight_Aggregated = pd.DataFrame(Type_Weight_Aggregated.groupby(["carrier","Day","type"],sort=True)["weight"].mean()).reset_index()



Type_Weight_Aggregated.head(3)
Type_distance_Aggregated = Type_distance_Aggregated.drop(["originLocation","destinationLocation"],axis=1)

Type_distance_Aggregated = pd.DataFrame(Type_distance_Aggregated.groupby(["carrier","Day","type"],sort=True)["distance"].sum()).reset_index()



Type_distance_Aggregated.head(3)
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12, 20)



sns.pointplot(x= Type_Weight_Aggregated["Day"], y= Type_Weight_Aggregated["weight"],hue=(Type_Weight_Aggregated["carrier"]+"-"+Type_Weight_Aggregated["type"]), data=weather_Weight_Aggregated,join=False,ax=ax1)

ax1.set(xlabel='Day of the month', ylabel='weight',title="Average weight carried with Expedite option",label='big')



sns.pointplot(x= Type_distance_Aggregated["Day"], y= Type_distance_Aggregated["distance"],hue=(Type_distance_Aggregated["carrier"]+"-"+Type_distance_Aggregated["type"]), data=weather_distance_Aggregated,join=False,ax=ax2)

ax2.set(xlabel='Day of the month', ylabel='distance',title="Total distance travelled by carrier D (with Expedite) across days",label='big')
f, ax = plt.subplots(ncols=2, figsize=(10,5))

sns.kdeplot(DataFile.weight, color='b', shade=True, ax=ax[0])

sns.kdeplot(DataFile.weight, color='r', shade=True, bw=100, ax=ax[1])



ax[0].set_title('KDE')

ax[1].set_title('KDE, bandwidth = 100')



plt.show()
DataFile.info()
#Create Dummy variable for daypart and type

DataFile['dayPart'].replace(['day','night'],[1,0],inplace=True)
#Making weight Bins - they were not used as they are not significant

DataFile['weight_bin'] = pd.qcut(DataFile['weight'], 5)

label = LabelEncoder()

DataFile['weight_bin'] = label.fit_transform(DataFile['weight_bin'])
#Making weight Bins - not used as they were not significant

DataFile['distance_bin'] = pd.qcut(DataFile['distance'], 5)

label = LabelEncoder()

DataFile['distance_bin'] = label.fit_transform(DataFile['distance_bin'])
DataFile = DataFile.drop(['trip','date',"weight_bin","distance_bin",'originLocation','destinationLocation','exWeatherTag','type','Year'],axis=1)
DataFile.head(5)
DataFile.shape
train_set, val_set = train_test_split(DataFile, test_size=0.10,shuffle=False)
# Now define x and y.



#the Y Variable

train_set_y = train_set["cost"].copy()

val_set_y = val_set["cost"].copy()



#the X variables

train_set_X = train_set.drop("cost", axis=1)

val_set_X = val_set.drop("cost", axis=1)
# Reference from Hands on Machine learning - one of my text books!

# The CategoricalEncoder class will allow us to convert categorical attributes to one-hot vectors.



class CategoricalEncoder(BaseEstimator, TransformerMixin):

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
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
cat_pipeline = Pipeline([

        ("selector", DataFrameSelector(['carrier'])),

        ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),

    ])



num_pipeline = Pipeline([

        ("selector", DataFrameSelector(['distance','weight'])),

        ('std_scaler', MinMaxScaler()),

      ])



no_pipeline = Pipeline([

        ("selector", DataFrameSelector(["dayPart","Month","Day"]))

    ])



y_pipeline = Pipeline([

        ("selector", DataFrameSelector(['cost'])),

        ('std_scaler', StandardScaler()),

      ])
full_pipeline = FeatureUnion(transformer_list=[

    ("no_pipeline", no_pipeline),

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

    ])



final_train_X = full_pipeline.fit_transform(train_set_X)

final_val_X = full_pipeline.transform(val_set_X)



final_train_y = y_pipeline.fit_transform(pd.DataFrame(train_set_y))

final_val_y = y_pipeline.transform(pd.DataFrame(val_set_y))