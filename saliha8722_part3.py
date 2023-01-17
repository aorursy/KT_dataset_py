# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(os.path.join(dirname, filename))

data.head(5)
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100kpop', 'country-year', 'HDI_for_year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']

data.columns.values
del data['country-year']

#del data['HDI_for_year'], silmek yerine NaN değerleri ortalama değeri ile dolduracağız
data['gdp_for_year_dollars'] = data['gdp_for_year_dollars'].str.replace(',','').astype(int)
#½20 test için ayırdık

from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 1)

for train_index, test_index in split.split(data, data['generation']):

    train = data.loc[train_index]

    test = data.loc[test_index]
data = train.copy()
data_labels = train['suicides/100kpop']
#categorical sütunları numerical olarak çeviriyoruz:

from sklearn.preprocessing import LabelEncoder



category_features = data[[column for column in data.columns if data[column].dtype == 'object']]

le = LabelEncoder()



data_category = category_features.apply(lambda col: le.fit_transform(col))

data_category.head(10)
#overfitting önlemek için ayrı ayrı sütunlar üreterek binary şeklinde getiriyoruz;

data_category_dummies = pd.get_dummies(data, columns=category_features.columns, drop_first=True )

data_category_dummies

#117 sütun elde ettik
#imputer kütüphanesi tek seferde tüm sütunlardaki missing değeri doldurmamıza yardımcı olur.

#tek NaN değerleri sütun HDI_for_year olmasına rağmen yine de bu yöntemi uygulayalım



#from sklearn.preprocessing import Imputer ---> versiondan dolayı hata aldım bu yüzden simpleimputer kullandım.

from sklearn.impute import SimpleImputer 

simple_imputer = SimpleImputer(strategy='median')



numerical_features = data[data.columns[data.dtypes != 'object']]



#imputer fonksiyonunu numerical sütunlarda uygulayacağız bu yüzden categorical olanları drop edeilm

data_numerical = simple_imputer.fit_transform(numerical_features)

data_numerical = pd.DataFrame(data_numerical,columns=data.columns[data.dtypes != 'object'])

data_numerical['HDI_for_year'].describe()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_data_numerical = scaler.fit_transform(data_numerical)



scaled_data_numerical = pd.DataFrame(scaled_data_numerical,columns=data.columns[data.dtypes != 'object'])



scaled_data_numerical
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



numerical_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler(with_mean=False))

])



categorical_pipeline = Pipeline([

    ('encoder', OneHotEncoder(handle_unknown='ignore')),

    ('scaler', StandardScaler(with_mean=False))

])



full_pipeline = ColumnTransformer([

    ('numerical_pipeline', numerical_pipeline, list(numerical_features.columns)),

    ('categorical_pipeline', categorical_pipeline, list(category_features.columns)),

])
data_prepared = full_pipeline.fit_transform(data)

type(data_prepared)
categorical_pipeline
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

dr = DecisionTreeRegressor(random_state=0)

dr.fit(data_prepared,data_labels)

dr_predictions = dr.predict(data_prepared)



drmse = np.sqrt(mean_squared_error(data_labels,dr_predictions))

drmse
from sklearn.model_selection import cross_val_score



scores_1 = cross_val_score(dr, data_prepared, data_labels, scoring = "neg_mean_squared_error", cv = 10)

tree_scores = np.sqrt(-scores_1)

tree_scores.mean()