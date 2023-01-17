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



pokemon_file_path = "/kaggle/input/pokemon/pokemon.csv"

X_full = pd.read_csv(pokemon_file_path)

import seaborn as sns

import matplotlib.pyplot as plt



colors = ["green", "purple"]

g = sns.factorplot(

    x='generation', 

    data=X_full,

    kind='count', 

    hue='is_legendary',

    palette=colors, 

    size=5, 

    aspect=1.5,

    legend=False,

    ).set_axis_labels('Generation', '# of Pokemon')



g.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),  shadow=True, ncol=2, labels=['Non Legendary','Legendary'])

plt.show()
X_capturerate = X_full.copy()

X_capturerate.capture_rate.iloc[773] = 255  

X_capturerate.capture_rate = pd.to_numeric(X_capturerate.capture_rate)



sns.swarmplot(x=X_capturerate['is_legendary'],

              y=X_capturerate['capture_rate'])

#sns.boxplot(x='generation', y='capture_rate', hue='is_legendary', data=X_capturerate)
sns.swarmplot(x=X_full['is_legendary'],

              y=X_full['base_total'])


X_capturerate['capture_rate'].sort_values()

sns.lmplot(x="base_total", y="capture_rate", hue="is_legendary", data=X_capturerate)

sns.swarmplot(x=X_full['is_legendary'],

              y=X_full['experience_growth'])
X_full.info()
pokemon_mfeatures = ['percentage_male','type1', 'type2', 'height_m', 'weight_kg', 

                     'experience_growth', 'hp', 'attack','base_total',

                     'defense','sp_attack', 'sp_defense', 'speed', 'capture_rate']



print(pokemon_mfeatures)





poke_mval = X_full[pokemon_mfeatures]

X = pd.DataFrame(poke_mval)



# setting my target data

y = X_full.is_legendary

# Returning all columns with the amount of null values in each column

X.isnull().sum()
plt.subplots(figsize=(10, 10))

sns.heatmap(

    X[X['type2'] != 'None'].groupby(['type1', 'type2']).size().unstack(),

    linewidths = 1,

    annot = True,

    cmap = "RdYlBu_r" # color

)



plt.show()

X['type2'].fillna('None', inplace=True)

X['percentage_male'].fillna(0, inplace=True)

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size= 0.8, test_size=0.2, random_state=0)



# select categorical columns

categorical_cols = [cname for cname in X_train.columns if

                    X_train[cname].dtype == 'object']



# select numerical columns

numerical_cols = [cname for cname in X_train.columns if 

                 X_train[cname].dtype in ['int64', 'float64']]



print (categorical_cols)

# Impute weight and height values



# preprocessing the numerical data

numerical_transform = SimpleImputer(strategy='mean')



# preprocesing the categorical transform

categorical_transform = Pipeline(steps= [('imputer', SimpleImputer(strategy = 'most_frequent')),

                                        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

                                        ])



# bundling the preprocessing transformers

preprocessor = ColumnTransformer(transformers = [('num', numerical_transform, numerical_cols),

                                              ('cat', categorical_transform, categorical_cols)])

# Define models...

rf_model = RandomForestClassifier (n_estimators=100, random_state=0)



# combine preprocesser with model in a pipeline

p1 = Pipeline(steps=[('preprocessorrf', preprocessor), ('modelrf', rf_model)])



# preprocess of training data, fit the model

p1.fit(X_train, y_train)





# proprocess validation data, obtain predictions

p1pred = p1.predict(X_valid)





preds1 = X.index[p1pred]



pd.crosstab(y_valid, preds1, rownames=['Actual Legendary'], colnames=['Predicted Legendary'])

print("Model Accuracy:", (154/161)*100)
from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer





# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in categorical_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])



# Imputation

#my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(numerical_transform.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(numerical_transform.transform(label_X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = label_X_train.columns

imputed_X_valid.columns = label_X_valid.columns



# Using RandomForest Model

rf_model.fit(imputed_X_train, y_train)



rf_preds = rf_model.predict(imputed_X_valid)

visualizerf = X.index[rf_preds]



#p2pred = XGB_model.predict(label_X_valid)

#preds2 = X.index[p2pred]



pd.crosstab(y_valid, visualizerf, rownames=['Actual Legendary'], colnames=['Predicted Legendary'])

from sklearn.naive_bayes import GaussianNB



# defining the Bayes model 

NB_model = GaussianNB()

NB_model.fit(imputed_X_train, y_train)

p3pred = NB_model.predict(imputed_X_valid)



preds3 = X.index[p3pred]



pd.crosstab(y_valid, preds3, rownames=['Actual Legendary'], colnames=['Predicted Legendary'])
from xgboost import XGBClassifier



XGB_model = XGBClassifier (n_estimators = 100, learning_rate=0.1)

XGB_model.fit(imputed_X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(imputed_X_valid, y_valid)], 

             verbose=False)



p2pred = XGB_model.predict(imputed_X_valid)

preds2 = X.index[p2pred]



pd.crosstab(y_valid, preds2, rownames=['Actual Legendary'], colnames=['Predicted Legendary'])

print("Model Accuracy:", (160/161)*100)