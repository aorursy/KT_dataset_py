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
#Getting data from data set

import pandas as pd

data = pd.read_csv("../input/titanic/train.csv")
#Checking data info

data.info()
data['Sex'].value_counts()
data['Cabin'].value_counts()
data['Embarked'].value_counts()
data.describe() 
import matplotlib.pyplot as plt



data.hist(bins=50, figsize=(20, 15))

plt.show()
corr_matrix = data.corr()

corr_matrix["Survived"].sort_values(ascending=False)
gen_cat = data[["Sex"]]

gen_cat.head(10)
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()



gen_cat_encoded = ordinal_encoder.fit_transform(gen_cat)

gen_cat_encoded
cabin_cat = data[['Cabin']].fillna('U')

cabin_cat.head(10)
cabin_cat_encoded = ordinal_encoder.fit_transform(cabin_cat)

cabin_cat_encoded[:10]
emb_cat = data[["Embarked"]].fillna('U')

emb_cat.head(10)
emb_cat_encoded = ordinal_encoder.fit_transform(emb_cat)

emb_cat_encoded[:10]
data["Sex_cat"] = gen_cat_encoded

data["Cabin_cat"] = cabin_cat_encoded

data["emb_cat"] = emb_cat_encoded
corr_matrix = data.corr()

corr_matrix["Survived"].sort_values(ascending=False)
features = ["Fare", "Cabin_cat", "Pclass", "Sex_cat", "emb_cat"]
X = data[features]

y = data.Survived
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=1)

rfc_model.fit(train_X, train_y)



from sklearn.metrics import mean_absolute_error

val_predictions = rfc_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
test_data_path = "../input/titanic/test.csv"



# Read test data file using pandas

test_data = pd.read_csv(test_data_path)

tmp_test_data = test_data

test_data.info()
#Making catagorysis in real number



test_gen_cat = test_data[["Sex"]]



test_gen_cat_encoded = ordinal_encoder.fit_transform(test_gen_cat)



test_cabin_cat = test_data[["Cabin"]].fillna("U")



test_cabin_cat_encoded = ordinal_encoder.fit_transform(test_cabin_cat)



test_emb_cat = test_data[["Embarked"]].fillna("U")



test_emb_cat_encoded = ordinal_encoder.fit_transform(test_emb_cat)



test_data["Sex_cat"] = test_gen_cat_encoded

test_data["Cabin_cat"] = test_cabin_cat_encoded

test_data["emb_cat"] = test_emb_cat_encoded
test_data = test_data[features]



cols_with_missing = [col for col in test_data.columns

                    if test_data[col].isnull().any()]
#Filling null data

#Make copy to avoid changing original data (when imputing)



from sklearn.impute import SimpleImputer



#Make new columns indicating what will be imputed

for col in cols_with_missing:

    test_data[col + '_was_missing'] = test_data[col].isnull()

    

#Imputation

my_imputer = SimpleImputer()

imputed_test_data = pd.DataFrame(my_imputer.fit_transform(test_data))



#Imputation removed column name; put them back

imputed_test_data.columns = test_data.columns



imputed_test_data.info()
# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

#test_X = imputed_test_data.drop('Fare_was_missing', axis = 1)



test_X = imputed_test_data.drop('Fare_was_missing', axis = 1)



# make predictions which we will submit. 

test_preds = rfc_model.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'PassengerId': tmp_test_data.PassengerId,

                       'Survived': test_preds})





output.to_csv('submission.csv', index=False)