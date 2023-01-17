# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Typical imports we might use

import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
my_data = pd.read_csv('/kaggle/input/fis-pt012120-mod2-project-warmup/train.csv')
my_data.head()
my_data.info()
#TODO



X = my_data.iloc[:,:-1]

y = my_data.iloc[:,-1]
from sklearn.model_selection import train_test_split



#TODO

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
X_train.head()
X_train.head()
def get_needed_features(my_dataframe):

    '''

    Only keep the features that are needed (get rid of id)

    Return dataframe without id column

    '''

    return my_dataframe.iloc[:,1:]
X_train_features = get_needed_features(X_train)

X_train_features.head()
X_train_features.shape
X_train_features.head()
X_train_features.columns
def get_numerical_features(my_dataframe):

    '''

    Return dataframe with only numerical features.

    '''

    columns_to_keep = ['host_id', 'latitude', 'longitude',

       'minimum_nights', 'number_of_reviews', 

       'reviews_per_month', 'calculated_host_listings_count',

       'availability_365'

    ]

    return my_dataframe[columns_to_keep]
X_train_numerics = get_numerical_features(X_train_features)

X_train_numerics.head()
def fill_null_values(my_dataframe, train_dataframe):

    '''

    '''

    # Find the medians for each column

    values_to_fills = {

        col: train_dataframe[col].median()

        for col in train_dataframe.columns

    }

    

    # Fill with the medians

    return my_dataframe.fillna(values_to_fills)
X_train_numerics_filled = fill_null_values(X_train_numerics, X_train_numerics)

X_train_numerics_filled.head()
X_train_numerics_filled.info()
# Note that labels was fixed to be `y_train` instead of `y`



features = X_train_numerics_filled 

labels = y_train
from sklearn.linear_model import LinearRegression



#TODO

# Note you can add more parameters to sklearn's LinearRegression function

my_model = LinearRegression()

my_model.fit(features, labels)
display(my_model.coef_)

display(my_model.intercept_)
my_model.coef_.reshape(-1,1)
pd.DataFrame(data=my_model.coef_.reshape(1,-1), columns= features.columns)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(

            my_model, 

            features,

            labels,

            cv=10,

            scoring="neg_mean_squared_error"

)



rmse_scores = np.sqrt(-scores)
display(rmse_scores)

display(rmse_scores.mean())

display(rmse_scores.std())
y.std()
# Note `get_numerical_features` was corrected from the lecture

test_set = pd.read_csv('/kaggle/input/fis-pt012120-mod2-project-warmup/test.csv')

X_test_final = fill_null_values(

    get_numerical_features(

        get_needed_features(test_set)

    ),

    X_train_numerics

)
X_test_final.head()
X_test_final.head()
y_hat = my_model.predict(X_test_final)
ids = test_set.id.values.reshape(-1,1)

prices = y_hat.reshape(-1,1)

ids
data = np.concatenate((ids,prices),axis=1)

df_final = pd.DataFrame(data=data, columns=['id','price'])

df_final = df_final.astype({'id': 'int32'})

df_final.head()
df_final.info()
# File to be submitted

df_final.to_csv('submission.csv', index=False)

# np.savetxt('example.csv',y_hat)
import pickle



pickle.dump(my_model, open('my_model_simple_sol.pkl','wb'))
# Load the model from earlier

model_loaded = pickle.load(open('my_model_pickle.pkl','rb'))
model_loaded
import joblib



joblib.dump(my_model, "my_model.pkl")
# Load the model from earlier

my_model_loaded = joblib.load("my_model.pkl")