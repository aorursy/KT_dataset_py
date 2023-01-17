# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
# just take continuous variables for now

cols=['bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors','sqft_above', 'sqft_basement','sqft_lot15']
X = df[cols]

y = df['price']
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import Normalizer
#Normalize

scalar = Normalizer()

X_scaled = scalar.fit_transform(X)
## Function to calculate permutation weight

from math import factorial

def shapWt(num_total_features,num_current_features):

    return ( ( factorial(num_current_features) ) * ( factorial( num_total_features - num_current_features -1 ) ) )/( factorial(num_total_features) )

    
# function to get powerset



from itertools import chain, combinations



def powerset(iterable):

    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"

    s = list(iterable)

    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
#Computes shapely regression value for a feature value



def shapleyRegression(X,y,feature_index_find,total_features,instance):

    

    fts = [i for i in range(total_features)]

    fts.remove(feature_index_find)

    

    all_subsets = list(powerset(fts))

    shap = 0.0

    total_wt = shapWt(total_features,0)

    shap = shap + shapWt(total_features,0)*LinearRegression().fit(X[:,[feature_index_find]],y).predict(instance[feature_index_find].reshape(-1,1))

    for each_set in all_subsets[1:]:

        wt = shapWt(total_features,len(each_set))

        total_wt += wt

        cols = list(each_set)

        model1 = LinearRegression().fit(X[:,cols],y)

        cols_with_ft = cols.copy()

        cols_with_ft.append(feature_index_find)

        #print(cols)

        model2 = LinearRegression().fit(X[:,cols_with_ft],y)

        #print(instance.shape)

        #print(instance.reshape(1,-1).shape)

        f1 = model1.predict(instance[cols].reshape(1,-1))

        f2 = model2.predict(instance[cols_with_ft].reshape(1,-1))

        

        shap = shap + (wt*(f2-f1))

    

    return shap

        

    
# find shapely value for all features

shap_vals=[]

for i in range(8):

    shap_vals.append(shapleyRegression(X_scaled,y,i,8,X_scaled[0,:]))
# Compute phi(i)*b(i)

weighted_shaps = np.multiply(np.asarray(shap_vals).reshape(-1,1),(X_scaled[0,:].reshape(-1,1)))

#find phi(0) which is prediction - summation(phi(i)*b(i)). This should ideally 

#be a constant for different prediction. But if you run this whole experiment for 

#data point index 1 instead of index 0 and compare phi_0 for both, they are different.



phi_0 = LinearRegression().fit(X_scaled,y).predict(X_scaled[0].reshape(1,-1)) - weighted_shaps.sum()