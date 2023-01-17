import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



## for feature slection



from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel



# to visualise al the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)
dataset=pd.read_csv('../input/final-house-price-dataset/X_train.csv')
dataset.head()
## Capture the dependent feature

y_train=dataset[['SalePrice']]
## drop dependent feature from dataset

X_train=dataset.drop(['SalePrice'],axis=1)
### Apply Feature Selection

# Firstly,specify the Lasso Regression model & select a suitable alpha (equivalent of penalty).

# remember that The bigger the alpha the less features that will be selected.



# Then will make use of the selectFromModel object from sklearn, 

# which will select the features which coefficients are non-zero



feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))

feature_sel_model.fit(X_train, y_train)

feature_sel_model.get_support()
# let's print the number of total and selected features



selected_feat = X_train.columns[(feature_sel_model.get_support())]

print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(

    np.sum(feature_sel_model.estimator_.coef_ == 0)))
selected_feat
X_train=X_train[selected_feat]
X_train.head()