import numpy as np 

import pandas as pd 



import sklearn.model_selection as model_selection



from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer



#For Missing Value Treatment

from sklearn.impute import SimpleImputer

from sklearn.impute import KNNImputer



#For Binning and creating Dummy Variables

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from sklearn.preprocessing import QuantileTransformer



from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_error
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

sub_test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')
train.head()
y = train['revenue']

X = train.drop(['revenue','Id'], axis = 1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2, random_state = 200)
#Condidering Numerical Features only

numerical_features = [c for c, dtype in zip(X.columns, X.dtypes) if dtype.kind in ['i','f'] ]



print('Numerical : ' + str(numerical_features))
#Data Processing Steps

preprocessor = make_column_transformer(

    

    (make_pipeline(

    KNNImputer(n_neighbors=10),

    KBinsDiscretizer(n_bins = 6),

    SelectKBest(chi2, k=15),

    ), numerical_features)

    

)
#Model Steps

regModel = make_pipeline(preprocessor, LinearRegression())
regModel.fit(X_train, y_train)
train_score = regModel.score(X_train,y_train)

test_score = regModel.score(X_test,y_test)



print(f'Train Accuracy : {train_score:.3f}')

print(f'Test Accuracy : {test_score:.3f}')
#Check RMSE

y_pred = regModel.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
y_sub_pred = regModel.predict(sub_test.drop(['Id'], axis = 1))
submission_df = pd.DataFrame({'Id' : sub_test['Id'], 'Prediction' : y_sub_pred})
submission_df.to_csv('Reg_Model_Pipeline.csv', index = False)

submission_df.head()