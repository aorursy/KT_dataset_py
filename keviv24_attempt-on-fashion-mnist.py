%reset -f

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

#%matplotlib qt5

# 1.0.1 For measuring time elapsed

from time import time

from imblearn.over_sampling import SMOTE, ADASYN



# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct





# 1.3 Data imputation

from sklearn.impute import SimpleImputer



# 1.4 Model building

#     Install h2o as: conda install -c h2oai h2o=3.22.1.2

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics





# 1.6 Change ipython options to display all data columns

pd.options.display.max_columns = 300
trainf = pd.read_csv("../input/fashion-mnist_train.csv")

testf = pd.read_csv("../input/fashion-mnist_test.csv")
# 2.1 Explore data

trainf.head(3)
# 2 Examine distribution of continuous variables

trainf.describe()   
trainf.shape      

trainf['label'].value_counts() # balanced classes 



trainf.columns.values
trainf.dtypes.value_counts() 

## . int64    785

## . dtype: int64



trainf.shape #(60000, 785)

# 3 Start h2o

h2o.init()
# 13.2 Transform data to h2o dataframe

df = h2o.H2OFrame(trainf)



len(df.columns)    # 785

df.shape           # (60000, 785)

df.columns
# 4. Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to

#     H2o dataframe

X_columns = df.columns[1:]        # Only column names. No data

X_columns       # 1 to 785



#preditor column name

y_columns = df.columns[0]

y_columns



df['label'].head()      # Just to be sure, Does not show anything in spyder. BUG

# 14.1 For classification, target column must be factor----convert to a factor

#      Required by h2o

df['label'] = df['label'].asfactor()



# 5. Build a deeplearning model on balanced data

dl_model = H2ODeepLearningEstimator(

                             distribution="multinomial",

                             activation = "RectifierWithDropout",

                             hidden = [32,32,32],

                             input_dropout_ratio=0.2,  # It is just as in Random Forest

                          #   and in this case for each record 

                             epochs = 100

                             )
# 6 . Train our model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60
# 7  covert to H2O frame before running it for prediction



testx=h2o.H2OFrame(testf)



# Now predict

result = dl_model.predict(testx[: , 1:])
# 8 Ground truth

#      Convert H2O frame back to pandas dataframe

xe = testx['label'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.head()

xe.columns
xe.head()



# 19. So compare ground truth with predicted



out = (xe['result'] == xe['label'])

np.sum(out)/out.size

# 81.81 accracy

# 9 .  Create confusion matrix using pandas dataframe

f  = confusion_matrix( xe['label'], xe['result'] )

f
# 10.... How were the classes distributed

#xe=xe.values

xe['label'].value_counts()
# 11 . 

xe['result'].value_counts() # skewed towards classes 5,6,3,4
