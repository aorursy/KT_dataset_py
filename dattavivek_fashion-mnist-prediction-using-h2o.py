

#Summary about the Fashion MNIST Dataset:

    

#Each row is a separate image with 785 columns.

#Column 1 is the class label.

#Remaining 784 columns are pixel numbers (As each image is of 28*28 in pixel size).

#Each value is the darkness of the pixel raning from (1 to 255)



#Each training and test example is assigned to one of the following labels:

#T-shirt/top as 0,Trouser as 1,Pullover as 2,Dress as 3,Coat as 4,Sandal as 5,Shirt as 6,Sneaker as 7,Bag as 8,Ankle boot as 9"""



%reset -f

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

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



# 1.6 Change ipython options to display all data columns

pd.options.display.max_columns = 300
# 2.0 Read data

# os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\education_analytics\\data")

os.chdir("../input")

fashion_trn = pd.read_csv("fashion-mnist_train.csv")

fashion_trn.shape

fashion_trn.dtypes

#To see the sample image we consider the first row.

abc = fashion_trn.values[25, 1:]

abc.shape    # (784,)

image = abc.reshape(28,28)   # Reshape to 28 X 28



# And plot it

plt.imshow(image)
#Start h2o

h2o.init()

#Read the train and test data as h2o dataframes.

train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
#Seperate the predictors and target columns as y_columns and X_columns respectively.

X_columns = train.columns[1:785] 

y_columns = train.columns[0]
#For classification, target column must be factor So, convert target column "label" into factor 

train["label"]=train["label"].asfactor()

train['label'].levels()
#Build basic deeplearning model on data

dl_model = H2ODeepLearningEstimator(epochs=300, #how many times it reads the data to understand the pattern(like the backward travel in neural network to adjust weights)

                                    distribution = 'bernoulli',                 # Response has two levels. if not bernoullis it takes multinomial approach

                                    missing_values_handling = "MeanImputation", 

                                    variable_importances=True,

									

									#below three are for cross validation strategy

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully#all folds have the same ratio of majority and minority ratio

                                    keep_cross_validation_predictions = True,  # For analysis

									

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [32,32,32],                  # ## more hidden layers -> more complex interactions#100,100 means 2 hidden layer of 100 neurons each

									)
#Train our model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = train)

end = time()

(end - start)/60
#Transform X_test to h2o dataframe



test['label'] = test['label'].asfactor()
# Make prediction on X_test

result = dl_model.predict(test[: , 1:785])

result.shape       # 5730 X 3

result.as_data_frame().head()   # Class-wise predictions
# Convert H2O frame back to pandas dataframe

xe = test['label'].as_data_frame()

xe['result'] = result[0].as_data_frame()
#So compare the actual label values with predicted

out = (xe['result'] == xe['label'])

np.sum(out)/out.size