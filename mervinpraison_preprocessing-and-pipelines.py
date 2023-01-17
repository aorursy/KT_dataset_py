import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# First, look at everything.

from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))
df = pd.read_csv('../input/automobile/auto.csv')

df.head()
df.boxplot(column='mpg', by='origin', figsize=(10,10), fontsize=10);
df.info()
# Read 'gapminder.csv' into a DataFrame: df

df = pd.read_csv('../input/gapminder/gapminder.csv')



# Create a boxplot of life expectancy per region

df.boxplot('life', 'Region', rot=60, figsize=(5,5));
df.head()
# Create dummy variables: df_region

df_region = pd.get_dummies(df)



# Print the columns of df_region

print(df_region.columns)



# Create dummy variables with drop_first=True: df_region

df_region2 = pd.get_dummies(df, drop_first=True)



# Print the new columns of df_region

print(df_region2.columns)
df_region2.shape
y = df_region2.life.values

X = df_region2.drop('life', axis=1).values
# Import necessary modules

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score



# Instantiate a ridge regressor: ridge

ridge = Ridge(alpha=.5, normalize=True)



# Perform 5-fold cross-validation: ridge_cv

ridge_cv = cross_val_score(ridge, X, y, cv=5)



# Print the cross-validated scores

print(ridge_cv)

# Read the CSV file into a DataFrame: df

df = pd.read_csv('../input/house-votes-non-index/house-votes-non-index.csv')

df.head()
# Convert '?' to NaN

df[df == '?'] = np.nan



# Print the number of NaNs

print(df.isnull().sum())



# Print shape of original DataFrame

print("Shape of Original DataFrame: {}".format(df.shape))



# Drop missing values and print shape of new DataFrame

df = df.dropna(axis=0)



# Print shape of new DataFrame

print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))
df.shape
# Import the Imputer module

from sklearn.preprocessing import Imputer

#from sklearn.impute import SimpleImputer as Imputer

from sklearn.svm import SVC



# Setup the Imputation transformer: imp

##################

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

#imp = Imputer(missing_values='NaN', strategy='most_frequent')



# Instantiate the SVC classifier: clf

clf = SVC()



# Setup the pipeline with the required steps: steps

steps = [('imputation', imp),

        ('SVM', clf)]
y = df.party



X = df.drop('party', axis=1)
X.shape
df.info()
# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC



# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=.3, random_state=42)



# Fit the pipeline to the train set

pipeline.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = pipeline.predict(X_test)



# Compute metrics

print(classification_report(y_test, y_pred))
w = pd.read_csv('../input/white-wine/white-wine.csv')

w.head()
X = w.drop('quality', axis=1).values
X.shape
# Import scale

from sklearn.preprocessing import scale



# Scale the features: X_scaled

X_scaled = scale(X)



# Print the mean and standard deviation of the unscaled features

print("Mean of Unscaled Features: {}".format(np.mean(X))) 

print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))



# Print the mean and standard deviation of the scaled features

print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 

print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))
y = w.quality.apply(lambda x: True if x < 6 else False) # or without .values
y.shape
y[:20]
# Import the necessary modules

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier



# Setup the pipeline steps: steps

steps = [('scaler', StandardScaler()),

        ('knn', KNeighborsClassifier())]

        

# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=42)



# Fit the pipeline to the training set: knn_scaled

knn_scaled = pipeline.fit(X_train, y_train)



# Instantiate and fit a k-NN classifier to the unscaled data

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)



# Compute and print metrics

print('Accuracy with Scaling: {}'.format(pipeline.score(X_test, y_test)))

print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

from sklearn.model_selection import GridSearchCV



# Setup the pipeline

steps = [('scaler', StandardScaler()),

         ('SVM', SVC())]



pipeline = Pipeline(steps)



# Specify the hyperparameter space

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=21)



# Instantiate the GridSearchCV object: cv

cv = GridSearchCV(pipeline, parameters, cv=3)



# Fit to the training set

cv.fit(X_train, y_train)



# Predict the labels of the test set: y_pred

y_pred = cv.predict(X_test)



# Compute and print metrics

print("Accuracy: {}".format(cv.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

print("Tuned Model Parameters: {}".format(cv.best_params_))
from sklearn.metrics import recall_score



recall_score(y_test, y_pred)
y_pred[:10]
y_test[:10].values
df = pd.read_csv('../input/gapminder/gapminder.csv')



# Create arrays for features and target variable

y = df.life

X = df.drop(['life', 'Region'], axis=1)
df.head()
X.shape
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



# Setup the pipeline steps: steps

steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),

         ('scaler', StandardScaler()),

         ('elasticnet', ElasticNet(max_iter=10000))]

#steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),

#         ('scaler', StandardScaler()),

#         ('elasticnet', ElasticNet())]



# Create the pipeline: pipeline 

pipeline = Pipeline(steps)



# Specify the hyperparameter space

parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)



# Create the GridSearchCV object: gm_cv

gm_cv = GridSearchCV(pipeline, parameters, cv=3)



# Fit to the training set

gm_cv.fit(X_train,y_train)



# Compute and print the metrics

r2 = gm_cv.score(X_test, y_test)

print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))

print("Tuned ElasticNet R squared: {}".format(r2))
