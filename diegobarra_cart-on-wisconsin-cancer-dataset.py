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

import seaborn as sns 

%matplotlib inline



# Import data and select "id" column as index



path = '../input/breast-cancer-csv/breastCancer.csv'

df = pd.read_csv(path, index_col = "id")



# Examine the dataframe



df.info()
# Convert the "class" column (target feature for classifier) to categorical data



df["class"] = df["class"].astype("category")



# Examine dataframe again



df.info()
from sklearn.impute import SimpleImputer

import numpy as np



# apply lambda function to change "?" for None values



df["bare_nucleoli"] = df["bare_nucleoli"].apply(lambda x: None if x is "?" else x)



# Convert column to numeric type data



df["bare_nucleoli"] = pd.to_numeric(df.bare_nucleoli)



# Initialize SimpleImputer



imputer = SimpleImputer(missing_values = np.nan, strategy = "median")



# Reshape the imputer input 



imp_input = df.bare_nucleoli.to_numpy().reshape(-1,1)



# Fit and transform imputer



imputer.fit(imp_input)

imp_input_transformed = imputer.transform(imp_input)



# Save the imputer output into the dataframe column and convert to integer datatype



df["bare_nucleoli"] = imp_input_transformed.astype(int)

df.info()
print(df.describe())
import seaborn as sns 

import matplotlib.pyplot as plt



# Define target 



y = df["class"]



# compute corelation matrix



corr = df.corr()



# display heatmap from correlation matrix



sns.heatmap(corr,cmap="Blues",  annot=True)

# Extract ratio between shape uniformity and size uniformity



df["shape_size_uniformity"] = df["shape_uniformity"]/df["size_uniformity"]



# Drop redundant features



df.drop(["shape_uniformity", "size_uniformity"], axis = 1, inplace = True)



# Compute new correlation matrix



corr = df.corr()



# Plot heatmap from the correlation matrix



sns.heatmap(corr, cmap = "Blues", annot = True)



# Check the distribution of the newly created feature in a bee-swarm plot



plt.figure()

sns.swarmplot(x = df["class"], y = df["shape_size_uniformity"], data = df )



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Prepare features and target for the model



y = df["class"].replace({2:0, 4:1})

X = df.drop(["class"], axis = 1)



# Generate train and test data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



# Instantiate Random Forest Classifier



rfc = RandomForestClassifier(n_estimators = 100, max_depth = 4, random_state = 42)



# Fit model into training data



rfc.fit(X_train,y_train)



# Predict test data



y_pred_train = rfc.predict(X_train)

y_pred_test = rfc.predict(X_test)



# Compute scores predicted train and test data



accuracy_train = accuracy_score(y_train, y_pred_train)

accuracy_test = accuracy_score(y_test, y_pred_test)



# Compute confusion matrix



conf_matrix_rf = confusion_matrix(y_test,y_pred_test)



##### Display results



print("The accuracy score of the Random Forest Classifier on train data is {:.2f}".format(accuracy_train))

print("The accuracy score of the Random Forest Classifier on test data is {:.2f}".format(accuracy_test))

print("")

print("Confusion matrix:")

print("")

print(conf_matrix_rf)



# Compute a panda series of feature importances



importances = pd.Series(data=rfc.feature_importances_, index= X_train.columns)



# Sort importances



importances_sorted = importances.sort_values()



# Plot a horizontal bar plot of the feature importances



importances_sorted.plot(kind = "barh")

from sklearn.model_selection import GridSearchCV



# Establish parameters for GridSearchCV, as a dictionary



rfc_params = {"n_estimators":[50, 70, 90, 100, 110, 130, 140], 

              "max_features":["log2", "sqrt", "auto"],

              "min_samples_leaf":[2, 4, 8, 10]}



# Obtain the model with the optimal hyperparameters found in GridSearchCV



rfc_gscv = GridSearchCV(estimator = rfc,

                       param_grid = rfc_params,

                       cv = 5,

                       scoring = "accuracy",

                       verbose = 2,

                       n_jobs = -1)



# Fit model found with GridSearchCV on train data 



rfc_gscv.fit(X_train,y_train)



# show results

print("Best parameters for Grid Search CV Random Forest Classifier model:")

print(rfc_gscv.best_params_)



# Compute predictions from new model



y_pred = rfc_gscv.predict(X_test)



# Compute confusion matrix from test data and prediction



conf_matrix_rfc_gscv = confusion_matrix(y_test, y_pred)



# show results



print(conf_matrix_rfc_gscv)