# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex5 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

train_data = pd.read_csv('../input/train.csv', index_col='Id')

test_data = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data.SalePrice              

train_data.drop(['SalePrice'], axis=1, inplace=True)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in train_data.columns if

                    train_data[cname].nunique() < 10 and 

                    train_data[cname].dtype == "object"]



# Select numeric columns only

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols =  numeric_cols  + categorical_cols



X = train_data[my_cols].copy()

X_test = test_data[my_cols].copy()
X.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



# Preprocessing for numerical data

numerical_transformer = SimpleImputer() #strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numeric_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])





my_pipeline = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', RandomForestRegressor(n_estimators=50, random_state=0))

])
from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(clf, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Define model

    model = RandomForestRegressor(n_estimators, random_state=0)



    # Bundle preprocessing and modeling code in a pipeline

    clf = Pipeline(steps=[

            ('preprocessor', preprocessor),

            ('model', model)

                     ])



    scores = -1 * cross_val_score(clf, X, y,

                                  cv=3,

                                  scoring='neg_mean_absolute_error')

    return scores.mean()





# def get_score(n_estimators):

#     my_pipeline = Pipeline(steps=[

#         ('preprocessor', SimpleImputer()),

#         ('model', RandomForestRegressor(n_estimators, random_state=0))

#     ])

#     scores = -1 * cross_val_score(my_pipeline, X, y,

#                                   cv=3,

#                                   scoring='neg_mean_absolute_error')

#     return scores.mean()



# Check your answer

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
results = {i: get_score(i) for i in range(50, 450, 50)}



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use(['dark_background'])



plt.plot(results.keys(), results.values())

plt.show()
print(min(results, key=results.get))

print(results)
n_estimators_best = min(results, key=results.get)



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()
clf.set_params(model__n_estimators = n_estimators_best).fit(train_data, y)





preds_test = clf.predict(test_data)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)