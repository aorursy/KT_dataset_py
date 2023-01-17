import pandas as pd

housing_data_full = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
housing_data_full.describe()
housing_data_full.columns
housing_data = housing_data_full.dropna(axis=0)
y = housing_data.Price

y.head()
selected_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]

X = housing_data[selected_features]
from sklearn.tree import DecisionTreeRegressor

housing_model = DecisionTreeRegressor(random_state=0)

housing_model.fit(X, y)
predicted_values = housing_model.predict(X)

print(predicted_values)

print(y.to_numpy())
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)

housing_model = DecisionTreeRegressor()

housing_model.fit(X_train, y_train)

val_predictions = housing_model.predict(X_valid)
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_valid, val_predictions))
def get_mae_train(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    preds_val = model.predict(X_train)

    mae = mean_absolute_error(y_train, preds_val)

    return(mae)



def get_mae_test(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    preds_val = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds_val)

    return(mae)



for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae_train = get_mae_train(max_leaf_nodes, X_train, X_valid, y_train, y_valid)

    my_mae_test = get_mae_test(max_leaf_nodes, X_train, X_valid, y_train, y_valid)

    print("Max leaf nodes: %d  \t\t MAE train:  %d \t\t MAE test:  %d" %(max_leaf_nodes, my_mae_train, my_mae_test))
my_mae_train = []

my_mae_test = []

leaves = range(5,1050,5)



for idx, max_leaf_nodes in enumerate(leaves):

    my_mae_train.append(get_mae_train(max_leaf_nodes, X_train, X_valid, y_train, y_valid))

    my_mae_test.append(get_mae_test(max_leaf_nodes, X_train, X_valid, y_train, y_valid))
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

#plt.plot(leaves, my_mae_train)

#plt.plot(leaves, my_mae_test)



sns.lineplot(x=leaves, y=my_mae_train)

sns.lineplot(x=leaves, y=my_mae_test)

plt.xlabel("No. leaves", fontsize=15)

plt.ylabel("MAE", fontsize=15)

plt.legend(labels=['Train MAE', 'Test MAE']);
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X_train, y_train)

predictionsRF = forest_model.predict(X_valid)

print(mean_absolute_error(y_valid, predictionsRF))
# Create and evaluate model in a function

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=50, random_state=1)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



# Retrieve all the dataset and divide it intro train and val

y = housing_data_full.Price

X_wo_y = housing_data_full.drop(['Price'], axis=1)

X = X_wo_y.select_dtypes(exclude=['object'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)
# Get names of columns with missing values

features_with_missing = [feature for feature in X_train.columns if X_train[feature].isnull().any()]



# Drop columns in training and validation data

reduced_X_train = X_train.drop(features_with_missing, axis=1)

reduced_X_valid = X_valid.drop(features_with_missing, axis=1)



print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns



print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for feature in features_with_missing:

    X_train_plus[feature + '_was_missing'] = X_train_plus[feature].isnull()

    X_valid_plus[feature + '_was_missing'] = X_valid_plus[feature].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns



print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
# Read the data

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



# Separate target from predictors

y = data.Price

X = data.drop(['Price'], axis=1)



# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# Drop columns with missing values (simplest approach)

cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 

X_train_full.drop(cols_with_missing, axis=1, inplace=True)

X_valid_full.drop(cols_with_missing, axis=1, inplace=True)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()



# Get list of categorical variables

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)

drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])



print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Select categorical columns

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
# Define the model

model = RandomForestRegressor(n_estimators=100, random_state=0)
from sklearn.metrics import mean_absolute_error



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)
from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE scores:\n", scores)
from xgboost import XGBRegressor



# Read the data

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



# Select subset of predictors

feature_names = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X = data[feature_names]



# Select target

y = data.Price



# Separate data into training and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y)



my_model = XGBRegressor()

my_model.fit(X_train, y_train)



predictions = my_model.predict(X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=2)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)



predictions = my_model.predict(X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
import eli5

from eli5.sklearn import PermutationImportance



# Fill NaN values with the mean

X_valid = X_valid.fillna(X_valid.mean())



# Perforformance permutation importance

perm = PermutationImportance(my_model, random_state=1).fit(X_valid, y_valid)

eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
from sklearn.tree import DecisionTreeClassifier



# Separate data into training and validation sets

X = X.fillna(X.mean())

y_scaled = y.copy()/1000 # scale for visualization 

X_train, X_valid, y_train, y_valid = train_test_split(X, y_scaled)



# Train the model

tree_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(X_train, y_train)
from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_valid, model_features=feature_names, 

                            feature='Distance')



# Plot

pdp.pdp_plot(pdp_goals, 'Distance');
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_valid, model_features=feature_names, 

                            feature='YearBuilt')

# Plot

pdp.pdp_plot(pdp_goals, 'YearBuilt');
features_to_plot = ['Distance', 'Rooms']

inter  =  pdp.pdp_interact(model=tree_model, dataset=X_valid, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)

plt.show()
import shap

import numpy as np

from sklearn.ensemble import RandomForestClassifier



# Load Hospital Readmissions dataset

data = pd.read_csv("../input/hospital-readmissions/train.csv")

y = data["readmitted"]

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]

X = X.drop("readmitted", axis=1)

X
# Split data and train the classifier

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=1).fit(X_train, y_train)



# Select the row for prediction

row_to_show = 15

data_for_prediction = X_valid.iloc[row_to_show] 

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

my_model.predict(data_for_prediction_array)



# Calculate and plot SHAP values

explainer = shap.TreeExplainer(my_model)

shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# Plot the summary plot of the first 200 observations

shap_values = explainer.shap_values(X_valid[:200])

shap.summary_plot(shap_values[1], X_valid[:200])
# Plot the dependence plot of the first 200 observations

shap_values = explainer.shap_values(X[:200])

shap.dependence_plot('number_inpatient', shap_values[1], X[:200], interaction_index="number_diagnoses")