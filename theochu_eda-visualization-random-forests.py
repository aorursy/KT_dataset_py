import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
#load the dateset

input_path = "../input/medical-cost-personal-dataset/insurance.csv"

insurance_data = pd.read_csv(input_path)

insurance_data.head()
insurance_data.info()
insurance_data.describe()

# the 'children' column means:Number of children covered by health insurance / Number of dependents
#'sex' column descriptive statsistics

insurance_data.sex.describe()
#'smoker' column descriptive statsistics

insurance_data.smoker.describe()
#'region' column descriptive statsistics

insurance_data.region.describe()
#the correlation of each column

insurance_data.corr()

#we can see that only the columns of age and bmi are relatively correlated with charges.
# is there missing values 

insurance_data.isnull().sum()

# lucky that there is no missing values!
#scatterplots between bmi and charges

plt.figure(figsize=(10,6))

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

plt.title("the scatterplot between bmi and charges")
#plot regression line to check the slightly positive correlation above

plt.figure(figsize=(10,6))

sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

plt.title("the regression line between bmi and charges")
#add 'smoker' to check how smoking affect the relationship between bmi and charges

plt.figure(figsize=(10,6))

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
#display the relation between smoking_or_not and charges throug scatterplot

plt.figure(figsize=(10,6))

sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])
#display the distribution of age

sns.distplot(a=insurance_data['age'], kde=True)
sns.countplot(insurance_data.sex.value_counts())

plt.title("gender",color = 'blue',fontsize=15)
from sklearn.metrics import mean_absolute_error 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder #to change object columns to numerical



print("Setup Complete")
# Create target object and call it y

y = insurance_data.charges

# Create X

features = ['age', 'bmi', 'children']

X = insurance_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Define the model. Set random_state to 1

insurance_rf_model = RandomForestRegressor(random_state=1)



# fit your model

insurance_rf_model.fit(train_X, train_y)



# Calculate the mean absolute error of your Random Forest model on the validation data

charges_preds = insurance_rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(val_y, charges_preds)



print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
# Separate target from predictors

y = insurance_data.charges

X = insurance_data.drop(['charges'], axis=1)



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



#label_encoder object columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



print("MAE from Label Encoding:") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
#adjust max_leaf_nodes



def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    preds_val = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds_val)

    return(mae)



candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf_nodes in candidate_max_leaf_nodes :

    my_mae = get_mae(max_leaf_nodes, label_X_train, label_X_valid, y_train, y_valid)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))