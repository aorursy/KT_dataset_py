# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import data analysis tools 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/hackerearth-employee-attrition/Train.csv')
print("Data Info : ", df.info())

print("__________________________________________________")

print("Data Describe : ",df.describe())
df.shape[0],df.columns
df.dtypes
#Lets check the no. of unique items present in the categorical column

df.select_dtypes('object').nunique()
# plot the label 

df.Attrition_rate.plot.hist();
df.columns.values
#Lets check the Effect of age on employee’s Performnce





per=df[['Gender','growth_rate']].groupby(['Gender']).agg('median')

per
per.T.plot(kind="bar",figsize=(10

                               ,5));

plt.title("Growth Rate");

plt.ylabel("Rate");

plt.legend(loc="upper left")

plt.xticks(rotation=0);
# Create a plot

pd.crosstab(df.Gender, df.Attrition_rate).plot(kind="bar", figsize=(15,5), color=["salmon", "lightblue"])



# Add some attributes to it

plt.title("Attrition Rate  Frequency for Sex")

plt.xlabel("Gender")

plt.ylabel("Rate")

plt.legend(["Female", "Male"])

plt.xticks(rotation=0); # keep the labels on the x-axis vertical
# Create a plot

pd.crosstab(df.Education_Level, df.Attrition_rate).plot(kind="bar", figsize=(15,5), color=["salmon", "lightblue"])
import seaborn as sns

# Visualizing the number of male and female in the data set
plt.figure(figsize=(10, 5))

sns.countplot(df['Gender'], palette = 'bone')

plt.title('Comparison of Males and Females', fontweight = 30)

plt.xlabel('Gender')

plt.ylabel('Count');
plt.style.use('ggplot')

plt.figure(figsize=(10, 5))

sns.countplot(df['Hometown'], palette = 'pink')

plt.title('Comparison of various groups', fontweight = 30, fontsize = 20)

plt.xlabel('Groups')

plt.ylabel('count');

 
plt.style.use('ggplot')

plt.figure(figsize=(10, 5))

sns.countplot(df['Relationship_Status'], palette = 'pink')

plt.title('Comparison of various groups', fontweight = 30, fontsize = 20)

plt.xlabel('Groups')

plt.ylabel('count');

 
#Lets check the Effect of age on employee’s Performnce





per=df[['Relationship_Status','Attrition_rate']].groupby(['Relationship_Status']).agg('sum')

per
per.T.plot(kind="bar",figsize=(10,5));

plt.title("Relationship Status");

plt.ylabel("Rates");

plt.legend(loc="upper right");
# These columns contain strings

for label,content in df.items():

    if pd.api.types.is_string_dtype(content):

        print(label)
# This will turn all of the string values into category values

for label,content in df.items():

    if pd.api.types.is_string_dtype(content):

        df[label]=content.astype("category").cat.as_ordered()
df.info()
df.Gender.cat.categories

df.isna().sum()
df.isnull().sum()/len(df)

for label,content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        print(label)
#Check for which numeric columns have null values

for label,content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label)
# Fill numeric rows with the median

for label,content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            # Fill missing numeric values with median since it's more robust than the mean

            df[label] = content.fillna(content.median())
# Check if there's any null values

for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label)
# Turn categorical variables into numbers

for label, content in df.items():

    # Check columns which *aren't* numeric

    if not pd.api.types.is_numeric_dtype(content):

         

        # We add the +1 because pandas encodes missing categories as -1

        df[label] = pd.Categorical(content).codes+1
df.info()
from sklearn.ensemble import RandomForestRegressor

# Instantiate model

model = RandomForestRegressor(n_jobs=-1)



# Fit the model

model.fit(df.drop("Attrition_rate", axis=1), df.Attrition_rate.values)
# Score the model

model.score(df.drop("Attrition_rate", axis=1), df.Attrition_rate.values)
#Splitting the data set into training and test sets

x=df.drop("Attrition_rate", axis=1)

y=df.Attrition_rate.values



from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.20, random_state = 42)



print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)
# Create evaluation function (the competition uses Root Mean Square Log Error)

from sklearn.metrics import mean_squared_log_error, mean_absolute_error

def rmsle(y_test, y_preds):

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



# Create function to evaluate our model

def show_scores(model):

    train_preds = model.predict(x_train)

    val_preds = model.predict(x_val)

    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),

              "Valid MAE": mean_absolute_error(y_val, val_preds),

              "Training RMSLE": rmsle(y_train, train_preds),

              "Valid RMSLE": rmsle(y_val, val_preds),

              "Training R^2": model.score(x_train, y_train),

              "Valid R^2": model.score(x_val, y_val)}

    return scores

len(x_train)
#Change max samples in RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,

                              max_samples=2000)

model
# Cutting down the max number of samples each tree can see improves training time

model.fit(x_train, y_train)
show_scores(model)

from sklearn.model_selection import RandomizedSearchCV



# Different RandomForestClassifier hyperparameters

rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"]

           }



rs_model = RandomizedSearchCV(RandomForestRegressor(),

                              param_distributions=rf_grid,

                              n_iter=2,

                              cv=5,

                              verbose=True)



rs_model.fit(x_train, y_train)
rs_model.best_params_
ideal_model = RandomForestRegressor(n_estimators=80,

                                    min_samples_leaf=4,

                                    min_samples_split=17,

                                    max_features='auto',

                                    n_jobs=-1

                                )

ideal_model.fit(x_train, y_train)
show_scores(ideal_model)
df_test = pd.read_csv("/kaggle/input/hackerearth-employee-attrition/Test.csv")

df_test.head()
for label, content in df_test.items():

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                print(label)
for label, content in df_test.items():

        if not pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                print(label)
df_test.info()
def preprocess_data(df):

     

    # Fill numeric rows with the median

    for label, content in df.items():

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                

                df[label] = content.fillna(content.median())

                

        # Turn categorical variables into numbers

        if not pd.api.types.is_numeric_dtype(content):

            if pd.api.types.is_string_dtype(content):

                df[label]=content.astype("category").cat.as_ordered()

            

            # We add the +1 because pandas encodes missing categories as -1

            df[label] = pd.Categorical(content).codes+1        

    

    return df
df_test=preprocess_data(df_test)

df_test.info()
# Make predictions on the test dataset using the best model

test_preds=ideal_model.predict(df_test)
# Create DataFrame compatible with Kaggle submission requirements

df_preds=pd.DataFrame()

df_preds['Employee_ID']=df_test["Employee_ID"]

df_preds["Attrition_rate"] = test_preds

df_preds


df_preds.to_csv("predictions.csv",

               index=False)

# Find feature importance of our best model

ideal_model.feature_importances_
# Helper function for plotting feature importance

def plot_features(columns, importances, n=20):

    df = (pd.DataFrame({"features": columns,

                        "feature_importance": importances})

          .sort_values("feature_importance", ascending=False)

          .reset_index(drop=True))

    

    sns.barplot(x="feature_importance",

                y="features",

                data=df[:n],

                orient="h")
plot_features(x_train.columns, ideal_model.feature_importances_)

sum(ideal_model.feature_importances_)
