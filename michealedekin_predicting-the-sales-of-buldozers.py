# Importing Libraries for loading dataset
import numpy as np
import pandas as pd
# Iporting libraries for Data Visualisation
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
# Importing training and validation set
Buldozer_datasets = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', low_memory=False)
Buldozer_datasets.head()
# Exploring the data
Buldozer_datasets.info()
# Checking for missing values
Buldozer_datasets.isna().sum()
# Plotting a scatter graph for comparism of data format
fig, ax = plt.subplots()
ax.scatter(Buldozer_datasets["saledate"][:1000], Buldozer_datasets['SalePrice'][:1000])
Buldozer_datasets.saledate
# Loading the data set again with the correct saledate format
# with time pharse data
Buldozer_datasets = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', low_memory=False, parse_dates=["saledate"])
Buldozer_datasets.head()

# Checking to ensure the date format is correct
Buldozer_datasets.saledate
Buldozer_datasets.saledate.dtype
#ploting the same scatter plot with the correct date format
fig, ax = plt.subplots()
ax.scatter(Buldozer_datasets["saledate"][:1000], Buldozer_datasets['SalePrice'][:1000])
# Loading the transpose of the datasets
Buldozer_datasets.head().T
Buldozer_datasets.saledate.head(20)
# Sort dataset in date order
Buldozer_datasets.sort_values(by=["saledate"], inplace=True, ascending=True)
Buldozer_datasets.saledate.head(20)

# Copy the original dataset
Buldozer_datasets_df = Buldozer_datasets.copy()
Buldozer_datasets_df
Buldozer_datasets_df["saleYear"]=Buldozer_datasets_df.saledate.dt.year
Buldozer_datasets_df["saleMonth"]=Buldozer_datasets_df.saledate.dt.month
Buldozer_datasets_df["saleDay"]=Buldozer_datasets_df.saledate.dt.day
Buldozer_datasets_df["saleDayOfWeek"]=Buldozer_datasets_df.saledate.dt.dayofweek
Buldozer_datasets_df["saleDayOfYear"]=Buldozer_datasets_df.saledate.dt.dayofyear

Buldozer_datasets_df.head().T
Buldozer_datasets_df.drop("saledate", axis=1, inplace=True)
Buldozer_datasets_df.head(20).T
# Exploring state in he buldozer dataset
Buldozer_datasets_df.state.value_counts()
# Converting strings into categories data
pd.api.types.is_string_dtype(Buldozer_datasets_df["UsageBand"])
# Find the columns which contain strings
for label, content in Buldozer_datasets_df.items():
    if pd.api.types.is_string_dtype(content):
        print(label)
        
# .items() treats a dataset like a dictionary
# Converting strings values to categorical values
for label, content in Buldozer_datasets_df.items():
    if pd.api.types.is_string_dtype(content):
        Buldozer_datasets_df[label] = content.astype("category").cat.as_ordered()
Buldozer_datasets_df.info()
Buldozer_datasets_df.state.cat.categories
Buldozer_datasets_df.state.value_counts()
# Access all the data in the form of numbers
Buldozer_datasets_df.state.cat.codes
# Check for missing values
Buldozer_datasets_df.isna().sum()/len(Buldozer_datasets_df)
# Save Preprocessed data
Buldozer_datasets_df.to_csv("Buldozer_datasets.csv", index=False)
# Importing training and validation set
Buldozer_datasets_df = pd.read_csv('./Buldozer_datasets.csv', low_memory=False)
Buldozer_datasets_df.head().T

# Check for missing values
Buldozer_datasets_df.isna().sum()
# Checking for missing data
# Check for numerical data first
for label, content in Buldozer_datasets_df.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)
# Check for which numerical column has null values
for label, content in Buldozer_datasets_df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
# Fill numerical row with median
for label, content in Buldozer_datasets_df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum(): 
            
            # Add a binary column which tells us if the data is missing
            Buldozer_datasets_df[label+"_is_missing"] = pd.isnull(content)
    
            # Fill missing numerical values with median 
            Buldozer_datasets_df[label] = content.fillna(content.median())
# Check if there is any null numerical value
for label, content in Buldozer_datasets_df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
# Check how many examples are missing
Buldozer_datasets_df.auctioneerID_is_missing.value_counts()
# Fill all the numeric missing values
Buldozer_datasets_df.isna().sum()
# Finding and turning category variables into numbers
for label, content in Buldozer_datasets_df.items():
        if not pd.api.types.is_numeric_dtype(content):
            print(label)
pd.Categorical(Buldozer_datasets_df["state"]).codes
for label, content in Buldozer_datasets_df.items():
    if not pd.api.types.is_numeric_dtype(content):
            
            # Add a binary column which tells us if the data is missing
            Buldozer_datasets_df[label+"_is_missing"] = pd.isnull(content)
    
            # Turn categories into numbers and add 1 
            Buldozer_datasets_df[label] = pd.Categorical(content).codes +1  
Buldozer_datasets_df.head().T
# See if all missing data are being resolved
Buldozer_datasets_df.isna().sum()[:30]
%%time
# Instantiate model
model = RandomForestRegressor(n_jobs=-1, random_state=42)

# Fit the model
model.fit(Buldozer_datasets_df.drop("SalePrice", axis=1), Buldozer_datasets_df["SalePrice"])
# Score the model
# Calculating the coefficient of determination (R^2)
model.score(Buldozer_datasets_df.drop("SalePrice", axis=1), Buldozer_datasets_df["SalePrice"])
# Splitting data into train and validation set
Buldozer_datasets_df.saleYear
Buldozer_datasets_df.saleYear.value_counts()
# Creating your own train and validation set
df_val = Buldozer_datasets_df[Buldozer_datasets_df.saleYear == 2012]
df_train = Buldozer_datasets_df[Buldozer_datasets_df.saleYear != 2012]
len(df_val)
len(df_train)
# Splitting data into X and Y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
y_train
# Creating evaluation function(the competition uses RMLSE)

def RMSLE(y_test, y_preds):
    
    """
    Calculate the RMSLE between predictions and true labels
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))
    
    
# Create function to evaluate model on a few different level
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
             "Valid MAE": mean_absolute_error(y_valid, val_preds),
             "Training RMSLE": RMSLE(y_train, train_preds),
             "Valid RMSLE": RMSLE(y_valid, val_preds),
             "Training R^2": r2_score(y_train, train_preds),
             "Valid R^2": r2_score(y_valid, val_preds)}
    return scores
# Tune the hyperparameters
# Change the mix samples value
model = RandomForestRegressor(n_jobs=-1,
                             random_state=42,
                             max_samples= 10000)
model
%%time
model.fit(X_train, y_train)
# Show the scores
show_scores(model)
%%time
# Hyperparameter tunning with RandomizedSearchCV

rf_grid ={"n_estimators":np.arange(10, 100, 10),
          "max_depth":[None, 3, 5, 10],
          "min_samples_split":np.arange(2, 20, 2),
          "min_samples_leaf":np.arange(1, 20, 2),
          "max_features":[0.5, 1, "sqrt", "auto"],
          "max_samples":[10000]}

# Instatiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                    random_state=42),
                              param_distributions = rf_grid,
                              n_iter = 2,
                              cv = 5,
                              verbose = True)
                              
# Fit the RandomizedSearchCV
rs_model.fit(X_train, y_train)
# Find the best model hyperparmeters
rs_model.best_params_
# Evaluate the RandomizedSearchCV model
show_scores(model)
%%time
# Most Ideal hyperparameter
Ideal_model = RandomForestRegressor(n_estimators=10,
                                    min_samples_split=2,
                                    min_samples_leaf=19,
                                    max_samples=None,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    random_state=42)
# Fit the model
Ideal_model.fit(X_train, y_train)
show_scores(Ideal_model)
# Importing test csv
test_csv = pd.read_csv('../input/bluebook-for-bulldozers/Test.csv', low_memory=False, parse_dates=["saledate"])
test_csv.head()
# Make predictions on the test datasets
test_preds = Ideal_model.predict(test_csv)
test_csv.columns
X_train.columns
def preprocess_data(df):
    """
    Perform transformation on df and return transormed df.
    """
    df["saleYear"]=df.saledate.dt.year
    df["saleMonth"]=df.saledate.dt.month
    df["saleDay"]=df.saledate.dt.day
    df["saleDayOfWeek"]=df.saledate.dt.dayofweek
    df["saleDayOfYear"]=df.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill the numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum(): 
            
                # Add a binary column which tells us if the data is missing
                df[label+"_is_missing"] = pd.isnull(content)
    
                # Fill missing numerical values with median 
                df[label] = content.fillna(content.median())
    
        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
                df[label+"_is_missing"] = pd.isnull(content)
            
                # We would add +1 to the categorical code because pandas encode missing data as -1
                df[label] = pd.Categorical(content).codes+1
            
        return df
# Process the test data
test_csv = preprocess_data(test_csv)
test_csv.head()
X_train.head()
# We can find how he column differ using set
set(X_train.columns) - set(test_csv.columns)
# manually adjust the missing ones
test_csv["Backhoe_Mounting_is_missing"] = False
test_csv["state_is_missing"] = False
test_csv["fiSecondaryDesc_is_missing"] = False
test_csv["fiProductClassDesc_is_missing"] = False
test_csv["fiModelSeries_is_missing"] = False
test_csv["fiModelDescriptor_is_missing"] = False
test_csv["fiModelDesc_is_missing"] = False
test_csv["fiBaseModel_is_missing"] = False
test_csv["autioneerID_is_missing"] = False
test_csv["UsageBand_is_missingg"] = False
test_csv["Undercarriage_Pad_Width_is_missing"] = False
test_csv["Turbocharged_is_missing"] = False
test_csv["Travel_Controls_is_missing"] = False
test_csv["Transmission_is_missing"] = False
test_csv["Track_Type_is_missing"] = False
test_csv["Tire_Size_is_missing"] = False
test_csv["Tip_Control_is_missing"] = False
test_csv["Thumb_is_missing"] = False
test_csv["Stick_is_missing"] = False
test_csv["Stick_Length_is_missing"] = False
test_csv["Steering_Controls_is_missing"] = False
test_csv["Scarifier_is_missing"] = False
test_csv["Ripper_is_missing"] = False
test_csv["Ride_Control_is_missing"] = False
test_csv["Pushblock_is_missing"] = False
test_csv["ProductSize_is_missing"] = False
test_csv["ProductGroup_is_missing"] = False
test_csv["ProductGroupDesc_is_missing"] = False
test_csv["Pattern_Changer_is_missing"] = False
test_csv["Pad_Type_is_missing"] = False
test_csv["MachineHoursCurrentMeter_is_missing"] = False
test_csv["Hydraulics_is_missing"] = False
test_csv["Hydraulics_Flow_is_missing"] = False
test_csv["Grouser_Type_is_missing"] = False
test_csv["Grouser_Tracks_is_missing"] = False
test_csv["Forks_is_missing"] = False
test_csv["Engine_Horsepower_is_missing"] = False
test_csv["Enclosure_is_missing"] = False
test_csv["Enclosure_Type_is_missing"] = False
test_csv["Drive_System_is_missing"] = False
test_csv["Differential_Type_is_missing"] = False
test_csv["Coupler_is_missing"] = False
test_csv["Coupler_System_is_missing"] = False
test_csv["Blade_Width_is_missing"] = False
test_csv["Blade_Type_is_missing"] = False
test_csv["Blade_Extension_is_missing"] = False
test_csv["Backhoe_Mounting_is_missing"] = False
test_csv["autioneerID_is_missing"] = False
test_csv.head()
# We can find how he column differ using set again
set(X_train.columns) - set(test_csv.columns)
test_csv["autioneerID_is_missing"] = False
test_csv["UsageBand_is_missing"] = False
test_csv.head()
test_csv.columns
# Finally we make prdictions on the data
# Make predictions on the test datasets
test_preds = model.predict(test_csv)
test_csv.dtypes
