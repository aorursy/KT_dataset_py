# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as sm


# Location of dataset
file = "../input/kepler.csv"

# Create a dataframe from csv data
df = pd.read_csv(file)
discoveries_by_year = df["discovered"].value_counts()

plt.figure(figsize=(12, 4))
plt.bar(discoveries_by_year.index, discoveries_by_year.values, align='center', alpha=0.5)
plt.xticks(discoveries_by_year.index, rotation='vertical')
plt.ylabel('Planets Discovered')
plt.title('Planets by Year of Updated Date')
plt.show()

print("(Data made available Feb 2018)")
discoveries_by_year.sort_index()
column = 'detection_type'
plt.figure(figsize=(12, 4))
df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({df[column].dtype})")
plt.ylabel("Exoplanets Discovered")
plt.show()
print(df[column].value_counts())
print(f"Exoplanets discovered, 1988 - 2007: {df['discovered'].where(df.discovered < 2008).count()}")
print(f"Exoplanets discovered, 2008 - 2018: {df['discovered'].where(df.discovered >= 2008).count()}")

# Add a count column to make counts easy
df["count"] = 1

# more on as_index @
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
grouped = df[["count", "detection_type", "discovered"]].groupby(["discovered", "detection_type"], as_index=False).agg('count')

grouped.pivot(index="discovered", columns="detection_type").fillna(0)["count"]
# Purpose: Returns an index array that identifies rows which do not have np.inf or NaN values
# Parameters: Pandas dataframe and name of column in dataframe
def get_usable_indexes(dataframe, column):
    return ~((dataframe[column] == np.inf) | (pd.isnull(dataframe[column])))

# Bar plot
column = "mass_detection_type"
plt.figure(figsize=(12, 4))
df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({df[column].dtype})")
plt.ylabel("Exoplanets Discovered")
plt.show()
# Show values behind the barplot
print(df[column].value_counts())


# Define handy function to print info 
def report_sparseness(column):
    usable_indexes = get_usable_indexes(df, column)
    usable = df.loc[usable_indexes, column].count()
    total = df[column].shape[0]
    percent = str(round((usable / total) * 100, 1)) + "%"
    print(f"Rows with {column}: {usable} / {total} ({percent})")

print()
print()
[report_sparseness(column) for column in ["mass", "mass_detection_type"]]
print()
grouped_detection_types = df[[
    "detection_type", 
    "mass_detection_type", 
    "mass"]
].groupby(["detection_type", "mass_detection_type"])

grouped_detection_types["mass"].count()
# Show boxplots of mass per detection_type
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
df.loc[:,["detection_type", "mass"]].boxplot(by = ["detection_type"], ax = ax)
ax.set_title("Exoplanet Mass by Detection Type")
ax.set_ylabel("Mass (units of Planet Jupiter)")
fig.suptitle("") # Hide outer title
ax.set_xlabel("") # Auto-show x-labels
plt.show()

# Print counts per detection_type
grouped_detection_types = df[[
    "detection_type", 
    "mass"]
].groupby(["detection_type"])

grouped_detection_types["mass"].count()
# Show boxplots of radius per detection_type
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
df.loc[:,["detection_type", "radius"]].boxplot(by = ["detection_type"], ax = ax)
ax.set_title("Exoplanet Radius by Detection Type")
ax.set_ylabel("Radius (units of Planet Jupiter)")
fig.suptitle("") # Hide outer title
ax.set_xlabel("") # Auto-show x-labels
plt.show()

# Print counts per detection_type
grouped_detection_types = df[[
    "detection_type", 
    "radius"]
].groupby(["detection_type"])

grouped_detection_types["radius"].count()
# There are two insane outliers from the imaging type that hamper this boxplot.
# Let's keep orbital periods under ~41 years so it at least somewhat matches our 30 years of data.
outliers = df["orbital_period"] > 15000
print(f"Ignoring {outliers.sum()} outliers")

# Show boxplots of radius per detection_type
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
df.loc[~outliers,["detection_type", "orbital_period"]].boxplot(by = ["detection_type"], ax = ax)
ax.set_title("Exoplanet Orbital Period by Detection Type")
ax.set_ylabel("Orbital Period (days)")
fig.suptitle("") # Hide outer title
ax.set_xlabel("") # Auto-show x-labels
plt.show()

# Print counts per detection_type
grouped_detection_types = df.loc[~outliers, [
    "detection_type", 
    "orbital_period"]
].groupby(["detection_type"])

grouped_detection_types["orbital_period"].count()
# drop the count column added in visualization section
df = df.drop(columns=["count"])

print(f"Rows: {df.shape[0]} Columns: {df.shape[1]}")
missing_values = df["hot_point_lon"].isnull().sum()
print(f"Missing 'hot_point_lon' values: {missing_values} / {df.shape[0]}")
print()

print("All 'planet_status' values are the same. All are confirmed.")
print(df["planet_status"].unique())
print()

df = df.drop(["hot_point_lon", "planet_status"], axis=1)
print("Dropped hot_point_lon and planet_status")
df_with_rows_without_missing_values = df.dropna()
print(f"Rows without missing values: {df_with_rows_without_missing_values.shape[0]}")
for column in df:
    inf_indexes = (df[column] == np.inf)
    if inf_indexes.sum() > 0:
        df.loc[inf_indexes, column] = float("nan")
        print(F"{column}: {inf_indexes.sum()} inf -> nan")
# Suffixes for min and max errors
min_suffix = "_error_min"
max_suffix = "_error_max"

# List of columns having min and max errors
trio_columns = [
    "mass",
    "mass_sini",
    "radius",
    "orbital_period",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "omega",
    "tperi",
    "tconj",
    "tzero_tr",
    "tzero_tr_sec",
    "lambda_angle",
    "impact_parameter",
    "tzero_vr",
    "k",
    "temp_calculated",
    "geometric_albedo",
    "star_distance",
    "star_metallicity",
    "star_mass",
    "star_radius",
    "star_age",
    "star_teff"
]

# Purpose: Return a summary of usable data as 
#          both a numerical percent value and display string equivalent
def get_usable_row_counts(dataframe, column):
    # Find indexes of NaN or inf values
    usable_indexes = get_usable_indexes(dataframe, column)
    percent = round((dataframe.loc[usable_indexes, column].count() 
                     / dataframe[column].shape[0])*100, 1)
    return percent, str(percent) + "%"

# Relies on outside variables
def make_trio_dataframe(dataframe):
    trio_df = pd.DataFrame()
    for column in trio_columns:
        measure_raw, measure_pct = get_usable_row_counts(dataframe, column)
        min_raw, min_pct = get_usable_row_counts(dataframe, column+min_suffix)
        max_raw, max_pct = get_usable_row_counts(dataframe, column+max_suffix)
        trio_df = trio_df.append({
            "column": column,
            "measure": measure_pct, 
            "measure_raw": measure_raw, 
            "min_error": min_pct, 
            "min_error_raw": min_raw, 
            "max_error": max_pct,
            "max_error_raw": max_raw
        }, ignore_index=True)
    
    return trio_df

trio_df = make_trio_dataframe(df)
trio_df[["column", "measure", "min_error", "max_error"]]
# Create Primary Transit dataframe
pt_df = df.loc[df["detection_type"] == "Primary Transit"]
print(f"Shape of Primary Transit dataframe")
print(f"  Rows: {pt_df.shape[0]} Columns: {pt_df.shape[1]}")
print()

# Create Radial Velocity dataframe
rv_df = df.loc[df["detection_type"] == "Radial Velocity"]
print(f"Shape of Radial Velocity dataframe")
print(f"  Rows: {rv_df.shape[0]} Columns: {rv_df.shape[1]}")
trio_pt_df = make_trio_dataframe(pt_df)
trio_pt_df[["column", "measure", "min_error", "max_error"]]
# Drop columns from Primary Transit dataframe where utilization is below 20%
columns_to_drop = trio_pt_df.loc[trio_pt_df["measure_raw"] < 20.0, "column"]
for column in columns_to_drop:
    pt_df = pt_df.drop(columns=[column + min_suffix, column + max_suffix, column] )

# See description of new, slimmer Primary Transit dataframe
trio_pt_df[["column", "measure", "min_error", "max_error"]].where(trio_pt_df.measure_raw > 20).dropna()
trio_rv_df = make_trio_dataframe(rv_df)
trio_rv_df[["column", "measure", "min_error", "max_error"]]
# Drop columns from Primary Transit dataframe where utilization is below 20%
columns_to_drop = trio_rv_df.loc[trio_rv_df["measure_raw"] < 20.0, "column"]
for column in columns_to_drop:
    rv_df = rv_df.drop(columns=[column + min_suffix, column + max_suffix, column] )

# See description of new, slimmer Primary Transit dataframe
trio_rv_df[["column", "measure", "min_error", "max_error"]].where(trio_rv_df.measure_raw > 20).dropna()
# These columns contain too many unique string or date values to plot
columns_to_skip = [
    "# name", 
    "updated", 
    "alternate_names", 
    "star_name", 
    "star_sp_type", 
    "temp_measured",
    "log_g",
    "star_alternate_names", 
    "publication_status", 
    "detection_type",
    "star_detected_disc", 
    "star_magnetic_field"]
pt_df = pt_df.drop(columns=columns_to_skip)
rv_df = rv_df.drop(columns=columns_to_skip)

print(f"Dropped unused columns")
def zscore_norm(dataframe):
    for column in dataframe.columns:
        if df[column].dtypes == np.int64 or df[column].dtypes == np.float:
            # Copy the numpy array, then only normalize non-nan values
            normalized = dataframe[column].copy()
            usable_indexes = get_usable_indexes(dataframe, column)
            normalized[usable_indexes] = stats.zscore(dataframe.loc[usable_indexes, column])
            
            # Add new zscored values to dataframe
            dataframe[column] = normalized
            
zscore_norm(pt_df)
zscore_norm(rv_df)
column = "molecules"
plt.figure(figsize=(12, 4))
pt_df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({pt_df[column].dtype})")
plt.show()
print(pt_df[column].value_counts())
column = "molecules"
plt.figure(figsize=(12, 4))
rv_df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({rv_df[column].dtype})")
plt.show()
print(rv_df[column].value_counts())

plt.show()
def make_histogram(dataframe, column):
    usable_rows = get_usable_indexes(dataframe, column)
    mean = np.mean(dataframe.loc[usable_rows, column])
    std = np.std(dataframe.loc[usable_rows, column])
    dataframe.loc[usable_rows, column].plot.hist(bins=10)
    plt.axvline(mean, color = 'red', alpha=.8)
    plt.axvline(np.mean(mean + 2*std), color = 'red', alpha=.6, linestyle='--')
    plt.axvline(np.mean(mean - 2*std), color = 'red', alpha=.6, linestyle='--')
    plt.title(f"Histogram of {column.capitalize()} ({dataframe[column].dtype})")
    return plt

def run_plots(dataframe):
    columns_remaining = list(dataframe.columns)
    
    # Plot trio columns
    for column in dataframe.columns:
        if column in trio_columns:
            plt.figure(figsize=(20, 5))
            plt.tight_layout()
            plt.subplot(1, 3, 1)
            make_histogram(dataframe, column)
            columns_remaining.remove(column)

            plt.subplot(1, 3, 2)
            make_histogram(dataframe, column + min_suffix)
            columns_remaining.remove(column + min_suffix)

            plt.subplot(1, 3, 3)
            make_histogram(dataframe, column + max_suffix)
            columns_remaining.remove(column + max_suffix)
            plt.show()
    
    for column in columns_remaining:
        # Plot categorical data
        try:
            if (dataframe[column].dtype == object) and column != "molecules":
                dataframe[column].value_counts().plot.bar()
                plt.title(f"Bar Plot of {column.capitalize()} ({dataframe[column].dtype})")
                plt.show()
                print(dataframe[column].value_counts())
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"No plot for {column}?")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Plot numerical data not in trio columns
        try:
            if dataframe[column].dtypes == np.int64 or dataframe[column].dtypes == np.float:
                usable_rows = get_usable_indexes(dataframe, column)
                mean = np.mean(dataframe.loc[usable_rows, column])
                std = np.std(dataframe.loc[usable_rows, column])

                dataframe.loc[usable_rows, column].plot.hist(bins=20)
                plt.axvline(mean, color = 'red', alpha=.8)
                plt.axvline(np.mean(mean + 2*std), color = 'red', alpha=.6, linestyle='--')
                plt.axvline(np.mean(mean - 2*std), color = 'red', alpha=.6, linestyle='--')
                plt.title(f"Histogram of {column.capitalize()} ({dataframe[column].dtype})")
                plt.show()
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"No plot for {column}?")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
run_plots(pt_df)
run_plots(rv_df)
# Purpose: Create linear model and plot points
def present_linear_model(x_axis_values, y_axis_values, labels=["x_label", "y_label"]):
    from sklearn import linear_model
    n_points = len(x_axis_values)
    
    # First initialize the model.
    linear_model = linear_model.LinearRegression()
    
    # Fit the model to the data
    x_input = x_axis_values.values.reshape(n_points, 1)
    y_output = y_axis_values.values.reshape(n_points, 1)
    linear_model.fit(x_input, y_output)
    
    # Get predictions
    y_pred = linear_model.predict(x_input)
    
    # Plot output
    plt.scatter(x_input, y_output, alpha=.1)
    plt.plot(x_input, y_pred, linewidth=2, color="black")
    plt.grid(True)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(f"{labels[0]} vs {labels[1]}")
    plt.show()
    
    # Return model parameters
    # slope (m) and y-intercept (b)
    intercept = linear_model.intercept_[0] #'Intercept: {0:.5f}'.format(linear_model.intercept_[0])
    slope = linear_model.coef_[0][0] #'Slope : {0:.5f}'.format(linear_model.coef_[0][0])
    return intercept, slope
# Create Mass-Radius dataframe
mr_df = pd.DataFrame()
mr_df["mass"] = pt_df["mass"]
mr_df["radius"] = pt_df["radius"]
mr_df = mr_df.dropna()

# Show plot of mass and radius
intercept, slope = present_linear_model(mr_df["mass"], mr_df["radius"], ["mass", "radius"])
print("Intercept: {0:.5f}".format(intercept))
print("Slope : {0:.5f}".format(slope))

# Show Regression results
ols_model = sm.ols(formula = 'radius ~ mass', data=mr_df)
results = ols_model.fit()

print('\nSSE, SST, SSR, and RMSE:')
sst = np.sum((mr_df["radius"] - np.mean(mr_df["radius"]))**2)
sse = sst - results.ssr
print('SSE: {}'.format(sse))
print('SST: {}'.format(sst))
print('SSR: {}'.format(results.ssr))
print('RMSE: {}'.format(np.sqrt(results.mse_model)))

# Get most of the linear regression statistics we are interested in:
print(results.summary())

# Plot a histogram of the residuals
sns.distplot(results.resid, hist=True)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.show()
# Create Mass-Orbital Period dataframe
mop_df = pd.DataFrame()
mop_df["mass"] = np.concatenate((pt_df["mass"], rv_df["mass"]))
mop_df["orbital_period"] = np.concatenate((pt_df["orbital_period"], rv_df["orbital_period"]))
mop_df = mop_df.dropna()

# Show plot of mass and radius
intercept, slope = present_linear_model(mop_df["mass"], mop_df["orbital_period"], ["mass", "orbital_period"])
print("Intercept: {0:.5f}".format(intercept))
print("Slope : {0:.5f}".format(slope))

# Show Regression results
ols_model = sm.ols(formula = 'mass ~ orbital_period', data=mop_df)
results = ols_model.fit()

print('\nSSE, SST, SSR, and RMSE:')
sst = np.sum((mop_df["mass"] - np.mean(mop_df["mass"]))**2)
sse = sst - results.ssr
print('SSE: {}'.format(sse))
print('SST: {}'.format(sst))
print('SSR: {}'.format(results.ssr))
print('RMSE: {}'.format(np.sqrt(results.mse_model)))

# Get most of the linear regression statistics we are interested in:
print(results.summary())

# Plot a histogram of the residuals
sns.distplot(results.resid, hist=True)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.show()