import numpy as np



import matplotlib.pyplot as plt



import pandas as pd



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error
df = pd.read_csv("../input/1000-cameras-dataset/camera_dataset.csv")
print(df.shape)
print(df.count())
print(df.head())
print((df == 0).sum(axis = 0))
df = df.replace(0, np.nan)



print(df.count())
plt.hist(df["Price"])



plt.title("Fig 1. Camera Price Ranges and Frequencies from 1994 to 2007")

plt.xlabel("Approximate Price")

plt.ylabel("Frequency")



plt.show()
print(

    (df[[

        

        "Model",

        "Zoom wide (W)",

        "Zoom tele (T)",

        "Normal focus range",

        "Macro focus range",

        

    ]]).iloc[47:65]

)
# Main line graph



# Threshold between the "majority" and the "outliers".

threshold = 2500



# Version of the DataFrame that excludes price outliers,

# as determined via histogram.

exclude_outliers = df[df["Price"] <= threshold]



# The `x_mean` array contains the unique year values in the "Release date" column,

# sorted in chronological order.

x_mean = pd.Series(exclude_outliers["Release date"].unique()).sort_values()



# Group the observations by year,

# then calculate the mean of each column per year,

# then access the mean price.

y_mean = exclude_outliers.groupby(["Release date"]).mean()["Price"]



plt.plot(

    x_mean,

    y_mean,

    marker = "o", # Set the point marker to a circle.

)



plt.grid(True)







# Trendline



# Use least squares polynomial fit to get the coefficients for a trendline.

coeffs = np.polyfit(x_mean, y_mean, 1)



# Make a polynomial from the coefficients.

poly = np.poly1d(coeffs)



# Evaluate the polynomial for each value in `x_mean`.

y_trend = poly(x_mean)



# Plot the trendline.

plt.plot(

    x_mean,

    y_trend,

    "r", # Set color to red.

)







print("Note: Prices above {threshold} USD were excluded.".format(

    threshold = threshold,

))



plt.title("Fig. 2. Change in Average Camera Price by Year from 1994 to 2007")

plt.xlabel("Year")

plt.ylabel("Price (USD)")

plt.show()
cumulative = 0.3991

total_years = 2007 - 1994



to_add = cumulative / total_years

print(to_add)
current_year = 1996



years_passed = current_year - 1994



conversion = years_passed * 0.0307 + 1



print(conversion)
orig_price = 100



final_price = orig_price / conversion



print(final_price)
def convert_1994(orig_price, current_year):

    

    if current_year < 1994 or current_year > 2007:

        raise ValueError("This function is for prices in the years 1994 to 2007.")

    

    years_passed = current_year - 1994

    

    conversion = years_passed * 0.0307 + 1

    

    final_price = orig_price / conversion

    

    return final_price
print(convert_1994(100, 1996))
# Perform `convert_1994` on all datapoints,

# using "Price" and "Release date" values as arguments to the function.



# Then append the new list as a column "1994 Price" of the original DataFrame.



df["1994 Price"] = list(map(

    

    convert_1994,

    df["Price"],

    df["Release date"]

    

))



print(df[["Price", "1994 Price"]])
plt.hist(df["1994 Price"])



plt.title("Fig. 3. Camera Price Ranges and Frequencies from 1994 to 2007, Adjusted for Inflation")

plt.xlabel("Approximate Price")

plt.ylabel("Frequency")



plt.show()
# Main line graph



# Threshold between the "majority" and the "outliers".

threshold = 2000



# Version of the DataFrame that excludes price outliers,

# as determined via histogram.

exclude_outliers = df[df["1994 Price"] <= threshold]



# The `x_mean` array contains the unique year values in the "Release date" column,

# sorted in chronological order.

x_mean = pd.Series(exclude_outliers["Release date"].unique()).sort_values()



# Group the observations by year,

# then calculate the mean of each column per year,

# then access the mean price.

y_mean = exclude_outliers.groupby(["Release date"]).mean()["1994 Price"]



plt.plot(

    x_mean,

    y_mean,

    marker = "o", # Set the point marker to a circle.

)



plt.grid(True)







# Trendline



# Use least squares polynomial fit to get the coefficients for a trendline.

coeffs = np.polyfit(x_mean, y_mean, 1)



# Make a polynomial from the coefficients.

poly = np.poly1d(coeffs)



# Evaluate the polynomial for each value in `x_mean`.

y_trend = poly(x_mean)



# Plot the trendline.

plt.plot(

    x_mean,

    y_trend,

    "r", # Set color to red.

)



print("Note: Prices over {threshold} USD were excluded.".format(

    threshold = threshold,

))



plt.title("Fig. 4. Change in Average Camera Price by Year from 1994 to 2007, Adjusted for Inflation")

plt.xlabel("Year")

plt.ylabel("Price (USD)")

plt.show()
# Check whether each value is NaN (missing).

dslr_mask = np.isnan(

    

    df[[

        "Zoom wide (W)",

        "Zoom tele (T)",

        "Normal focus range",

        "Macro focus range",

    ]]

    

# For each row, return True if all values in the row are True.

# This is done by setting `axis = 1` so it goes from left to right.

).all(axis = 1).values



print(dslr_mask.shape)

print(dslr_mask)

print("\nThere are {} DSLR cameras.".format(

    

    # Sum the True values to count the number of DSLR cameras.

    sum(dslr_mask)

    

))
dslr_df = df[dslr_mask]



# Use the tilde `~` to get the bitwise NOT of the array.

# This now identifies non-DSLR cameras.

non_dslr_df = df[~dslr_mask]
print(dslr_df.count())
dslr_df = dslr_df.drop(

    

    [

        "Model",

        "Zoom wide (W)",

        "Zoom tele (T)",

        "Normal focus range",

        "Macro focus range",

        "Storage included",

        "Price",

    ],

    

    axis = 1,

)



print("Columns in `dslr_df`:\n")

print(list(dslr_df.columns))
dslr_df = dslr_df.dropna()



print(dslr_df.shape)
# List of feature names.

feature_list = list(dslr_df.drop("1994 Price", axis = 1).columns)
plt.hist(dslr_df["1994 Price"])



plt.title("Fig. 5. Camera Price Ranges and Frequencies (DSLR) from 1994 to 2007")

plt.xlabel("Approximate Price")

plt.ylabel("Frequency")



plt.show()
# The `corr()` method of a DataFrame returns a correlation matrix.

# Values were rounded to the 2nd decimal place for ease of reading.



dslr_corr = dslr_df.corr().round(2)



print(dslr_corr)
print(dslr_corr.iloc[1:4, 1:4])
print(dslr_corr.iloc[4:7, 4:7])
print(dslr_corr["1994 Price"])
print(dslr_corr["1994 Price"].drop("1994 Price").sort_values())
# Create subplots to neatly arrange the plots.

fig, axes = plt.subplots(

    nrows = 3,

    ncols = 2,

    figsize = (14, 7),

)



# Adjust the figure, and set a title.

fig.subplots_adjust(

        hspace = 1.5,

        wspace = 0.5,

    )

fig.suptitle('Fig. 6. Correlation of Each Feature to the Target (DSLR)')





for feature, ax in zip(feature_list, axes.flatten()):

    

    # Main line graph



    x = dslr_df[feature]

    y = dslr_df["1994 Price"]



    ax.scatter(x, y)







    # Trendline



    # Use least squares polynomial fit to get the coefficients for a trendline.

    coeffs = np.polyfit(x, y, 1)



    # Make a polynomial from the coefficients.

    poly = np.poly1d(coeffs)



    # Evaluate the polynomial for each value in `x`.

    y_trend = poly(x)



    # Plot the trendline.

    ax.plot(

        x,

        y_trend,

        "r", # Set color to red.

    )



    

    ax.set(

        title = ("{} and 1994 Price".format(feature)),

        xlabel = feature,

        ylabel = "1994 Price (USD)",

    )
dslr_df_majority = dslr_df[dslr_df["1994 Price"] <= 2500]



print(dslr_df_majority["1994 Price"].describe())
# Create subplots to neatly arrange the plots.

fig, axes = plt.subplots(

    nrows = 3,

    ncols = 2,

    figsize = (14, 7),

)



# Adjust the figure and set a title.

fig.subplots_adjust(

        hspace = 1.5,

        wspace = 0.5,

    )

fig.suptitle('Fig. 7. Correlation of Each Feature to the Target, Excluding Outliers (DSLR)')





# Start the loop.

for feature, ax in zip(feature_list, axes.flatten()):

    

    # Main line graph



    x = dslr_df_majority[feature]

    y = dslr_df_majority["1994 Price"]



    ax.scatter(x, y)







    # Trendline



    # Use least squares polynomial fit to get the coefficients for a trendline.

    coeffs = np.polyfit(x, y, 1)



    # Make a polynomial from the coefficients.

    poly = np.poly1d(coeffs)



    # Evaluate the polynomial for each value in `x`.

    y_trend = poly(x)



    # Plot the trendline.

    ax.plot(

        x,

        y_trend,

        "r", # Set color to red.

    )





    

    # Set subplot labels.

    

    ax.set(

        title = ("{} and 1994 Price".format(feature)),

        xlabel = feature,

        ylabel = "1994 Price (USD)",

    )
dslr_df_majority_corr = dslr_df_majority.corr().round(2)



print(dslr_df_majority_corr)
print(dslr_df_majority_corr["1994 Price"].drop("1994 Price").sort_values())
X = dslr_df_majority.drop(

    ["Release date", "1994 Price"],

    axis = 1,

).values



# Print the first 5 observations.

print(X[:5])
y = dslr_df_majority["1994 Price"].values



# Print the first 5 observations.

print(y[:5])
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(

    X, 

    y,

    test_size = 0.3,

    random_state = 0, # Use arbitrary random state for reproducible results.

)



model.fit(X_train, y_train)



print(y_train.shape, y_test.shape)
y_pred = model.predict(X_test)



print(y_pred.shape)

print(y_pred[:5])
print(mean_squared_error(y_test, y_pred))
# Use the `var()` method to calculate variance.

variance = dslr_df_majority["1994 Price"].var()



print(variance)
print(model.score(X_test, y_test))
def score_linreg(X, y, kf):

    mse_scores = []

    rsquared_scores = []



    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        model = LinearRegression()

        model.fit(X_train, y_train)



        y_pred = model.predict(X_test)

        

        mse_scores.append(mean_squared_error(y_test, y_pred))

        rsquared_scores.append(model.score(X_test, y_test))

        

    scores_str = """Variance (to compare to MSE): {variance}

MSE: {mse}

R-Squared: {rsquared}""".format(

        mse = np.mean(mse_scores),

        rsquared = np.mean(rsquared_scores),

        variance = np.var(y)

    )

    

    return scores_str
kf = KFold(n_splits = 5)



print(score_linreg(X, y, kf))
X = dslr_df_majority[["Max resolution", "Low resolution", "Effective pixels"]].values

y = dslr_df_majority["1994 Price"].values



kf = KFold(n_splits = 5)



print(score_linreg(X, y, kf))
print(non_dslr_df.count())
non_dslr_df = non_dslr_df.dropna()



print("Observations:", non_dslr_df.shape[0])
non_dslr_df = non_dslr_df.drop(["Model", "Price"], axis = 1)



print(list(non_dslr_df.columns))
feature_list = list(non_dslr_df.drop("1994 Price", axis = 1).columns)
plt.hist(non_dslr_df["1994 Price"])



plt.title("Fig. 9. Camera Price Ranges and Frequencies from 1994 to 2007 (non-DSLR)")

plt.xlabel("Approximate Price")

plt.ylabel("Frequency")



plt.show()
non_dslr_df_corr = non_dslr_df.corr().round(2)



print(non_dslr_df_corr)
print(non_dslr_df_corr["1994 Price"].drop("1994 Price").sort_values())
# Create subplots to neatly arrange the plots.

fig, axes = plt.subplots(

    nrows = 6,

    ncols = 2,

    figsize = (14, 22),

)



# Delete the extra subplot. Only 11 are needed.

fig.delaxes(axes[5, 1])



# Adjust the figure, and set a title.

fig.subplots_adjust(

        hspace = 0.5,

        wspace = 0.4,

    )

fig.suptitle('Fig. 10. Correlation of Each Feature to the Target (non-DSLR)')







for feature, ax in zip(feature_list, axes.flatten()):

    

    # Main line graph



    x = non_dslr_df[feature]

    y = non_dslr_df["1994 Price"]



    ax.scatter(x, y)





    # Trendline



    # Use least squares polynomial fit to get the coefficients for a trendline.

    coeffs = np.polyfit(x, y, 1)



    # Make a polynomial from the coefficients.

    poly = np.poly1d(coeffs)



    # Evaluate the polynomial for each value in `x`.

    y_trend = poly(x)



    # Plot the trendline.

    ax.plot(

        x,

        y_trend,

        "r", # Set color to red.

    )



    

    ax.set(

        title = ("{} and 1994 Price".format(feature)),

        xlabel = feature,

        ylabel = "1994 Price (USD)",

    )
non_dslr_df_majority = non_dslr_df[non_dslr_df["1994 Price"] <= 750]



print("Observations:", non_dslr_df_majority.shape[0])
# Create subplots to neatly arrange the plots.

fig, axes = plt.subplots(

    nrows = 6,

    ncols = 2,

    figsize = (14, 22),

)



# Delete the extra subplot. Only 11 are needed.

fig.delaxes(axes[5, 1])



# Adjust the figure, and set a title.

fig.subplots_adjust(

        hspace = 0.5,

        wspace = 0.4,

    )

fig.suptitle('Fig. 11. Correlation of Each Feature to the Target, Excluding Outliers (non-DSLR)')







for feature, ax in zip(feature_list, axes.flatten()):

    

    # Main line graph



    x = non_dslr_df_majority[feature]

    y = non_dslr_df_majority["1994 Price"]



    ax.scatter(x, y)





    # Trendline



    # Use least squares polynomial fit to get the coefficients for a trendline.

    coeffs = np.polyfit(x, y, 1)



    # Make a polynomial from the coefficients.

    poly = np.poly1d(coeffs)



    # Evaluate the polynomial for each value in `x`.

    y_trend = poly(x)



    # Plot the trendline.

    ax.plot(

        x,

        y_trend,

        "r", # Set color to red.

    )



    

    ax.set(

        title = ("{} and 1994 Price".format(feature)),

        xlabel = feature,

        ylabel = "1994 Price (USD)",

    )
non_dslr_df_majority_corr = non_dslr_df_majority.corr().round(2)



print(non_dslr_df_majority_corr)
print(non_dslr_df_majority_corr["1994 Price"].drop("1994 Price").sort_values())
X = non_dslr_df_majority.drop(["Release date", "1994 Price"], axis = 1).values

y = non_dslr_df_majority["1994 Price"].values
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(

    X, 

    y,

    test_size = 0.3,

    random_state = 0, # Use arbitrary random state for reproducible results.

)



model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print("Variance: ", non_dslr_df_majority["1994 Price"].var())

print("MSE: ", mean_squared_error(y_test, y_pred))

print("R-Squared: ", model.score(X_test, y_test))
kf = KFold(n_splits = 5)



# `score_linreg` is a function that was defined in 6.3.3. of this notebook.

print(score_linreg(X, y, kf))
X = non_dslr_df_majority[["Max resolution", "Low resolution", "Effective pixels"]].values

y = non_dslr_df_majority["1994 Price"].values



kf = KFold(n_splits = 5)



# `score_linreg` is a function that was defined in 6.3.3. of this notebook.

print(score_linreg(X, y, kf))