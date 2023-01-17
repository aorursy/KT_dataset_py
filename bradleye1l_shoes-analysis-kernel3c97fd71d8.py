import pandas as pd
import numpy as np

df = pd.read_csv('../input/7210_1.csv', parse_dates=['dateAdded', 'dateUpdated'])
df.head()
df = df[['brand', 'categories', 'colors', 'dateAdded', 'dateUpdated', 'prices.amountMin', 'reviews', 'prices.merchant','prices.size']] #I'm not taking the Max Prices, because most of them are the same
df.head()
df = df.rename(columns={'prices.merchant': 'Merchant', 'brand': 'Brand', 'categories': 'Categories','colors': 'Colors', 'prices.amountMin' : 'Price', 'reviews': 'Rating', 'prices.size': 'Size'}) 
df.head()
df.info()
print("Null values per column:")
df.isnull().sum()
Brand_df = df[['Brand', 'Price']].dropna()
Colors_df = df[["Colors", "Price"]].dropna()
Rating_df = df[["Rating", "Price"]].dropna()
Merchant_df = df[['Merchant', 'Price']].dropna()
Size_df = df[["Size", "Price"]].dropna()
NoNull_df = df.dropna()
# Honestly I'm not yet skilled enough to clean it up the way I want to.  Basically I was wanting to have only the "rating" info show up in the Rating
# column and have the "categorical" info (such as "Boots" and "Athelitic") show up in the Category column.  If someone can show me how to do this,
# I'd greatly appreciate it!  

df.drop(['Categories', 'Rating'], axis=1, inplace=True)
df.head()
print("How many entries we have over a period of time:")

df['dateAdded'].value_counts().resample('Y').sum().plot.line()
dfp = df[['dateAdded', 'Price']].dropna()
dfp = dfp.drop_duplicates(subset=['dateAdded', 'Price'], keep=False)
dfp = dfp.sort_values(by='dateAdded')

import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.plot(dfp['dateAdded'],dfp['Price'])
plt.xticks(rotation=45)
plt.xlabel('dateAdded')
plt.ylabel('Price')
plt.title('Price over the Dates Added:')
dfp = df[['dateUpdated', 'Price']].dropna()
dfp = dfp.drop_duplicates(subset=['dateUpdated', 'Price'], keep=False)
dfp = dfp.sort_values(by='dateUpdated')

import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.plot(dfp['dateUpdated'],dfp['Price'])
plt.xticks(rotation=45)
plt.xlabel('dateUpdated')
plt.ylabel('Price')
plt.title('Price over the Dates Updated:')
dfp = df[['dateAdded', 'dateUpdated', 'Price']].dropna()
dfp = dfp.drop_duplicates(subset=['dateUpdated', 'dateAdded', 'Price'], keep=False)
dfp = dfp.sort_values(by='dateAdded')

import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.plot(dfp['dateUpdated'],dfp['Price'], label='dateUpdated')
plt.plot(dfp['dateAdded'],dfp['Price'], label='dateAdded')
plt.xticks(rotation=45)
plt.xlabel('Dates Added/Updated')
plt.ylabel('Price')
plt.title('Price over the Dates Added/Updated:')
plt.legend(loc='upper right')
dfp.head(20)
dfa = dfp[['dateAdded', 'Price']].dropna()
dfu = dfp[['dateUpdated', 'Price']].dropna()
dfd = dfu['dateUpdated'] - dfa['dateAdded']
dfdwp = [dfd, dfa[['Price']]]
dfdwp = pd.concat(dfdwp, axis=1)
dfdwp.head(20)
dfdwp = dfdwp.rename(columns={0: 'Time_Until_Update'})
dfdwp.head(20)
dfdwp = dfdwp.reset_index().groupby("Time_Until_Update").mean()
dfdwp.head()
dfdwp.loc[dfdwp['index'] == 11157]  #This is a test to make sure we did it right
dfdwp.plot(y='Price', use_index=True)
plt.xticks(rotation=45)  #I'll need someone to show me how to get the x tick mark labels to show up here
plt.xlabel('Time Until Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')
plt.legend(loc='upper right')
dfdwp.reset_index(level=0, inplace=True)
dfdwp.head()
plt.plot(dfdwp['Time_Until_Update'],dfdwp['Price'])
plt.xticks(rotation=45)  # Never mind, I've figured it out :)
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')
dfdwp.plot(y='Price', use_index=True)
plt.plot(dfdwp['Time_Until_Update'],dfdwp['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')
plt.xlim([0, 4000])
dfdwp.plot(y='Price', use_index=True)
plt.xlim([0, 10000])
dfdwp.info()
plt.plot(dfdwp['Time_Until_Update'],dfdwp['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')
from scipy import stats

dfdwpno = dfdwp[dfdwp["Price"] < 501]
dfdwpno.head()
print("Number of outliers:")
print("")
dfdwpoo = dfdwp[dfdwp["Price"] > 500]
dfdwpoo.info()
plt.plot(dfdwpno['Time_Until_Update'],dfdwpno['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update (no outliers):')
Brand_dfm = Brand_df.reset_index().groupby("Brand").mean()
Brand_dfm.reset_index(level=0, inplace=True)
Brand_dfm.head(10)
Brando = Brand_dfm['Brand'].apply(lambda x: x.upper())
Brando.head()

Brand_dfm = [Brando, Brand_dfm[['Price', 'index']]]
Brand_dfm = pd.concat(Brand_dfm, axis=1)
Brand_dfm.head(10)
Brand_dfm = Brand_dfm.reset_index().groupby("Brand").mean()
Brand_dfm.reset_index(level=0, inplace=True)
Brand_dfm.head(10)
plt.bar(Brand_dfm['Brand'],Brand_dfm['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Brand')
plt.ylabel('Price')
plt.title('Average Price for each Brand:')
Brand_dfm = Brand_dfm.sort_values(by='Price')
Brand_dfm.head(10)
Brand_dfm.describe()
# some dummy lists with unordered values 
x_axis = Brand_dfm['Brand']
y_axis = Brand_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()
Colors_dfm = Colors_df.reset_index().groupby("Colors").mean()
Colors_dfm.reset_index(level=0, inplace=True)
Colors_dfm.head(10)
Colors_dfm.info()
Colors_dfm = Colors_dfm[Colors_dfm.Colors != ","]
Colors_dfm.head()
Colors_dfm.describe()
# some dummy lists with unordered values 
x_axis = Colors_dfm['Colors']
y_axis = Colors_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()
Colors_dfm = Colors_dfm.sort_values(by='Price')
print(Colors_dfm.nlargest(10, 'Price'))
Colors_dfm.head(10)
Merchant_dfm = Merchant_df.reset_index().groupby("Merchant").mean()
Merchant_dfm.reset_index(level=0, inplace=True)
Merchant_dfm.head(10)
Merchant_dfm.describe()
# some dummy lists with unordered values 
x_axis = Merchant_dfm['Merchant']
y_axis = Merchant_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()
Merchant_dfm = Merchant_dfm.sort_values(by='Price', ascending=False)
Merchant_dfm.head(10)
Merchant_dfm = Merchant_dfm.sort_values(by='Price', ascending=True)
Merchant_dfm.head(10)
Size_dfm = Size_df.reset_index().groupby("Size").mean()
Size_dfm.reset_index(level=0, inplace=True)
Size_dfm.head(10)
Size_dfm.info()
Size_dfm.describe()
# some dummy lists with unordered values 
x_axis = Size_dfm['Size']
y_axis = Size_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()
Size_dfm = Size_dfm.sort_values(by='Price', ascending=False)
print("Top 10 Shoe Sizes:")
Size_dfm.head(10)
Size_dfm = Size_dfm.sort_values(by='Price', ascending=True)
print("Bottom 10 Shoe Sizes:")
Size_dfm.head(10)
NoNull_df.head()
from sklearn.model_selection import cross_val_score

LABELS = ['Brand', 'Colors', 'Merchant', 'Size']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
NoNull_df[LABELS] = NoNull_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(NoNull_df[LABELS])

NoNull_df[['dateAdded', 'dateUpdated']] = NoNull_df[['dateAdded', 'dateUpdated']].astype(int)
merged = pd.concat([NoNull_df[['dateAdded', 'dateUpdated']], label_dummies], axis=1)

X = merged
y = NoNull_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("All Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
from sklearn.model_selection import cross_val_score

LABELS = ['Brand']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Brand_df[LABELS] = Brand_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Brand_df[LABELS])

X = label_dummies
y = Brand_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Brand Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
from sklearn.model_selection import cross_val_score

LABELS = ['Colors']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Colors_df[LABELS] = Colors_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Colors_df[LABELS])

X = label_dummies
y = Colors_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Color Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
from sklearn.model_selection import cross_val_score

LABELS = ['Merchant']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Merchant_df[LABELS] = Merchant_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Merchant_df[LABELS])

X = label_dummies
y = Merchant_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Merchant Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
from sklearn.model_selection import cross_val_score

LABELS = ['Size']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Size_df[LABELS] = Size_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Size_df[LABELS])

X = label_dummies
y = Size_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Size Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)
df.info()
dfml = df

dfml[['dateAdded', 'dateUpdated']] = df[['dateAdded', 'dateUpdated']].astype(int)

y = dfml.Price
X = dfml.dateAdded

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))
# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
reg.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print('R^2 Value:')
print(reg.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.xlabel("Economy")
plt.ylabel("Happiness Score")
plt.ylim([0,10])
plt.show()
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
X = dfml.dateUpdated
X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("dateUpdated Model:")
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
dfdwpno.head()
dfdwpno[['Time_Until_Update']] = dfdwpno[['Time_Until_Update']].astype(int)

X = dfdwpno.Time_Until_Update
X = X.reshape(-1,1)

y = dfdwpno.Price
y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("Time Until Update Model:")
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
