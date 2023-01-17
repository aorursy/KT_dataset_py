import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/weatherAUS.csv')
data.shape
data.columns
data.info()
data.head()
data_null_percent = pd.Series(index=data.columns)

for column_name in data:
    data_null_percent[column_name] = data[column_name].count()/data.shape[0]
    
data_null_percent_sorted = data_null_percent.sort_values()
data_null_percent_sorted.plot.barh()
data = data.drop(columns=['Cloud9am','Cloud3pm', 'Evaporation', 'Sunshine','RISK_MM'])
data = data.dropna()
data.isnull().any()
data.shape
data.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
print("train: " + str(train.shape) + ", test: " + str(test.shape))
train["RainToday"] = train["RainToday"].map({"No":0, "Yes":1})
train["RainTomorrow"] = train["RainTomorrow"].map({"No":0, "Yes":1})

test["RainToday"] = test["RainToday"].map({"No":0, "Yes":1})
test["RainTomorrow"] = test["RainTomorrow"].map({"No":0, "Yes":1})
def category_impact_plot(variable, subplot_position):
    plt.subplot(subplot_position)
    pd.pivot_table(train, index=variable, values='RainTomorrow').plot.bar(figsize=(25,5), ax=plt.gca()) 
   
plt.figure(1)
category_impact_plot("WindGustDir", 131)
category_impact_plot("WindDir9am", 132)
category_impact_plot("WindDir3pm", 133)

categorical_variables = ["WindGustDir", "WindDir9am", "WindDir3pm"]

train = pd.get_dummies(train, columns=categorical_variables)
test = pd.get_dummies(test, columns=categorical_variables)
train.head()
location_pivot = train.pivot_table(index="Location", values="RainTomorrow")
location_pivot_sorted = location_pivot.sort_values(by=["RainTomorrow"])

location_pivot_sorted.plot.barh(figsize=(10,12))
plt.ylabel('')
train = pd.get_dummies(train, columns=["Location"])
test = pd.get_dummies(test, columns=["Location"])
train["Month"] = pd.to_datetime(train["Date"]).dt.month
test["Month"] = pd.to_datetime(test["Date"]).dt.month
date_pivot = train.pivot_table(index="Month", values="RainTomorrow")#.sort_index(ascending=False)

date_pivot.plot.barh()
plt.ylabel('')
train = pd.get_dummies(train, columns=["Month"])
test = pd.get_dummies(test, columns=["Month"])
# the preprocessing.minmax_scale() function allows us to quickly and easily rescale our data
from sklearn.preprocessing import minmax_scale

# Added 2 backets to make it a dataframe. Otherwise you will get a type error stating cannot iterate over 0-d array.
def apply_minmax_scale(dataset, features):
    for feature in features:
        dataset[feature] = minmax_scale(dataset[[feature]])
        
numerical_features = ["MinTemp","MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am",
                     "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", 
                     "Pressure3pm", "Temp9am", "Temp3pm"]

apply_minmax_scale(train, numerical_features)
apply_minmax_scale(test, numerical_features)

train[numerical_features].head()
rainTomorrow_yes = train[train["RainTomorrow"] == 1]
rainTomorrow_no = train[train["RainTomorrow"] == 0]
def variable_impact_plot(variable, subplot_position):
    plt.subplot(subplot_position)
    rainTomorrow_yes[variable].plot.hist(figsize=(25,10), alpha=0.5, color="blue", bins=50, ax=plt.gca())
    rainTomorrow_no[variable].plot.hist(figsize=(25,10), alpha=0.5, color="yellow", bins=50, ax=plt.gca())
    plt.ylabel('')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(variable)

plt.figure(1)
variable_impact_plot("MinTemp", 341)
variable_impact_plot("MaxTemp", 342)
variable_impact_plot("Rainfall", 343)
variable_impact_plot("WindGustSpeed", 344)
variable_impact_plot("WindSpeed9am", 345)
variable_impact_plot("WindSpeed3pm", 346)
variable_impact_plot("Humidity9am", 347)
variable_impact_plot("Humidity3pm", 348)
plt.figure(2)
variable_impact_plot("Pressure9am", 341)
variable_impact_plot("Pressure3pm", 342)
variable_impact_plot("Temp9am", 343)
variable_impact_plot("Temp3pm", 344)
# columns we will be using all the way down
columns = list(train.columns[1:])
columns.remove("RainTomorrow")
import seaborn as sns

# custom function to set the style for heatmap
def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(30, 25))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

plot_correlation_heatmap(train[columns])
# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(train[columns], train["RainTomorrow"])
coefficients = logisticRegression.coef_
print(coefficients)
feature_importance = pd.Series(coefficients[0], index=columns)
print(feature_importance)
# Plotting as a horizontal Bar chart
feature_importance.plot.barh(figsize=(10,25))
plt.show()
ordered_feature_importance = feature_importance.abs().sort_values()
ordered_feature_importance.plot.barh(figsize=(10,25))
plt.show()
predictors = ["Pressure3pm", "WindGustSpeed", "Pressure9am", "Humidity3pm"]

lr = LogisticRegression()
lr.fit(train[predictors], train["RainTomorrow"])
predictions = lr.predict(test[predictors])
print(predictions)
# Calculating the accuracy using the k-fold cross validation method with k=10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, train[predictors], train["RainTomorrow"], cv=10)
print(scores)
# Taking the mean of all the scores
accuracy = scores.mean()
print(accuracy)