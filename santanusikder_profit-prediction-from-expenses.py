# Import the necessary modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
# Load the dataset containing the various expenses of 50 startups and their profits

df = pd.read_csv("../input/various-expenses-and-the-profits-of-50-startups/50_Startups.csv")

# Preview

df.head()
# Check for missing values

df.info()
# LabelEncoder from sklearn.preprocessing can be used to convert multiple categories in a categorical variable into numerical values

catToNum = LabelEncoder()

df["State"] = catToNum.fit_transform(df["State"])

# Preview

df.head()
# Change the name of the R&D Spend column to RD and Marketing Spend to Marketing

df.rename(columns = {"R&D Spend" : "RD", "Marketing Spend" : "Marketing"}, inplace = True)

# Preview

df.head()
df.describe()
df.info()
plt.figure(figsize = (14, 8))

sns.boxplot(x = df.columns, y = [df[col] for col in df.columns])
df.corr()
# Check the differences in the means

# California: 0, Florida: 1, New York: 2

df.groupby("State", as_index = True).mean()["Profit"].to_frame()
# Check the differences in minimums and maximums

maximums = df.groupby("State").max()["Profit"].to_frame()

minimums = df.groupby("State").min()["Profit"].to_frame()

minimums.merge(maximums, on = "State").rename(columns = {"Profit_x" : "min", "Profit_y" : "max"})
sns.boxplot(data = df, x = "State", y = "Profit")
stateGroups = df.groupby("State", as_index = True)

gCal, gFlor, gNY = stateGroups.get_group(0)["Profit"], stateGroups.get_group(1)["Profit"], stateGroups.get_group(2)["Profit"]

# So I got the groups above and now I'll create a list for making the pairing easy during the F-tests

profitGroups = [gCal, gFlor, gNY, gCal]

for i in range(3):

    f_score, p_value = stats.f_oneway(profitGroups[i], profitGroups[i + 1])

    print('''\

    Category pair: (%d, %d)

    F-Score = %f

    P-Value (Confidence Score) = %f

    '''%(i, (i + 1) % 3, f_score, p_value))
fig, axes = plt.subplots(2, 2, figsize = (14, 12))

fig.suptitle("Regression plots: Profit vs expenses and State", fontsize = 20)

axesList = list(axes[0])

axesList.extend(list(axes[1]))



for i, axis in enumerate(axesList):

    col = df.columns[i]

    sns.regplot(data = df, x = col, y = "Profit", ax = axis)

    axis.set_title("Profit vs %s %s"%(col, "costs" if col != "State" else "categories"), fontsize = 15)



# plt.savefig("Profit_vs_Expenses.jpg")

plt.show()
trainX, testX, trainy, testy = train_test_split(df[df.columns[:-1]], df[["Profit"]], test_size = 1/5, random_state = 0)
# Instantiate the linear regression object

regr = LinearRegression()
# Train using the training set

regr.fit(trainX, trainy)
# Print out the coefficients matrix (m * n) and the intercept vector (m,)

print('''\

Coefficients: %s

Intercepts: %s

'''%(regr.coef_, regr.intercept_))
# Firstly, let's create the trainyCap and testyCap arrays by predicting values based on trainX and testX, respectively

trainyCap = regr.predict(trainX)

testyCap = regr.predict(testX)
# Training set's evaluation

mse = mean_squared_error(trainy, trainyCap)

rmse = np.sqrt(mse) # or mean_squared_error(trainy, trainyCap, squared = False)

r2 = r2_score(trainy, trainyCap)
# Test set's evaluation

mse2 = mean_squared_error(testy, testyCap)

rmse2 = np.sqrt(mse2) # or mean_squared_error(trainy, trainyCap, squared = False)

r22 = r2_score(testy, testyCap)
# Convert the above results into a dataframe

dfEvaluation = pd.DataFrame({"Train" : [mse, rmse, r2], "Test" : [mse2, rmse2, r22]}, index = ["MSE", "RMSE", "R2 Score"])
# View

dfEvaluation