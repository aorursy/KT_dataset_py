import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
src = r'../input/insurance.csv'
data = pd.read_csv(src)
data.info()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
plt.figure(figsize = (10,6))
sns.scatterplot(x = "age", y = "charges", data = data, hue = "sex")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Distribution of charges by age and sex")
smokers = data["smoker"].unique()
colors = ["Reds", "Greens"]
for i, smoker in enumerate(smokers):
    temp = data[data["smoker"] == smoker]
    sns.scatterplot(temp["bmi"], temp["charges"], cmap = colors[i])
plt.legend(smokers)
plt.figure(figsize = (10,5))
sns.boxplot(x = "region", y = "charges", hue = "sex", data = data)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
plt.figure(figsize = (10,6))
sns.scatterplot(x = "age", y = "charges", data = data, hue = "smoker")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Distribution of charges by age and sex")
plt.figure(figsize = (10,8))
sns.boxplot(x = "children", y = "charges",hue = "smoker", data = data)
plt.title("Distribution of charges by number of children")
sns.heatmap(data.corr(), annot = True)
data.head()
# Transforming categorical features to numerical values
data["smoker"] = data["smoker"].replace(["yes","no"], [1,0])
data["sex"] = data["sex"].replace(["male","female"], [1,0])
data["region_southeast"] = data["region"].apply(lambda x: 1 if x == "southeast" else 0)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# Data are split into training and test data
y_data = data["charges"]
x_data = data.drop(["charges","region"], axis = 1)
x_train, x_test, y_train ,y_test = train_test_split(x_data, y_data, test_size = 0.25)

# Model is trained and then used on test dataset
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)

# Coefficients and intercept of linear regression model extracted
model_coef = pd.DataFrame(data = model1.coef_, index = x_test.columns)
model_coef.loc["intercept", 0] = model1.intercept_ 
display(model_coef)

# Model's performance
model_performance = pd.DataFrame(data = [r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))],
                                 index = ["R2","RMSE"])
display(model_performance)
residual = y_test - y_pred
# Positive residual means that the actual charge > predicted charge
# Negative residual means that the actual charge < predicted charge
plt.scatter(y_test, residual)
plt.title("Residual vs actual charges")
plt.xlabel("Actual charges")
plt.ylabel("Residual")
