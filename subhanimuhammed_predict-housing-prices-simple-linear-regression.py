import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("../input/kc_house_data.csv")
df.head()
df.info()
df.date = pd.to_datetime(df['date'])
df.yr_built = pd.to_datetime(df.yr_built)
df.yr_renovated = pd.to_datetime(df.yr_renovated)
df.info()
df = df.drop_duplicates()
df.isnull().sum()
df.price.skew()
df.price.plot(kind='hist')
np.log(df.price).skew()
np.log(df.price).plot(kind = 'hist')
df.price = np.log(df.price)
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(df.columns[to_drop], axis=1)
print (corr['price'].sort_values(ascending=False)[:5], '\n')
print (corr['price'].sort_values(ascending=False)[-5:])
df.grade.unique()
grade_pivot = df.pivot_table(index='grade',
                                  values='price', aggfunc=np.median)
grade_pivot
grade_pivot.plot(kind='bar', color='blue')
plt.xlabel('Grade')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)

x = df.sqft_living
y = df.price

z = np.polyfit(x,y,1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()
df.grade.plot(kind = 'hist')
plt.show()
df.grade.skew()
df.sqft_living.skew()
df.sqft_living = np.log(df.sqft_living)
df.sqft_living.plot(kind='hist',color = 'blue')
df.sqft_living.skew()
plt.scatter(x=df.sqft_living,y=df.price)
plt.xlabel('Sqft_Living')
plt.ylabel('Sale_Price')
plt.xticks(rotation=0)

x = df.sqft_living
y = df.price

z = np.polyfit(x,y,1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()
df.sqft_living15.skew()
np.log(df.sqft_living15).skew()
df.sqft_living15 = np.log(df.sqft_living15)

df.sqft_living15.plot(kind='hist')
plt.scatter(x=df.sqft_living15,y=df.price)
plt.xlabel('Sqft_Living15')
plt.ylabel('Sale_Price')
plt.xticks(rotation=0)

x = df.sqft_living15
y = df.price

z = np.polyfit(x,y,1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()
df.sqft_above.skew()
np.log(df.sqft_above).skew()
df.sqft_above = np.log(df.sqft_above)
df.sqft_above.plot(kind='hist')
plt.scatter(x=df.sqft_above,y=df.price)
plt.xlabel('Sqft Above')
plt.ylabel('Sale_Price')
plt.xticks(rotation=0)

x = df.sqft_above
y = df.price

z = np.polyfit(x,y,1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()
print (corr['price'].sort_values(ascending=False)[:7], '\n')
print (corr['price'].sort_values(ascending=False)[-5:])
df['age_house'] = df.date - df.yr_built
df.age_house = pd.to_numeric(df.age_house.dt.days/365)
df.age_house.head()
df = df.drop(['date','yr_built'],axis = 'columns')
df = df[df.bedrooms > 1]
df.yr_renovated = pd.to_numeric(df.yr_renovated)
c = df.yr_renovated != 0
c = c.map({False:0, True:1})
df.yr_renovated = c
df.info()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
y = df.price
X = df.drop(['price', 'id'], axis=1)
cv = cross_val_score(reg,X,y,cv = 5)
np.mean(cv)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size = 0.3)
reg.fit(X_train,y_train)
result = reg.score(X_train,y_train)
print("Accuracy: %.3f%%" % (result*100.0))
import statsmodels.api as sm
X1 = sm.add_constant(X_train)
result = sm.OLS(y_train, X1).fit()
#print dir(result)
print('The R2 and Adjusted R2 are : {0} %, {1} % ;respectively'.format(result.rsquared*100, result.rsquared_adj*100))
y_pred = reg.predict(X_test)
from sklearn import metrics

mae = metrics.mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error is: ", mae)
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Square Error is: ", rms)
mape = np.mean(metrics.mean_absolute_error(y_test,y_pred)/y_test *100)
print('So the Mean Absolute Percentage Error is: {0} %'.format(mape))
#np.mean(np.abs((y_test - y_pred) / y_test)) * 100
actual_values = y_test
predictions = y_pred
sns.scatterplot(actual_values, predictions)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 11.5, xmax = 15.5, color = "red")
plt.show()
# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([11.5, 15.5], [11.5, 15.5], c = "red")
plt.show()
