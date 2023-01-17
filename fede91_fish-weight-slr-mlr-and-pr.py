# Import the main libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
# Import the dataset and print the first 5 rows

path = '../input/fish-market/Fish.csv'

df = pd.read_csv(path)

df.head()
# Let's count the null values in the dataset

df_null = df.isnull()

df_null.head()
# Let's print the missing values for each column name

for column in df_null.columns.to_list():

  print(column)

  print(df_null[column].value_counts())

  print('')
df.shape
df.dtypes
df.describe()
# Identify the rows where the Weight is missing

df.loc[df['Weight']==0]
# Drop the row where weight=0

df = df[df['Weight'] != 0]

print(df.shape)

df.head()
df.describe(include='object')
df['Species'].value_counts()
# Define the x_labels

species = df['Species'].unique()



# Define the bar chart

plt.figure(figsize=(8,6))

plt.bar(species, df['Species'].value_counts(), color='G')



# Graphics

plt.xlabel('Species', fontsize=12, color='W')

plt.ylabel('Number of records', fontsize=12, color ='W')

plt.title('Barplot of number of species in the dataset', fontsize=16, color ='W')
# I want to identify the average weight, length1, length2, lenght3, height and width grouped by species

df_s_group = df.groupby('Species').mean()

df_s_group
sp_list = df['Species'].unique()

for sp in sp_list:

  print(sp)

  print(df[df['Species'] == sp].describe())

  print('')
sns.pairplot(df, kind='scatter', hue='Species')
sp_list = df['Species'].unique()

for sp in sp_list:

  print(sp)

  print(df[df['Species'] == sp].corr())

  print('')
df.corr()
df_Perch = df[df['Species'] == 'Perch']

sns.pairplot(df_Perch, kind='scatter')
# Import from scipy library the stats module

from scipy import stats
col_list = df.columns.to_list()[2:]

Y = df['Weight']

for x_pearson in col_list:

  pearson_coef, p_value = stats.pearsonr(df[x_pearson], Y)

  print(x_pearson)

  print('The Pearson Correlation Coefficient is ', pearson_coef, ' and the P-value is ', p_value)

  print('')
# Import LinearRegression from sklearn

from sklearn.linear_model import LinearRegression
# Let's define a new LinearRegression model

lm = LinearRegression()

lm
# Let's identify the X variable and the Y variable. Considering the Pearson test, the best variable to use to  develop a Simple Linear Regression is 'Length3'

X = df[['Length3']]

Y = df[['Weight']]



lm.fit(X,Y)
# I want to print the intercept and the coefficient of the Linear Regression

print('The coefficient is: ', lm.coef_)

print('The intercept is: ', lm.intercept_)
# Use seaborn to plot the Linear Regression model

sns.regplot(X,Y)
# Import the metrics I need

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
# Let's plot the residual

sns.residplot(X,Y)
Yhat_lm = lm.predict(X)

Yhat_lm[0:4]
mse_lm = mean_squared_error(Y,Yhat_lm)

r_score_lm = r2_score(Y, Yhat_lm)

print('The Mean Squared Error is ', mse_lm, ' and the R^2 score is ', r_score_lm)
# Use the col_list to define the independent variable.



for x in col_list:

  print(x)

  loop_lm = LinearRegression().fit(df[[x]], Y)

  print('The intercept is ', loop_lm.intercept_, ' and the coefficient is ', loop_lm.coef_)

  Yhat_loop_lm = loop_lm.predict(df[[x]])

  print('The Mean Squared Error is ', mean_squared_error(Y, Yhat_loop_lm), ' and the R^2 score is ', r2_score(Y,Yhat_loop_lm))

  print('')

# Define the Multiple Linear Regression Model and the independent variables. Train it on dependent and independent variables.

X = df[col_list]

mlrm = LinearRegression().fit(X,Y)

mlrm
print('The intercept is ', mlrm.intercept_, ' and the coefficients are ', mlrm.coef_)
# Let's plot the distribution of the Y and the predicted Y

Yhat_mlrm = mlrm.predict(X)



plt.figure(figsize=(8,6))



ax1 =  sns.distplot(Y, hist=False, color='R', label='Actual weight')

sns.distplot(Yhat_mlrm, hist=False, color='B', ax=ax1, label='Predicted weight')
# Calculate the Mean Squared Error and the R^2 score value

print('The Mean Squared Error is ', mean_squared_error(Y, Yhat_mlrm), ' and the R^2 Score is ', r2_score(Y, Yhat_mlrm))
def PlotPolly(model, independent_variable, dependent_variable, Name):

    x_new = np.linspace(min(independent_variable)*0.98, max(independent_variable)*1.01, 100)

    y_new = model(x_new)



    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')

    plt.title('Polynomial Fit with Matplotlib for Weight')

    ax = plt.gca()

    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.gcf()

    plt.xlabel(Name)

    plt.ylabel('Weight of fish')



    plt.show()

    #plt.close()
# Train a 5 degrees polynome

pol = np.polyfit(df['Width'], df['Weight'], 5)

func = np.poly1d(pol)

print(func)
# Plot the function

PlotPolly(func, df['Width'], df['Weight'], 'Width')
for x in col_list:

  pol_loop = np.polyfit(df[x], df['Weight'], 5)

  func_loop = np.poly1d(pol_loop)

  print(func_loop)

  plt.figure()

  PlotPolly(func_loop, df[x], Y, Name=x)

  plt.show()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
# Define the features we need to take in consideration for this model

X_polF = df[col_list]
# Define the Input for the pipeline

Input = [('standardscaler', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2, include_bias=False)), ('model', LinearRegression()) ]
pipe = Pipeline(Input)

pipe
pipe.fit(X_polF, Y)
Yhat_pipe = pipe.predict(X_polF)

Yhat_pipe[0:4]
# We can visualise the distribution of Yhat_pipe and the actual Y values in order to understand if the Polynomial Feature Regression model is a better model.

plt.figure(figsize=(12,10))

ax2 = sns.distplot(Y, hist=False, color='R', label='Actual values')

sns.distplot(Yhat_pipe, hist=False, color='G', label='Predicted values')

plt.show()
print("The Mean Squared Error of the Polynomial Multiple Linear Regression is ",  mean_squared_error(Y, Yhat_pipe), ' and the R^2 score is ', r2_score(Y, Yhat_pipe))
Y_pred = pd.DataFrame(data=Yhat_pipe, columns=['Estimate Weight'])

prediction_df = pd.concat([Y_pred, Y], axis=1)

prediction_df