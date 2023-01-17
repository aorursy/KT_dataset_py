# linear algebra

import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

# data visualization

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)
Auto = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
Auto.head()
Auto.info()
Auto.describe()
# Let's convert the special characters to NaN

missing_values = ['?','--','-','??','.']



# new dataframe after replacing special characters from the set

Auto = Auto.replace(missing_values, np.nan)

Auto.head()
# Let us look at the total missing values in the data set

# Looking for any missing values in the dataframe column

miss_val = Auto.columns[Auto.isnull().any()]



# printing out the columns and the total number of missing values of all the column

for column in miss_val:

    print(column, Auto[column].isnull().sum())
# defining two empty lists for columns and its values

nan_columns = []

nan_values = []



for column in miss_val:

    nan_columns.append(column)

    nan_values.append(Auto[column].isnull().sum())



# plotting the graph

fig, ax = plt.subplots(figsize=(15,8))

plt.bar(nan_columns, nan_values, color = 'purple', width = 0.5)

ax.set_xlabel("Column Names")

ax.set_ylabel("Count of missing values")

ax.set_title("Variables with missing values");
# Lets take the median to fill the missing values for the following features.

# In this process, we use the limit direction as both.



# Normalized losses

median_value= Auto['normalized-losses'].median()

Auto['normalized-losses']=Auto['normalized-losses'].fillna(median_value)



# Bore

median_value= Auto['bore'].median()

Auto['bore']=Auto['bore'].fillna(median_value)



# Stroke

median_value= Auto['stroke'].median()

Auto['stroke']=Auto['stroke'].fillna(median_value)



# Horsepower

median_value= Auto['horsepower'].median()

Auto['horsepower']=Auto['horsepower'].fillna(median_value)



# Peak-RPM

median_value= Auto['peak-rpm'].median()

Auto['peak-rpm']=Auto['peak-rpm'].fillna(median_value)



# Price

median_value= Auto['price'].median()

Auto['price']=Auto['price'].fillna(median_value)
# Looking what body_style and make our missing values have

Auto[['make','body-style']][Auto['num-of-doors'].isnull()==True]
# Looking for number of doors for a sedan model of Mazda

Auto['num-of-doors'][(Auto['body-style']=='sedan') & (Auto['make']=='mazda')]
# Similarly, looking for number of doors for a sedan model of Dodge

Auto['num-of-doors'][(Auto['body-style']=='sedan') & (Auto['make']=='dodge')]
Auto['num-of-doors'] = Auto['num-of-doors'].fillna('four')



# dictionary mapping for num of doors

a=Auto['num-of-doors'].map({'two':2,'four':4})

Auto['num-of-doors']=a
# converting data type to int

Auto['num-of-doors'] = Auto['num-of-doors'].astype(str).astype(int)   
Auto.info()
# Heatmap



plt.subplots(figsize=(20,8))

sns.heatmap(Auto.isnull(),yticklabels=False,cbar=False,cmap='Greens_r')
Auto.isna().sum()
from scipy.stats import norm

from scipy import stats



sns.distplot(Auto['price'], fit=norm);

fig = plt.figure()
#skewness and kurtosis

print("Skewness: %f" % Auto['price'].skew())

print("Kurtosis: %f" % Auto['price'].kurt())
sns.pairplot(Auto[['city-mpg', 'engine-size', 'wheel-base']], palette='Set1')
sns.pairplot(Auto[['highway-mpg', 'engine-size', 'wheel-base']], palette='Set1')
fig = px.histogram(Auto, x="make", title='Count of cars based on OEM')

fig.show()
# Comparing Symboling and make



fig, ax = plt.subplots(figsize=(30,10)) 

sns.violinplot(x="make", y="symboling", data=Auto, palette='deep')



plt.title("Risk factor based on make")
label = Auto["body-style"].unique()

sizes = Auto["body-style"].value_counts().values



# Now we could define the Pie chart

# pull is given as a fraction of the pie radius. This serves the same purpose as explode 

fig_pie1 = go.Figure(data=[go.Pie(labels=label, values=sizes, pull=[0.1, 0, 0, 0])])

# Defining the layout

fig_pie1.update_layout(title="Body-style Propotion",    

        font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    ))

fig_pie1.show()
# Plotting multiple violinplots, including a box and scatter diagram

fig_vio1 = px.violin(x = Auto['body-style'], y = Auto["price"], box=True, points="all")

# Defining the layout

fig_vio1.update_layout(

    title="Body-style and Price",

    xaxis_title="Body-style",

    yaxis_title="Price",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig_vio1.show()
# since the datatype of horsepower and peak-rpm is object, let's convert it first into integer

Auto['horsepower'] = Auto['horsepower'].astype(int)

Auto['peak-rpm'] = Auto['peak-rpm'].astype(int)



# let's create the new column for torque

Auto['torque'] = ((Auto['horsepower'] * 5252) / Auto['peak-rpm'])

Auto.head()
# Extracting the first 10 largest Horsepower values

HP = Auto['horsepower'].nlargest(10)

HP_Index = [129,49,126,127,128,105,73,74,15,16]

# Extracting the corresponding 10 values from the column 'make'

make_hp = Auto['make'].iloc[HP_Index]



#creating a new dataframe with this values.

data = {'HP': HP, 'Make':make_hp} 

df = pd.DataFrame(data) 

df
plt.subplots(figsize=(15,8))

ax = sns.barplot(x="Make", y="HP", data=df, palette='deep')

ax.set_xlabel("Make")

ax.set_ylabel("HP")

ax.set_title("Fastest accelerating car");
sns.pairplot(Auto[['horsepower','torque','peak-rpm']], palette='Set1', kind="reg")
plt.figure(figsize=(30,15))

sns.heatmap(Auto.corr(),annot=True,cmap='viridis',linecolor='white',linewidths=1);
# Calculating the combined mpg and creating a new dataframe

Comb_mpg = ((Auto['highway-mpg'] * 0.4) + (Auto['city-mpg'] * 0.55))

data1 = {'comb-mpg':Comb_mpg, 'make':Auto['make'], 'fuel-type':Auto['fuel-type']}



df1 = pd.DataFrame(data1) # for the easiness of visualising
# We use a violin plot mwith hue as the fuel type to know which car is fuel efficient



#fuel_effi = df1.nlargest(5, ['comb-mpg'])  ----------- to know the first most fuel efficient cars

#fuel_effi



plt.subplots(figsize=(30,10))

vx = sns.violinplot(x="make", y="comb-mpg", data=df1,hue='fuel-type',split=True,palette='Set1')

vx.set_xlabel("Make")

vx.set_ylabel("Comb-mpg")

vx.set_title("Fuel efficient cars");
fig = px.violin(Auto, y="price", x="drive-wheels", box=True, points="all", 

                color='fuel-type', hover_data=Auto.columns)

fig.show()
Auto.head()
Auto.info()
#Let's split fuel type, drive wheels and engine location 



fuel_type = pd.get_dummies(Auto['fuel-type'], drop_first=True) 

# this would remove the diesel column after spliting the fuel-type column, which is automatically predictable



drive_wheels = pd.get_dummies(Auto['drive-wheels'], drop_first=True) 

# this would remove the 4wd column after spliting the drive-wheels column, which is automatically predictable



engine_location = pd.get_dummies(Auto['engine-location'], drop_first=True) 

# this would remove the front column after spliting the engine-location column, which is automatically predictable



aspiration = pd.get_dummies(Auto['aspiration'], drop_first=True) 

# this would remove the std column after spliting the aspiration column, which is automatically predictable
Auto.drop(['fuel-type','drive-wheels','engine-location', 'aspiration'],axis=1,inplace=True)
Auto_df = pd.concat([Auto,fuel_type,drive_wheels,engine_location, aspiration], axis=1)
Auto_df.head()
# converting the following data type to float

Auto_df['normalized-losses'] = Auto_df['normalized-losses'].astype(float)

Auto_df['bore'] = Auto_df['bore'].astype(float)

Auto_df['stroke'] = Auto_df['stroke'].astype(float)

Auto_df['price'] = Auto_df['price'].astype(float)
Auto_df.info()
Auto_df.select_dtypes(include='object')
Auto_df.drop(['make','body-style','engine-type','num-of-cylinders', 'fuel-system'],axis=1,inplace=True)
Auto_clean = pd.concat([Auto_df],axis=1)
Auto_clean.info()
from sklearn.model_selection import train_test_split
X = Auto_clean.drop('price', axis=1) # This would consider all the columns except price

y = Auto_clean['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
# printing the y intercept

print("y intercept is :", lm.intercept_)
# printing the coefficients or the slope value

print("coefficients are :", lm.coef_)
# creating the dataframe with the coefficient

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
# lets look at the predictions

predictions
# now let us look at the y_test

y_test
plt.subplots(figsize=(10,8))

ax = sns.scatterplot(x=y_test, y=predictions,

                    sizes=(20, 200), legend="full", palette="Set2")
# Residuals are the difference between the actual and the predicted values

sns.jointplot(x=y_test,y=predictions,kind='hex')
# ploting the prediction error

pred_error = y_test - predictions



sns.distplot((pred_error))
# Linear regression using Statsmodel

import statsmodels.api as sm



# Unlike sklearn that adds an intercept to our data for the best fit, statsmodel doesn't. We need to add it ourselves

# Remember, we want to predict the price based off our features.

# X represents our predictor variables, and y our predicted variable.

# We need now to add manually the intercepts



X_endog = sm.add_constant(X_train)
# fitting the model

ls = sm.OLS(y_train, X_endog).fit() # OLS = Ordinary Least Squares

# summary of the model

ls.summary() 
# Printing the P values 

print(ls.pvalues)
from sklearn import metrics
# Mean Absolute Error (MAE)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))



# Mean Squared Error (MSE)

print('MSE:', metrics.mean_squared_error(y_test, predictions))



# Root Mean Squared Error (RMSE)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# R-squared (R2)

print('R^2 Score:', metrics.r2_score(y_test, predictions))