import pandas as pd
import numpy as np
dataset = pd.read_csv('../input/scour-depth-analysis/scour_datasheet.csv')
df = pd.DataFrame(dataset)
df.describe()
# importing all libraries and dependencies for dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# importing all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker

# importing all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
# shape of the data
df.shape
df.info()
df['D95 (mm)'].dtype
#all diameter columns will be analysed separatly
for i in df.index:
    if df['D95 (mm)'][i] == '–' or df['D95 (mm)'][i] == '--':
        df['D95 (mm)'][i] = "199999"        
    if df['D16 (mm)'][i] == '–' or df['D16 (mm)'][i] == '--':
        df['D16 (mm)'][i] = "199999" 
    if df['D50 (mm)'][i] == '–' or df['D50 (mm)'][i] == '--':
        df['D50 (mm)'][i] = "199999" 
    if df['D84 (mm)'][i] == '–' or df['D84 (mm)'][i] == '--':
        df['D84 (mm)'][i] = "199999" 
    if df['Grad'][i] == '–' or df['Grad'][i] == '--':
        df['Grad'][i] = "199999" 
df['D95 (mm)'] = df['D95 (mm)'].astype(str)
df['D95 (mm)'] = df['D95 (mm)'].astype(float)
df['D16 (mm)'] = df['D16 (mm)'].astype(str)
df['D16 (mm)'] = df['D16 (mm)'].astype(float)
df['D50 (mm)'] = df['D50 (mm)'].astype(str)
df['D50 (mm)'] = df['D50 (mm)'].astype(float)
df['D84 (mm)'] = df['D84 (mm)'].astype(str)
df['D84 (mm)'] = df['D84 (mm)'].astype(float)
df['Grad'] = df['Grad'].astype(str)
df['Grad'] = df['Grad'].astype(float)
df.dtypes
# dropping pier based on Acadamic knowledge

df = df.drop('Pier',axis=1)
sns.set(style="whitegrid")
ax = sns.boxplot(x=df["Scour Depth (m)"])
Q1 =  df['Scour Depth (m)'].quantile(0.25)
Q3 = df['Scour Depth (m)'].quantile(0.75)
IQR = Q3 - Q1

#removing outliers from data bsed on price
outliers = []
for i in df.index :
    if df['Scour Depth (m)'][i] > Q1+1.5*IQR or df['Scour Depth (m)'][i] < Q1 -  1.5*IQR:
        outliers.append(i)
        
df.drop(index = outliers,inplace = True)
sns.set(style="whitegrid")
ax = sns.boxplot(x=df["Scour Depth (m)"])
df.shape
df.head()
df.drop(index = 0,inplace = True)
print(df.head())
d = df
df['Pier Shape'].unique()
# Visualizing the different Pier shape available

plt.rcParams['figure.figsize'] = [10,6]
ax=df['Pier Shape'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set1')
ax.title.set_text('Pier Shape')
plt.xlabel("Pier Shpe",fontweight = 'bold')
plt.ylabel("Frequency",fontweight = 'bold')
plt.figure(figsize=(8,8))

plt.title('Pier scour Distribution Plot')
sns.distplot(df["Scour Depth (m)"])
# Segregation of Numerical and Categorical Variables/Columns

cat_col = df.select_dtypes(include=['object']).columns
num_col = df.select_dtypes(exclude=['object']).columns
df_cat = df[cat_col]
df_num = df[num_col]
accuracy = df['Accuracy (m)']
print(accuracy.head())
print(accuracy.unique())
df.drop(columns = 'Accuracy (m)',inplace = True)
df.head()
df['Scour Depth (m)'].unique()
df_num.drop(columns = 'Accuracy (m)',inplace = True)
df_num.head()
diameter = ['D16 (mm)', 'D50 (mm)', 'D84 (mm)', 'D95 (mm)', 'Grad']
df_num.drop(columns = diameter,inplace = True)
from sklearn import preprocessing
normalized_X = preprocessing.normalize(df_num)
print(normalized_X.shape)
d_df = df.loc[:,diameter]
d_df.head()
d_df.dtypes
print(normalized_X)
nor_df = pd.DataFrame(data = normalized_X ,columns=["Pier_Width", "Pier_Length", "Skew", "Velocity", "depth", "scour_depth"])
print(nor_df.head())
ax = sns.pairplot(nor_df)
print(df_cat)
plt.figure(figsize=(20, 15))
plt.subplot(2,2,1)
sns.boxplot(x = 'Upstream/Downstream', y = 'Scour Depth (m)', data = df)
plt.subplot(2,2,2)
sns.boxplot(x = 'Pier Type', y = 'Scour Depth (m)', data = df)
plt.subplot(2,2,3)
sns.boxplot(x = 'Pier Shape', y = 'Scour Depth (m)', data = df)
plt.subplot(2,2,4)
sns.boxplot(x = 'Bed Material', y = 'Scour Depth (m)', data = df)
unknown = ['Unknown',' Unknown','Unknown     Noncohesive']
Insignificant = ['Insignificant',' Insignificant','Insignificant Cohesive','Insignificant Noncohesive']
Substantial = [' Substantial','Substantial   Noncohesive','Substantial']
Moderate = ['Moderate      Noncohesive','Moderate']
for i in df.index:
    if df['Debris Effects'][i] in unknown:
        df['Debris Effects'][i] = "Unknown"
    if df['Debris Effects'][i] in Insignificant:
        df['Debris Effects'][i] = "Insignificant"
    if df['Debris Effects'][i] in Substantial:
        df['Debris Effects'][i] = "Substantial"
    if df['Debris Effects'][i] in Moderate:
        df['Debris Effects'][i] = "Moderate"
plt.figure(figsize=(25, 15))
sns.boxplot(x = 'Debris Effects', y = 'Scour Depth (m)', data = df)
df.head()
d_df['scour_depth'] = df['Scour Depth (m)']
d_df.head()
droping_value = []
for i in d_df.index:
    if d_df['D16 (mm)'][i] > 1999:
        droping_value.append(i)
print(droping_value)
d_df.drop(index = droping_value,inplace = True)
d_df.describe()
ax = sns.pairplot(d_df)
d_df.corr()
nor_d_df = preprocessing.normalize(d_df)
print(nor_d_df)
diameter_normalised = pd.DataFrame(data = nor_d_df ,columns=["D16", "D50", "D84", "D95",'Grad',"scour_depth"])
print(diameter_normalised.head())
droping_diameters = ['D16','D84','D95']
diameter_df = diameter_normalised.drop(columns = droping_diameters)
diameter_df.head()
ax = sns.pairplot(diameter_df)
diameter_df.corr()
X = pd.DataFrame(nor_df)
X.head()
cat_col = df.select_dtypes(include=['object']).columns
print(cat_col)
# Get the dummy variables for the categorical feature and store it in a new variable - 'dummies'

dummies = pd.get_dummies(df[cat_col])
dummies.shape
dummies.head()
dummies = pd.get_dummies(df[cat_col], drop_first = True)
dummies.shape
dummies.reset_index(drop=True, inplace=True)
X.reset_index(drop=True, inplace=True)
dummies.head()
# Add the results to the original dataframe

X = pd.concat([X, dummies], axis = 1)
X.shape
X.head()
df_train, df_test = train_test_split(X, train_size = 0.9, test_size = 0.1, random_state = 0)
df_train.corr()
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (15, 15))
sns.heatmap(df_train.corr(), cmap="RdYlGn")
plt.show()
plt.scatter(X['Pier_Width'], X['scour_depth'])
plt.title('scattere plot of correlated variables')
plt.xlabel('pier width')
plt.ylabel('scour depth')
plt.show()
plt.scatter(X['Velocity'], X['scour_depth'])
plt.title('scattere plot of correlated variables')
plt.xlabel('velocity')
plt.ylabel('scour depth')
plt.show()
plt.scatter(X['depth'],X['scour_depth'])
plt.title('scattere plot of correlated variables')
plt.xlabel('depth')
plt.ylabel('scour depth')
plt.show()
y_train = df_train.pop('scour_depth')
X_train = df_train
y_test = df_test.pop('scour_depth')
X_test = df_test
X_train_1 = X_train['Pier_Width']
# Add a constant
X_train_1c = sm.add_constant(X_train_1)

# Create a first fitted model
lr_1 = sm.OLS(y_train, X_train_1c).fit()
# Check parameters created

lr_1.params
# Let's visualise the data with a scatter plot and the fitted regression line

plt.scatter(X_train_1c.iloc[:, 1], y_train)
plt.plot(X_train_1c.iloc[:, 1], 0.3345*X_train_1c.iloc[:, 1] + 0.0173, 'r')
plt.show()
# Print a summary of the linear regression model obtained
print(lr_1.summary())
X_train_poly = pd.DataFrame(X_train_1)
X_train_poly.head()
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
x_poly2 = polynomial_features.fit_transform(X_train_poly)
x_poly2.shape
import statsmodels.api as sm

model2 = sm.OLS(y_train, X_train_poly).fit()
model2.params
model2.summary()
X_train_poly2 = X_train[['Velocity']]
X_train_poly2.head()
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
x_poly3 = polynomial_features.fit_transform(X_train_poly2)
x_poly3.shape
import statsmodels.api as sm

model3 = sm.OLS(y_train, x_poly3).fit()
model3.params
print(model3.summary())