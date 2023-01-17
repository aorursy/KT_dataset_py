#importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
%pylab inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data2017 = pd.read_csv("../input/world-happiness/2017.csv")
data2017.head()
data2017.info()
data2017.isnull().sum().sum()
data2017.describe()
plt.rcParams["figure.figsize"] = (20,10)
data_plot = data2017.loc[:,['Whisker.high',
       'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual', "Happiness.Score" ]]
data_plot.plot()
plt.grid()
plt.ylabel("Score")
plt.xlabel("Country Ranking")
data_plot = data2017.loc[:,['Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual']]
data_plot.plot()
plt.grid()
plt.ylabel("Score")
plt.xlabel("Country Ranking")
sns.pairplot(data2017.iloc[:,2:])
plt.show()
#creating correlation matrix

corr = data2017.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(13, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,annot= True, mask=mask, cmap= 'coolwarm', vmax=.9, center=0,
            square=True, linewidths=.5, fmt= '.1f', cbar_kws={"shrink": .5})
c = corr.abs()

sol = (c.where(np.triu(np.ones(c.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
sol[:10]
corr.iloc[:,1].sort_values()
features_to_analyse = ['Whisker.high',
       'Whisker.low','Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual', 'Happiness.Score']
fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))
for i, ax in enumerate(fig.axes):
    if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='Happiness.Score', data=data2017[features_to_analyse], ax=ax)
data2017 = data2017.drop(['Whisker.high','Whisker.low'], axis=1)
filter1 = data2017['Happiness.Score']>=7
happy_countries = data2017[filter1]
happy_countries[:10]
from sklearn.model_selection import train_test_split

X= data2017.iloc[:,3:]
y= data2017["Happiness.Score"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,shuffle=True)
from sklearn.ensemble import RandomForestRegressor 
  
 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(X_train, y_train) 
from sklearn.metrics import mean_squared_error
print('train score: {:.2f}'.format(regressor.score(X_train, y_train)))
print('test score: {:.2f}'.format(regressor.score(X_test, y_test)))
y_predicted = regressor.predict(X_test)
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predicted))
y_predicted = regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted Happiness Score")
plt.show()
names=X.columns
sorted(zip(map(lambda x: round(x, 4), regressor.feature_importances_), names), reverse=True)
Importance = pd.DataFrame({'Importance':regressor.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by ='Importance',
                      axis = 0,
                      ascending = True).plot(kind = 'barh',
                                            color = 'r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

print('train score: {:.2f}'.format(model.score(X_train, y_train)))
print('test score: {:.2f}'.format(model.score(X_test, y_test)))
y_predicted = model.predict(X_test)
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predicted))
y_predicted = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted Happiness Score")
plt.show()
data2017_2 = data2017.drop('Economy..GDP.per.Capita.', axis=1)
X= data2017_2.iloc[:,3:]
y= data2017_2["Happiness.Score"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,shuffle=True)
model = LinearRegression().fit(X_train, y_train)

print('train score: {:.2f}'.format(model.score(X_train, y_train)))
print('test score: {:.2f}'.format(model.score(X_test, y_test)))
y_predicted = model.predict(X_test)
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predicted))
undata = pd.read_csv("../input/un-2017-country-profile-variables/country_profile_variables.csv")
undata.head()
undata.isnull().sum().sum()
undata.shape
c = undata.corr().abs()

sol = (c.where(np.triu(np.ones(c.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
sol[:10]
corr_matrix = c

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]

to_drop

undata.drop(to_drop, axis=1, inplace=True)
undata.shape
data_all= pd.merge(left=data2017_2, right=undata,on=None, left_on='Country', right_on='country')
data_all.head()
data_all = data_all.drop("country", axis=1)
data_all.shape
data_all.info()
data_last2 = data_all.drop(['GDP: Gross domestic product (million current US$)',
       'GDP growth rate (annual %, const. 2005 prices)', "GDP per capita (current US$)"
       , 'Economy: Agriculture (% of GVA)',
       'Economy: Industry (% of GVA)',
       'Employment: Agriculture (% of employed)',
       'Unemployment (% of labour force)', 'International trade: Exports (million US$)',
       'International trade: Imports (million US$)',
       'International trade: Balance (million US$)' , "Net Official Development Assist. received (% of GNI)"], axis=1)
data_last2.head(3)
data_last2.describe(include=['object'])
data_last2.info()
data_last2 = data_last2.replace(to_replace='~0.0', value=0.0, regex=True)
data_last2['Population growth rate (average annual %)'] = data_last2['Population growth rate (average annual %)'].astype(float)
data_last2["Fertility rate, total (live births per woman)"] = data_last2["Fertility rate, total (live births per woman)"].astype(float)
data_last2["Mobile-cellular subscriptions (per 100 inhabitants)"] = data_last2["Mobile-cellular subscriptions (per 100 inhabitants)"].astype(float)
data_last2["Threatened species (number)"] = data_last2["Threatened species (number)"].astype(float)

data_last2["Population age distribution (0-14 / 60+ years, %)"].head()
data_last2["0-14 Age"] =data_last2["Population age distribution (0-14 / 60+ years, %)"].str.replace("/"," ")
data_last2["60+ years Age"] =data_last2["Population age distribution (0-14 / 60+ years, %)"].str.replace("/"," ")

child= []
old= []
for i in range(data_last2.shape[0]):
    child.append(data_last2["0-14 Age"][i].split(" ")[0])
    old.append(data_last2["60+ years Age"][i].split(" ")[1])
data_last2["0-14 Age"] = child
data_last2["60+ years Age"]= old

data_last2["0-14 Age"] = data_last2["0-14 Age"].astype(float)
data_last2["60+ years Age"] = data_last2["60+ years Age"].astype(float)
data_last2[["0-14 Age","60+ years Age"] ]
data_last2 = data_last2[data_last2["Forested area (% of land area)"] != 0.0]
data_last2["Forested area (% of land area)"]
data_last2["Forested area Urban"] =data_last2["Forested area (% of land area)"].str.replace("/"," ")
data_last2["Forested area Rural"] =data_last2["Forested area (% of land area)"].str.replace("/"," ")

data_last2 = data_last2.reset_index()
data_last2 = data_last2.drop("index", axis=1)
data_last2["Forested area Urban"]
urban= []
rural= []
for i in range(data_last2.shape[0]):
    urban.append(data_last2["Forested area Urban"][i].split(" ")[0])
    rural.append(data_last2["Forested area Rural"][i].split(" ")[1])
data_last2["Forested area Urban"] = urban
data_last2["Forested area Rural"] = rural
data_last2["Forested area Urban"] = data_last2["Forested area Urban"].astype(float)
data_last2["Forested area Rural"] = data_last2["Forested area Rural"].astype(float)
data_last2 = data_last2[data_last2['Balance of payments, current account (million US$)']!= "..."]
data_last2["Refugees and others of concern to UNHCR (in thousands)"] = data_last2["Refugees and others of concern to UNHCR (in thousands)"].astype(float)
data_last2["Infant mortality rate (per 1000 live births"] = data_last2["Infant mortality rate (per 1000 live births"].astype(float)
data_last2['Balance of payments, current account (million US$)'] = data_last2['Balance of payments, current account (million US$)'].astype(float)
data_last2['Employment: Services (% of employed)'] = data_last2['Employment: Services (% of employed)'].astype(float)
data_last2['Employment: Industry (% of employed)'] = data_last2['Employment: Industry (% of employed)'].astype(float)

data_last2.corr().iloc[:,1][data_last2.corr().iloc[:,1].abs().sort_values() >=0.4]
data_last3= data_last2[['Country', 'Happiness.Score', 'Family', 'Health..Life.Expectancy.',
       'Freedom', 'Trust..Government.Corruption.', 'Dystopia.Residual',
       'Urban population (% of total population)',
       'Fertility rate, total (live births per woman)',
       'Energy production, primary (Petajoules)',
       'Mobile-cellular subscriptions (per 100 inhabitants)',
       'Forested area Rural', 'Employment: Services (% of employed)', '0-14 Age',
       '60+ years Age']]
X= data_last3.iloc[:,2:]
y= data_last3["Happiness.Score"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, shuffle= True)

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

print('train score: {:.2f}'.format(model.score(X_train, y_train)))
print('test score: {:.2f}'.format(model.score(X_test, y_test)))
y_predicted = model.predict(X_test)
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predicted))
from sklearn.covariance import EmpiricalCovariance, MinCovDet
# fit a Minimum Covariance Determinant (MCD) robust estimator to data 
robust_cov = MinCovDet().fit(data_last3.iloc[:,2:])
# Get the Mahalanobis distance
data_last3["mahalanobis"] = robust_cov.mahalanobis(data_last3.iloc[:,2:])
data_last3.head()
from scipy.stats import chi2
data_last3['p_value'] = 1 - chi2.cdf(data_last3['mahalanobis'], 12)

# Extreme values with a significance level of 0.01
data_last3.loc[data_last3.p_value < 0.01].head(30).sort_values(by= "mahalanobis", ascending=False)
data_last4 = data_last3[data_last3["mahalanobis"] <=3000]
data_last4 = data_last4.drop("mahalanobis", axis=1)
data_last4 = data_last4.drop("p_value", axis=1)
data_last3 = data_last3.drop("mahalanobis", axis=1)
data_last3 = data_last3.drop("p_value", axis=1)
data_last3.shape
data_last4.shape
data_last4.columns
features_to_analyse = ['Urban population (% of total population)',
       'Fertility rate, total (live births per woman)',
       'Energy production, primary (Petajoules)',
       'Mobile-cellular subscriptions (per 100 inhabitants)',
       'Forested area Rural', 'Employment: Services (% of employed)',
       '0-14 Age', '60+ years Age', 'Happiness.Score']
fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))
for i, ax in enumerate(fig.axes):
    if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='Happiness.Score', data=data_last4[features_to_analyse], ax=ax)
X= data_last4.iloc[:,2:]
y= data_last4["Happiness.Score"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

print('train score: {:.2f}'.format(model.score(X_train, y_train)))
print('test score: {:.2f}'.format(model.score(X_test, y_test)))
y_predicted = model.predict(X_test)
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predicted))
y_predicted = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted Happiness Score")
plt.show()