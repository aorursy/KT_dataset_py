
import pandas as pd
df = pd.read_csv("../input/Document.csv", header = None)
df.head(10)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
headers

df.columns = headers
df
df.dropna(subset=["price"], axis = 0)
df
df.dtypes
df.describe()
df.describe(include = "all")
df.info
import numpy as np
df.replace("?", np.NaN, inplace = True)
df.head(5)
missing_data = df.isnull()
missing_data.head(3)
for column in missing_data.columns.values.tolist():
    print(column)
    
    print(missing_data[column].value_counts())
    print(" ")
    
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)
df["normalized-losses"].replace(np.NaN, avg_1, inplace = True)
avg_2 = df["bore"].astype("float").mean(axis = 0)
df["bore"].replace(np.NaN, avg_2, inplace = True)
avg_3 = df["stroke"].astype("float").mean(axis = 0)
df["stroke"].replace(np.NaN, avg_3, inplace = True)

avg_4 = df["horsepower"].astype("float").mean(axis = 0)
df["horsepower"].replace(np.NaN, avg_4, inplace = True)
avg_5 = df["peak-rpm"].astype("float").mean(axis = 0)
df["peak-rpm"].replace(np.NaN, avg_5, inplace = True)
df["num-of-doors"].value_counts()
df["num-of-doors"].value_counts().idxmax()
df["num-of-doors"].replace(np.NaN, "four", inplace = True)
df.dropna(subset = ["price"], axis = 0, inplace = True)
df.reset_index(drop = True,inplace = True)
df
df.dtypes
df[["bore", "stroke", "peak-rpm"]] = df[["bore", "stroke", "peak-rpm"]].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["price"] = df["price"].astype(float)

df.dtypes
df["city L/100km"] = 235/df["city-mpg"]
df["highway L/100km"] = 235/df["highway-mpg"]
df.head(5)
df["horsepower"] = df["horsepower"].astype(float)
binwidth = (max(df["horsepower"]) - min(df["horsepower"]))/4
bins  = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
bins
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels = group_names,include_lowest = True)
df['horsepower-binned'].head(5)
df.head(10)
%matplotlib inline 
import matplotlib as plt
from matplotlib import pyplot


plt.pyplot.hist(df["horsepower"], bins = 3)

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("hp bins")
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis =1, inplace = True)
df.head()
dummy_variable2 = pd.get_dummies(df['aspiration'])
dummy_variable2.rename(columns = {'std': 'aspiration-std' , 'turbo' : 'aspiration-turbo'}, inplace = True)
df = pd.concat([df,dummy_variable2], axis = 1)
df.drop("aspiration", axis =1, inplace = True)
df.head()
df.to_csv('clean_df.csv')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df.dtypes

df.corr()
df[['bore','stroke','compression-ratio','horsepower']].corr()
sns.regplot(x = "engine-size", y = "price", data = df)
plt.ylim(0,)
df[["engine-size","price"]].corr()
sns.regplot(x = "highway-mpg", y ="price", data = df)
plt.ylim(0,)
df[["highway-mpg", "price"]].corr()
sns.regplot(x = "peak-rpm", y = "price", data = df)
plt.ylim(0,)
df[["peak-rpm", "price"]].corr()


sns.regplot(x="stroke",y = "price", data = df)
plt.ylim(0,)
df[["stroke","price"]].corr()
sns.boxplot(x= "body-style", y = "price", data = df)
plt.ylim(0,)
sns.boxplot(x = "engine-location", y = "price", data = df)


plt.ylim(0,)
sns.boxplot(x = "drive-wheels", y = "price", data = df)
plt.ylim(0,)
df.describe(include=['object'])
df['drive-wheels'].value_counts()
dw_count = df['drive-wheels'].value_counts().to_frame()
dw_count.rename(columns={'drive-wheels': 'count'}, inplace = True)
dw_count.index.name = 'drive-wheels'
dw_count
el_count = df['engine-location'].value_counts().to_frame()
el_count.rename(columns={'engine-location':'count'}, inplace = True)
el_count.index.name = 'engine-location'
el_count
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style', 'price']]
df_group_one = df_group_one.groupby(['drive-wheels','body-style'], as_index = False).mean()
df_group_one
pivot = df_group_one.pivot(index='drive-wheels', columns ='body-style')
pivot
pivot = pivot.fillna(0)
pivot
df['body-style'].unique()
df_group_two = df[['body-style', 'price']]
df_group_two = df_group_two.groupby(['body-style'], as_index = False).mean()
df_group_two
pivot2 = df_group_two.pivot(columns = 'body-style')
pivot2 = pivot2.fillna(0)
pivot2
plt.pcolor(pivot,cmap= 'RdBu')
plt.colorbar()
plt.show()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
pearson_coef, p_value
df_anova = df[["drive-wheels","price"]]
grouped_anova = df_anova.groupby(['drive-wheels'])
grouped_anova.head(1)
grouped_anova.get_group('4wd')['price']
f_val, p_val = stats.f_oneway(grouped_anova.get_group('4wd')['price'], grouped_anova.get_group('rwd')['price'])
f_val, p_val
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat = lm.predict(X)
Yhat[0:5]
print(lm.intercept_)
print(lm.coef_)
lm1 = LinearRegression()
lm1.fit(df[['engine-size']], df['price'])
print(lm1.intercept_)
print(lm1.coef_)

Z = df[['horsepower', 'curb-weight', 'engine-size','highway-mpg']]
lm.fit(Z, df['price'])
Y_hat = lm.predict(Z)

lm.fit(df[['normalized-losses', 'highway-mpg']], df['price'])
print(lm.intercept_)
print(lm.coef_)
import seaborn as sns
%matplotlib inline
width = 12
height  = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y ="price", data = df)
plt.ylim(0,)
plt.figure(figsize=(12,10))
sns.regplot(x='peak-rpm', y ='price', data = df)
plt.ylim(0,)
df[["peak-rpm", "highway-mpg", "price"]].corr()

plt.figure(figsize=(12,10))
sns.residplot(df['highway-mpg'], df['price'])
plt.ylim(0,)
sns.residplot(df['peak-rpm'],df['price'])
plt.ylim(0,)

Y_hat = lm.predict(Z)

plt.figure(figsize=(12,10))
ax1 = sns.distplot(df[['price']], hist=False, color ='r', label = 'Actual Values')
sns.distplot(Y_hat, hist=False, color = 'b', label = 'fitted values',ax= ax1)
plt.title('Actual vs fitted')
plt.xlabel('Price')
plt.ylabel('Proportion in cars')


def PlotPolly(model, independent_variable, dependent_variable, name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable,'.', x_new, y_new, '-')
    plt.title('Polynomial fit')
    ax = plt.gca()
    ax.set_facecolor((0.88, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(name)
    plt.ylabel('price')
    plt.show()
x = df['highway-mpg']
y = df['price']

f = np.polyfit(x,y,3)
p = np.poly1d(f)

PlotPolly(p,x,y,'highway-mpg')
f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'highway-mpg')
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
Z_pr.shape

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe
pipe.fit(Z,y)
ypipe = pipe.predict(Z)
ypipe[0:5]
lm.fit(X,Y)
lm.score(X,Y)
from sklearn.metrics import mean_squared_error
Yhat = lm.predict(X)
mean_squared_error(df['price'], Yhat)

lm.fit(Z,Y)
lm.score(Z,Y)
Y_hat = lm.predict(Z)

mean_squared_error(df['price'], Y_hat)
from sklearn.metrics import r2_score
r_squared = r2_score(y,p(x))
print(r_squared)
mean_squared_error(y,p(x))
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

new_input = np.arange(1,100,1).reshape(-1,1)
lm.fit(X,Y)
yhat = lm.predict(new_input)
print(yhat[0:5])


plt.plot(new_input,yhat)
