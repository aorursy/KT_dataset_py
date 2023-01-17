import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/nyc-property-sales/nyc-rolling-sales.csv', index_col = 0)
df.info()
df.isna().sum()
categorical = df.select_dtypes(include=['object'])
categorical.head().transpose() # Transposing make visualization easier for big datasets
categorical.describe().transpose()
df['SALE PRICE'] = df['SALE PRICE'].apply(lambda s: int(s) if not "-" in s else 0)
df['LAND SQUARE FEET'] = df['LAND SQUARE FEET'].apply(lambda s: int(s) if not '-' in s else 0)
df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].apply(lambda s: int(s) if not '-' in s else 0)
df['SALE PRICE'] = df['SALE PRICE'].apply(lambda s: s if type(s) == int else 0)
df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], format= '%Y-%m-%d %H:%M:%S')
df.drop(["EASE-MENT", "ADDRESS", "APARTMENT NUMBER"], axis = 1, inplace = True)
numerical = df.select_dtypes(exclude=['object'])
numerical.info()
numerical.describe().transpose()
weird_zeros_cols = [
    "ZIP CODE",
    "GROSS SQUARE FEET",
    "YEAR BUILT",
    "SALE PRICE"
   ]

l = len(df)
for col in weird_zeros_cols:
    print(f"{col:.10}\t{len(df[df[col] == 0])/l:0.2f}% missing")
for col in weird_zeros_cols:
    df = df[df[col] != 0]
categorical = df.select_dtypes(include=['object'])
categorical.describe().transpose()
sns.countplot(
    x="TAX CLASS AT PRESENT",
    data = df,
    order = df["TAX CLASS AT PRESENT"].value_counts().index,
)
plt.show()
pivot = df.pivot_table(index='TAX CLASS AT PRESENT',
                       values='SALE PRICE',
                       aggfunc=np.sum,).sort_values("SALE PRICE")

pivot.plot(
    kind='bar',
    color='orange',
    title="Total Sale Price per Tax Class"
)
pivot = df.pivot_table(index='TAX CLASS AT PRESENT',
                       values='SALE PRICE',
                       aggfunc=np.mean).sort_values("SALE PRICE")
pivot.plot(kind='bar',
           color='black',
           title="Average Price per Tax Class"
          )
g = sns.countplot(
    x='BUILDING CLASS CATEGORY',
    data = df,
    order = df["BUILDING CLASS CATEGORY"].value_counts().index,
)
g.set_yscale('log')
g.set_xticklabels(g.get_xticklabels(), rotation = 90)
plt.show()
pivot = df.pivot_table(index='BUILDING CLASS CATEGORY',
                       values='SALE PRICE',
                       aggfunc=np.sum).sort_values("SALE PRICE")
pivot.plot(kind='bar', color = 'green')
df['BUILDING CLASS CATEGORY'].value_counts().head(6)
top_vals = df['BUILDING CLASS CATEGORY'].value_counts().index[:5]
df = df[df["BUILDING CLASS CATEGORY"].isin(top_vals)]
numerical = df.select_dtypes(exclude=['object', 'datetime'])
numerical.describe().transpose()
sns.heatmap(numerical.corr()) #, annot= True)
df.drop(["RESIDENTIAL UNITS", "LAND SQUARE FEET"], axis = 1, inplace = True)
plt.hist(
    df["SALE PRICE"],
    bins = 20,
    log = True,
    rwidth = 0.8
)
plt.show()
df = df[(df['SALE PRICE'] < 5e8) & (df['SALE PRICE'] > 1e5)]
# Plot histogram.
n, bins, patches = plt.hist(
    df["SALE PRICE"],
    bins = 20,
    log = True,
    rwidth = 0.8
) 
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

cm = plt.cm.get_cmap('plasma')
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

n, bins, patches = plt.hist(
    df["SALE PRICE"],
    bins = 20,
    log = True,
    rwidth = 0.8
) 

bin_centers = 0.5 * (bins[:-1] + bins[1:])

sorted_patches = [p for _,p in sorted(zip(n,patches), key=lambda pair: pair[0])] #sorts patches in respect to n
sorted_centers = [c for _,c in sorted(zip(n, bin_centers), key=lambda pair: pair[0])] #sorts bin_centers in respect to n

# scale values to interval [0,1]
col = sorted_centers - min(sorted_centers)
col /= max(col)
col = sorted(col)
cm = plt.cm.get_cmap('plasma')
for c, p in zip(col, sorted_patches):
    plt.setp(p, 'facecolor', cm(c))

plt.show()
df["SALE PRICE"].skew()
price = np.log(df["SALE PRICE"])
print(price.skew())
plt.hist(price, bins =20)
plt.show()
df.groupby("BOROUGH").mean()['SALE PRICE'].plot(kind='bar') # An alternative to pivot tables
df['TOTAL UNITS'].value_counts()
sns.countplot(x='TOTAL UNITS', data = df, log=True)
plt.show()
df = df[df['TOTAL UNITS'] < 50]
df['TAX CLASS AT TIME OF SALE'].value_counts()
df['BOROUGH'].value_counts()
plt.hist(df['SALE DATE'], bins=20)
plt.show()
sns.countplot(df["SALE DATE"].dt.dayofweek)
df['day'] = df["SALE DATE"].dt.dayofweek
df = df[df["day"] < 5 ]
df.drop(["day"], axis =1, inplace = True)
sns.countplot(df["SALE DATE"].dt.day)
month = np.empty(5 * 7)
for day, count in df["SALE DATE"].dt.day.value_counts().iteritems():
    month[int(day) -1] = count
month = month.reshape((5,7))
sns.heatmap(month)
import joypy
month_df = pd.DataFrame(month)
joypy.joyplot(month_df, overlap=2, colormap=plt.cm.OrRd_r, linecolor='w', linewidth=.5)
plt.show()
axes = df["SALE PRICE"].plot(
    marker='.',
    alpha=0.5,
    linestyle='',
    figsize=(11, 9),
    subplots=True
)
df.info()
df.drop([
    "BUILDING CLASS AT PRESENT",
    "BUILDING CLASS AT TIME OF SALE",
    "NEIGHBORHOOD",
    'TAX CLASS AT PRESENT'
], axis = 1, inplace = True)
df['TAX CLASS AT TIME OF SALE'].value_counts()
df["SALE DATE"] = pd.to_numeric(df["SALE DATE"])
numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
numeric_cols.remove("SALE DATE")

# Removing exceding skewness from features
for col in numeric_cols:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col])
one_hot = ['BUILDING CLASS CATEGORY','TAX CLASS AT TIME OF SALE']
dummies = pd.get_dummies(df[one_hot])
dummies = pd.concat([dummies, pd.get_dummies(df["BOROUGH"])], axis=1) #BOROUGH are integers, so we need to do it seperately
dummies.info(verbose=True, memory_usage=True) #Its nice to check how much memory the dummies will require
df.drop(['BUILDING CLASS CATEGORY', 'TAX CLASS AT TIME OF SALE', 'BOROUGH'], axis = 1, inplace = True)
df = pd.concat([df, dummies], axis =1)
df.info()
from sklearn.model_selection import train_test_split

features = df.drop(["SALE PRICE"], axis = 1)
target = df["SALE PRICE"]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)
# Outlier detection
from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples = 1000)
clf.fit(x_train)

outliers = clf.predict(x_train)
np.unique(outliers, return_counts = True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(lr.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred) ** 0.5)
sns.residplot(y_test, y_pred, color="orange", scatter_kws={"s": 3})
from sklearn.linear_model import Lars
lars = Lars()
lars.fit(x_train, y_train)
y_pred = lars.predict(x_test)
print(lars.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred))
sns.residplot(y_test, y_pred, color="g", scatter_kws={"s": 3})
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 100)
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)
print(dtr.score(x_test,y_test))
print(mean_squared_error(y_test, y_pred))
sns.residplot(y_test, y_pred, color="g", scatter_kws={"s": 3})
from sklearn.ensemble import AdaBoostRegressor
adar = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators = 600)
adar.fit(x_train, y_train)
y_pred = adar.predict(x_test)
adar.score(x_train, y_train)
sns.residplot(y_test, y_pred, color="purple", scatter_kws={"s": 3}) #The best result is plotted in royal purple
fig, ax = plt.subplots(1)
ax.scatter(y_test, y_pred, s=2)
ax.plot([min(y_test.to_list()), max(y_test.to_list())], [min(y_test.to_list()), max(y_test.to_list())], ls='--', c='black', lw=2)
plt.show()
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.05, max_depth = 1, loss='ls')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(clf.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred))
sns.residplot(y_test, y_pred, color="blue", scatter_kws={"s": 3})
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lreg = LinearRegression(normalize = True)
y_pred = cross_val_predict(lreg, x_test, y_test, cv=50)
sns.residplot(y_test, y_pred, color="pink", scatter_kws={"s": 3})
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(features)
feat_pca = pca.transform(features)
fig, ax = plt.subplots(1, figsize=(15,7))
ax.scatter(feat_pca[:,0], feat_pca[:,1], cmap='plasma', c= target, alpha = 0.5, s=2)
#ax.set_ylim(-1e1, 3e1)
plt.show()
pca = PCA(n_components = 3)
pca.fit(features)
feat_pca = pca.transform(features)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feat_pca[:,0], feat_pca[:,1],feat_pca[:,2],cmap='plasma', c = target, alpha = 0.5)
ax.view_init(30,30)
