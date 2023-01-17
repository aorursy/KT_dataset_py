import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from scipy import stats



# Prevent Pandas from truncating displayed dataframes

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



sns.set(style="white", font_scale=1.2)

plt.rcParams["figure.figsize"] = [10,8]
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
submission.head()
print("Training Size: {} observations, {} features\nTest Size: {} observations, {} features\n".format(train.shape[0], train.shape[1], test.shape[0], test.shape[1]))
set(train.columns) - set(test.columns)
train.describe()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.columns
categorical_features = train.select_dtypes(include=[np.object])

categorical_features.columns
unique_categories = pd.DataFrame(index=categorical_features.columns, columns=["TrainCount", "TestCount"])

for c in categorical_features.columns:

    unique_categories.loc[c, "TrainCount"] = len(train[c].value_counts())

    unique_categories.loc[c, "TestCount"] = len(test[c].value_counts())

    

unique_categories = unique_categories.sort_values(by="TrainCount", ascending=False)

unique_categories.head()
temp = pd.melt(unique_categories.reset_index(), id_vars="index")

g = sns.catplot(y="index", x="value", hue="variable", data=temp, kind="bar", height=9)

g.set_ylabels("Count")

g.set_xlabels("Categorical Variable")

g.set_xticklabels(rotation=90)

plt.title("Number of Unique Categories by Feature")

plt.show()
nulls = train.isnull().sum()[train.isnull().sum() > 0].sort_values(ascending=False).to_frame().rename(columns={0: "MissingVals"})

nulls["MissingValsPct"] = nulls["MissingVals"] / len(train)

nulls
sns.barplot(y=nulls.index, x=nulls["MissingValsPct"], orient="h")

plt.title("% of Values Missing by Feature")

plt.show()
msno.matrix(train, labels=True)

plt.show()
z_threshold = 3

z = pd.DataFrame(np.abs(stats.zscore(train[numeric_features.columns])))

outlier_rows = z[z[z > z_threshold].any(axis=1)] # Rows with outliers

print("# Rows with potential outliers: {}".format(len(outlier_rows)))

outlier_rows.head()
fig, ax = plt.subplots(1,3, figsize=(15,5))

sns.distplot(train["SalePrice"], ax=ax[0], fit=stats.norm)

sns.boxplot(train["SalePrice"], orient='v', ax=ax[1])

stats.probplot(train["SalePrice"], plot=plt)



ax[0].set_title("SalePrice Distribution vs. Normal Distribution")

ax[1].set_title("Boxplot of SalePrice")

ax[2].set_title("Q-Q Plot of SalePrice")

ax[0].set_ylabel("SalePrice")

ax[1].set_xlabel("All Homes")



for a in ax:

    for label in a.get_xticklabels():

        label.set_rotation(90)

plt.tight_layout()

plt.show()
log_SalePrice = np.log1p(train["SalePrice"]) # Applies log(1+x) to all elements of column

fig, ax = plt.subplots(1,3, figsize=(15,5))

sns.distplot(log_SalePrice, ax=ax[0], fit=stats.norm)

sns.boxplot(log_SalePrice, orient='v', ax=ax[1])

stats.probplot(log_SalePrice, plot=plt)



ax[0].set_title("Log(SalePrice + 1) vs. Normal Distribution")

ax[1].set_title("Boxplot of Log(SalePrice + 1)")

ax[2].set_title("Q-Q Plot of Log(SalePrice + 1)")

ax[0].set_ylabel("SalePrice")

ax[1].set_xlabel("All Homes")





plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(15,12))

corr_matrix = train.corr()

sns.heatmap(corr_matrix, square=True)

plt.title("Heatmap of All Numerical Features")

plt.show()
fig = plt.figure(figsize=(15,12))

sns.heatmap(corr_matrix[(corr_matrix > 0.5) | (corr_matrix < -0.5)], annot=True, annot_kws={"size": 9}, linewidths=0.1, square=True)

plt.title("Heatmap of Highest Correlated Features")

plt.show()
k = 11 #number of variables for heatmap (including SalePrice)

cols_positive = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols_positive].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_positive.values, xticklabels=cols_positive.values)

plt.show()
k = 10 #number of variables for heatmap

cols_negative = np.append(['SalePrice'], corr_matrix.nsmallest(k, 'SalePrice')['SalePrice'].index.values)

cm = np.corrcoef(train[cols_negative].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_negative, xticklabels=cols_negative)

plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']

sns.pairplot(train[cols_positive], diag_kind='kde')

plt.show()
x = train.copy()

for c in categorical_features.columns:

    x[c] = x[c].astype('category')

    if x[c].isnull().any():

        x[c] = x[c].cat.add_categories(['Missing'])

        x[c] = x[c].fillna('Missing')

x["SalePrice"] = train["SalePrice"]

x.head()
def boxplot_custom(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x = plt.xticks(rotation=90)



df = pd.melt(x, id_vars=["SalePrice"], value_vars=categorical_features)

g = sns.FacetGrid(df, col="variable", col_wrap=3, sharex=False, sharey=False, height=5)

g = g.map(boxplot_custom, "value", "SalePrice")

plt.show()