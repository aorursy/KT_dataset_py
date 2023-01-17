import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.io import output_notebook
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import sklearn
import sklearn.metrics
from sklearn import ensemble
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
output_notebook()
%matplotlib inline
url = "../input/winequality-red.csv"
wine = pd.read_csv(url)
wine.head(n=5)
print("Shape of Red Wine dataset: {s}".format(s = wine.shape))
print("Column headers/names: {s}".format(s = list(wine)))
# Now, let's check the information about different variables/column from the dataset:
wine.info()
# Let's look at the summary of the dataset,
wine.describe()
wine.isnull().sum()
wine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)
wine.head(n=5)
wine['quality'].unique()
wine.quality.value_counts().sort_index()
sns.countplot(x='quality', data=wine)
conditions = [
    (wine['quality'] >= 7),
    (wine['quality'] <= 4)
]
rating = ['good', 'bad']
wine['rating'] = np.select(conditions, rating, default='average')
wine.rating.value_counts()
wine.groupby('rating').mean()
correlation = wine.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
correlation['quality'].sort_values(ascending=False)
bx = sns.boxplot(x="rating", y='sulphates', data = wine)
bx.set(xlabel='Wine Ratings', ylabel='Sulphates', title='Sulphates in different types of Wine ratings')
bx = sns.violinplot(x="rating", y='citric_acid', data = wine)
bx.set(xlabel='Wine Ratings', ylabel='Citric Acid', title='Xitric_acid in different types of Wine ratings')
bx = sns.boxplot(x="rating", y='fixed_acidity', data = wine)
bx.set(xlabel='Wine Ratings', ylabel='Fixed Acidity', title='Fixed Acidity in different types of Wine ratings')
bx = sns.swarmplot(x="rating", y="pH", data = wine);
bx.set(xlabel='Wine Ratings', ylabel='pH', title='pH in different types of Wine ratings')
g = sns.pairplot(wine[['alcohol', 'sulphates', 'citric_acid',  'fixed_acidity', 'residual_sugar', "rating"]], hue="rating", diag_kind="hist")
for ax in g.axes.flat: 
    plt.setp(ax.get_xticklabels(), rotation=45)
g1 = sns.FacetGrid(wine, col="rating", col_wrap=6)
g1.map(sns.kdeplot, "alcohol")
g2 = sns.FacetGrid(wine, col="rating", col_wrap=6)
g2.map(sns.kdeplot, "sulphates")
g3 = sns.FacetGrid(wine, col="rating", col_wrap=6)
g3.map(sns.kdeplot, "citric_acid")
g4 = sns.FacetGrid(wine, col="rating", col_wrap=6)
g4.map(sns.kdeplot, "fixed_acidity")
g5 = sns.FacetGrid(wine, col="rating", col_wrap=6)
g5.map(sns.kdeplot, "residual_sugar")
sns.lmplot(x = "alcohol", y = "residual_sugar", col = "rating", data = wine)
y,X = dmatrices('quality ~ alcohol', data=wine, return_type='dataframe')
print("X:", type(X))
print(X.columns)
model=smf.OLS(y, X)
result=model.fit()
result.summary()
wine['rate_code'] = (wine['quality'] > 4).astype(np.float32)
y, X = dmatrices('rate_code ~ alcohol', data = wine)
sns.distplot(X[y[:,0] > 0, 1])
sns.distplot(X[y[:,0] == 0, 1])
model = smf.Logit(y, X)
result = model.fit()
result.summary2()
yhat = result.predict(X)
sns.distplot(yhat[y[:,0] > 0])
sns.distplot(yhat[y[:,0] == 0])
yhat = result.predict(X) > 0.955
print(sklearn.metrics.classification_report(y, yhat))
model = sklearn.linear_model.LogisticRegression()
y,X = dmatrices('rate_code ~ alcohol + sulphates + citric_acid + fixed_acidity', data = wine)
model.fit(X, y)
yhat = model.predict(X)
print(sklearn.metrics.classification_report(y, yhat))
y, X = dmatrices('rate_code ~ alcohol', data = wine)
model = sklearn.ensemble.RandomForestClassifier()
model.fit(X, y)
yhat = model.predict(X)
print(sklearn.metrics.classification_report(y, yhat))
