import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from plotnine import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
wine_df = pd.read_csv('../input/winequality-red.csv')
wine_df.describe()
wine_df.info()
wine_df.head()
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Wine Quality Correlation Heatmap")
corr = wine_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           center=0)

corr = wine_df.corr()
corr
bins_quality = [0, 5.5, 6, 7, 8]
group_quality = ['bad','average','above average','good']
wine_df['binned_quality'] = pd.cut(wine_df['quality'], bins = bins_quality, labels = group_quality)
wine_df.head()
bins_volatile_acid = [0, 0.5, 0.7, 2]
group_volatile_acid = ['low','average','high']
wine_df['binned_volatile_acid'] = pd.cut(wine_df['volatile acidity'], bins = bins_volatile_acid, labels = group_volatile_acid)
wine_df.head()
bins_citric_acid = [-0.1, 0.3, 0.6, 1]
group_citric_acid = ['low','average','high']
wine_df['binned_citric_acid'] = pd.cut(wine_df['citric acid'], bins = bins_citric_acid, labels = group_citric_acid)
wine_df.head()
bins_alcohol = [0, 10, 11, 15]
group_alcohol = ['low','average','high']
wine_df['binned_alcohol'] = pd.cut(wine_df['alcohol'], bins = bins_alcohol, labels = group_alcohol)
wine_df.head()
wine_df.info()
ggplot(wine_df, aes(x='sulphates', y='chlorides', color='binned_alcohol', shape='binned_volatile_acid')) + geom_point() +\
    facet_wrap('binned_quality', ncol=2) + scale_color_brewer(type = 'qual', palette = 'Dark2')
ggplot(wine_df, aes(x='sulphates', y='chlorides', color='binned_alcohol', size='binned_citric_acid')) + geom_point() +\
    facet_wrap('binned_quality', ncol=2) + scale_color_brewer(type = 'qual', palette = 'Dark2')
new_wine_df = pd.read_csv('../input/winequality-red.csv')
features_df = new_wine_df.drop(['quality'], axis = 1)
y = new_wine_df['quality']
features_df.head()
X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

y_pred_regression = [round(x) for x in predictions]
y_pred_regression[0:10]
accuracy_score(y_test, y_pred_regression)
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
y_pred_knn = knn.predict(X_test)  

accuracy_score(y_test, y_pred_knn)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

accuracy_score(y_test, y_pred_rf)
plt.scatter(y_test, y_pred_rf)
plt.xlabel("True Values")
plt.ylabel("Predicted Results")
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, features_df, y, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,
                            'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)