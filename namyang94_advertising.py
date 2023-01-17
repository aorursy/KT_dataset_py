import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style('whitegrid')
df = pd.read_csv('../input/advertising.csv')
df.head()
df.describe()
np.sum(df.isnull(), axis = 0)
sns.heatmap(df.corr());
sns.scatterplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = df, hue = 'Clicked on Ad');
sns.distplot(df[df['Clicked on Ad'] == 0]['Daily Internet Usage'], label = 'Did not click');

sns.distplot(df[df['Clicked on Ad'] == 1]['Daily Internet Usage'], label = 'Clicked');

plt.legend();
sns.distplot(df[df['Clicked on Ad'] == 0]['Daily Time Spent on Site'], label = 'Did not click');

sns.distplot(df[df['Clicked on Ad'] == 1]['Daily Time Spent on Site'], label = 'Clicked');

plt.legend();
sns.boxplot(x = 'Clicked on Ad', y = 'Age', data = df);
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
x = df[['Daily Time Spent on Site', 'Daily Internet Usage']]

y = df['Clicked on Ad']



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
pipe = Pipeline([

    ('scaler', StandardScaler()),

    ('clf', LogisticRegression())

])
pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
from sklearn.metrics import classification_report as cr, confusion_matrix as cm
print(cm(y_test,y_pred))

print(cr(y_test,y_pred))
def plotBoundary(x,classifier):

    x1_max = np.max(x['Daily Time Spent on Site']) + 1

    x1_min = np.min(x['Daily Time Spent on Site']) - 1

    x2_max = np.max(x['Daily Internet Usage']) + 1

    x2_min = np.min(x['Daily Internet Usage']) - 1



    xx1,xx2 = np.meshgrid( np.arange(x1_min,x1_max,0.1), np.arange(x2_min, x2_max,0.1))

    features = np.array([xx1.ravel(), xx2.ravel()]).T

    predictions = classifier.predict(features).reshape(xx1.shape)

    plt.contour(xx1,xx2,predictions);
plotBoundary(x,pipe)

sns.scatterplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = x_test, hue = y_test);
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
pipe2 = Pipeline([

    ('scaler', StandardScaler()),

    ('clf', KNeighborsClassifier())

])



param_grid = {

    'clf__n_neighbors': list(range(5,55,5))

}



grid = GridSearchCV(pipe2, param_grid = param_grid, cv = 5)

grid.fit(x_train,y_train)
print(grid.best_params_)

print(grid.best_score_)
y_pred2 = grid.predict(x_test)

print(cm(y_test,y_pred2))

print(cr(y_test,y_pred2))
plotBoundary(x,grid)

sns.scatterplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = x_test, hue = y_test);