import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_red_wine = pd.read_csv('../input/winequality-red.csv', sep=',')
df_white_wine = pd.read_csv('../input/winequality-white.csv', sep=',')
df_red_wine.head()
df_white_wine.head()
df_red_wine.info()
df_white_wine.info()
#df = df_red_wine
df = pd.concat([df_red_wine, df_white_wine], ignore_index=True)
df.plot(kind='box', fontsize = 10, figsize= (20, 10), subplots = True)
plt.show()
df['quality'].plot(kind='hist', fontsize = 20, figsize= (20,10))
plt.xlabel('Quality', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()
df.quality.unique()
#Splitting Data
df_quality_3 = df[df['quality'] == 3]
df_quality_4 = df[df['quality'] == 4]
df_quality_5 = df[df['quality'] == 5]
df_quality_6 = df[df['quality'] == 6]
df_quality_7 = df[df['quality'] == 7]
df_quality_8 = df[df['quality'] == 8]
df_quality_9 = df[df['quality'] == 9]

#Under sampling quality 5 & 6 to quality 7
df_sample_5 = df_quality_5.sample(300)
df_sample_6 = df_quality_6.sample(300)
df_sample_7 = df_quality_7.sample(300)
df = pd.concat([df_quality_3, df_quality_4, df_sample_5, df_sample_6, df_sample_7, df_quality_8, df_quality_9], ignore_index=True)

df['quality'].plot(kind='hist', fontsize = 20, figsize= (20,10))
plt.xlabel('Quality', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()

df.quality.replace((3, 4, 5, 6, 7, 8, 9), (0, 0, 0, 0, 1, 1, 1), inplace=True)
df['quality'].plot(kind='hist', fontsize = 20, figsize= (20,10))
plt.xlabel('Quality', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()
measurements = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
for measurement in measurements:
    df[measurement].plot(kind='hist', bins=50, fontsize = 20, figsize= (20, 10), edgecolor='black', linewidth=1.2)
    plt.xlabel(measurement, fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.show()
df_translated=pd.DataFrame()

for measurement in measurements:
    df_translated[measurement] = df[measurement].apply(lambda x: x+1-(df[measurement].min()))

df_translated.head()

from scipy import stats
df_translated_normalized = pd.DataFrame()
for measurement in measurements:
    df_translated_normalized[measurement] = stats.boxcox(df_translated[measurement])[0]
    df_translated_normalized[measurement].plot(kind='hist', bins=50, fontsize = 20, figsize= (20, 10), edgecolor='black', linewidth=1.2)
    plt.xlabel(measurement, fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.show()
df_translated_normalized['quality'] = df['quality']
grouped = df_translated_normalized.groupby('quality')[measurements].mean()
grouped
for measurement in measurements:
    plt.bar(grouped.index, grouped[measurement])
    plt.xlabel(grouped.index.name)
    plt.ylabel(measurement)
    plt.show()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
X = df_translated_normalized[measurements].values
y = df_translated_normalized['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

k = np.arange(1,30)
params_neighbors = {'knn__n_neighbors' : k}

steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)


cv_knn = GridSearchCV(pipeline, param_grid=params_neighbors, cv=5)
cv_knn.fit(X_train, y_train)
print("Tuned k neighbors {}".format(cv_knn.best_params_))
print("Highest score {}".format(cv_knn.best_score_))
c_space= np.logspace(-5, 8, 15)
param_C = {'logreg__C': c_space}

steps = [('scaler', StandardScaler()),
        ('logreg', LogisticRegression())]
pipeline = Pipeline(steps)


cv_logreg = GridSearchCV(pipeline, param_grid=param_C, cv=5)

cv_logreg.fit(X_train, y_train)

print("Tuned C parameter {}".format(cv_logreg.best_params_))
print("Highest score {}".format(cv_logreg.best_score_))
maxdepth_space = np.arange(1, 10, 1)
param_maxdepth = {'DTreeReg__max_depth': maxdepth_space}

steps = [('scaler', StandardScaler()),
        ('DTreeReg', DecisionTreeRegressor())]
pipeline = Pipeline(steps)
pipeline.get_params().keys()
cv_DecTree = GridSearchCV(pipeline, param_grid=param_maxdepth, cv=5)

cv_DecTree.fit(X_train, y_train)

print("Tuned C parameter {}".format(cv_DecTree.best_params_))
print("Highest score {}".format(cv_DecTree.best_score_))
knn_tuned = KNeighborsClassifier(n_neighbors = 1)
logreg_tuned = LogisticRegression(C=31.6)
knn_tuned.fit(X_train, y_train)
knn_tuned.predict(X_test)
knn_tuned.score(X_test, y_test)


