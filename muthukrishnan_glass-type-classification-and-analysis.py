import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/glass.csv')
features = data.loc[:,data.columns != 'Type']
target = data.loc[:,'Type']
features.describe()
corrmat = features.corr()
corrmat
corrmat.iloc[0,:].plot(kind='bar')
import matplotlib.pyplot as plt
chartlocation = 0
plt.figure(figsize=(15,12))
columns = np.copy(corrmat.columns.values)
for index, row in corrmat.iterrows():
    column_name = columns[chartlocation]
    chartlocation = chartlocation + 1
    plt.subplot(3,3,chartlocation)
    row.drop(index).plot(kind='bar', title=column_name)
chartlocation = 0
plt.figure(figsize=(15,12))
columns = features.columns.values
for column in columns:
    chartlocation = chartlocation + 1
    plt.subplot(3,3,chartlocation)
    features.boxplot(column=column)
fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(1, 1, 1)
features.hist(ax=ax)
plt.show()
features.skew().plot(kind='bar')
plt.show()
def find_outlier_fences_IQR(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    return [fence_low, fence_high]

fences = {}
for column in features.columns.values:
    fences[column] = find_outlier_fences_IQR(features, column)
print(fences)

#lets find rows with more than one or two outliers and drop them.
outliers_index = []
for index, row in features.iterrows():
    outliers_detected = 0
    for column in features.columns.values:
        fence_low = fences[column][0]
        fence_high = fences[column][1]
        if row[column] < fence_low or row[column] > fence_high:
            outliers_detected = outliers_detected + 1
    
    if outliers_detected > 1:
        outliers_index.append(index)

print("\nthere are %d rows found with more than 1 outlier" %(len(outliers_index)))
outliers_removed_featureset = features.drop(outliers_index)
outliers_removed_targetset = target.drop(outliers_index)
from sklearn.preprocessing import StandardScaler
autoscaler = StandardScaler()
features_scaled = autoscaler.fit_transform(outliers_removed_featureset)
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig)
X_reduced = PCA(n_components=3).fit_transform(features_scaled.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=outliers_removed_targetset)
plt.title("Priciple components 3")
plt.show()
X_reduced = PCA(n_components=2).fit_transform(features_scaled.data)
plt.title("Priciple components 2")
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=outliers_removed_targetset)
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_scaled,outliers_removed_targetset, test_size=0.20, random_state=42)
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
feature_with_importance = pd.DataFrame()
feature_with_importance['columns'] = outliers_removed_featureset.columns
feature_with_importance['importance'] = clf.feature_importances_
feature_with_importance.sort_values(by=['importance'], ascending=True, inplace=True)
feature_with_importance.set_index('columns', inplace=True)
feature_with_importance.plot(kind='bar')
plt.show()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

models = [
    SVC(),
    KNeighborsClassifier(),
    GradientBoostingClassifier(n_estimators=100)
]

for model in models:
    clf = model.fit(X_train, y_train)
    print('score:',clf.score(X_test,y_test))
from sklearn.model_selection import GridSearchCV
parameter_grid = {
    'C' :  [1, 10, 100, 1000, 1500],
    'gamma' : [0.001, 0.01, 0.1, 1],
    'kernel': [ 'rbf', 'sigmoid']
}

gsv = GridSearchCV(SVC(),parameter_grid)
gsv = gsv.fit(X_train, y_train)
print('score:',gsv.score(X_test,y_test))
gsv.best_params_
