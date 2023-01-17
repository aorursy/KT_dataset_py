import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
df = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
df.head()
plt.figure(figsize = (16, 8))
plt.scatter(df['country'], df['gdpp'])
plt.ylabel("GDP")
plt.xlabel("Country")
plt.xticks([])
plt.show()
# removing outliers
df = df.loc[df['gdpp'] <= 50000]
# calculating mean GDP
mean_gdp = np.mean(df['gdpp'])
df['Class'] = 1
# dividing classes
for i in df.index:
    if df.loc[i, 'gdpp'] < mean_gdp:
        df.loc[i, 'Class'] = 0
    elif df.loc[i, 'gdpp'] > 2*mean_gdp:
        df.loc[i, 'Class'] = 2
plt.figure(figsize = (16, 8))
plt.scatter(df['country'], df['gdpp'])
plt.ylabel("GDP")
plt.xlabel("Country")
plt.xticks([])
plt.show()
gdp_above_twoavg = df.loc[df['Class'] == 2]
gdp_above_avg = df.loc[df['Class'] == 1]
gdp_below_avg = df.loc[df['Class'] == 0]
plt.figure(figsize = (16, 8))
plt.scatter(gdp_above_twoavg['country'], gdp_above_twoavg['gdpp'], label = "GDP above 2*avg")
plt.scatter(gdp_above_avg['country'], gdp_above_avg['gdpp'], label = "GDP btw avg-2*avg")
plt.scatter(gdp_below_avg['country'], gdp_below_avg['gdpp'], label = "GDP below avg")
plt.xlabel("Country")
plt.ylabel("GDP")
plt.xticks([])
plt.legend()
plt.show()
y = df['Class']
X = df.drop(['Class', 'country'], axis = 1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
X_train.head()
def plot_classes(X, y, clf, title):
    values = {}
    ranges = {}
    for i in range(0, 8):
        if i == 4:
            pass
        else:
            values[i] = 50
            ranges[i] = 500
    plot_decision_regions(X, y, clf=clf,
                          legend=2, feature_index = [4, 8],
                          filler_feature_values = values,
                          filler_feature_ranges = ranges)
    plt.xlabel("Income per capita")
    plt.ylabel("Country GDP")
    plt.title(title)
    plt.show()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(C = 10, max_iter = 1000).fit(X_train, y_train)
print(f"Train Score: {logreg.score(X_train, y_train)}")
print(f"Test Score: {logreg.score(X_test, y_test)}")
plot_classes(X_train.values, y_train.values, logreg, 'Logistic Regression on train set')
plot_classes(X_test.values, y_test.values, logreg, 'Logistic Regression on test set')
logreg = LogisticRegression(C = 10, max_iter = 1000).fit(X_train_scaled, y_train)
print(f"Train Score: {logreg.score(X_train, y_train)}")
print(f"Test Score: {logreg.score(X_test, y_test)}")
values = {}
ranges = {}
for i in range(0, 8):
    if i == 4:
        pass
    else:
        values[i] = 1
        ranges[i] = 5
plot_decision_regions(X_train_scaled, y_train.values, clf=logreg,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("Logistic Regression on train set with Feature Normalization")
plt.show()
plt.figure()
plot_decision_regions(X_test_scaled, y_test.values, clf=logreg,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("Logistic Regression on test set with Feature Normalization")
plt.show()
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
print(f"Train Score: {knn.score(X_train, y_train)}")
print(f"Test Score: {knn.score(X_test, y_test)}")
plot_classes(X_train.values, y_train.values, knn, 'kNN on train set with k=3')
plot_classes(X_test.values, y_test.values, knn, 'kNN on test set with k=3')
knn.fit(X_train_scaled, y_train)
print(f"Train Score: {knn.score(X_train_scaled, y_train)}")
print(f"Test Score: {knn.score(X_test_scaled, y_test)}")
values = {}
ranges = {}
for i in range(0, 8):
    if i == 4:
        pass
    else:
        values[i] = 0.5
        ranges[i] = 5
plot_decision_regions(X_train_scaled, y_train.values, clf=knn,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("kNN on train set with Feature Normalization & k=3")
plt.show()
plt.figure()
plot_decision_regions(X_test_scaled, y_test.values, clf=knn,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("kNN on test set with Feature Normalization & k=3")
plt.show()
svm = SVC(gamma = 'auto', C = 10).fit(X_train, y_train)
print(f"Train Score: {svm.score(X_train, y_train)}")
print(f"Test Score: {svm.score(X_test, y_test)}")
values = {}
ranges = {}
for i in range(0, 8):
    if i == 4:
        pass
    else:
        values[i] = 100
        ranges[i] = 500
plot_decision_regions(X_train.values, y_train.values, clf=svm,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("SVM on train set")
plt.show()
plt.figure()
plot_decision_regions(X_test.values, y_test.values, clf=svm,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("SVM on test set")
plt.show()
svm = SVC(gamma = 'auto', C = 10).fit(X_train_scaled, y_train)
print(f"Train Score: {svm.score(X_train_scaled, y_train)}")
print(f"Test Score: {svm.score(X_test_scaled, y_test)}")
values = {}
ranges = {}
for i in range(0, 8):
    if i == 4:
        pass
    else:
        values[i] = 0.8
        ranges[i] = 4
plot_decision_regions(X_train_scaled, y_train.values, clf=svm,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("SVM on train set with Feature Normalization")
plt.show()
plt.figure()
plot_decision_regions(X_test_scaled, y_test.values, clf=svm,
                      legend=2, feature_index = [4, 8],
                      filler_feature_values = values,
                      filler_feature_ranges = ranges)
plt.xlabel("Income per capita")
plt.ylabel("Country GDP")
plt.title("SVM on test set with Feature Normalization")
plt.show()
tree = DecisionTreeClassifier().fit(X_train, y_train)
print(f"Train Score: {tree.score(X_train, y_train)}")
print(f"Test Score: {tree.score(X_test, y_test)}")
plot_classes(X_train.values, y_train.values, tree, 'Decision Tree on train set')
plot_classes(X_test.values, y_test.values, tree, 'Decision Tree on test set')
tree = DecisionTreeClassifier().fit(X_train_scaled, y_train)
print(f"Train Score: {tree.score(X_train_scaled, y_train)}")
print(f"Test Score: {tree.score(X_test_scaled, y_test)}")
plot_classes(X_train_scaled, y_train.values, tree, 'Decision Tree on train set with Feature Normalization')
plot_classes(X_test_scaled, y_test.values, tree, 'Decision Tree on test set with Feature Normalization')
poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
logreg = LogisticRegression(C = 10, max_iter = 1000).fit(X_train_poly, y_train)
print(f"Train Score: {logreg.score(X_train_poly, y_train)}")
print(f"Test Score: {logreg.score(X_test_poly, y_test)}")
knn.fit(X_train_poly, y_train)
print(f"Train Score: {knn.score(X_train_poly, y_train)}")
print(f"Test Score: {knn.score(X_test_poly, y_test)}")
svm = SVC(gamma = 5, C = 10).fit(X_train_poly, y_train)
print(f"Train Score: {svm.score(X_train_poly, y_train)}")
print(f"Test Score: {svm.score(X_test_poly, y_test)}")
tree = DecisionTreeClassifier().fit(X_train_poly, y_train)
print(f"Train Score: {tree.score(X_train_poly, y_train)}")
print(f"Test Score: {tree.score(X_test_poly, y_test)}")