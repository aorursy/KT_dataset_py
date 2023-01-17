import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Warning
import warnings
warnings.filterwarnings("ignore")
column_name = ["MPG", "Cylinders", "Displacement", "Horsepower","Weight","Acceleration","Model Year","Origin"]
df_raw = pd.read_csv("../input/autompg-data/auto-mpg.data", names = column_name, na_values = "?", comment = "\t",sep = " ", skipinitialspace = True)
data = df_raw.copy()
data.head()
data = data.rename(columns = {"MPG":"target"})
data.head()
data.shape
data.info()
#Horsepower içerisinde 6 tane missing value var 
#Kategorik veri yok, daha sonra origin kategorik olmalı
data.describe()
# in acceleration little bit skewness

#model year no skewness

#displacement, horsepower, target, cylinder have skewness


data.isna().sum()
sns.distplot(data.Horsepower);
data.isna().sum()
data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())
data.isna().sum()
sns.distplot(data.Horsepower);
for feature in data.select_dtypes("number").columns:
    if feature == "model_year":
        continue
    plt.figure(figsize=(16,5))
    sns.distplot(data[feature], hist_kws={"rwidth": 0.9})
    plt.xlim(data[feature].min(), data[feature].max())
    data[feature].plot(kind="hist", rwidth=0.9, bins=50)
    plt.title(f"{feature.capitalize()}")
    plt.tight_layout()
    plt.show()
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()
threshold = 0.75
filtre = np.abs(corr_matrix["target"])>threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.show()
sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()
sns.countplot(data["Cylinders"]);
print(data["Cylinders"].value_counts())
sns.countplot(data["Origin"]);
print(data["Origin"].value_counts())
#boxplot
for c in data.columns:
    plt.figure()
    sns.boxplot( x = c, data = data, orient = "v")

## BoxPlot

th = 2

Q1_acc = data["Acceleration"].quantile(0.25)
Q3_acc = data["Acceleration"].quantile(0.75)
IQR_acc = Q3_acc - Q1_acc 

lower_bound_acc = Q1_acc - th * IQR_acc
upper_bound_acc = Q3_acc + th * IQR_acc

outliers_lower_vector_acc = (data["Acceleration"] < (lower_bound_acc))
outliers_upper_vector_acc = (data["Acceleration"] > (upper_bound_acc))
outliers_vector_acc = (outliers_lower_vector_acc) | (outliers_upper_vector_acc)
#outliers_vector_acc = (data["Acceleration"] < lower_bound_acc) | (data["Acceleration"] > upper_bound_acc)

print("Total Sample Size : ", data["Acceleration"].shape[0])
print("Lower Bound : ", "%.2f" %(lower_bound_acc))
print("Upper Bound : ", "%.2f" %(upper_bound_acc))
print("Total Outlier for Threshold {} * IQR : ".format(th), data["Acceleration"][outliers_vector_acc].shape[0])
#Boxplot 

th = 2

Q1_hp = data["Horsepower"].quantile(0.25)
Q3_hp = data["Horsepower"].quantile(0.75)
IQR_hp = Q3_hp - Q1_hp

lower_bound_hp = Q1_hp - th * IQR_hp
upper_bound_hp = Q3_hp + th * IQR_hp

outliers_lower_vector_hp = ( data["Horsepower"] < (lower_bound_hp))
outliers_upper_vector_hp = ( data["Horsepower"] > (upper_bound_hp))
outliers_vector_hp = (outliers_lower_vector_hp) | (outliers_upper_vector_hp)


print("Total Sample Size :", data["Horsepower"].shape[0])
print("Lower Bound : ", "%.2f" % (lower_bound_hp))
print("Upper Bound : ", "%.2f" % (upper_bound_hp))
print("Total Outlier for Threshold {} * IQR :".format(th), data["Horsepower"][outliers_vector_hp].shape[0])
data = data[~outliers_vector_acc]
data = data[~outliers_vector_hp]
data.shape
#target dependent variable

sns.distplot(data.target, fit = norm);
(mu, sigma) = norm.fit(data["target"])
print("mu : {} | sigma : {}".format(mu,sigma))
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()
data["target"] = np.log1p(data["target"])
sns.distplot(data.target, fit = norm);
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()
# Independent Variables

skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False )

skewness = pd.DataFrame(skewed_feats, columns=["skewed_feats"])
skewness
data["Cylinders"] = data["Cylinders"].astype(str)  
data["Origin"] = data["Origin"].astype(str)

data = pd.get_dummies(data)
data.head()
x = data.drop(["target"], axis = 1)
y = data.target
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42)
print("X train shape : {} | X test shape : {} \nY train shape : {} | Y test shape : {} ".format(X_train.shape[0],X_test.shape[0],Y_train.shape[0],Y_test.shape[0]))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef: ",lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Linear Regression MSE: ",mse)

ridge = Ridge(random_state =42, max_iter = 10000)
alphas = np.logspace(-4,-0.5,30)
tuned_parameters = [{"alpha":alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef : ",clf.best_estimator_.coef_)

ridge = clf.best_estimator_

print("Ridge Best Estimator : ", ridge)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)

print("Ridge MSE : ",mse)
print("-------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")
lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Lasso Coef: ",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ",lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Lasso MSE: ",mse)
print("---------------------------------------------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")
parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, Y_train)


print("ElasticNet Coef: ",clf.best_estimator_.coef_)
print("ElasticNet Best Estimator: ",clf.best_estimator_)


y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("ElasticNet MSE: ",mse)
scaler = RobustScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef: ",lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Linear Regression MSE: ",mse)

ridge = Ridge(random_state =42, max_iter = 10000)
alphas = np.logspace(-4,-0.5,30)
tuned_parameters = [{"alpha":alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef : ",clf.best_estimator_.coef_)

ridge = clf.best_estimator_

print("Ridge Best Estimator : ", ridge)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)

print("Ridge MSE : ",mse)
print("-------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")
lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Lasso Coef: ",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ",lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Lasso MSE: ",mse)
print("---------------------------------------------------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")
parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, Y_train)


print("ElasticNet Coef: ",clf.best_estimator_.coef_)
print("ElasticNet Best Estimator: ",clf.best_estimator_)


y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("ElasticNet MSE: ",mse)