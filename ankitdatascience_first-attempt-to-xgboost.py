import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
import matplotlib.pyplot as plt
!pip install kneed
# kneed is not installed in kaggle. uncomment the above line.
from kneed import KneeLocator
from sklearn.linear_model import LassoCV
import xgboost as xg
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
dataset = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv", index_col = 0)
dataset.head()
missing_info = dataset.drop("target", axis = 1).isna().sum()
missing_info[missing_info > 0]
dtype_column = dataset.drop("target", axis = 1).dtypes
len(dtype_column[dtype_column == 'float64'])
dataset.target.value_counts()
# Scaling the data is quite important in PCA. So let's do that first.
X = scale(dataset.drop('target', axis = 1).values) 

# Fitting a PCA
pca200comp = PCA(n_components = 200).fit(X)

# Plotting the variance explained by each component.
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (14,5))
ax1.plot(list(range(1,201)),pca200comp.explained_variance_ratio_, marker = "o")
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Prop. Variance Explained')

# Plotting the cumulative variance explained by each component.
ax2.plot(list(range(1,201)),pca200comp.explained_variance_ratio_.cumsum(), marker = "o")
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Cumulative Prop. Variance Explained')

plt.show()
# We can find the elbow using KneeLocator.
kl = KneeLocator(range(1, 201), pca200comp.explained_variance_ratio_, curve="convex", direction="decreasing")
kl.elbow
# Code for fitting ridge regression model for different values of lambda.
X = dataset.drop('target', axis = 1).values
y = np.array(dataset['target'])

# Scaling the variables
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

#lambda_range = np.linspace(1,1000,100) # Setting a range of lambda

cv_Lasso = LassoCV().fit(X,y) # CV is 5 fold by default.
print(cv_Lasso.alpha_)
print(len(cv_Lasso.coef_[cv_Lasso.coef_ == 0]))
Lasso_coef = pd.DataFrame(cv_Lasso.coef_, index = dataset.drop('target', axis = 1).columns, columns = ["Coef"])
Lasso_coef["abs_coef"] = np.abs(Lasso_coef["Coef"])
Lasso_coef.sort_values(["abs_coef"], ascending = False, inplace = True)
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 7)

# Running a XGBoost with default settings.
model = xg.XGBClassifier()
model.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# Creating a confusion matrix 
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
# Selecting the top 25 variables.
data_top_lasso = dataset[Lasso_coef[:75].index]
data_top_lasso = data_top_lasso.values

# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(data_top_lasso, y, test_size = 0.33, random_state = 7)

# Running a XGBoost with default settings.
model = xg.XGBClassifier(tree_method='gpu_hist')
model.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# Creating a confusion matrix 
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 7)

classifier = xg.XGBClassifier(n_thread = -1, tree_method='gpu_hist')
param_grid = {
    "n_estimators" : np.arange(100, 500, 50),
    "max_depth" : np.arange(1, 20, 3),
    "colsample_bytree": np.arange(0.5,1, 0.1),
    "criterion": ["gini",'entropy']
}
model = RandomizedSearchCV(estimator = classifier,
                          param_distributions = param_grid,
                          n_iter = 10,
                          scoring = "accuracy",
                          verbose = 10,
                          n_jobs = -1,
                          cv = 5)
model.fit(X_train, y_train)
model.best_score_
print(model.best_estimator_.get_params())
accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
