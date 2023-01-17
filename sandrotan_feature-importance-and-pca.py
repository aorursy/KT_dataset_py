import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load train and test sets 

X_train = pd.read_csv('../input/X_train_preprocessed.csv')

X_test = pd.read_csv('../input/X_test_preprocessed.csv').values

y_train = pd.read_csv('../input/y_train_preprocessed.csv').values

print(X_train.shape, y_train.shape)
# Create a RF regressor
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train.ravel())
feature_imp = pd.Series(clf.feature_importances_, index=X_train.keys()).sort_values(ascending=False)
print(feature_imp)
print(sum(feature_imp[:20]))
# Creating a bar plot
sns.barplot(x=feature_imp[:10], y=feature_imp[:10].index)

# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# Create PCA transformer that retains 99.9% of original variance
from sklearn.decomposition import PCA

pca = PCA(0.999)
X_train_pca = pca.fit_transform(X_train)
print(X_train_pca.shape)
from sklearn.model_selection import KFold, cross_val_score

kfolds = KFold(n_splits=10, shuffle=True, random_state=1)

def cv_r2(model, train, test):
    r2 = cross_val_score(model,train, test, scoring="r2", cv=kfolds, n_jobs=-1)
    return np.mean(r2)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=20000, learning_rate=0.02, max_depth=3, max_features='sqrt',
                                min_samples_leaf=5, min_samples_split=5, loss='ls')


# Compare the results with and without PCA
cv_r2(gbr, X_train, y_train)

cv_r2(gbr, X_train_pca, y_train)
