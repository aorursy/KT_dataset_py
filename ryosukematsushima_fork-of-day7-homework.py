import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import graphviz
from IPython.display import Image
from sklearn.externals.six import StringIO
from IPython.display import display 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
df_data = pd.read_csv("../input/kc_house_data.csv")
df_data["price"] = df_data["price"]/10**5
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
ex_ver = ["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]
for ver in ex_ver:
    sns.jointplot(x=ver, y="price", data=df_data,kind = 'reg', size = 10)
    plt.show()
df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=0)
df_train["price"] = df_train["price"].astype(np.int64)
from sklearn.ensemble import RandomForestClassifier

X_train = df_train[["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]].values
y_train = df_train["price"].values

clf = RandomForestClassifier(n_estimators=10, max_depth=2, criterion="gini",
                                                 min_samples_leaf=2, min_samples_split=2, random_state=1234)
clf.fit(X_train, y_train)

print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
### MSE,RMSEの算出
X_test = df_test[["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]].values
y_test = df_test["price"].values

y_pred = clf.predict(X_test)

y_test *= 10**5
y_pred *= 10**5 #元の価格の単位に直す

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

X_train = df_train[["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]].values
y_train = df_train["price"].values

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2,
                                                                                 min_samples_leaf=2,
                                                                                 min_samples_split=2, 
                                                                                 random_state=1234,
                                                                                 criterion="gini"),
                                           n_estimators=10, random_state=1234)
clf.fit(X_train, y_train)

# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()


X_test = df_test[["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]].values
y_test = df_test["price"].values

y_pred = clf.predict(X_test)

y_test *= 10**5
y_pred *= 10**5 #元の価格の単位に直す

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
