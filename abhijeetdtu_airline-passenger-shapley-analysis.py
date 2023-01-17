import pandas as pd
df = pd.read_csv("/kaggle/input/train.csv")
tdf = pd.read_csv("/kaggle/input/test.csv")
df = pd.concat([df, tdf])
df.head()
df = df.reset_index()
df
df.shape
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
from sklearn.model_selection import train_test_split
df.columns
df = df.iloc[: , 3:]
cat_cols = ["Gender"  , "Customer Type" ,"Class", "Type of Travel"]
num_cols = [c for c in df.columns if (c not in cat_cols) and (c != "satisfaction")]
cols_to_scale = ["Age" , "Departure Delay in Minutes" , "Arrival Delay in Minutes" ,"Flight Distance"]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df.loc[: , cols_to_scale] = ss.fit_transform(df[cols_to_scale])
df
ohedf = pd.DataFrame(ohe.fit_transform(df[cat_cols]) , columns= ohe.get_feature_names())
df.drop(cat_cols, axis=1)
fdf = ohedf.join(df.drop(cat_cols, axis=1))
fdf.shape
X = fdf.drop("satisfaction" , axis=1)
y= df["satisfaction"].astype("category").cat.codes
X = X.fillna(0)
X
X.shape
X_train , X_test , y_train , y_test = train_test_split(X,y)
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train , y_train)
from sklearn.metrics import accuracy_score
accuracy_score(clf.predict(X_train) , y_train)
accuracy_score(clf.predict(X_test) , y_test)
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
from sklearn.model_selection import cross_validate
cross_validate(clf , X,y)