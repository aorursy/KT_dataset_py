#Number manipulation
import numpy as np

#Data Manipulation
import pandas as pd

#Plotting Libraries
from matplotlib import pyplot as plt
import seaborn as sns
#Some configuration settings
%matplotlib inline
pd.set_option("display.max_columns", 100)
df = pd.read_csv("../input/diamonds.csv", index_col=0)
df.shape
df.head()
df.describe()
df.describe(include=object)
df.isna().any().any()
df.hist(figsize=(20, 20))
plt.show()
for feature in df.dtypes[df.dtypes == object].index:
    sns.countplot(y= feature, data= df)
    plt.show()
for feature in df.dtypes[df.dtypes != object].index:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.violinplot(x= feature, data= df)
    plt.subplot(1, 2, 2)
    sns.boxplot(x= feature, data= df)
    plt.show()
pd.concat([df[df["x"] == 0], df[df["y"] == 0], df[df["z"] == 0]]).drop_duplicates()
len(pd.concat([df[df["x"] == 0], df[df["y"] == 0], df[df["z"] == 0]]).drop_duplicates())
df = df[(df[['x','y','z']] != 0).all(axis=1)]
df[df["z"] == 0]
df["volume"] = df["x"] * df["y"] * df["z"]
df["density"] = df["carat"]*0.2/df["volume"]
df.head()
sns.countplot(y="clarity", data= df)
df.clarity.replace(["VVS1", "VVS2"], "VVS", inplace=True)
df.clarity.replace(["VS1", "VS2"], "VS", inplace= True)
df.clarity.replace(["SI1", "SI2"], "SI", inplace= True)
df.clarity.replace("I1", "I", inplace= True)
sns.countplot(y="clarity", data= df)
color_grades = {
    "Colorless": ["D", "E", "F"],
    "Near Colorless": ["G", "H", "I", "J"],
    "Faint Yellow": ["K", "L", "M"],
    "Very Light Yellow": ["N", "O", "P", "Q", "R"],
    "Light Yellow": ["S", "T", "U", "V", "W", "X", "Y", "Z"]
}
c_l = []
for color in df.color:
    for key, item in color_grades.items():
        if color in item:
            c_l.append(key)
            break
df["ColorGrade"] = c_l
sns.countplot(y="color", data= df)
sns.lmplot(y="carat", x="price", hue="clarity", data= df, fit_reg= False)
sns.lmplot(y="carat", x="price", hue="clarity", data= df[df.clarity == "I"], fit_reg= False)
sns.lmplot(y="carat", x="price", hue="color", data= df, fit_reg= False)
sns.lmplot(y="carat", x="price", hue="cut", data= df, fit_reg= False)
df.head()
new_df = pd.get_dummies(df)
plt.figure(figsize=(20, 20))
corr = new_df.corr()
sns.heatmap(corr*100, cmap="YlGn", annot= True, fmt=".0f")
df.to_csv("cleaned.csv", index= False)
df = pd.get_dummies(df)
df.head()
X = df.drop(["price"], axis= 1).astype(float)
y = df.price.astype(float)
#Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#Building everything
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Saving the model
import pickle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)
print( len(X_train), len(X_test), len(y_train), len(y_test) )
X_train.shape, y_train.shape
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}
rf_hyperparameters = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingregressor__n_estimators": [100, 200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, pred))
    print("MSE:", "\n", mean_squared_error(y_test, pred))
    
print(np.mean(y_test))
plt.scatter(y, fitted_models["rf"].predict(X))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)