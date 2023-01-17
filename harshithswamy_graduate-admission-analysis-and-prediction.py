import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error





from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
data_frame = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data_frame.head()
data_frame.drop(labels="Serial No.", axis=1, inplace=True)
data_frame.shape
data_frame.describe()
data_frame.info()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.countplot(x="University Rating", data=data_frame)

plt.title("University Rating Count Plot")

plt.subplot(2,2,2)

sns.countplot(x="SOP", data=data_frame)

plt.title("SOP Count Plot")

plt.subplot(2,2,3)

sns.countplot(x="LOR ", data=data_frame)

plt.title("LOR Count Plot")

plt.subplot(2,2,4)

sns.countplot(x="Research", data=data_frame)

plt.title("Research Count Plot")

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(data_frame["CGPA"], color="blue")

plt.title("CGPA Distribution")

plt.subplot(2,2,2)

sns.distplot(data_frame["TOEFL Score"], color="red")

plt.title("TOEFL Score Distribution")

plt.subplot(2,2,3)

sns.distplot(data_frame["GRE Score"], color="orange")

plt.title("GRE Score Distribution")

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.scatterplot(x="University Rating", y="CGPA", data=data_frame, color="blue")

plt.title("University Rating vs CGPA")

plt.subplot(2,2,2)

sns.scatterplot(x="University Rating", y="TOEFL Score", data=data_frame, color="red")

plt.title("University Rating vs TOEFL Score")

plt.subplot(2,2,3)

sns.scatterplot(x="University Rating", y="GRE Score", data=data_frame, color="green")

plt.title("University Rating vs GRE Score")

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.scatterplot(x="GRE Score", y="CGPA", data=data_frame, color="blue")

plt.title("GRE Score vs CGPA")

plt.subplot(2,2,2)

sns.scatterplot(x="TOEFL Score", y="CGPA", data=data_frame, color="red")

plt.title("TOEFL Score vs CGPA")

plt.subplot(2,2,3)

sns.scatterplot(x="GRE Score", y="CGPA", data=data_frame, hue="Research")

plt.title("GRE Score vs CGPA With Research")

plt.subplot(2,2,4)

sns.scatterplot(x="TOEFL Score", y="CGPA", data=data_frame, hue="Research")

plt.title("TOEFL Score vs CGPA With Research")

plt.show()
plt.rcParams['figure.figsize'] = (7, 5)

stats.probplot(data_frame['Chance of Admit '], plot = plt)

plt.show()
sns.heatmap(data_frame.corr(), annot=True, square=True, cmap="YlGnBu")

plt.title("Correlation Between Fetures")
X = data_frame.drop("Chance of Admit ", axis=1)

Y = data_frame["Chance of Admit "].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print("X Train Shape:", x_train.shape)

print("X Test Shape:", x_test.shape)

print("Y Train Shape:", y_train.shape)

print("Y Test Shape:", y_test.shape)
def model_evaluation(model_name, y_test, y_preds):

    mse = mean_squared_error(y_test, y_preds)

    rmse = np.sqrt(mse)

    r2_value = r2_score(y_test, y_preds)

    

    print("************* {} Model *************".format(model_name))

    print("MSE Value: {}".format(mse))

    print("RMSE Value: {}".format(rmse))

    print("R2 Score: {}".format(r2_value))

    print("************************************")

    

    x = np.arange(0, len(y_test))

    plt.scatter(x, list(y_test), label="Actual")

    plt.scatter(x, list(y_preds), marker="*", label="Predicted")

    plt.title("Actual vs Predicted Values")

    plt.legend()

    plt.show()

    return r2_value
svr_model = SVR(kernel="poly")

svr_model.fit(x_train, y_train)

svr_preds = svr_model.predict(x_test)

svr_score = model_evaluation("Support Vector Regression", y_test, svr_preds)
lr_model = LinearRegression()

lr_model.fit(x_train, y_train)

lr_preds = lr_model.predict(x_test)

lr_score = model_evaluation("Linear Regression", y_test, lr_preds)
rf_model = RandomForestRegressor()

rf_model.fit(x_train, y_train)

rf_preds = rf_model.predict(x_test)

rf_score = model_evaluation("Random Forest Regression", y_test, rf_preds)
xgb_model = XGBRegressor()

xgb_model.fit(x_train, y_train)

xgb_preds = xgb_model.predict(x_test)

xgb_score = model_evaluation("XGB Regression", y_test, xgb_preds)
plt.bar(["Linear Regression","Random ForestRegression", "SVR", "XGB Regression"], 

        [lr_score, rf_score, svr_score, xgb_score])



plt.xticks(rotation=30)

plt.title("R2 Score Comparison")

plt.show()