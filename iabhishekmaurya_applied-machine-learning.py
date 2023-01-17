import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv") 
data.head()
data.info()
data["is_canceled"].value_counts()
corr_matrix = data.corr()
corr_matrix["is_canceled"].sort_values(ascending=False)
nulls = data.isnull().sum()
nulls[nulls > 0]
data.iloc[:,23].fillna(data.iloc[:,23].mean(), inplace=True)
data.iloc[:,10].fillna(data.iloc[:,10].mean(), inplace=True)
nulls = data.isnull().sum()
nulls[nulls > 0]
data = data.drop(['stays_in_weekend_nights','arrival_date_day_of_month', 'children', 'arrival_date_week_number', 'company', 'reservation_status_date'], axis=1)
data["country"].value_counts()
print("Data shape BEFORE drop of rows where country is not especified : ",data.shape)
data = data[data['country'].notna()]
print("Data shape AFTER drop of rows where country is not especified : ",data.shape)
data["country"].value_counts()
data = data.drop(['country'], axis=1)
# as it contains a lot of variety
data = data.drop(['reservation_status'], axis=1)
X = (data.loc[:, data.columns != 'is_canceled'])
y = (data.loc[:, data.columns == 'is_canceled'])
x_columns = X.columns
object_column_name = X.select_dtypes('object').columns
print (object_column_name)

object_column_index = X.columns.get_indexer(X.select_dtypes('object').columns)
print (object_column_index)
print(X.shape)

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), object_column_index)], remainder='passthrough')

X = columnTransformer.fit_transform(X)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 25)
def model(algo):
    algo_model = algo.fit(X_train, y_train)
    global y_prob, y_pred
    y_prob = algo.predict_proba(X_test)[:,1]
    y_pred = algo_model.predict(X_test)

    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred),roc_auc_score(y_test,y_pred)))
print('Logistic Regression\n')
model(LogisticRegression(solver = "saga"))
print('Decision Tree\n')
model(DecisionTreeClassifier(max_depth = 12))
print('Random Forest\n')
model(RandomForestClassifier())
print('Gaussian Naive Bayes\n')
model(GaussianNB())
print('Model: XGBoost\n')
model(XGBClassifier())