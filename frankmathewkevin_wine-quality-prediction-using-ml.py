#Loading the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Loading the dataset
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head(10)
#Basic summary of the data
df.info()
#Basic Statistics information about the data
df.describe()
#Since all the values are numeric, we will check the correlation
fig = plt.figure(figsize=(10,5))
sns.heatmap(round(df.corr(),2),cmap = "viridis", annot = True)
plt.title("Correlation")
#EDA
fig = plt.figure(figsize = (10,5))
gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax1 = sns.scatterplot(data = df, x = "citric acid", y = "fixed acidity", hue = "quality", palette = "rainbow", legend = "full")
ax2 = fig.add_subplot(gs[0,1])
ax2 = sns.scatterplot(data = df, x = "density", y = "fixed acidity", hue = "quality", palette = "rainbow", legend = "full")
ax1.set_title("Citric Acid vs Fixed Acidity")
ax2.set_title("Density vs Fixed Acidity")
plt.tight_layout()
sns.distplot(df.quality, kde = True, color = "y")
fig = plt.figure(figsize = (20,10))
df1 = df[df["quality"].isin([6,7,8])]
sns.lmplot("citric acid","fixed acidity", df1, hue = "quality", col = "quality")
#plt.suptitle("Scatter plot with regression lines on different axes", fontsize = 10)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Splitting the dataset into train and test
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']
target = ['quality']

X = sc.fit_transform(df[features])
y = sc.fit_transform(df[target])


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20, random_state = 1)

print("Training Dataset for features:", X_train.shape)
print("Training Dataset for target:", y_train.shape)
print("Testing Dataset for features:", X_test.shape)
print("Testing Dataset for target:", y_test.shape)
#Lets use Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
#Check the Intercept
print('Intercept:',lr_model.intercept_) 
df_X_train = pd.DataFrame(data = X_train)
print(type(df_X_train))
df_X_train.head()
#Changing columns names
df_X_train.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
df_X_train.head()
#Checking Coefficients
pd.DataFrame((lr_model.coef_).T, index = df_X_train.columns, columns = ["Coefficients"]).sort_values(by = "Coefficients", ascending = False)
#Prediction for training & testing set
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)
#Model Evaluation
from sklearn import metrics
#MAE
MAE_train = metrics.mean_absolute_error(y_train,y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test,y_pred_test)

#MSE
MSE_train = metrics.mean_squared_error(y_train,y_pred_train)
MSE_test = metrics.mean_squared_error(y_test,y_pred_test)

#RMSE
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)

#R-Squared
R2_train = metrics.r2_score(y_train,y_pred_train)
R2_test = metrics.r2_score(y_test,y_pred_test)

print("RMSE for train dataset is:", RMSE_train)
print("RMSE for test dataset is:", RMSE_test)
print("R-Squared value for train dataset is:", R2_train)
print("R-Squared value for test dataset is:", R2_test)
#Lets change the target variable into a classification problem (Good(1) vs Bad(0))
df.head()
#The quality is bad if it falls between 2.5 to 6 and the quality is good if it falls between 6 to 8.5
bins = (2.5,6,8.5)
groups_name = ["Bad","Good"]
df["quality_new"] = pd.cut(df["quality"], bins = bins, labels = groups_name)
df.sample(3)
from sklearn.preprocessing import LabelEncoder
label_quality = LabelEncoder()
df["quality_new"].dtype
df["quality_new"] = label_quality.fit_transform(df["quality_new"])
#Lets use Logistic Regression
df.head()
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']
target = ['quality_new']
X = df[features]
y = df[target]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20, random_state = 9)
#Applying Standard scaling to get optimized result

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
#Predicting the testing set
y_pred_log = log_model.predict(X_test)
y_pred_log
from sklearn.metrics import accuracy_score
print("Accuracy Score is:", accuracy_score(y_test,y_pred_log))
from sklearn.metrics import confusion_matrix
c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_log))
c_matrix.index = ["Actual Bad Quality", "Actual Good Quality"]
c_matrix.columns = ["Predicted Bad Quality", "Predicted Good Quality"]
print(c_matrix)
from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()
sgd_model.fit(X_train,y_train)
y_pred_sgd = sgd_model.predict(X_test)
print("Accuracy Score is:", accuracy_score(y_test,y_pred_sgd))
c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_sgd))
c_matrix.index = ["Actual Bad Quality", "Actual Good Quality"]
c_matrix.columns = ["Predicted Bad Quality", "Predicted Good Quality"]
print(c_matrix)
#Lets use Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state = 0)
dt_model.fit(X_train,y_train)
y_pred_dt = dt_model.predict(X_test)
print("Accuracy Score is:", accuracy_score(y_test,y_pred_dt))
c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_dt))
c_matrix.index = ["Actual Bad Quality", "Actual Good Quality"]
c_matrix.columns = ["Predicted Bad Quality", "Predicted Good Quality"]
print(c_matrix)
#Lets use Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 300)
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
print("Accuracy Score is:", accuracy_score(y_test,y_pred_dt))
c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_rf))
c_matrix.index = ["Actual Bad Quality", "Actual Good Quality"]
c_matrix.columns = ["Predicted Bad Quality", "Predicted Good Quality"]
print(c_matrix)
#HyperParameter Tuning

from sklearn.model_selection import RandomizedSearchCV
parameters_rf = {
    'n_estimators' : [300,700],
    'criterion' : ['gini','entropy'],
    'min_samples_split' : range(6,11),
    'min_samples_leaf' : range(1,5),
    'max_features' : ['sqrt','log2',5],
    'bootstrap' : [True, False]
    }
tuned_rf_model = RandomizedSearchCV(rf_model, param_distributions = parameters_rf, n_iter = 100, n_jobs = -1)
tuned_rf_model.fit(X_train,y_train)
y_pred_tuned_rf_model = tuned_rf_model.predict(X_test)
print("Accuracy Score is:", accuracy_score(y_test,y_pred_tuned_rf_model))
tuned_rf_model.best_params_
c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_tuned_rf_model))
c_matrix.index = ["Actual Bad Quality", "Actual Good Quality"]
c_matrix.columns = ["Predicted Bad Quality", "Predicted Good Quality"]
print(c_matrix)
