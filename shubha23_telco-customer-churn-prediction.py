# Import necessary packages to read, process and visualize data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt     # Generate plots
import seaborn as sns               # Visualization
%matplotlib inline

# Read the data
filename = "../input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(filename)

# Let us see the shape of data
print(data.shape)   
# Following output shows there are 7043 rows and 21 columns in our data
# Overview and statistical details of the data..
# Let us see first five rows to understand what type of values exist for each columns
data.head()
# To view all column names and their respective data types
data.columns
data.info()
data.describe() # Shows statistical summaries for all numeric columns
# Plot distribution of dependent/target variable - Churn column
data['Churn'].value_counts().head().plot.bar()   # To generate a bar plot

# To generate a pie chart. Since there are only two classes, a pie chart may look more appealing
sizes = data['Churn'].value_counts(sort = True)
labels = np.unique(data.Churn)

# Visualize the data
plt.figure(figsize = (6,6))
plt.subplot(212)
plt.title("Customer churn rate:")
plt.pie(sizes, labels = labels, autopct='%1.1f%%')

# Bar & pie plots below show that number of customers churned is less than half of not churned.
# Convert following object type columns to numeric        
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = 'coerce')
# Let us find if there are any missing values in our data.
print("No. of missing values: \n",data.isnull().sum())
# Drop CustomerId column as it is not required
data.drop(['customerID'], axis = 1, inplace = True)

# Fill the missing values with 0
data['TotalCharges'] = data['TotalCharges'].fillna(0.0)

# Check for any existing missing values
print("Missing values now: \n", data.isnull().sum())
# Now let us work on categorical features. 
data.gender = [1 if x == "Male" else 0 for x in data.gender]
for col in ('Partner', 'Dependents', 'PhoneService' , 'OnlineSecurity',
        'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV',
        'StreamingMovies','PaperlessBilling','MultipleLines','Churn'):
    data[col] = [1 if x == "Yes" else 0 for x in data[col]]        
data.head(10)   # See how data looks like now
# Print correlation between all features and target variable
data.corr()['Churn'].sort_values()
# Plot heatmap using Seaborn to visualize correlation amongst ftrs.
sns.heatmap(data.corr(), annot = True)
# For following features, let us generate bar plots w.r.t. target variable
for col in ('Partner', 'Dependents', 'PhoneService' , 'OnlineSecurity',
        'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV',
        'StreamingMovies','PaperlessBilling','MultipleLines'):
    sns.barplot(x = col, y = 'Churn', data = data)
    plt.show()
# Following plots show Churn rate for each category of these categorical features.    
# Generate pairplots for all features.
highCorrCols = ['MonthlyCharges','TotalCharges','tenure', 'Churn']
sns.pairplot(data[highCorrCols], hue = 'Churn')
# Prepare data for model training and testing input.
y = data.Churn.values     # Target feature

# All features except class (target)
data = pd.get_dummies(data)
X = data.drop(["Churn"],axis=1)

from sklearn.metrics import accuracy_score, mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

# Split the data into training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state=1)

# Classification using RBF SVM  
svc_rbf = SVC(kernel = "rbf")
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print("Mean-squared error using SVM RBF:", mse(y_test, prediction))
print("Accuracy with SVM RBF:",accuracy_score(y_test, prediction))

# Classification using Random Forest Classifier
rfc = RF(max_depth= 5, n_estimators= 10, max_features= 'auto')
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("Mean-squared error using Random Forest Classifier:", mse(y_test, prediction))
print("Accuracy with Random Forest Classifier:",accuracy_score(y_test, prediction))

# Classification using Logistic Regression
logreg = LR(C = 1)
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("Mean-squared error using Logistic Regression:", mse(y_test, prediction))
print("Accuracy with Logistic Regression:",accuracy_score(y_test, prediction))

# Classification using Multi-layer perceptron 
ann = MLPClassifier(solver='lbfgs', alpha = 1e-5,
                    hidden_layer_sizes = (5, 2), random_state = 1)
ann = ann.fit(X_train, y_train)
prediction = ann.predict(X_test)
print("Mean-squared error using Neural networks MLP:", mse(y_test, prediction))
print("Accuracy with Neural networks MLP:",accuracy_score(y_test, prediction))