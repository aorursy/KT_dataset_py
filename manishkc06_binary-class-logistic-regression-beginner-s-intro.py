import pandas as pd

import matplotlib.pyplot as plt
# In read_csv() function, we have passed the raw data link at github

data_location = "https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv"

data = pd.read_csv(data_location)
data.head()
plt.scatter(data.age,data.bought_insurance,marker='+',color='red')
X = data[['age']]     # input variable



y = data['bought_insurance']    # output variable
print("Shape: ", X.shape, "Dimension: ", X.ndim)

print("Shape: ", y.shape, "Dimension: ", y.ndim)
# import train_test_split

from sklearn.model_selection import train_test_split
# split the data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 42)



# X_train: independent/input feature data for training the model

# y_train: dependent/output feature data for training the model

# X_test: independent/input feature data for testing the model; will be used to predict the output values

# y_test: original dependent/output values of X_test; We will compare this values with our predicted values to check the performance of our built model.

 

# test_size = 0.30: 30% of the data will go for test set and 70% of the data will go for train set

# random_state = 42: this will fix the split i.e. there will be same split for each time you run the code
# import Logistic Regression from sklearn.linear_model

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
# Fit the model

log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)
y_test.values
predictions
res = pd.DataFrame(predictions)

res.index = X_test.index # its important for comparison

res.columns = ["prediction"]
from google.colab import files

res.to_csv('filename.csv') 

files.download('filename.csv')
# The confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions) 
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()   # ravel() will convert the 2D numpy array into 1D.

print(tn, fp, fn, tp)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)