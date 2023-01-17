""""
Context
This dataset is created for prediction of Graduate Admissions from an Indian perspective.

Content
The dataset contains several parameters which are considered important during the application for Masters Programs.
The parameters included are :

GRE Scores ( out of 340 )
TOEFL Scores ( out of 120 )
University Rating ( out of 5 )
Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
Undergraduate GPA ( out of 10 )
Research Experience ( either 0 or 1 )
Chance of Admit ( ranging from 0 to 1 )
Acknowledgements
This dataset is inspired by the UCLA Graduate Dataset. The test scores and GPA are in the older format.
The dataset is owned by Mohan S Acharya.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Read the data from the input CSV file
# skipinitialspace=True is used to remove trailing and lagging spaces in header colunmns
admission_data=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv',header=0,skipinitialspace=True)
admission_data.head()
# Remove the unwanted column from the dataset
del admission_data['Serial No.']
print(admission_data.head())
print(admission_data.keys())
# Let's dig down to the co-relation matrix of the dataset
admission_data.corr()
#Let's find out some interesting facts
import matplotlib.pyplot as plt 
plt.scatter(admission_data['GRE Score'], admission_data['University Rating'], alpha=0.5)
plt.title('Scatter plot of GRE vs Ranking')
plt.xlabel('GRE Score')
plt.ylabel('University Ranking')
plt.show()
### Those who have a score above 320 tends to apply to the university with ranking 4 and 5
plt.scatter(admission_data['GRE Score'], admission_data['Chance of Admit '], alpha=0.5)
plt.title('Scatter plot of GRE vs Chance of Admit')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit ')
plt.show()

# Those with higher GRE score have a high chance of admission
# Creating the input and output variable
X = admission_data.drop('Chance of Admit ', axis=1)
Y = admission_data['Chance of Admit ']
#Splitting the train and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Formualting the derived weight vector
X_train_matrix=np.empty
Y_train_array=np.empty
X_train_transpose=np.empty
X_train_dot=np.empty
X_tarin_dot_inverse=np.empty

X_train_matrix=X_train.to_numpy()
X_train_transpose=X_train.to_numpy().transpose()
Y_train_array=Y_train.to_numpy()

X_train_dot=np.dot(X_train_transpose,X_train_matrix)
X_tarin_dot_inverse=np.linalg.inv(X_train_dot)
# print(X_tarin_dot_inverse)

weight_vect=np.dot(np.dot(X_tarin_dot_inverse,X_train_transpose),Y_train.to_numpy())

print(weight_vect)
# Prediction of the Test dataset
Y_preidiction=np.dot(weight_vect,X_test.to_numpy().transpose())
print(Y_preidiction)
# Creating the Actual vs Predicted DF and comapring the results
df_dd = pd.DataFrame({'Actual': Y_test, 'Predicted': np.dot(weight_vect,X_test.to_numpy().transpose())})
# y_test.to_numpy()
df_dd.head(25).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
# Accuracy Measurement
df_dd['error']=df_dd['Actual']-df_dd['Predicted']
sum_squared=0
for i in df_dd['error']:
    sum_squared=sum_squared+(i*i)
mse=sum_squared/len(df_dd['error'])
print("MSE: " ,mse)
import math
print("RMSE: ",math.sqrt(mse))
