# https://www.kaggle.com/mohansacharya/graduate-admissions
import pandas as pd
import numpy as np

df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")

df.columns = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Chance of Admit']

df.head()
df.describe()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(26, 12))
fig.suptitle("Histogram of Columns Data", fontsize=16)

ax1 = fig.add_subplot(241)
ax2 = fig.add_subplot(242)
ax3 = fig.add_subplot(243)
ax4 = fig.add_subplot(244)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)

ax1.set_title(df.columns[1])
ax2.set_title(df.columns[2])
ax3.set_title(df.columns[3])
ax4.set_title(df.columns[4])
ax5.set_title(df.columns[5])
ax6.set_title(df.columns[6])
ax7.set_title(df.columns[7])
ax8.set_title(df.columns[8])

ax1.hist(df[df.columns[1]])
ax2.hist(df[df.columns[2]])
ax3.hist(df[df.columns[3]])
ax4.hist(df[df.columns[4]])
ax5.hist(df[df.columns[5]])
ax6.hist(df[df.columns[6]])
ax7.hist(df[df.columns[7]])
ax8.hist(df[df.columns[8]])


fig.subplots_adjust(wspace = 0.2, hspace = 0.2)

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]

y = df[["Chance of Admit"]]

#preparing the data for train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn import datasets, linear_model


#building and training the model 
regr = linear_model.LinearRegression()
regr.fit (X_train,y_train)

#predicting using the linear regresssion model
y_pred = regr.predict(X_test)

#evaluating the linear regression model 
print('Coefficients: \n')
coff = X.columns.to_series()
for i in np.arange(len(X.columns)):
    print (X.columns.to_series()[i] + ": " + str(regr.coef_[0][i]))

print ("\n") 
    
print("Mean squared error: " + str(mean_squared_error(y_test, y_pred)))
print('R2 score: ', str(r2_score(y_test, y_pred)))


#plt.figure() 
fig, ax = plt.subplots()
ax.plot(np.arange(len(X_test)), y_test,  color='black')
ax.plot(np.arange(len(X_test)), y_pred,  color='blue', linewidth=1)
ax.set_title('Real Values vs. Linear Regression Predictions')
plt.show()


#generate predictions 
clf = linear_model.Lasso(alpha=0.01)
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

#evaluate results 
print("Mean squared error: " + str(mean_squared_error(y_test, y_predict)))
print('R2 score: ', str(r2_score(y_test, y_predict)))

#plt.figure() 
fig, ax = plt.subplots()
ax.plot(np.arange(len(X_test)), y_test,  color='black')
ax.plot(np.arange(len(X_test)), y_pred,  color='green', linewidth=1)
ax.set_title('Real Values vs. Linear Regression With Lasso Predictions')
plt.show()

from sklearn import tree

#generate predictions 
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

#evaluate results 
print("Mean squared error: " + str(mean_squared_error(y_test, y_predict)))
print('R2 score: ', str(r2_score(y_test, y_predict)))

#plt.figure() 
fig, ax = plt.subplots()
ax.plot(np.arange(len(X_test)), y_test,  color='black')
ax.plot(np.arange(len(X_test)), y_pred,  color='red', linewidth=1)
ax.set_title('Real Values vs. Decision Tree Predictions')
plt.show()

