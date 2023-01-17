import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

print(dataset.shape)
# Overview of the first 5 rows

dataset.head(5)
dataset.describe()
dataset.drop(columns=['Serial No.'], axis=1, inplace=True)
dataset.apply(lambda x: sum(x.isnull()))
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

list(dataset)
sns.distplot(dataset['chance_of_admit'].dropna(), hist=True, kde=True, 

             bins=int(350/50), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
dataset['university_rating']= dataset.university_rating.astype(object)

dataset['research']= dataset.research.astype(object)
dataset.dtypes
pd.DataFrame(dataset.corr()['chance_of_admit'])
sns.pairplot(data=dataset,

                  y_vars=['chance_of_admit'],

                  x_vars=['gre_score','toefl_score','university_rating','sop','lor','cgpa','research'])
corr = dataset.corr()

corr.style.background_gradient(cmap='coolwarm')
fig, axs = plt.subplots(2, 3)

# gre_score plot

axs[0, 0].boxplot(dataset['gre_score'])

axs[0, 0].set_title('gre_score')



# toefl_score plot

axs[0, 1].boxplot(dataset['toefl_score'])

axs[0, 1].set_title('toefl_score')



# sop plot

axs[0, 2].boxplot(dataset['sop'])

axs[0, 2].set_title("sop")



# lor plot

axs[1, 0].boxplot(dataset['lor'])

axs[1, 0].set_title('lor')



# cgpa plot

axs[1, 1].boxplot(dataset['cgpa'])

axs[1, 1].set_title('cgpa')



# chance of admission plot

axs[1, 2].boxplot(dataset['chance_of_admit'])

axs[1, 2].set_title('chance_of_admit')



fig.subplots_adjust(left=0.15, right=1.98, bottom=0.09, top=0.95,

                    hspace=0.5, wspace=0.5)



plt.show()

sns.boxplot(data = dataset['lor'], orient="v", palette="Set2")

X = dataset[['toefl_score','university_rating','sop','lor','cgpa','research']]

Y = dataset['chance_of_admit']

X = pd.get_dummies(data=X, drop_first=True)
#Treatment of the outlier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn import linear_model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
coef_dframe = pd.DataFrame(lm.coef_,X.columns, columns=['Coefficients'])

coef_dframe
lm_pred = model.predict(X_test)
fig = plt.figure()

c = [i for i in range(1,101,1)]

plt.plot(c,y_test, color = 'grey', linewidth = 2.5, label='Test')

plt.plot(c,lm_pred, color = 'blue', linewidth = 2.5, label='Predicted')

plt.grid(alpha = 0.3)

plt.legend()

fig.suptitle('Actual vs Predicted')
from sklearn import metrics

np.sqrt(metrics.mean_squared_error(y_test,lm_pred))
metrics.mean_absolute_error(y_test,lm_pred)
SS_Residual = sum((y_test-lm_pred)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

R_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_R_squared = 1 - (1-R_squared)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)



print (R_squared, adjusted_R_squared)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf_model = rf.fit(X_train, y_train);

rf_model
# Use the forest's predict method on the test data

rf_pred = rf.predict(X_test)
# Calculate the absolute errors

errors_rf = abs(rf_pred - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors_rf), 2))
rmse = np.sqrt(metrics.mean_squared_error(y_test,rf_pred))



print('Root Mean Squared Error:', round(rmse,2))
SS_Residual = sum((y_test-rf_pred)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

R_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_R_squared = 1 - (1-R_squared)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)



print (R_squared, adjusted_R_squared)
feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances
X_test
Predictions = model.predict(X_test)

Predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame(Predictions,columns=['Predictions'])



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)