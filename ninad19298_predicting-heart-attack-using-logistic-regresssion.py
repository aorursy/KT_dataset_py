import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
raw_data=pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

raw_data.head()
raw_data.describe(include='all') ### describing the data
plt.scatter(raw_data['age'],raw_data['target'])

plt.xlabel('age')

plt.ylabel('deaths from heart attack')

plt.show()
sns.distplot(raw_data['age'])
data=raw_data.copy()

data.columns.values
estimators=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',

       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
y=data['target'] ### defining the target value (dependent variable)

x1=data[estimators] ### defining the features (independent variable)
x=sm.add_constant(x1) ### adding constant

reg_log =sm.Logit(y,x) 

result_log = reg_log.fit() ### fitting the regression
result_log.summary() ### MLE table
#### DROPPING THE  INSIGNIFICANT VARIABLES

data_new=data.drop(['age','fbs'],axis=1)
data_new.head()
data_new.columns.values
estimators_new=['sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']
y_=data_new['target']

x1_= data_new[estimators_new]
x_=sm.add_constant(x1_)

reg_log_ =sm.Logit(y_,x_)

result_log_ = reg_log_.fit()
result_log_.summary()
np.exp(-1.7640)
def confusion_matrix(data,actual_values,model):

        

        # Confusion matrix 

        

        # Parameters

        # ----------

        # data: data frame or array

            # data is a data frame formatted in the same way as your input data (without the actual values)

            # e.g. const, var1, var2, etc. Order is very important!

        # actual_values: data frame or array

            # These are the actual values from the test_data

            # In the case of a logistic regression, it should be a single column with 0s and 1s

            

        # model: a LogitResults object

            # this is the variable where you have the fitted model 

            # e.g. results_log in this course

        # ----------

        

        #Predict the values using the Logit model

        pred_values = model.predict(data)

        # Specify the bins 

        bins=np.array([0,0.5,1])

        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0

        # if they are between 0.5 and 1, they will be considered 1

        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

        # Calculate the accuracy

        accuracy = (cm[0,0]+cm[1,1])/cm.sum()

        # Return the confusion matrix and 

        return cm, accuracy
confusion_matrix(x_,y_,result_log_)