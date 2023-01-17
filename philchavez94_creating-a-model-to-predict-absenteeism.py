import pandas as pd

import numpy as np
data_preprocessed = pd.read_csv('../input/absenteeism-preprocessed/Absenteeism_preprocessed.csv')
data_preprocessed.head()
data_preprocessed['Absenteeism Time in Hours'].median()
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
data_preprocessed['Excessive Absenteeism'] = targets
data_preprocessed.head()
targets.sum() / targets.shape[0]
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Day of Week','Daily Work Load Average','Distance to Work'],axis=1)
data_with_targets.head()
data_with_targets.iloc[:,:-1]
unscaled_inputs = data_with_targets.iloc[:,:-1]
#from sklearn.preprocessing import StandardScaler



#absenteeism_scaler = StandardScaler()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator, TransformerMixin):

    

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):

        self.scaler = StandardScaler(copy,with_mean,with_std)

        self.columns = columns

        self.mean_ = None

        self.var_ = None

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        return self

    

    def transform(self, X, y=None, copy=None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

    
unscaled_inputs.columns.values
# columns_to_scale = ['Month Value',

      # 'Day of Week', 'Transportation Expense', 'Distance to Work', 'Age',

     # 'Daily Work Load Average', 'Body Mass Index',

      # 'Children', 'Pets']



columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs
from sklearn.model_selection import train_test_split
train_test_split(scaled_inputs, targets)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets)
print (x_test.shape, y_test.shape)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = .8, random_state = 20)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
reg = LogisticRegression()
reg.fit(x_train, y_train)
reg.score(x_train, y_train)
model_outputs = reg.predict(x_train)

model_outputs
y_train
model_outputs == y_train
np.sum((model_outputs==y_train))
model_outputs.shape[0]
np.sum((model_outputs==y_train)) / model_outputs.shape[0]
reg.intercept_
reg.coef_
unscaled_inputs.columns.values
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature Name'], data = feature_name)



summary_table['Coefficient'] = np.transpose(reg.coef_)



summary_table
summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
summary_table
summary_table.sort_values('Odds_ratio', ascending = False)
reg.score(x_test, y_test)
predicted_proba = reg.predict_proba(x_test)

predicted_proba
predicted_proba[:,1]
import pickle
with open('model','wb') as file:

    pickle.dump(reg, file)
with open('scaler','wb') as file:

    pickle.dump(absenteeism_scaler, file)