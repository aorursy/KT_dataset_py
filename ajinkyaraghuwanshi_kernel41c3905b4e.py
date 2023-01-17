# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from IPython.display import display

import csv

from sklearn import preprocessing



import seaborn as sns

from pylab import rcParams

from preprocessing import *

from sklearn.preprocessing import scale, normalize

from math import *

from keras.models import Sequential

from keras.layers import Dense
# load data

df = pd.read_csv('/kaggle/input/ee-769-assignment1/train.csv')

# df.shape

# # split data into X and y

# df.isnull().sum()

# col = ['Age','Attrition','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

# df[col] = df[col].fillna(df.mode().iloc[0])

# ########################################################

# X= df[['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]#.values

# Y= df.iloc[:,1]#.values



# X.shape

# type(X)

# Y

# display(X)

# display(Y)
#print(X)

# print('Original Features: \n',list(X.columns),'\n')

# data= pd.get_dummies(df.)

# data

# print('Features after One-Hot Encoding:\n',list(data.columns))

# imputer = SimpleImputer(missing_values = ','NAN',strategy = 'mean')

# imputer.fit(X[:,[1,3,5,6,8,9,10,12,13,14,16,18,19,20,22,23,24,25,26,27,28,29,30,31]])

# X[:,[1,3,5,6,8,9,10,12,13,14,16,18,19,20,22,23,24,25,26,27,28,29,30,31]] = imputer.fit_transform(X[:,[1,3,5,6,8,9,10,12,13,14,16,18,19,20,22,23,24,25,26,27,28,29,30,31]])
dummy0 = pd.get_dummies(df.BusinessTravel)

dummy1 = pd.get_dummies(df.Department)

dummy2 = pd.get_dummies(df.EducationField)

dummy3 = pd.get_dummies(df.Gender)

dummy4 = pd.get_dummies(df.JobRole)

dummy5 = pd.get_dummies(df.MaritalStatus)

dummy6 = pd.get_dummies(df.OverTime)

# X = pd.concat([X,dummy0,dummy1,dummy2,dummy3,dummy4,dummy5,dummy6])

dummy0

# X.shape
merged =pd.concat([df,dummy0,dummy1,dummy2,dummy3,dummy4,dummy5,dummy6],axis='columns')

display(merged)

final = merged.drop(['BusinessTravel','Department','EducationField','EmployeeCount','EmployeeNumber','Gender','JobRole','MaritalStatus','OverTime'],axis= 'columns')

display(final)
X = final.drop(['Attrition','ID'],axis= 'columns')

Y = final[['Attrition']]



# encode = LabelEncoder()

print('Original Features: \n',list(X.columns),'\n')

print('Original Features: \n',list(Y.columns),'\n')

# # Now encoding the data

# X[:,1] = encode.fit_transform(X[:,1])

# X[:,3] = encode.fit_transform(X[:,3])

# X[:,6] = encode.fit_transform(X[:,6])

# X[:,10] = encode.fit_transform(X[:,10])

# X[:,14] = encode.fit_transform(X[:,14])

# X[:,16] = encode.fit_transform(X[:,16])

# X[:,20] = encode.fit_transform(X[:,20])

# print(X)

# print(type(X))

# print(X.shape)

# X1 = pd.DataFrame(X)

# X1[20].unique()





# print(X1)

# ##################################################

# from sklearn.compose import make_column_transformer

# # Now applying One Hot Encoding

# # hotencode = OneHotEncoder(sparse=False)

# # X = hotencode.fit_transform(X[['BusinessTravel']])

# column_trans = make_column_transformer((OneHotEncoder(),['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']),remainder = 'passthrough')

# column_trans.fit_transform(X)

# # ########################################################

def sigmoid(x):

  return 1 / (1 + np.exp(-x))
####################################################################################################################

#Making Neural Network

class NNClassifier:



    def __init__(self, n_classes, n_features, n_hidden_units=30,

                 l1=0.0, l2=0.0, epochs=500, learning_rate=0.01,

                 n_batches=1, random_seed=None):



        if random_seed:

            np.random.seed(random_seed)

        self.n_classes = n_classes

        self.n_features = n_features

        self.n_hidden_units = n_hidden_units

        self.w1, self.w2 = self._init_weights()

        self.l1 = l1

        self.l2 = l2

        self.epochs = epochs

        self.learning_rate = learning_rate

        self.n_batches = n_batches



    def _init_weights(self):

        w1 = np.random.uniform(-1.0, 1.0, 

                               size=self.n_hidden_units * (self.n_features + 1))

        w1 = w1.reshape(self.n_hidden_units, self.n_features + 1)

        w2 = np.random.uniform(-1.0, 1.0, 

                               size=self.n_classes * (self.n_hidden_units + 1))

        w2 = w2.reshape(self.n_classes, self.n_hidden_units + 1)

        return w1, w2



    def _add_bias_unit(self, X, how='column'):

        if how == 'column':

            X_new = np.ones((X.shape[0], X.shape[1] + 1))

            X_new[:, 1:] = X

        elif how == 'row':

            X_new = np.ones((X.shape[0] + 1, X.shape[1]))

            X_new[1:, :] = X

        return X_new



    def _forward(self, X):

        net_input = self._add_bias_unit(X, how='column')

        net_hidden = self.w1.dot(net_input.T)

        act_hidden = sigmoid(net_hidden)

        act_hidden = self._add_bias_unit(act_hidden, how='row')

        net_out = self.w2.dot(act_hidden)

        act_out = sigmoid(net_out)

        return net_input, net_hidden, act_hidden, net_out, act_out

    

    def _backward(self, net_input, net_hidden, act_hidden, act_out, y):

        sigma3 = act_out - y

        net_hidden = self._add_bias_unit(net_hidden, how='row')

        sigma2 = self.w2.T.dot(sigma3) * sigmoid_prime(net_hidden)

        sigma2 = sigma2[1:, :]

        grad1 = sigma2.dot(net_input)

        grad2 = sigma3.dot(act_hidden.T)

        return grad1, grad2



    def _error(self, y, output):

        L1_term = L1_reg(self.l1, self.w1, self.w2)

        L2_term = L2_reg(self.l2, self.w1, self.w2)

        error = cross_entropy(output, y) + L1_term + L2_term

        return 0.5 * np.mean(error)



    def _backprop_step(self, X, y):

        net_input, net_hidden, act_hidden, net_out, act_out = self._forward(X)

        y = y.T



        grad1, grad2 = self._backward(net_input, net_hidden, act_hidden, act_out, y)



        # regularize

        grad1[:, 1:] += (self.w1[:, 1:] * (self.l1 + self.l2))

        grad2[:, 1:] += (self.w2[:, 1:] * (self.l1 + self.l2))



        error = self._error(y, act_out)

        

        return error, grad1, grad2



    def predict(self, X):

        Xt = X.copy()

        net_input, net_hidden, act_hidden, net_out, act_out = self._forward(Xt)

        return mle(net_out.T)

    

    def predict_proba(self, X):

        Xt = X.copy()

        net_input, net_hidden, act_hidden, net_out, act_out = self._forward(Xt)

        return softmax(act_out.T)



    def fit(self, X, y):

        self.error_ = []

        X_data, y_data = X.copy(), y.copy()

        #y_data_enc = one_hot(y_data, self.n_classes)

        for i in range(self.epochs):



            X_mb = np.array_split(X_data, self.n_batches)

            y_mb = np.array_split(y_data, self.n_batches)

            

            epoch_errors = []



            for Xi, yi in zip(X_mb, y_mb):

                

                # update weights

                error, grad1, grad2 = self._backprop_step(Xi, yi)

                epoch_errors.append(error)

                self.w1 -= (self.learning_rate * grad1)

                self.w2 -= (self.learning_rate * grad2)

            self.error_.append(np.mean(epoch_errors))

        return self

    

    def score(self, X, y):

        y_hat = self.predict(X)

        return np.sum(y == y_hat, axis=0) / float(X.shape[0])

########################################################################################################                                                
###################################################################

# Now Neural network usin keras

model = Sequential()

model.add(Dense(12, input_dim = 51, activation = ''))

model.add(Dense(3, activation = 'sigmoid'))

model.add(Dense(8, activation = 'sigmoid'))

model.add(Dense(10, activation = 'sigmoid'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=['accuracy'])

###################################################################


# split data into train and test sets

# seed = 7

# test_size = 0.33

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# # fit model no training data

duplicate_columns = X.columns[X.columns.duplicated()]

print(duplicate_columns)

# print(X[['Human Resources']])

# X.rename(columns={X[:,29]:"Human Resources1",X[:,32]:"Human Resources2",X[:,41]:"Human Resources3"})

# cols=pd.Series(X.columns)



# for dup in cols[cols.duplicated()].unique(): 

#     cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]







cols = []

count = 1

for column in X.columns:

    if column == 'Human Resources':

        cols.append(f'Human Resources{count}')

        count+=1

        continue

    cols.append(column)

X.columns = cols



duplicate_columns = X.columns[X.columns.duplicated()]

print(duplicate_columns)



x = X.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(X)

X = pd.DataFrame(x_scaled)

print("X")

display(X)

# classifier = XGBClassifier()

# classifier.fit(X,Y)



# # make predictions for test data

td = pd.read_csv('/kaggle/input/ee-769-assignment1/test.csv')

dummy00 = pd.get_dummies(td.BusinessTravel)

dummy11 = pd.get_dummies(td.Department)

dummy22 = pd.get_dummies(td.EducationField)

dummy33 = pd.get_dummies(td.Gender)

dummy44 = pd.get_dummies(td.JobRole)

dummy55 = pd.get_dummies(td.MaritalStatus)

dummy66 = pd.get_dummies(td.OverTime)

print('\n Dummy00')

display(dummy00)

merged1 =pd.concat([td,dummy00,dummy11,dummy22,dummy33,dummy44,dummy55,dummy66],axis='columns')

print('\n merged')

display(merged1)

Test_data = merged1.drop(['BusinessTravel','Department','EducationField','EmployeeCount','EmployeeNumber','Gender','JobRole','MaritalStatus','OverTime','ID'],axis= 'columns')

print('\n Test_data')

display(Test_data)





cols = []

count = 1

for column in Test_data.columns:

    if column == 'Human Resources':

        cols.append(f'Human Resources{count}')

        count+=1

        continue

    cols.append(column)

Test_data.columns = cols



print('\n Test_data dropped')

display(Test_data)

x = Test_data.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

Test_data = pd.DataFrame(x_scaled)

#######################################################################################

# y_pred = classifier.predict(Test_data)

# print(Test_data.shape)

# print('\n Test_data Normalized')

# display(Test_data)

############################################################################################3





# N_FEATURES = 51

# N_CLASSES = 10

# RANDOM_SEED = 42

# seed = 7

# test_size = 0.9

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



# nn = NNClassifier(n_classes=N_CLASSES, 

#                   n_features=N_FEATURES,

#                   n_hidden_units=250,

#                   l2=0.5,

#                   l1=0.0,

#                   epochs=500,

#                   learning_rate=0.001,

#                   n_batches=25,

#                   random_seed=RANDOM_SEED)



# nn.fit(X_train, y_train);



# print('Test Accuracy: %.2f%%' % (nn.score(X_test, y_test) * 100))

##############################################################################################











#split data into train and test sets

# seed = 7

# test_size = 0.1

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

      

      



        

###########################################################################

# Keras model fitting and try



model.fit(X,Y,epochs = 1000, batch_size =10)



## scores = model.evaluate(X_test,y_test)



## print("\n %s: %.2f%%" % model.metric_names[1], scores[1]*100)



y_pred = model.predict_classes(Test_data)



y_pred = y_pred[:,0]

print(y_pred)

print(y_pred.shape)

# #######################################################################



# y_pred = classifier.predict(X_test)

# predictions = [round(value) for value in y_pred]

# # evaluate predictions

# accuracy = accuracy_score(y_test, predictions)

# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# print(predictions)









#######################################################################################################

# Now taking the output

test_id=np.arange(1029,1470,1)

print(test_id.shape)



new_l = list(zip(test_id,y_pred))

print(y_pred.shape)

print(test_id.shape)

data_frame =pd.DataFrame(data={"ID": test_id,"Attrition": y_pred})

data_frame.to_csv("./Out.csv",sep=',',index= False)

      

    