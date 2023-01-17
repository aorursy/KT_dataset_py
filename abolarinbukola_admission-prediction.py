#Importing the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')



df.head(3)
df.info()
df['admit'] =  np.where(df['Chance of Admit '] > 0.5,1,0)
df.head()
#Dropping useless variables

df.drop(['Chance of Admit ', 'Serial No.'], axis = 1, inplace = True)
df.head(3)
df.describe()
df.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant





# the target column (in this case 'admit') should not be included in variables

#Categorical variables already turned into dummy indicator may or maynot be added if any

variables = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ',

       'CGPA',]]

X = add_constant(variables)

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]

vif['features'] = X.columns

vif



#Using 10 as the minimum vif values i.e any independent variable 10 and above will have to be dropped

#From the results all independent variable are below 10
#Declaring our target variable as y

#Declaring our independent variables as x

y = df['admit']

x = df.drop(['admit'], axis = 1)
scaler = StandardScaler() #Selecting the standardscaler



scaler.fit(x)#fitting our independent variables
scaled_x = scaler.transform(x)#scaling
#Splitting our data into train and test dataframe

x_train, x_test, y_train, y_test = train_test_split(scaled_x,y , test_size = 0.2, random_state = 49)
reg = LogisticRegression()#Selecting our model

reg.fit(x_train,y_train)
y_new = reg.predict(x_test) #Predicting with our already trained model using x_test
#Getting the accuracy of our model

acc = metrics.accuracy_score(y_new,y_test)

acc
#The intercept for our regression

reg.intercept_
#Coefficient for all our variables

reg.coef_
cm = confusion_matrix(y_new, y_test)

cm
# Format for easier understanding

cm_df = pd.DataFrame(cm)

cm_df.columns = ['Predicted 0','Predicted 1']

cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})

cm_df
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



dnew = dt.predict(x_test)



acc2 = metrics.accuracy_score(dnew,y_test)

acc2
sv = svm.SVC() #select the algorithm

sv.fit(x_train,y_train) # we train the algorithm with the training data and the training output

y_pred = sv.predict(x_test) #now we pass the testing data to the trained algorithm

acc_svm = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the SVM is:', acc_svm)
knc = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

knc.fit(x_train,y_train)

y_pred = knc.predict(x_test)

acc_knn = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the KNN is', acc_knn)
df1 = pd.DataFrame(data = x.columns.values, columns = ['Features'])



df1['weight'] = np.transpose(reg.coef_)

df1['odds'] = np.exp(np.transpose(reg.coef_))

df1