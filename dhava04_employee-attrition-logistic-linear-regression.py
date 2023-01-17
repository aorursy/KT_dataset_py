# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,classification_report

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def logreg_model():

    model = LogisticRegression()

    model.fit(train_x,train_y)

    train_predict = model.predict(train_x)

    test_predict = model.predict(test_x)



    print('*** Confusion matrix Train Data ***')

    print(confusion_matrix(train_y,train_predict))

    print(' ')

    print('*** Confusion matrix Test Data ***')

    print(confusion_matrix(test_y,test_predict))

    print(' ')

    print('*** Classification Report Train Data ***')

    print(classification_report(train_y,train_predict))

    print(' ')

    print('*** Classification Report Test Data ***')

    print(classification_report(test_y,test_predict))

    print(' ')

    print('*** Accuracy Score Train Data ***')

    print("Train Accuracy : ",accuracy_score(train_y,train_predict))

    print(' ')

    print('*** Accuracy Score Test Data ***')

    print("Test Accuracy : ",accuracy_score(test_y,test_predict))

    print(' ')
def linreg_model():

    model = LinearRegression().fit(train_x,train_y)

    train_predict = model.predict(train_x)

    test_predict = model.predict(test_x)



    print('======== INTERCEPT AND SLOPE =========')

    print('Intercept:',model.intercept_,'Slope:',model.coef_)



    print('========== PRINT MODEL METRICS =========')

    print( 'MAE Train :', mean_absolute_error(train_y,train_predict))

    print( 'MAE test  :', mean_absolute_error(test_y,test_predict))

    print( 'MSE Train :', mean_squared_error(train_y,train_predict))

    print( 'MSE test  :', mean_squared_error(test_y,test_predict))

    print( 'MAPE Train:',np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

    print( 'MAPE Test :',np.mean(np.abs((test_y - test_predict) / test_y)) * 100)

    print( 'R^2 Train :', r2_score(train_y,train_predict))

    print( 'R^2 Test  :', r2_score(test_y,test_predict))



    #Plot actual vs predicted value

    plt.figure(figsize=(10,7))

    plt.title("Actual vs. predicted expenses",fontsize=25)

    plt.xlabel("Actual expenses",fontsize=18)

    plt.ylabel("Predicted expenses", fontsize=18)

    plt.scatter(x=test_y,y=test_predict)

class col_encode:

    

    @staticmethod

    def vect(dataframe,column,regex):

        vector = CountVectorizer(token_pattern=regex)

        data_ohe_vct = vector.fit_transform(column)

        dataframe = dataframe.join(pd.DataFrame(data_ohe_vct.toarray(), columns=vector.get_feature_names()))

        return(dataframe)
df=pd.read_csv("../input/HR-Employee-Attrition.csv")

df.head()
print(df.info())

print(df.isna().sum())

# it is a clean dataset - No nulls
%matplotlib inline

pd.crosstab(df.Department,df.Attrition).plot(kind='bar')

plt.title('Department influence on attrition')

plt.xlabel('Department')

plt.ylabel('Attrition')
%matplotlib inline

pd.crosstab(df.Gender,df.Attrition).plot(kind='bar')

plt.title('Gender influence on attrition')

plt.xlabel('Gender')

plt.ylabel('Attrition')
%matplotlib inline

pd.crosstab(df.MaritalStatus,df.Attrition).plot(kind='bar')

plt.title('MaritalStatus influence on attrition')

plt.xlabel('MaritalStatus')

plt.ylabel('Attrition')
#Attrition has only Yes and No. Changing it to numerical

df.Attrition.replace({"Yes":1,"No":0}, inplace=True)

print(df["Attrition"].unique())
print(df["BusinessTravel"].unique())

#perform one hot encoding for BusinessTravel using count vectorizer

ohe = col_encode()

df = ohe.vect(df,df["BusinessTravel"],r'\b[^,]+\b')

df.head()
print(df["Department"].unique())
df = ohe.vect(df,df["Department"],r'\b[^,]+\b')

df.head()
print(df["Gender"].unique())
df = ohe.vect(df,df["Gender"],r'\b[^,]+\b')

df.head()
df.rename(columns={'human resources': 'human resources dept'}, inplace=True)

print(df["EducationField"].unique())

df = ohe.vect(df,df["EducationField"],r'\b[^,]+\b')

df.head()
print(df["MaritalStatus"].unique())

df = ohe.vect(df,df["MaritalStatus"],r'\b[^,]+\b')

df.head()
print(df["Over18"].unique())

print(df["OverTime"].unique())

df_overcols = pd.DataFrame()

cols =['Over18','OverTime']

df_overcols = pd.get_dummies(df[cols], prefix=['Over18','OverTime'])

df = df.join(df_overcols)

df.head()
df.drop(columns=df.select_dtypes(include=['object']).columns,inplace=True)

df.head()
attrition_corr = df.corr()['Attrition'] 

attrition_corr
Model_features = attrition_corr[(abs(attrition_corr) > 0.0) & (abs(attrition_corr) != 1.0)].sort_values(ascending=False)

print("{} correlated values:\n{}".format(len(Model_features), Model_features))
y = df['Attrition']

x = df.drop(columns=['Attrition'])

train_x, test_x,train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1)

print(train_x.shape, test_x.shape,train_y.shape, test_y.shape)
print(' *** Logistic Regression ***')

logreg_model()



print(' *** Linear Regression ***')

linreg_model()
df.drop(columns=["EmployeeCount","Over18_Y","StandardHours"])

df.head()

print(' *** Logistic Regression ***')

logreg_model()



print(' *** Linear Regression ***')

linreg_model()