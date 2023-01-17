import numpy as numpyInstance
import pandas as pandasInstance
import seaborn as seabornInstance
import matplotlib.pyplot as matplotLibInstance
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
%matplotlib inline
candidates_Data = pandasInstance.read_csv('../input/Admission_Predict_Ver1.1.csv')
candidates_Data.head()
candidates_Data.info()
candidates_Data.describe()
matplotLibInstance.figure(figsize=(12,10))
seabornInstance.countplot(x='University Rating',data=candidates_Data,palette='winter')
matplotLibInstance.figure(figsize=(12,10))
seabornInstance.distplot(candidates_Data['GRE Score'],color='seagreen')
matplotLibInstance.figure(figsize=(12,10))
seabornInstance.distplot(candidates_Data['CGPA'],color='red')
matplotLibInstance.figure(figsize=(12,10))
seabornInstance.jointplot(x='GRE Score',y='TOEFL Score',data=candidates_Data,kind='hex')
from sklearn.model_selection import train_test_split
candidates_Data.columns
X = candidates_Data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]
Y = candidates_Data['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
linearRegressionInstance = LinearRegression()
linearRegressionInstance.fit(X_train,y_train)
linearRegressionInstance.intercept_
linearRegressionInstance.coef_
coefficientsDescription = pandasInstance.DataFrame(data=linearRegressionInstance.coef_,index=X_train.columns,columns=['Values'])
coefficientsDescription
predictionsOfAdmission = linearRegressionInstance.predict(X_test)
matplotLibInstance.figure(figsize=(12,10))
seabornInstance.scatterplot(x=y_test,y=predictionsOfAdmission,color='crimson')
matplotLibInstance.figure(figsize=(12,10))
seabornInstance.distplot(y_test - predictionsOfAdmission,color='purple')
coefficientsDescription
def calGetAdmssionOrNot(cal):
    if cal >=0.60:
        return 1
    else:
        return 0
    
gotAdmissionOrNot = candidates_Data['Chance of Admit '].apply(calGetAdmssionOrNot)
candidates_Data['Get Admission'] = gotAdmissionOrNot
candidates_Data
candidates_Data.head()
candidates_Data.columns
X = candidates_Data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]
Y_Predict = candidates_Data['Get Admission']
X_train, X_test, y_train, y_test = train_test_split(X, Y_Predict, test_size=0.4, random_state=101)
from sklearn.linear_model import LogisticRegression
logisticRegressionInstance = LogisticRegression()
logisticRegressionInstance.fit(X_train,y_train)
admissionPredictions = logisticRegressionInstance.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,admissionPredictions))