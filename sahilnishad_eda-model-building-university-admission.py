import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
#reading csv file 
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
df.head()
df.info()
#dropping 'Serial No.' column
df.drop(columns=['Serial No.'],axis=1,inplace=True)
df.head()
#checking null values
df.isnull().sum()
#checking statistical summary
df.describe()
#histogram plot of df
df.hist(bins=20,figsize=(20,20))
#pairplot of dataframe df
sns.pairplot(df)
#plotting annotated correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
#creating dataframe with all feature except 'Chance of Admit'
#thus we are going to drop 'Chance of Admit' column from df and save it in x

x = df.drop(columns = ['Chance of Admit '])
x
#after that we are going to create a dataframe that has only 'Chance of Admit' column which will behave as output 

y = df['Chance of Admit ']
y
#converting into numpy arrays

x = np.array(x)
y = np.array(y)
x.shape
y.shape
#reshaping y

y = y.reshape(-1,1)
y.shape
#splitting data into 85% to training & 15% to testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
#Model building
model = LinearRegression()
#Fitting the model
model.fit(x_train, y_train)
#intercept of linear model
model.intercept_
#coefficients of linear model
model.coef_
#accuracy of our model
accuracy = model.score(x_test, y_test)
accuracy
#predicting output from x_test
y_pred = model.predict(x_test)
y_pred
#comparing y_pred with y_test
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1