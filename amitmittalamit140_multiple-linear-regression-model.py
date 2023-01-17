#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#loading dataset

df = pd.read_csv("../input/startup50/50_Startups.csv")

df.head()
#shape of our dataset

df.shape
#information about the data

df.info()
#checking for missing data

df.isnull().sum()

#there is no missing value in the data
#description about data

df.describe()
#Box Plot of independent variables an it seems we dont have outliers in our independent varables

plt.figure(figsize=(20,3))

for i,col in zip(range(1,4),df.columns):

    plt.subplot(1,4,i)

    sns.boxplot(x=col,data=df,color='pink')

    plt.title(f"Box Plot of {col}")

    plt.tight_layout()
#Distribution Plot of independent variables an it seems  all variables are Normally distributed.

plt.figure(figsize=(20,3))

for i,col in zip(range(1,4),df.columns):

    plt.subplot(1,4,i)

    sns.distplot(a=df[col],color='orange')

    plt.tight_layout()
#Box Plot and Distribution Plot for Dependent variable PROFIT

plt.figure(figsize=(20,3))



plt.subplot(1,2,1)

sns.boxplot(df.Profit,color='#005030')

plt.title('Box Plot of Profit')



plt.subplot(1,2,2)

sns.distplot(a=df.Profit,color='#500050')

plt.title('Distribution Plot of Profit')

plt.show()
#This is the outlier, since we see blow the R&D is higly correlated to the Proft

#Here R&D spend is zero so its obious the profit is very low for this startup

df[df['Profit']<25000]
df[df['R&D Spend']<10000]
#After observing above few startup data, We can say that for Index 49 (which is Outlier) there is some error in Profit

#typo error maybe. Sonce Profit for other startups with very R&D Spend is much higher than this

#We will be removing the outlier from the dataset

df.drop(index=49,axis=0,inplace=True)

df.shape
#Distribution Plot of independent variables an it seems  all variables are Normally distributed.

plt.figure(figsize=(20,3))

for i,col in zip(range(1,4),df.columns):

    plt.subplot(1,4,i)

    sns.scatterplot(x=col,y='Profit',data=df,color='blue')

    plt.title(f"{col} vs Profit")

    plt.tight_layout()
#we can see that R&D is highly linearly correalted with Profit

#let us look at correlation matrix

plt.title("Correlation Matrix")

sns.heatmap(data=df.corr(),annot=True,cmap='coolwarm',linewidths=0.1)
#This shows linear relationship between R&D Spend and Marketing Spend

sns.scatterplot(x='R&D Spend',y='Marketing Spend',data=df)

plt.title("R&D Spend vs Marketing Spend")
df1 = df.copy()

df1.head()
#droping column Marketing Spend (because of Multicollinearity with R&D Spend) 

#droping column Adminstration(because of very low correlation with Proft)

df2 = df1.drop(columns=['Marketing Spend','Administration'],axis=1)

df2.head()
sns.heatmap(df2.corr(),annot=True)
#we have one Catgorical variable column also 'State'

#Lets explore and analyse it



df2.State.unique()
#There are three unique states and their counts are given below. They are equally distributed.



df2.groupby('State')['State'].count()
#We will convert this column into dummy variables

df3 = pd.get_dummies(data=df2)

df3.head()
#To avoid dummy variable Trap, we will drop one dummy variable. Let us remove State_California

df4 = df3.drop(labels=['State_California'],axis=1)

df4.head()
#Now we are done with data preprocessing steps

#Now will split our dataset into Dependent variable and Independent variable



X = df4.iloc[:,[0,2,3]].values

y = df4.iloc[:,1].values
print(f"Shape of Dependent Variable X = {X.shape}")

print(f"Shape of Independent Variable y = {y.shape}")
#Now we will spit our data into Train set and Test Set



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 101)
print(f"Shape of X_train = {X_train.shape}")

print(f"Shape of X_test = {X_test.shape}")

print(f"Shape of y_train = {y_train.shape}")

print(f"Shape of y_test = {y_test.shape}")
#Now we will build regression model on Training Set and Test it on our Test Set



from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X=X_train,y=y_train)
#Now it's time to test the accuracy of the model on our Test Data

#this is very good accuracy on training set

lm.score(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(lm,X_train,y_train,cv=5)

print(f"Accuracies obtained from 5-cross validation = {accuracies}")

print(f'Mean of all accuracies = {accuracies.mean()}')

print(f"Standard Deviation of accuracies = {accuracies.std()}")
#from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=lm,param_grid={'normalize':[True,False]})

grid_search = grid_search.fit(X_train,y_train)

print(f"Best Parameter for our model is {grid_search.best_params_}")

print(f"Best score for the model is {grid_search.best_score_}")
# Taking best parameter

lm = LinearRegression(normalize=True)

lm.fit(X_train,y_train)
#Now it's time to test the accuracy of the model on our Test Data

#this is very good accuracy on training set

lm.score(X_train,y_train)
#Since we have already taken best parameter for our linear model.

#Now we can see how model performs on test dataset

y_pred = lm.predict(X_test)

data = {'y_test':y_test,'y_pred':y_pred.round(2)}

pd.DataFrame(data=data)
#coefficients of regression model

coeff = f'Profit = ({lm.intercept_} x Bias) '

for i,col in zip(range(3),df4.columns[[0,2,3]]):

    coeff+=f'+\n ({lm.coef_[i]} x {col}) '



print(coeff)
plt.title('Residual Plot',size=20)

sns.residplot(y_test,y_pred,color='purple')

plt.xlabel('y_pred',size=15)

plt.ylabel('Residues',size=15)



#we can not see any pattern in the plot => model is good
sns.scatterplot(y_test,y_pred)

plt.xlabel('y_test',size=15)

plt.ylabel('y_pred',size=15)
from sklearn import metrics

r2= metrics.r2_score(y_test,y_pred)

N,p = X_test.shape

adj_r2 = 1-((1-r2)*(N-1))/(N-p-1)

print(f'R^2 = {r2}')

print(f'Adjusted R^2 = {adj_r2}')