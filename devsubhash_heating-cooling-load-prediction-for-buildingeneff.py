# getting started with the model 

# importing required libraries/packages 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Data has been converted from .xlsx to .csv before importing



# Importing and Reading the Dataset

df_en= pd.read_csv('../input/EnergyEff_UCI.csv')
df_en.describe().iloc[:3]
# check to see if there are any missing entries

df_en.info()
#check column names

df_en.columns
#correlation map for features

f,ax = plt.subplots(figsize=(7, 7))

ax.set_title('Correlation map for variables')

sns.heatmap(df_en.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap="YlOrRd")
#Getting an idea about the distribution of Relative Compactness vs Heating Load

p = sns.barplot(data=df_en, x = 'Relative Compactness',y='Heating Load', palette='muted')
#Getting an idea about the distribution of Relative Compactness vs Cooling Load

p = sns.barplot(data=df_en, x = 'Relative Compactness',y='Cooling Load', palette='muted')
#Getting an idea about Surface Area vs Heating Load

p = sns.barplot(data=df_en, x = 'Surface Area',y='Heating Load', palette='muted')
#Getting an idea about Surface Area vs Cooling Load

p = sns.barplot(data=df_en, x = 'Surface Area',y='Cooling Load', palette='muted')
#Getting an idea about Wall Area vs Heating Load

p = sns.barplot(data=df_en, x = 'Wall Area',y='Heating Load', palette='muted')
#Getting an idea about Wall Area vs Cooling Load

p = sns.barplot(data=df_en, x = 'Wall Area',y='Cooling Load', palette='muted')
#Getting an idea about Roof Area vs Heating Load

p = sns.barplot(data=df_en, x = 'Roof Area',y='Heating Load', palette='muted')
#Getting an idea about Roof Area vs Cooling Load

p = sns.barplot(data=df_en, x = 'Roof Area',y='Cooling Load', palette='muted')
#Getting an idea about Overall Height vs Heating Load

p = sns.barplot(data=df_en, x = 'Overall Height',y='Heating Load', palette='muted')
#Getting an idea about Overall Height vs Cooling Load

p = sns.barplot(data=df_en, x = 'Overall Height',y='Cooling Load', palette='muted')
g = sns.pairplot(df_en,palette="husl")
#converting float outcomes to int for the regressor and checking Datatypes

df_en.astype({'Heating Load': 'int64','Cooling Load': 'int64'}).dtypes
# Defining X and y

X = df_en.drop(['Heating Load','Cooling Load'], axis=1)

y = df_en['Heating Load']



# Training the model

from sklearn.model_selection import train_test_split

X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42) # 80-20 split



# Checking split 

print('X_train:', X_train.shape)

print('y_train:', y_train.shape)

print('X_test:', X_test.shape)

print('y_test:', y_test.shape)
#Random Forest Trial

# Load random forest classifier 

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor 

 

 # create regressor object 

reg= RandomForestRegressor(n_estimators = 200, random_state = 42) 

  

# fit the regressor with x and y data 

reg.fit(X_train, y_train)   



# predict the response



acc_rfr = round(reg.score(X_test,y_test)* 100, 2)

print("Random Forest Regressor Accuracy:",acc_rfr,"%")
y_pred = reg.predict(X_test)

df_en = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_en.iloc[:5]
plt.scatter(y_test,y_pred)
# create regressor object



from sklearn.linear_model import LinearRegression

reg = LinearRegression()



# fit the regressor with x and y data 

reg.fit(X_train, y_train)



# predict the response



acc_linreg = round(reg.score(X_test,y_test)* 100, 2)

print("Linear Regression Accuracy:",acc_linreg,"%")
y_pred = reg.predict(X_test)

df_en = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_en.iloc[:5]
plt.scatter(y_test,y_pred)
#visualizing both algorithm accuracies for Heating Load using Matplotlib

acc = ('Random Forest Regression', 'Linear Regression')

x_pos = np.arange(len(acc))

accuracy = [acc_rfr, acc_linreg]

plt.figure(figsize = (7,4))

plt.bar(x_pos, accuracy, alpha=0.7,align='center', color='b')

plt.xticks(x_pos, acc)

plt.ylabel('Accuracy (%)')

plt.title('Regressor Accuracies')

plt.show()