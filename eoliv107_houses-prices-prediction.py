# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing packages that will be used
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#Reading the dataset
data = pd.read_csv('../input/housepricing/HousePrices_HalfMil.csv')
#Viewing first 5 rows of the database
data.head()
#Treating the names of the variables
variables = {'White Marble': 'White_Marble',
             'Black Marble': 'Black_Marble',
             'Indian Marble': 'Indian_Marble',
             'Glass Doors': 'Glass_Doors',
             'Swiming Pool': 'Swiming_Pool'}
data.rename(columns=variables, inplace = True)
data.head()
# Analysing nulls contents
pd.DataFrame(data.isnull().sum(), columns=['Qty_Null_Values'])
#Size of the DataSet
print('The size of the dataset is {} objects and {} variables'.format(data.shape[0], data.shape[1]))
#Type of variables
data.info()
#Descriptive statistics preliminary analysis
data.describe()
#Correlation matrix 
data.corr()
#Ploting correlatinig graph to indetify which variables are more relevant for the regression
ax = sns.heatmap(data.corr(), annot= True, linewidths=.1, annot_kws={'size':9})
ax.figure.set_size_inches(16,12)
ax.set_title('Correlation of variables', fontsize = 16, loc = 'left')
sns.set_style('darkgrid')
ax = sns.boxplot(data.Prices, width = 0.3, orient= 'v', palette='Oranges')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_title("Boxplot of the prices", fontsize = 18, loc = 'left')
ax = sns.boxplot(y = data.Prices, x = data.White_Marble, width = 0.3, orient= 'v', palette= 'prism')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_title("Boxplot of the prices vs White Marble placed", fontsize = 18, loc = 'left')
ax.set_xticklabels(['White Marble - Not', 'White Marble - Yes'], fontsize = 12)
ax = ax
ax = sns.boxplot(y = data.Prices, x = data.Floors, width = 0.3, orient= 'v', palette= 'prism')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_xlabel('')
ax.set_title("Boxplot of the prices vs Quantity of floors", fontsize = 18, loc = 'left')
ax.set_xticklabels(['One-story house', 'Double-story house'], fontsize = 12)
ax = ax
ax = sns.boxplot(y = data.Prices, x = data.City, width = 0.3, orient= 'v', palette= 'prism')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_title("Boxplot of the prices vs City", fontsize = 18, loc = 'left')
ax.set_xticklabels(['City A', 'City B', 'City C'], fontsize = 12)
ax = ax
ax = sns.boxplot(y = data.Prices, x = data.Fiber, width = 0.3, orient= 'v', palette= 'prism')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_title("Boxplot of the prices vs Type of Fiber", fontsize = 18, loc = 'left')
ax.set_xticklabels(['Fiber A', 'Fiber B'], fontsize = 12)
ax = ax
ax = sns.boxplot(y = data.Prices, x = data.Baths, width = 0.4, orient= 'v', palette= 'prism')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_title("Boxplot of the prices vs Baths", fontsize = 18, loc = 'left')
ax = ax
ax = sns.lineplot(y= data.Prices, x = data.Area)
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_xlabel('Area')
ax.set_title("Prices vs Area", fontsize = 18, loc = 'left')
ax = ax
ax = sns.boxplot(y = data.Prices, x = data.Glass_Doors, width = 0.4, orient= 'v', palette= 'prism')
ax.figure.set_size_inches(12,8)
ax.set_ylabel('House Prices', fontsize = 12)
ax.set_title("Boxplot of the prices vs Glass Doors", fontsize = 18, loc = 'left')
ax.set_xticklabels(['Glass Door - No', 'Glass Door - Yes'], fontsize = 12)
ax = ax
ax = sns.distplot(data.Prices, color='Blue')
ax.figure.set_size_inches(18,8)
ax.set_xlabel('Prices of the houses', fontsize = 12)
ax.set_title("Frequency Distribution of House Prices", fontsize = 18, loc = 'left')
ax = ax
#Crianting the variables to consider on the prediction
Y = data['Prices']
X = data[['White_Marble', 'Floors', 'City','Fiber', 'Baths', 'Area', 'Glass_Doors']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.35)
#Instantiating and estimating the regression model
regression_model = LinearRegression().fit(X_train, Y_train)
#Obtaining the coefficient of determination (R²) by train datas
print('R² = {}'.format(regression_model.score(X_train, Y_train).round(2)))
#Loading prediction for test datas
Y_previsto = regression_model.predict(X_test)
#Obtaining the coefficient of determination (R²) for model datas
print('R² = %s' % metrics.r2_score(Y_test, Y_previsto).round(2))
#Analysing how the predicting were fited to the model
print('Real price on database $ {0:.2f}'.format(data[0:1]['Prices'][0]))
print('Predicted Price considering the same variables of the database $ {0:.2f}.'.format(regression_model.predict(data[0:1][['White_Marble', 'Floors', 'City','Fiber', 'Baths', 'Area', 'Glass_Doors']])[0]))
# Considering the punctual values above we can see the error of the model
print('Error $ {0:.2f}'.format(data[0:1]['Prices'][0] - regression_model.predict(data[0:1][['White_Marble', 'Floors', 'City','Fiber', 'Baths', 'Area', 'Glass_Doors']])[0]))
print('Error {0:.1f} %'.format((data[0:1]['Prices'][0] - regression_model.predict(data[0:1][['White_Marble', 'Floors', 'City','Fiber', 'Baths', 'Area', 'Glass_Doors']])[0])/data[0:1]['Prices'][0] * 100))
#We can export for a program the model os prediction
import pickle
output = open('modelo_projeto', 'wb')
pickle.dump(regression_model, output)
output.close()
#Importing model
modelo = open('modelo_projeto','rb')
lm_new = pickle.load(modelo)
modelo.close()
# Importing libraries
from ipywidgets import widgets, HBox, VBox
from IPython.display import display

# Creating forms
White_Marble = widgets.Text(description="White Marble")
Floors = widgets.Text(description="Floors")
City = widgets.Text(description="City")
Fiber = widgets.Text(description="Fiber")
Baths = widgets.Text(description="Baths")
Area = widgets.Text(description="Area")
Glass_Door = widgets.Text(description='Glass Door')

botao = widgets.Button(description="Simular")

# Creating and positioning the screen
left = VBox([White_Marble, Floors, City, Fiber])
right = VBox([Baths, Area, Glass_Door])
inputs = HBox([left, right])

# Function for the simulation
def simulador(sender):
    entrada=[[
                float(White_Marble.value if White_Marble.value else 0), 
                float(Floors.value if Floors.value else 0), 
                float(City.value if City.value else 0), 
                float(Fiber.value if Fiber.value else 0), 
                float(Baths.value if Baths.value else 0), 
                float(Area.value if Area.value else 0),
                float(Glass_Door.value if Glass_Door.value else 0)
             ]]
    print('$ {0:.2f}'.format(lm_new.predict(entrada)[0]))
    
# Setting function for the button
botao.on_click(simulador)
display(inputs, botao)
