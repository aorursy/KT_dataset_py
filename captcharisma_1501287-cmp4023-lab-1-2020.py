# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt # import

from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from numpy import cov

%matplotlib inline

plt.rcParams['figure.figsize']=10,8

matplotlib.style.use('ggplot')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


DWDM_Dirt_Lab = pd.read_csv("../input/DWDM Dirt Lab 1.csv",encoding='ISO-8859-1')



DWDM_Dirt_Lab.head()

DWDM_Dirt_Lab.shape
region_df = DWDM_Dirt_Lab['Region'].value_counts().rename_axis('regions').reset_index(name='counts') 

region_df
region_df.iloc[2] #3rd most occuring Region is Victoria
DWDM_Dirt_Lab.columns

region_df= DWDM_Dirt_Lab['Region'].value_counts().to_frame() #create dataframe



region_df[region_df.Region > 8].plot(kind='bar', y="Region", title="Region Occurence over 8", figsize=(12,9))
gender_df = DWDM_Dirt_Lab['Gender'].value_counts().rename_axis('gender').reset_index(name='count')

gender_df
#changing F to Female

DWDM_Dirt_Lab['Gender'] = np.where(DWDM_Dirt_Lab['Gender']!='Male', 'Female', DWDM_Dirt_Lab['Gender'])



gender_df = DWDM_Dirt_Lab['Gender'].value_counts().rename_axis('gender').reset_index(name='counts')

gender_df
# Create a list of colors

colors = ["#346beb", "#D69A80"]

DWDM_Dirt_Lab['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')

DWDM_Dirt_Lab.dtypes
DWDM_Dirt_Lab = pd.read_csv("../input/DWDM Dirt Lab 1.csv")

DWDM_Dirt_Lab['Footlength_cm'] =  np.where(DWDM_Dirt_Lab['Footlength_cm'].str.isalnum()==True,DWDM_Dirt_Lab['Footlength_cm'],'NaN')

DWDM_Dirt_Lab['Armspan_cm'] =  np.where(DWDM_Dirt_Lab['Armspan_cm'].str.isalnum()==True,DWDM_Dirt_Lab['Armspan_cm'],'NaN')

DWDM_Dirt_Lab['Height_cm'] =  np.where(DWDM_Dirt_Lab['Height_cm'].str.isalnum()==True,DWDM_Dirt_Lab['Height_cm'],'NaN')



DWDM_Dirt_Lab.head(10)
DWDM_Dirt_Lab.isnull().sum(axis=0)
DWDM_Dirt_Lab= DWDM_Dirt_Lab.drop([0], axis=0 ) 

DWDM_Dirt_Lab

#casewise deletion
DWDM_Dirt_Lab.dropna(axis=0)

DWDM_Dirt_Lab.isnull().sum(axis=0)
DWDM_Dirt_Lab
DWDM_Dirt_Lab[['Age-years', 'Height_cm', 'Footlength_cm','Armspan_cm','Languages_spoken', 'Travel_time_to_School','Reaction_time', 'Score_in_memory_game']].astype(float).corr()

extract = DWDM_Dirt_Lab[['Armspan_cm','Footlength_cm', 'Reaction_time','Height_cm']] 

extract
print(extract)
extract["Armspan_cm"] = extract.Armspan_cm.astype(float)

extract["Footlength_cm"] = extract.Footlength_cm.astype(float)

extract["Height_cm"] = extract.Height_cm.astype(float)
extract.plot(kind='scatter', 

             x='Height_cm', y="Armspan_cm", 

             color="purple",

             figsize=(10,8)

            )

plt.ylabel("Y - Target Variable", size=15)

plt.xlabel('X - Independent Variable', size=15)

plt.suptitle("Relationship between Height and Arm Span"

,size = 15, color='black')

            

# ensure there is not missing data

extract = extract.dropna(axis=0)



extract.isnull().sum(axis=0)
# separate our data into dependent (Y) and independent(X) variables

X_data = extract[['Height_cm','Footlength_cm','Armspan_cm']]

Y_data = extract['Reaction_time']
#splits data into 70/30 for training and testing repectively

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30) 

 
#import linear model package

from sklearn.linear_model import LinearRegression

# Create an instance of linear regression

reg = LinearRegression()
#fit model to training dataset

model_cv=reg.fit(X_train, y_train)
reg.coef_
X_train.columns
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
#intercept

reg.intercept_
#validate regression model

test_predicted = reg.predict(X_test)

test_predicted

#determine residuals

y_test_ar = y_test.values

residuals = y_test_ar - test_predicted



# create dataframe for plot

df1 = pd.DataFrame(residuals)



df1 = df1.rename(index=str, columns={0:'Residuals'})



df2 = pd.DataFrame(test_predicted)

df2 = df2.rename(index=str, columns={0:'Predicted'})





plot_df = pd.concat([df1,df2], axis=1)



plot_df.plot(kind="scatter",

    x='Predicted',

    y='Residuals',

    title="Residuals vs Predicted Values",

    figsize=(12,8)

)
print('Score: ', reg.score(X_test,y_test))
#Intercept

intercept = model_cv.intercept_

#Regression Coefficient

slope = reg.coef_

#R**2

gof = reg.score(X_test,y_test)

line = pd.DataFrame({'y-intercept':intercept,'Regression Coefficient':slope,'Coefficient of Determination':gof})

line
def regFormula(x,y):

    x_df = pd.DataFrame(x)

    y_df = pd.DataFrame("Reaction_time")

    y_pred = (slope * x_df) + intercept #formula

    results = x_df.copy()

    results["Actual_y_value"] =y_df

    results['Predicted_y_value']=y_pred

    #print(results.head(10)) #only show first 10 records

    return results
rng = np.random.RandomState(1)

x2 = 5 * rng.rand(50) #create new x-value

D2 = pd.DataFrame(x2, columns=['x2'])

df_xy_2 = pd.concat([extract,D2], axis =1 )

df_xy_2.head()
reg.predict([[170,20,120]])
import sklearn.metrics as metrics



mae = metrics.mean_absolute_error(y_test, test_predicted)

mse = metrics.mean_squared_error(y_test, test_predicted)

rmse = np.sqrt(mse) # or mse**(0.5)  

r2 = metrics.r2_score(y_test, test_predicted)



print("Results of sklearn.metrics:")

print("MAE:",mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R-Squared:", r2)