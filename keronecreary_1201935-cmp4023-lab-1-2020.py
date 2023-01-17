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


student_data = pd.read_csv("../input/DWDM Dirt Lab 1.csv")

student_data.head()
student_data.shape
region_df = student_data['Region'].value_counts().rename_axis('regions').reset_index(name='counts') 

region_df
region_df.iloc[2] # 3rd most occuring Region is Victoria
import matplotlib.pyplot as plt # import


region_df[region_df.counts > 8].plot(kind='bar', x= "regions", y="counts", title= "Region Occurance over 8", figsize=(12,9))
gender_df = student_data['Gender'].value_counts().rename_axis('gender').reset_index(name='counts')

gender_df
#changing F to Female

student_data['Gender'] = np.where(student_data['Gender']!='Male', 'Female', student_data['Gender'])



gender_df = student_data['Gender'].value_counts().rename_axis('gender').reset_index(name='counts')

gender_df

# Create a list of colors

colors = ["#346beb", "#D69A80"]



# Create a pie chart

student_data['Gender'].value_counts().plot.pie(colors=colors,autopct='%1.1f%%', fontsize=9, figsize=(6, 6))
student_data.dtypes #detemine the datatype of the column




student_data['Footlength_cm'] =  np.where(student_data['Footlength_cm'].str.isdecimal()==True,student_data['Footlength_cm'],'NaN')

student_data['Armspan_cm'] =  np.where(student_data['Armspan_cm'].str.isdecimal()==True,student_data['Armspan_cm'],'NaN')

student_data['Height_cm'] =  np.where(student_data['Height_cm'].str.isdecimal()==True,student_data['Height_cm'],'NaN')



# Reaction time is a float variable therefore it must be numerical





student_data.head(10)
student_data.isnull().sum(axis=0)
student_data = student_data.drop([0], axis=0 )

student_data

student_data = student_data.dropna(axis=0)

student_data.isnull().sum(axis=0)

student_data


student_data[[ 'Age-years', 'Height_cm', 'Footlength_cm','Armspan_cm','Languages_spoken', 'Travel_time_to_School','Reaction_time', 'Score_in_memory_game' ]].astype(float).corr()
extract = student_data[['Armspan_cm','Footlength_cm', 'Reaction_time','Height_cm']]

extract



#convert columns to float datatype

extract["Armspan_cm"] = extract.Armspan_cm.astype(float)

extract["Footlength_cm"] = extract.Footlength_cm.astype(float)

extract["Height_cm"] = extract.Height_cm.astype(float)


extract.plot(kind='scatter', x='Footlength_cm', y="Armspan_cm", title="Relationship between Foot Length and Arm Span", figsize=(12,9))
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

from sklearn import linear_model

# Create an instance of linear regression

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.coef_
X_train.columns
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
#intercept

reg.intercept_


# Make predictions using the testing set

test_predicted = reg.predict(X_test)





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



reg.score(X_test,y_test)
reg.predict([[170,20,120]])
import sklearn.metrics as metrics
# mean squared error MSE

print("Mean squared error: %.2f" % metrics.mean_squared_error(y_test, test_predicted))
# root mean squared error RMSE

import math  

print("Root Mean squared error: %.2f" % math.sqrt(metrics.mean_squared_error(y_test, test_predicted)))
# mean absolute error MAE

print("Mean absolute error: %.2f" % metrics.mean_absolute_error(y_test, test_predicted))