import pandas as pd

import matplotlib.pyplot as plot 

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression





df=pd.read_csv('weatherhistory.csv')





#df['Formatted Date'].unique()

#df.describe()



#label encoding the text input data



label_encoder=preprocessing.LabelEncoder()

df['Formatted Date']=label_encoder.fit_transform(df['Formatted Date'])

df['Formatted Date']

df['Formatted Date'].unique()





#scatterplot for visualising the data pattern



scatter_plot=df.plot.scatter(x='Formatted Date',y='Humidity',c='Apparent Temperature (C)',colormap='viridis')

scatter_plot





#taking specific columns into consideration



df=df.filter(['Humidity','Apparent Temperature (C)','Temperature (C)','Wind Bearing (degrees)','Pressure (millibars)','Wind Speed (km/h)'])





#train_test split

x_train,x_test,y_train,y_test=train_test_split(df.drop('Apparent Temperature (C)',axis=1),df['Apparent Temperature (C)'],test_size=0.30,random_state=276)





#using linear regression model





model=LinearRegression()

model.fit(x_train,y_train)

predictions=model.predict(x_test)

predictions





#evaluating the performance of the model



from  sklearn import metrics



mse=metrics.mean_absolute_error(y_test,predictions)



from sklearn.metrics import r2_score



r2=metrics.r2_score(y_test,predictions)





r2,mse
