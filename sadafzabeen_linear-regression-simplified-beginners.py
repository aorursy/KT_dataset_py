#This is a very simplified notebook to understand how to code Linear Regression model.



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
data=pd.read_csv("../input/student-performance-data-set/student-por.csv") #Read the file through pandas library & if you are running the code in your local use 'Sep=;' since its comma separated file





data.head()  #To display the top 5 rows & columns from the dataframe
data.shape #Check the shape of the dataframe
data.isnull().sum() #Check if you have any null values in your dataset
#We have 33 cols or Attributes & we are not going to use all of them but a few which are actually useful for our prediction



data=data[['absences','failures','G1','G2','G3','studytime']] #Attributes of our use , You can use different attributes of interest & play around



#Now we are going to create 2 arrays, One which is going to store our predictors & one to store our output variable

#in this case G3(GRADE FOR SEM3 ) is our output variable or label



predict='G3'                    # We just stored our output variable in a new variable predict

x=np.array(data.drop(['G3'],1)) #We want to drop G3 since its our output variable & 'axis=1'simply means drop the entire column

y=np.array(data[predict])       #storing our output variable in y

#now we are going to split our data into 4 arrays. We will call the train_test_split from sklearn model & store 33% of our data in test set

 

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.33) 
linear=LinearRegression().fit(x_train, y_train)  #Fit our model on x_train & y_train

accuracy=linear.score(x_test,y_test)  

print(accuracy)
predictions=linear.predict(x_test)   #Call the predict function on x_test



for i in range(len(predictions)):

    print (predictions[i],x_test[i],y_test[i])  #Printing predictions , original x_test values,& original y_test values