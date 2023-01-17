# House price prediction using linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
% matplotlib inline 
#import dataset of  House Sales in King County, USA
df=pd.read_csv("../input/kc_house_data.csv")
# Storing values of sqft_living column into space object
space=df['sqft_living']
type(space)
#check howmuch data is missing in square feet living column
sum(np.isnan(space)) / space.shape[0] 
#it dispaled 0.0 means space array  has no nan value

price=df['price']
#Finding NAN value in price 
sum(np.isnan(price)) / price.shape[0] 
#it displayed 0.0 means price array has no nan value
# Storing space data into array x
x=np.array(space).reshape(space.shape[0],1)
y=np.array(price)

#Splitting space and price data into training set and testing set
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=40)
lin_regressor=LinearRegression()
# Trained the linear regression algorithm
lin_regressor=lin_regressor.fit(xtrain,ytrain)
predict=lin_regressor.predict(xtest)
# measuring accuracy of the model
accuracy=lin_regressor.score(xtest,ytest)

#Visualizing the training results
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,lin_regressor.predict(xtrain),color='blue')
plt.xlabel('Living Square feet ',color='blue',fontsize=20)
plt.ylabel('price',color='red',fontsize=20)
plt.title('Training Result visualization',color='green',fontsize=20)
plt.show()

#Visualizing the testing results
plt.scatter(xtest,ytest,color='blue')
plt.plot(xtest,lin_regressor.predict(xtest),color='red')
plt.xlabel('Living Square feet ',color='blue',fontsize=20)
plt.ylabel('price',color='red',fontsize=20)
plt.title('Testing Result visualization',color='green',fontsize=20)
plt.show()


