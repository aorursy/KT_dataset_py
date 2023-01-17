import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns



data_fire = pd.read_csv('../input/forestfires.csv')
from sklearn import metrics #import metrices from sklearn for evaluation

from sklearn.model_selection import train_test_split #import train_test_split

from sklearn import preprocessing #import preprocessing package

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge 

import numpy as np



Xtemp=data_fire.temp#get X values from dataset

X=np.array(Xtemp).astype("float")

print(X)

Ytemp=data_fire.area #get Y values from dataset

Y=np.array(Ytemp).astype("float")

print(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)



model=LinearRegression()

#x_train=x_train[:,5]

x_train=x_train.reshape(-1,1)

model.fit(x_train,y_train)

#x_test=x_test[:,5]

x_test=x_test.reshape(-1,1)



y_pred=model.predict(x_test)



#put title

plt.title("Linear Regression on Forest fire data")

#pt label in x axis

plt.xlabel("temperature")

#put label in y axis

plt.ylabel("area")

plt.scatter(x_test,y_test,color='red',s=8)

plt.plot(x_test,y_pred,color='green')

#display the graph

plt.show()



#Score



print('\tRed dotes indicate the actual values in the data set')

print('\tGreen dotes indicate the predicted values in the data set\n')

print("\t:Mean_absolute_error:",metrics.mean_absolute_error(y_test,y_pred))

print("\t:Mean_squared_error:",metrics.mean_squared_error(y_test,y_pred))

print("\t:Root_mean_squared_error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("\t:R--squared measure:",metrics.r2_score(y_test,y_pred))



#RIDGE REGRESSION



ridgeReg=Ridge(alpha=1.5,normalize=True)

ridgeReg.fit(x_train,y_train)

y_pred=ridgeReg.predict(x_test)



#put title

plt.title("Ridge Regression on Boston house price data")

#pt label in x axis

plt.xlabel("CRIS Value")

#put label in y axis

plt.ylabel("house price")

plt.scatter(x_test,y_test,color='red',s=8)

plt.plot(x_test,y_pred,color='green')

#display the graph

plt.show()



print('\tRed dotes indicate the actual values in the data set')

print('\tGreen dotes indicate the predicted values in the data set\n')

print("\t:Mean_absolute_error:",metrics.mean_absolute_error(y_test,y_pred))

print("\t:Mean_squared_error:",metrics.mean_squared_error(y_test,y_pred))

print("\t:Root_mean_squared_error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("\t:R--squared measure:",metrics.r2_score(y_test,y_pred))
#LASSO REGRESSION



import numpy as np

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt



data_fire = pd.read_csv('../input/forestfires.csv')

x,y=data_fire['X'],data_fire['Y'] #extract data target from dataset

all_feature_names=data_fire.feature_names #get all feature names

print("\nInitial Feature\n") #print initial feature names

features_count=0

for feature in all_feature_names:

    features_count+=1

    print(str(features_count)+""+feature)



alpha_value=1

while(alpha_value<=10):

        clf=Lasso(alpha=alpha_value)

        sfm=SelectFromModel(clf)

        sfm.fit(x,y)

        n_features=sfm.transform(x).shape[1]

        objects=('Before LASSO','After LASSO')

        count=[features_count,n_features]

        plt.bar(objects, count, align='center')

        plt.ylabel('Features coun')

        plt.title('Feature selection using LASSO(alpha='+str(alpha_value)+')')

        plt.show()

        print ('The selected features when alpha ='+str(alpha_value)+"are \n")

        selected_feature_indices=sfm.get_support(indices=True)

        count=1

        for index in selected_feature_indices:

                print(str(count)+" "+all_feature_names[index])

                count+=1

        alpha_value+=3