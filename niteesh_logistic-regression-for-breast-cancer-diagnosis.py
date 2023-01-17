#Adding all the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot

%matplotlib inline

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.cross_validation import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn import metrics #
data = pd.read_csv("../input/data.csv",header=0)# here header 0 means the 0 th row is our coloumn 

                                                # header in data
#printing top two rows of data

print(data.head(2))
#look what kind of data we have

data.info()
# we will drop the unnamed column

data.drop("Unnamed: 32",axis=1,inplace=True)

#axis=1 means it will drop column
#check the remaining column

data.columns
# we also don't need ID column for aur analysis so we will also drop it

data.drop("id",axis=1,inplace=True)
# now we will divide data into three parts

features_mean= list(data.columns[1:11])

features_se= list(data.columns[11:20])

features_worst=list(data.columns[21:31])

print(features_mean)

print("-----------------------------------")

print(features_se)

print("------------------------------------")

print(features_worst)
# now we will map diagnosis column into integer values M=1 and B=0

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
# this will describe the all statistical function of our data

data.describe()
#get the frequency of cancer stages

sns.countplot(data['diagnosis'],label='count')
# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are

# dependenig on each other so we should avoid it because what is the use of using same column twice

# lets check the correlation between features

# now we will do this analysis only for features_mean then we will do for others and will see who is

#doing best

corr = data[features_mean].corr() # .corr is used for find corelation

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           xticklabels= features_mean, yticklabels= features_mean,

           cmap= 'coolwarm') 

# for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)
# now these are the variables which will use for prediction

prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
#now split our data into train and test

train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test

# we can check their dimension

print(train.shape)

print(test.shape)
# taking training and testing data

train_X = train[prediction_var]# taking the training data input 

train_y=train.diagnosis# This is output of our training data

# same we have to do for test

test_X= test[prediction_var] # taking test data inputs

test_y =test.diagnosis   #output value of test dat

# Now explore a little bit more

# now from features_mean i will try to find the variable which can be use for classify

# so lets plot a scatter plot for identify those variable who have a separable boundary between two class

#of cancer

# Lets start with the data analysis for features_mean

# Just try to understand which features can be used for prediction

# I will plot scatter plot for the all features_mean for both of diagnosis Category

# and from it we will find which are easily can used for differenciate between two category



color_function = {0: "blue", 1: "red"}

# Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis"].map(lambda x: color_function.get(x))

# mapping the color fuction with diagnosis column

pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (15, 15)); 

# plotting scatter plot matrix
# So predicton features will be 

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
def model(model,data,prediction,outcome):

    # This function will be used for to check accuracy of different model

    # model is the m

    kf = KFold(data.shape[0], n_folds=10) # if you have refer the link then you must understand what is n_folds

    
# As we are going to use many models lets make a function

# Which we can use with different models

def classification_model(model,data,prediction_input,output):

    # here the model means the model 

    # data is used for the data 

    #prediction_input means the inputs used for prediction

    # output mean the value which are to be predicted

    # here we will try to find out the Accuarcy of model by using same data for fiiting and 

    #comparison for same data

    #Fit the model:

    model.fit(data[prediction_input],data[output]) #Here we fit the model using training set

  

    #Make predictions on training set:

    predictions = model.predict(data[prediction_input])

  

    #Print accuracy

    # now checkin accuracy for same data

    accuracy = metrics.accuracy_score(predictions,data[output])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

 

    

    kf = KFold(data.shape[0], n_folds=5)

    # About cross validitaion please follow this link

    #https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-

    #python-r/

    #let me explain a little bit data.shape[0] means number of rows in data

    #n_folds is for number of folds

    error = []

    for train, test in kf:

        # as the data is divided into train and test using KFold

        # now as explained above we have fit many models 

        # so here also we are going to fit model

        #in the cross validation the data in train and test will change for evry iteration

        train_X = (data[prediction_input].iloc[train,:])# in this iloc is used for index of trainig data

        # here iloc[train,:] means all row in train in kf amd the all columns

        train_y = data[output].iloc[train]# here is only column so it repersenting only row in train

        # Training the algorithm using the predictors and target.

        model.fit(train_X, train_y)

    

        # now do this for test data also

        test_X=data[prediction_input].iloc[test,:]

        test_y=data[output].iloc[test]

        error.append(model.score(test_X,test_y))

        # printing the score 

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
outcome_var= "diagnosis"

# lets try with logistic regression

model=LogisticRegression()

classification_model(model,data,prediction_var,outcome_var)