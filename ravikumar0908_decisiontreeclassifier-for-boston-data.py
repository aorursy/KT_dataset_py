#problems with Linear/Regressor model

#1. they are bulild around LINE assumptions

#2. They nedd preprocessing of missing values, outlier handling

#3. They can only work better if data is linearly separable otherwise there would be high error rates in prediction

#4. They use sinlg esplit condition to describe the regression/regressor plot.

# so more chances that accuracy wont be good when data is non-linear distributed.





# Classfiation/Regression using the CART based sinlge -Decision Tree



# Accuracy of various models

# Acc of Linear/Logistic Regression < single decision tress <ensembile Trees < RANDOM FOREST





#benefit of Decision Trees

# 1. They use Aggregated Behaviours to decide the split conditions of the DAta, but linear regression/regressor model

# only use single split condition to describe the data.



# 2. No Assumption are required to perform regression/regressor on them

# 3. No missing value imputation required.

# 4. No variable selection required, but in Linear model the variable must be normally distributed with Target variable.



#problems of Single CART Trees

#1. Overfiting :- it a model could give good performance for Training data but on Test data the 

# result are poor so if Accuracy of training data != Accuracay of Test Data it means we have Overfitting in model



    #solution of overfitting 

        # 1. pruning (in case when we have single-tress)

        # 2. control growth :- a) based on depth 

                        #b)based on max no of obs for a split

                        #c) based on mini obs for split 

                        #d) minimum observation in the leaf node



#2. Class -Imbalance Issues :- 



    #solution :- Resolved using the Resamping with replacement



#note :- To cross check that this REsampling is not resulting in Biased model builing,

        # we use Cross-Validation on this resample with the model. if model passes

        # the cross-validation , then we are sure that the model would be a generalized mode

#whats a Generalized model:- its model which produce same errors across any new Test Dataset,

                #as it produced on the the Test Data, duing it model building process.



#3. Biasness towards continuous variables,as more split points are available for them.

    #solution :- we prefer CART based models for classification mostly,



# so we go for Ensembling ( parallel Learning)

# in it we divide the class imlbalance into multiple bags with improved proportions of observation

        # so we try to learn from each bag.



# This is called Ensembling 



#problem with Ensembling :- only do Observation Sampling for each bag, but no Feature sampling in each bag

#### so solution is given by RandomForest  :- it allows Observation Sampling + Feature Sampling for each bag      



# now doing classification based on Single -Cart Trees i.e Decision Tree 

        

import pandas as pd



#boston=pd.read_csv("Decision Trees\\workspace\\labelledBoston.csv")    

boston = pd.read_csv("../input/labelledBoston.csv")

  

       

boston.head()        

boston.shape        

#now split the data. using test_train

#to normalize the data_

from sklearn.preprocessing import normalize

#now import the decisionRegressor from sklearn

from sklearn.tree import DecisionTreeClassifier

#now import the test_train_split

from sklearn.model_selection import train_test_split

# to calculate the confusion matrix

from sklearn.metrics import confusion_matrix

# to calculate the accuracy score 

from sklearn.metrics import accuracy_score 



targetVariable=boston["labelled"]



type(targetVariable)

#how to know class imbalanced

counts=targetVariable.value_counts()

counts

#total class 0 percent

percentOfClass0=counts[0]/506

percentOfClass0

#total class 1 percent

percentOfClass1=counts[1]/506

percentOfClass1



#how to knoe class imbalance in the dataset

# count each class type



#no preprocessing is require as it is Single-Decision based learning

# syntax : train_test_split(X is dataset,y=target variable,test_split=.25,random_state=1234)

x_train,x_test,y_train,y_test=train_test_split(

                    boston,targetVariable,test_size=0.25,random_state=12

                                    )

#shape

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)







#now applying the Superviszed CART based model (1-single Decision Tree)

#to train the model and Test the model

clf=DecisionTreeClassifier()

clf=clf.fit(x_train,y_train)





#now check the training accuracy

#if training accuracy != Test Accuracy then it means OverFitting is happening



predictTrain=clf.predict(x_train)

confusion_matrix(y_train,predictTrain)

len(predictTrain)



x_train["labelled"].value_counts()







xTestActualLabels=x_test["labelled"]



xTestActualLabels.value_counts()





#now predict with classifier

predict=clf.predict(x_test)

len(predict)

#find the confusion matrix for training

confusion_matrix(y_test,predict)







### So 1. Accuracy is 100% means our classifier is good

### 2.The Test and Train is accuracy is matching it means No overfitting happening



boston["labelled"].value_counts()









#---- :) :) Enjoy the Data Science











































       

        

        