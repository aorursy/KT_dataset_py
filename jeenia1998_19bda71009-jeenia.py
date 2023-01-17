# importing all libraries for dataframe



import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter



# importing libraries for model



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
# get data from csv files



train1 = pd.read_csv('../input/bda-2019-ml-test/Train_Mask.csv') # using pandas library to read a csv file

test1 = pd.read_csv('../input/bda-2019-ml-test/Test_Mask_Dataset.csv')
# take a sample of what train and test data looks like
train = train1.drop(['timeindex'],axis=1) # dropping index column from the dataset

train.head() # head display top few rows of the dataset
test = test1.drop(['timeindex'],axis=1)

test.head()
# determine size of dataset



train.shape, test.shape
# mathematical description of train dataset



train.describe().T # T is used for Transpose
# checking for the missing value in both train and test dataset
train.isnull().sum() # displays all the columns with the no. of missing values
test.isnull().sum()
# provide information about types of data we are dealing with
train.info()
test.info()
# ploting histogram for all the columns in the dataset 



train.hist(figsize=(20,20))

plt.show()
# plotting boxplot and distribution plot indivually for all the features

features = {'flag','currentBack','motorTempBack','positionBack','refPositionBack','refVelocityBack','trackingDeviationBack',

            'velocityBack','currentFront','motorTempFront','positionFront','refPositionFront','refVelocityFront',

           'trackingDeviationFront','velocityFront'}



for i in features:

    ax,fig = plt.subplots(1,2,figsize=(20,5))

    box = sns.boxplot(x=train[i], ax = fig[0]) #boxplot using seaborn library

    box_title = box.set_title('Box Plot') # title for the boxplot

    dist = sns.distplot(train[i], ax = fig[1]) #distribution plot using seaborn library

    distplot_title = dist.set_title('distribution plot') #title for distribution plot
# shows the correlation between all the features in a matrix



train.corr() # corr is the function for correlation
# shows the correlation between all the features in a heatmap



corr = train.corr() # correlation matrix

fig = plt.figure(figsize=(15,15)) # for plot size



sns.heatmap(corr,annot=True) # plotting heatmap using seaborn library with correlataion matrix



plt.show() # for showing plot
# defining a function to identify outliers in the dataset



def outliers(df,features):

    outlier_indices = []

    

    for i in features:

        # 1st quartile

        q1 = np.percentile(df[i],25)

        # 3rd quartile

        q3 = np.percentile(df[i],75)

        # IQR

        IQR = q3 - q1

        # outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[i] < q1 - outlier_step) | (df[i] > q3 +outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(j for j,k in outlier_indices.items() if k>2)

    

    return multiple_outliers
# defining a variable tr1 and ts1 which will contain the dataset of important features



tr1 = train[['flag','currentBack', 'motorTempBack', 'positionBack','trackingDeviationBack','velocityBack',

            'currentFront', 'motorTempFront', 'positionFront','trackingDeviationFront']]



ts1 = test[['currentBack', 'motorTempBack', 'positionBack','trackingDeviationBack','velocityBack',

            'currentFront', 'motorTempFront', 'positionFront','trackingDeviationFront']]
# chceking the outliers in selected features



tr1.loc[outliers(tr1,['currentBack', 'positionBack','trackingDeviationBack', 'motorTempFront', 'trackingDeviationFront'])]
ts1.loc[outliers(ts1,['currentBack', 'positionBack','trackingDeviationBack', 'motorTempFront', 'trackingDeviationFront'])]
# dropping the outliers from the dataset



tr1 = tr1.drop(outliers(train,['currentBack', 'positionBack','trackingDeviationBack', 'motorTempFront', 

                             'trackingDeviationFront']), axis=0).reset_index(drop = True)
ts1 = ts1.drop(outliers(test,['currentBack', 'positionBack','trackingDeviationBack', 'motorTempFront', 

                             'trackingDeviationFront']), axis=0).reset_index(drop = True)
# Splitting of the dataset



x=train.drop(['flag'],axis=1) # independent variables

y=train['flag'] # target variable



# splitting dataset into 80% training and 20% testing

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=1) 

# chceking the shape of the dataset after splitting

x_train.shape,x_test.shape,y_train.shape,y_test.shape
# building KNN model 



model = RandomForestClassifier(n_estimators=10,random_state=1)#using in-built RandomForestClassifier function from sklearn

model.fit(x_train,y_train) # fitting the training dataset in the model
# making predictions on test dataset with the fitted model



predictions = model.predict(x_test)
# using sklearn library to build a classification report with precision, recall and f1-score 



print(classification_report(y_test,predictions))
# adding predicted values in the test dataset



test['Anomaly_Flag'] = model.predict(test) # saving the predicted values in Anomaly_Flag feature
# submitting the final predicted values in a form of csv



Sample_Submission = pd.read_csv('../input/bda-2019-ml-test/Sample Submission.csv')

Sample_Submission['flag'] = test['Anomaly_Flag']

Sample_Submission.to_csv('final.csv',index=False)