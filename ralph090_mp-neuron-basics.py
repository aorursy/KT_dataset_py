import sklearn.datasets

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
cancer=sklearn.datasets.load_breast_cancer()
X=cancer.data

Y=cancer.target

print(X.shape)

#30 Columns (Features) and 569 Samples
dataframe=pd.DataFrame(cancer.data,columns=cancer.feature_names)
dataframe['target(Y)']=cancer.target
print(cancer.target_names)

dataframe['target(Y)'].value_counts()
#O is for benign and 1 is for malignant

dataframe.groupby('target(Y)').mean()

#12.146524 is the mean radius for a malignant tumor
#Train on one part i.e on the training data and predict on test data

X=dataframe.drop('target(Y)',axis=1)

Y=dataframe['target(Y)']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

#from this 80% of the X data is X_train and rest is X_test

#In order to change the test size we can use test_size for both X & Y as an argument as

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
#As for the Y which is our true output we can check how many are malignant & benign after the split using Y_test_mean()

#By getting the mean we can observe that as per our split which is 90% train and 10% test the mean of the test data have

#changed significantly so to solve this we use an another argument named as stratify as

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y)
#One more thing at the input data split if we check the mean similarly to the Y train & test data we will get values

# for every features, for a good model deployment practice we use an another argument named as random_state which

#will ensure and apply a constant state to all features that are available to us

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
plt.plot(X_train.T,'_')

plt.xticks(rotation='vertical')

plt.ylabel('Range of Numerical Values')

plt.title('All features value range')

plt.show()
#Data Normalization or convertig into a binary form {0,1}

X_normal=X_train['mean area'].map(lambda x:0 if x<1000 else 1)

#In order to normalize all the data into binary format use pd.cut 

X_all_train=X_train.apply(pd.cut,bins=2,labels=[0,1]).values #used to convert the data into np arrays

X_all_test=X_test.apply(pd.cut,bins=2,labels=[0,1]).values
#b=3 Inference of the model

threshold=3

print('Threshold value: {}'.format(threshold))

i=100

print('Taking sample row number: {}'.format(i))

#100th row and all the features

if np.sum(X_all_train[i,:])>=threshold:

    print('Result: Malignant')

else:

    print('Result: Benign')



if Y_train[i]==1:

    print('Y_train output is Malignant')

else:

    print('Y_train output is Benign')

     
Y_train[100]
threshold=0

predictions=[]

correct_predictions=0

for x,y in zip(X_all_train,Y_train):

    predictions.append(np.sum(x)>=threshold)

    correct_predictions+=(y==(np.sum(x)>=threshold))

    



total_samples=X_all_train.shape[0]





print('Accuracy is {}%'.format(round((correct_predictions/total_samples)*100)))


for threshold in range(X_all_train.shape[1]+1):

    predictions=[]

    correct_predictions=0

    for x,y in zip(X_all_train,Y_train):

        predictions.append(np.sum(x)>=threshold)

        correct_predictions+=(y==(np.sum(x)>=threshold))

    

    total_samples=X_all_train.shape[0]

    print('At threshold value: {}, Accuracy is {}%'.format(threshold,round((correct_predictions/total_samples)*100)))
dataframe.groupby('target(Y)').mean()
#At threshold=0 which is of course true, when it is benign i.e. 0 the feature{mean radius} is greater than the malignant one

# and same for other features so the data is more shifted towards benign 

# When normalizing the data for values x<1000 change to 0 otherwise 1 but in actual manignant cases are less so the normalization

# or can say the binarization is not up to the mark ##

#In order to normalize all the data again as per our observation into binary format

X_all_trainew=X_train.apply(pd.cut,bins=2,labels=[1,0]).values#used to convert the data into np arrays

X_all_testnew=X_test.apply(pd.cut,bins=2,labels=[1,0]).values

for threshold in range(X_all_trainew.shape[1]+1):

    predictions=[]

    correct_predictions=0

    for x,y in zip(X_all_trainew,Y_train):

        predictions.append(np.sum(x)>=threshold)

        correct_predictions+=(y==(np.sum(x)>=threshold))

    

    total_samples=X_all_trainew.shape[0]

    print('At threshold value: {}, Accuracy is {}%'.format(threshold,round((correct_predictions/total_samples)*100)))
#On test data

threshold=28

predictions_test=[]

for x in (X_all_testnew):

    predictions_test.append((np.sum(x)>=threshold))

    

test_accuracy=accuracy_score(predictions_test,Y_test)

print(threshold,round(test_accuracy*100))
#Templating a Model/function

class MPNeuron:

    def __init__(self):

        self.threshold=None

    def model(self,x):

        return (sum(x)>=self.threshold)

    def predict(self,X):

        Y=[]

        for x in X:

            Y.append(self.model(x))

        return np.array(Y)

    

    def fit(self,X,Y):

        accuracy={}

        for threshold in range(X.shape[1]+1):

            self.threshold=threshold

            pred=self.predict(X)

            accuracy[threshold]=accuracy_score(pred,Y)

            

        maximum_accuracy=max(accuracy,key=accuracy.get)

        self.threshold=maximum_accuracy

        

        print(maximum_accuracy)

        print(accuracy[maximum_accuracy])
x=MPNeuron()

x.fit(X_all_trainew,Y_train)