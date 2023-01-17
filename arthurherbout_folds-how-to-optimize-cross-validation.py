# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Filling the blanks, standardizing the dataset

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



dataframe = pd.read_csv('../input/train.csv')



# I need to convert all strings to integers

# I need to fill in the missing values of the data and make it complete

# female = 0, Male = 1

dataframe['Gender'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Here we have added a colomn "Gender" at the end of our dataframe

#print(X)



# Embarked from 'C', 'Q', 'S'

# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.



# All missing Embarked -> just make them embark from most common place

if len(dataframe.Embarked[ dataframe.Embarked.isnull() ]) > 0:

    dataframe.Embarked[ dataframe.Embarked.isnull() ] = dataframe.Embarked.dropna().mode().values



Ports = list(enumerate(np.unique(dataframe['Embarked'])))    # determine all values of Embarked,

print(Ports)

Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

print(Ports_dict)

dataframe.Embarked = dataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# Here we have converted strings values in Embarked to integers 1, 2 or 3

#print(X)



#All the ages with no data : I will complete the empty entries with the median Age

median_age = dataframe['Age'].dropna().median()

if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:

    dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = median_age

#print(X)





# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

#print(y)



y=dataframe['Survived']

#print(y)



# I get rid of the label column

dataframe = dataframe.drop(['Survived'], axis=1)



train_data = dataframe.values

labels = y.values



#Standardization

from sklearn.preprocessing import StandardScaler

train_data_std = StandardScaler().fit_transform(train_data)
def StratifiedKFold_generator(X, n_folds = 10,features=None,label=False,labels=None):

    

    """

    Generate StratifiedKFold for multiclass features or label.

    This method does not require a (multi)classification problem.

    This method returns a tuple(features_chosen,folds)

    

    features_chosen is the list of features chosen to perform Stratified-K-Folds.

    folds is the dictionary of folds created by Stratified-K-Folds. 

    Its keys are the index of the features, its values the fold itself.

    

    Parameters

    ----------

    X : array-like, shape (n_samples, n_features)

    Training vector, where n_samples is the number of samples and

    n_features is the number of features.

        

    n_folds : int >=2

    Number of folds to create for each available feature

    Default value is set to 10  

    

    features : list, default 'None'

    List of the features to consider when performing StratifiedKFold.

    When 'None', all features are considered. 

              

    label : Boolean, default value is 'False'

    Indicates if the labels are to be tested, i.e. if the problem is a (multi)classification

    If set to "True", a label array is to be given to the algorithm.

            

    labels : array-like, shape(n_samples,1), default value is 'None'

    Labels of the dataset. For now this method only works for simple array(1D)          

    

    

    

    """

    

    from sklearn.cross_validation import StratifiedKFold

    import numpy as np

    # Getting the number of features and the number of samples

    (n_samples,n_features)=X.shape

    

    # Initializing our dictionary : keys are the indices choosen, values are the StratifiedKFold generator induced 

    folds= {}

    

    # We explore all the features to see which one can be choosen to create Stratified-K-Folds

    if features ==None:

        iterator = range(0,n_features-1)

    else :

        iterator = features

        

    for i in iterator:

        (unique,times)= np.unique(X[:,i],return_counts = True) 

        # unique is a list of the different values in X[;,i]

        # times is a list of the number of times each of the unique values 

        # comes up in X[:,i]

        if len(unique)<= n_folds : # our feature seems multi-class 

            minimun = times[0]

            for j in range(1,len(unique)-1):

                if times[j]<minimun :

                    minimum = times[j]

            if minimum >= n_folds : 

                # our feature will do just fine with StratifiedKFolds

                fold_i = StratifiedKFold(X[:,i],n_folds = n_folds,shuffle=True)    

                folds[i] = fold_i

    

    # We finish by using the labels if told to do so :

    if label==True:

        fold_labels = StratifiedKFold(labels,n_folds = n_folds,shuffle=True)    

        folds["labels"] = fold_labels

    

    # We want to know which features and/or label have been chosen to perform StratifiedKFolds.

    features_chosen=[]

    

    for key in folds.keys():

        features_chosen.append(key)          

    

    return (features_chosen,folds)

                

            

        

        

        

        
(features_index,folds) = StratifiedKFold_generator(train_data, n_folds = 10,label=True, labels=labels)
folds

l=['ene',5,6]

l[0] 
def Stratified_generator(X, n_folds = 10,n_iter = 10, features=None,label=False,labels=None,method=["Fold","ShuffleSplit"]):

    

    """

    Generate StratifiedKFold and/or StratifiedShuffleSplit for multiclass features or label.

    This method does not require a (multi)classification problem.

    This method returns a tuple(features_chosen,folds)

    

    features_chosen is the list of features chosen to perform the desired splitting.

    folds is the dictionary of folds created by Stratified-K-Folds. 

    Its keys are the index of the features, its values the fold itself.

    

    Parameters

    ----------

    X : array-like, shape (n_samples, n_features)

    Training vector, where n_samples is the number of samples and

    n_features is the number of features.

        

    n_folds : int >=2, necesary for StratifiedKFold

    Number of folds to create for each available feature

    Default value is set to 10  

    

    n_iter : int, necesary for StratifiedShuffleSplit

    Number of re-shuffling and splitting iterations.

    Default value is set to 10

    

    features : list, default 'None'

    List of the features to consider when performing StratifiedKFold.

    When 'None', all features are considered. 

              

    label : Boolean, default value is 'False'

    Indicates if the labels are to be tested, i.e. if the problem is a (multi)classification

    If set to "True", a label array is to be given to the algorithm.

            

    labels : array-like, shape(n_samples,1), default value is 'None'

    Labels of the dataset. For now this method only works for simple array(1D)          

    

    method: list of strings. 

    Indicates which methods to use to generate the folds.

    "Fold" and "Split" are the only possible values for now.

    

    """

    

    from sklearn.cross_validation import StratifiedKFold

    from sklearn.cross_validation import StratifiedShuffleSplit

    import numpy as np

    # Getting the number of features and the number of samples

    (n_samples,n_features)=X.shape

    

    # Initializing our dictionaries : keys are the indices choosen, values are the StratifiedKFold and/orStratifiedShuffleSplit generator induced 

    folds = {}

    splits = {}

    

    # We explore all the features to see which one can be choosen to create Stratified-K-Folds

    if features ==None:

        iterator = range(0,n_features-1)

    else :

        iterator = features

        

    for i in iterator:

        (unique,times)= np.unique(X[:,i],return_counts = True) 

        # unique is a list of the different values in X[;,i]

        # times is a list of the number of times each of the unique values 

        # comes up in X[:,i]

        if len(unique)<= n_folds : # our feature seems multi-class 

            minimun = times[0]

            for j in range(1,len(unique)-1):

                if times[j]<minimun :

                    minimum = times[j]

            if minimum >= n_folds : 

                if "Fold" in method :

                    fold_i = StratifiedKFold(X[:,i],n_folds = n_folds,shuffle=True)    

                    folds[i] = fold_i

                if "ShuffleSplit" in method:

                    split_i = StratifiedShuffleSplit(X[:,i],n_iter = n_iter)

                    splits[i]=split_i

    

    # We finish by using the labels if told to do so :

    if label==True:

        if "Fold" in method:

            fold_labels = StratifiedKFold(labels,n_folds = n_folds,shuffle=True)    

            folds["labels"] = fold_labels

        if "ShuffleSplit" in method:

            split_label = StratifiedShuffleSplit(labels,n_iter = n_iter)

            splits["labels"]=split_label

    # We want to know which features and/or label have been chosen to perform StratifiedKFolds.

    features_chosen=[]

        

    for key in folds.keys():

        features_chosen.append(key) 

   

    return (features_chosen,folds,splits)

                

            
(features_index,folds,splits)= Stratified_generator(train_data, n_folds = 10,n_iter = 10, features=None,label=True,labels=labels,method=["Fold","ShuffleSplit"])    
splits