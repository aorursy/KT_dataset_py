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
import pandas as pd

P39_Financial_Data = pd.read_csv("../input/P39-Financial-Data.csv")
P39_Financial_Data 
# predict if esign or not



#pay schedule - how often would be paid

#home owner if 1

#curr adress year- how many years at curr adress

# personal account m y - months had personal account or years

# amount requested

# has debt -- 1 is true

# bunch of risk scores - non normalized --> will userpays loan

# then normalize risk scores
# EDA



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sn





dataset = P39_Financial_Data 
dataset.head
dataset.columns
dataset.describe()
# clean the data - any columns have na? no, so no need to deal

dataset.isna().any()
"""



# not workign great-can fix later



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sn







# histograms - make sure columns are plottable - no categoricals (still can plot), and no NAs

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

#dataset2.columns





#fig = plt.figure(figsize = )



plt.suptitle('histogram', fontsize=20)



#iterate all features

for i in range(1, dataset2.shape[1] + 1):

    

    

    # generate a subplot witha given index

    plt.subplot(2,2,i)

    

    # what is this?!

    f = plt.gca()

    

    #get name of column at a given value

    f.set_title(dataset2.columns.values[i-1])

    

    # Why getting the unique values? 

    # so if vals>100, then vals = 100. cap number of bins

    vals = np.size(dataset2.iloc[:, i-1].unique())

    

    # plot histogram where i subset data, all rows at index i-1 (why not name --> easier to use), and number of bins is equal to num of uniques

    # is this a standard method?!

    plt.hist(dataset2.iloc[:, i-1], bins=vals, color = '#3F5D7D')

    

    plt.show()

    

    """
# #plt.subtitle('histogram', fontsize=20)

# dataset2.shape[1] #get number of columns

# dataset2.columns.values[1]

# #np.size(dataset2.iloc[:, 1].unique())

# dataset2.iloc[:, 2].unique() #get the array pandas of the column

# np.size(dataset2.iloc[:, 3].unique()) #get number of unique values




# my version



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sn





# drop some of the variables

# histograms - make sure columns are plottable - no categoricals (still can plot), and no NAs

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

#dataset2.columns





# suptitle things

plt.suptitle('histogram', fontsize=20)









#iterate all features: from 1 to rest

for i in range(0, dataset2.shape[1]):

    

    

    # so can iterate all the cols. now per each col do a histogram witha title that is pulled from it

    #print column

    print(dataset2.columns.values[i])

    

    #do histogram of column we are iterating

    plt.hist(dataset2.iloc[:, i])

    

    plt.show()

    

    

    
# corr plot and amtrix - do after!
dataset.columns
# data reprocessing 



#remove months employed-issue - some users 0 mistakenly

#dataset = dataset.drop(columns = ['months_employed'])



# feature engineering

dataset['personal_account_months'] = dataset.personal_account_m + (dataset.personal_account_y * 12)

dataset
dataset.columns
# preprocess data - hot encode and scale data



# one hot encode

dataset2 = pd.get_dummies(dataset)

dataset2.columns

#dataset2

#dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])
dataset2
dataset3.dtypes
# put aside

response = dataset["e_signed"]

users = dataset['entry_id']



#drop reponse, pay_schedule, and the ids --> how to reconcile IDs?

dataset3 = dataset.drop(columns = ['e_signed', 'entry_id', 'pay_schedule'])

dataset3
# split train test

from sklearn.model_selection import train_test_split



# have train and test -- and response variable is separate

X_train, X_Test, y_train, y_test = train_test_split(dataset3, response, test_size = 0.2, random_state = 0)





# feature scale

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()



#fit scaler to X_Train - column names lost - until i conert to df

X_Train2 = pd.DataFrame(sc_X.fit_transform(X_train))

X_Test2 = pd.DataFrame(sc_X.fit_transform(X_Test))



#map it all back

X_Train2.columns = X_train.columns.values

X_Test2.columns = X_Test.columns.values



# reset index - why?! can I not get it other way

X_Train2.index = X_train.index.values

X_Test2.index = X_Test.index.values



X_Train2

#X_Train2.columns = X_train.columns.values


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score



# function to get accuracy scores



def errors(y_test, y_pred, model_type):

    

    # accuracy

    acc = accuracy_score(y_test, y_pred)



    #precision - 

    prec = precision_score(y_test, y_pred)



    #recall - biased model

    rec = recall_score(y_test, y_pred)



    #f1

    f1 = f1_score(y_test, y_pred)



    #put to pandas

    errors = pd.DataFrame([[model_type, acc, prec, rec, f1]], columns = ['Model', 'accuracy', 'precision', 'recall', 'f1 score'])



    return errors
# model building



# logistic regression

from sklearn.linear_model import LogisticRegression



#lasso - penalizes if too much of a coefficient ona variable

classifier = LogisticRegression(random_state = 0, penalty = 'l1')



# fit classifier to train data --> fits the classifier object

classifier.fit(X_Train2, y_train)



# predicting test set - want to see accuracy

y_pred = classifier.predict(X_Test2)





error_logistic = errors(y_test, y_pred, 'logistic')

error_logistic
"""



from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score





# accuracy

acc = accuracy_score(y_test, y_pred)



#precision - 

prec = precision_score(y_test, y_pred)



#recall - biased model

rec = recall_score(y_test, y_pred)



#f1

f1 = f1_score(y_test, y_pred)



#put to pandas

errors = pd.DataFrame([['Linear Regression (lasso)', acc, prec, rec, f1]], columns = ['Model', 'accuracy', 'precision', 'recall', 'f1 score'])



"""
# let's do more models and see which one works the best





# model building



# SVC

from sklearn.svm import SVC



#lasso - penalizes if too much of a coefficient ona variable

classifier = SVC(random_state = 0, kernel = 'linear')



# fit classifier to train data --> fits the classifier object

classifier.fit(X_Train2, y_train)



# predicting test set - want to see accuracy

y_pred = classifier.predict(X_Test2)



error_svc = errors(y_test, y_pred, 'SVC')

error_svc
# model building



# SVC

from sklearn.svm import SVC



#lasso - penalizes if too much of a coefficient ona variable

classifier = SVC(random_state = 0, kernel = 'rbf')



# fit classifier to train data --> fits the classifier object

classifier.fit(X_Train2, y_train)



# predicting test set - want to see accuracy

y_pred = classifier.predict(X_Test2)



error_svc_rbf = errors(y_test, y_pred, 'SVC_rbf')

error_svc_rbf
# model building



# random forest

from sklearn.ensemble import RandomForestClassifier



#lasso - penalizes if too much of a coefficient ona variable

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=0)



# fit classifier to train data --> fits the classifier object

classifier.fit(X_Train2, y_train)



# predicting test set - want to see accuracy

y_pred = classifier.predict(X_Test2)



error_rf = errors(y_test, y_pred, 'rf')

error_rf
# append for comparison, have to reset idnex too - can see svc bit better - rf is best one

appended_error = error_svc.append([error_logistic, error_svc_rbf, error_rf], ignore_index=True)

appended_error
# do cross validation - k fold --- can be used in grid search

from sklearn.model_selection import cross_val_score



# cross validate the classifier X_train

accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)



#average accuracy is 62% --> where we batch and try --> we guarantee that model is consistent

print(accuracies.mean())

print(accuracies.std() * 2)
# grid search



# aside - learning stuff
#matplotlib

import matplotlib.pyplot as plt

import numpy as np



x = np.arange(1,5)

y = x**3



# plotting 2 arrays

plt.plot([1,2,3,4], [1,4,9,16], "go", x, y, "r^")

plt.title("First plot")

plt.xlabel("x")

plt.ylabel("y")

plt.show()
x = np.arange(1,5)

y = x**3



# subplot 1 returned

plt.subplot(1,2,1) #figure 1 returned

plt.plot([1,2,3,4], [1,4,9,16], "go")  #plot attached to figure 1



plt.subplot(1,2,2)

plt.plot( x, y, "r^")
#bar chart

# things are really layered on in a way





divisions = ["a", "b", "c"]



# plotting stuff for a given coordinate - what will be what

girls = [1,2,3]

boys = [5,6,7]



index = np.arange(3)

width = 0.2



plt.bar(index, girls, width)

plt.bar(index + width, boys, width)



#I give coordinates to all and divisions array are the xticks that will be used

plt.xticks(index + width/2, divisions)





plt.show()
x = np.random.randn(1000)



plt.title("x")

plt.hist(x)
# figure object created

fig = plt.figure()



# call suptitle method

fig.suptitle('abc')



#fig.show()



# generate subplots

plt.subplots(nrows = 2, ncols = 2)
import numpy as np





t = np.arange(0, 5, 0.2)

plt.plot(t,t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.show()
x = np.arange(0, 5, 0.2)

y = np.arange(0, 5, 0.2)







#working off one instance of pyplot - call methods on it which return visual object which are overlaid on top each other

#plt.figure is implicitly created for line2D object

plt.figure()



# return line2D object - it is overlaid on result of previous method that was called

plt.plot(x,y,linewidth=2.0, animated = False, color = 'red')

#plt.show()