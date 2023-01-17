# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/artists.csv')
df.loc[df['years'].str.contains('- 13', na=False), 'years'] = "14th century"



df.loc[df['years'].str.contains('- 14', na=False), 'years'] = "15th century"



df.loc[df['years'].str.contains('- 15', na=False), 'years'] = "16th century"



df.loc[df['years'].str.contains('– 15', na=False), 'years'] = "16th century"



df.loc[df['years'].str.contains('- 16', na=False), 'years'] = "17th century"



df.loc[df['years'].str.contains('- 17', na=False), 'years'] = "18th century"



df.loc[df['years'].str.contains('- 18', na=False), 'years'] = "19th century"



df.loc[df['years'].str.contains('– 18', na=False), 'years'] = "19th century"



df.loc[df['years'].str.contains('- 19', na=False), 'years'] = "20th century"



df.loc[df['years'].str.contains('– 19', na=False), 'years'] = "20th century"



df.loc[df['years'].str.contains('- 20', na=False), 'years'] = "21st century"
# separate multiple strings, or remove values after first comma in nationality and genre



#strings are immutable so assign the reference to the result of string.replace

df['nationality'] = df['nationality'].astype(str).str.replace(r"[,\/]((\w*)|(.)).*","")



df['genre'] = df['genre'].astype(str).str.replace(r"[,\/]((\w*)|(.)).*","")



df
from sklearn.preprocessing import LabelEncoder 

var_mod = ['years','genre','nationality'] 

le = LabelEncoder() 

for i in var_mod:

    df[i] = le.fit_transform(df[i]) 

df.dtypes



df
#Import models from scikit learn module: 

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier 

from sklearn.tree import DecisionTreeClassifier, export_graphviz 

from sklearn import metrics

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.linear_model  import LogisticRegression 

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
#Generic function for making a classification model and accessing performance: 

def classification_model(model, data, predictors, outcome):   

  #Fit the model:   

  model.fit(data[predictors],data[outcome])

     

  #Make predictions on training set:  

  predictions = model.predict(data[predictors])

     

  #Print accuracy   

  accuracy = metrics.accuracy_score(predictions,data[outcome])   

  print ("Accuracy of predictions1: %s" % "{0:.3%}".format(accuracy))   



  #Perform k-fold cross-validation with 5 folds   

  kf = KFold(n_splits=5)   

  error = []   

  for train, test in kf.split(data[predictors]):

    # Filter training data     

    train_predictors = (data[predictors].iloc[train,:])          



    # The target we're using to train the algorithm.     

    train_target = data[outcome].iloc[train]

       

    # Training the algorithm using the predictors and target.

    model.fit(train_predictors, train_target)



    #Record error from each cross-validation run     

    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))



  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))



  #Fit the model again so that it can be refered outside the  function:

  model.fit(data[predictors],data[outcome])



  return predictions
outcome_var = 'genre'

model = LogisticRegression()

predictor_var = ['years','nationality']

classification_model(model, df, predictor_var, outcome_var)
model = DecisionTreeClassifier() 

predictor_var = ['years','nationality'] 

classification_model(model, df,predictor_var,outcome_var)
model = RandomForestClassifier(n_estimators=100) 

predictor_var = ['years','nationality'] 

classification_model(model, df,predictor_var,outcome_var)
#another method, testing less data so not as good



#Load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.linear_model  import LogisticRegression 

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



# only going to use all rows from the following columns

array = df.values

columnsTitles=["id","name","years","nationality","genre"]

df=df.reindex(columns=columnsTitles)

X = array[:,2:3] # all rows of columns index 2 and 3

Y = array[:,4] # all rows of column index 4

Y=Y.astype('int')



# set training set as 80% of x, test set 20% of x, train set 80% y, set test 20% y

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
model = LogisticRegression()

model.fit(x_train,y_train) # train the model with 80% of x and 80% of y

predictions = model.predict(x_test) # use the trained model to make predictions with the x test data

print(accuracy_score(y_test, predictions)) # find accuracy of trained model using the y test data and the predictions

print(confusion_matrix(y_test,predictions)) # Scores and reports print(classification_report(y_test,predictions)) print(accuracy_score(y_test,predictions)*100)



predictions # show predictions for Loan_Status
model = DecisionTreeClassifier()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))



predictions
model = RandomForestClassifier(n_estimators=100)

model.fit(x_train,y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))



predictions