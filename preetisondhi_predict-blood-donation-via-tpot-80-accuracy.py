# Importing necessary libraries 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score



#Importing library for visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Filter the unwanted warning

import warnings

warnings.simplefilter("ignore")
#Lets get started exploring the data.



train = pd.read_csv("../input/predicting-blood-analysis/blood-train.csv")

test=pd.read_csv("../input/predicting-blood-analysis/blood-test.csv")

train.head()
#Printing the train and test size

print("Train Shape : ",train.shape)

print("Test Shape : ",test.shape)
# Print a concise summary of transfusion DataFrame

train.info()
# Rename target column as 'target' for brevity

train.rename(

    columns={'Made Donation in March 2007':'Target'},

    inplace=True

)   
#Counting the number of people who donated and not donated

train["Target"].value_counts()
test.head()
test.info()
# Statistics of the data

train.describe()
#Boxplot for Months since Last Donation

plt.figure(figsize=(20,10)) 

sns.boxplot(y="Months since Last Donation",data=train)
#Correlation between all variables [Checking how different variable are related]

corrmat=train.corr()

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1,fmt = ".2f",annot=True)
# Import train_test_split method

from sklearn.model_selection import train_test_split



# Split transfusion DataFrame into

# X_train, X_test, y_train and y_test datasets,

# stratifying on the `target` column

X_train, X_test, y_train, y_test = train_test_split(

    train.drop(columns=['Target','Unnamed: 0']),

    train.Target,

    test_size=0.2,

    random_state=0)
# Import TPOTClassifier and roc_auc_score

from tpot import TPOTClassifier

from sklearn.metrics import roc_auc_score



# Instantiate TPOTClassifier

tpot = TPOTClassifier(

    generations=5,

    population_size=20,

    verbosity=2,

    scoring='roc_auc',

    random_state=42,

    disable_update_check=True,

    config_dict='TPOT light'

)

tpot.fit(X_train, y_train)



# AUC score for tpot model

tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])

print(f'\nAUC score: {tpot_auc_score:.4f}')





# Print best pipeline steps

print('\nBest pipeline steps:', end='\n')

for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):

    # Print idx and transform

    print(f'{idx}. {transform}')

tpot.fitted_pipeline_
# Importing modules

from sklearn.linear_model import LogisticRegression

# Instantiate LogisticRegression

logreg = LogisticRegression(C=25.0, random_state=42)

#Fitting the model

logreg.fit(X_train,y_train)
#Predicting on the test data

pred=logreg.predict(X_test)
#printing the confusion matrix

confusion_matrix(pred,y_test)
# AUC score for tpot model

logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

print(f'\nAUC score: {logreg_auc_score:.4f}')