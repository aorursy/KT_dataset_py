import pandas as pd

import matplotlib.pyplot as plt



from sklearn import linear_model # We will use logistic model to predict whether there will be a negative consequence of discussing mental 

#health disorders with your coworkers.

from sklearn.ensemble import GradientBoostingClassifier # We will compare it with Gradient Boosting Classifier
survey=pd.read_csv("../input/survey.csv") #reading the dataset

survey.head() 

#survey2=survey.copy()

#survey2['self_employed']=['No','Yes','Cant say' ]

survey['work_interfere'].value_counts()

#survey.median()

#pd.isnull(survey)

#null_data = survey[survey.isnull().any(axis=1)]

#null_data
survey['state']=survey['state'].fillna('Other')

survey['self_employed']=survey['self_employed'].fillna('No')

survey['work_interfere']=survey['work_interfere'].fillna('cant say')

survey['comments']=survey['comments'].fillna('None')

survey_treatment=survey.treatment.fillna('No')
survey.work_interfere=survey.work_interfere.map({'Sometimes':0,'Never':1,'Rarely':2,'Often':3,'cant say':4 })

survey.treatment=survey.treatment.map({'No':0,'Yes':1 })

#survey.treatment
%matplotlib inline

pd.crosstab(survey.treatment, survey.obs_consequence.astype(bool)).plot(kind='line')
import statsmodels.api as sm

import matplotlib.pyplot as plt

from patsy import dmatrices

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn.cross_validation import cross_val_score

survey['self_employed']=survey['self_employed'].map({'No':0,'Yes':1})

survey['family_history']=survey['family_history'].map({'No':0,'Yes':1})

survey['mental_vs_physical']=survey['mental_vs_physical'].map({"Yes":1,"Don't know":2,"No":1})

survey['benefits']=survey['benefits'].map({"Yes":0,"Don't know":2,"No":1})

survey['obs_consequence']=survey['obs_consequence'].map({'No':1,'Yes':0})



#survey['obs_consequence'].value_counts()

#survey.head()

survey.dtypes

#survey.mental_vs_physical
y, X = dmatrices('obs_consequence~self_employed+treatment+work_interfere+mental_vs_physical+benefits+family_history',

                  survey, return_type="dataframe")

import numpy as np

y=np.ravel(y)





model=LogisticRegression()

model=model.fit(X,y)

survey.corr(method='pearson')
model.score(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

model2=LogisticRegression()

model2.fit(X_train,y_train)
predicted=model2.predict(X_test)

print (predicted)
probs=model2.predict_proba(X_test)

print (probs)
print (metrics.accuracy_score(y_test,predicted))

print (metrics.roc_auc_score(y_test,probs[:,1]))
print (metrics.confusion_matrix(y_test,predicted))
survey.describe()
# evaluate the model using 10-fold cross-validation

scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

print (scores)

print (scores.mean())



# The final accuracy of the model is 85.4 % . Thus you can correctly observe a negative consequence by discussing 

# mental health issues with your co-workers  85.4% of the time