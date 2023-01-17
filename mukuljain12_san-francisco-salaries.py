%matplotlib inline
from pandas import Series,DataFrame
import pandas as pd
%matplotlib inline
import nltk as nl
salaries=pd.read_csv("../input/Salaries.csv")
salaries.head(5)
# There are many features that won't be very useful in our prediction. We can remove this.
salaries=salaries.drop(['Status','Benefits','Notes','Agency','EmployeeName'],axis=1)


#if we observe the dataset, there are rows where no job title is provided. We need to remove this.
#We also need to remove rows where TotalPayment values do not exist -( They contain 0 values )
salaries=salaries[salaries['JobTitle']!='Not provided']
salaries=salaries[salaries['TotalPayBenefits']!=0.00]
salaries['TotalPayBenefits']=salaries['TotalPayBenefits']-salaries['TotalPay']
salaries=salaries[salaries['TotalPay']>=0]       
#removing negative salaries

# How many job titles are there? Can we use these key words in job titles to predict their salaries? 
salaries['JobTitle'].value_counts()[:20].plot(kind='bar')
#From the above it looks like each job has multiple rows. 
salaries['JobTitle'].value_counts().count()
#In total there are 1109 unique jobs

jobnames=salaries['JobTitle']
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
wordsinjobs=""
for word in jobnames:
    wordsinjobs=wordsinjobs+word
tokens=tokenizer.tokenize(wordsinjobs)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(tokens)
dtm=vectorizer.fit_transform(salaries['JobTitle'])
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(dtm,salaries['TotalPay'],test_size=0.33,random_state=42)
# Let us try a linear model
from sklearn.linear_model import LinearRegression
regr=LinearRegression()
lm=regr.fit(X_train,Y_train)
predy=lm.predict(X_test)
import math
MSE=(predy-Y_test)*(predy-Y_test)
MSE=MSE.mean()
MSE=math.sqrt(MSE)
MSE
from sklearn.metrics import r2_score
r2_score(predy,Y_test)