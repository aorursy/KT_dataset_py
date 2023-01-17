#import pandas for data manipulation

import pandas as pd



#import data

data = pd.read_csv('../input/diabetes.csv')



#change float64 values to integers for comparisons to work correctly

data.loc[:,['BMI','DiabetesPedigreeFunction']]=data.loc[:,['BMI','DiabetesPedigreeFunction']].astype(int)
#for columns glucose through age, remove rows with 3 or more 0's across those columns

data = data.loc[(data.loc[:,'Glucose':'Age']==0).sum(axis=1)<3,:]



#split independent and dependent

x = data.iloc[:,0:8].values

y = data.iloc[:,8].values
#change all x values to float for imputing and scaling

x = x.astype(float)
#replace remaining glucose and skin thickness 0's with the respective mean's

from sklearn.preprocessing import Imputer

xputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)

xputer = xputer.fit(x[:,[1,3]])

x[:,[1,3]] = xputer.transform(x[:,[1,3]])
#train test split

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=1)





#feature scaling x

from sklearn.preprocessing import StandardScaler

xscaler = StandardScaler()

xtrain = xscaler.fit_transform(xtrain)

xtest = xscaler.transform(xtest)



#fit model

from sklearn.svm import SVC

classifier = SVC(kernel='linear')

classifier.fit(xtrain,ytrain)
#predict

ypred = classifier.predict(xtest)
#confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest,ypred)

print(cm)
#single score

score = classifier.score(xtest,ytest)

print('single score: {0:.2f}%'.format(score*100))
#cross-validation score

from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, x, y, cv=5)

print('cv score: {0:.2f}% +/- {1:.2f}%'.format(scores.mean()*100,scores.std()*200))