import numpy as np # linear algebra

import pandas as pd 

import statsmodels.formula.api as smf

import statsmodels.api as sm

from sklearn.metrics import roc_curve, auc, confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train
train.Age.fillna(train.Age.mean(), inplace=True)

train.dropna(subset=["Embarked"], inplace = True)

train.drop(train[train.Fare == 0].index, inplace=True)



train.describe()



sns.regplot(x="Age", y="Survived", data=train, fit_reg=True).set_title("Age")
sns.barplot(x="Sex", y="Survived", data=train).set_title("Sex")
sns.regplot(x="PassengerId", y="Survived", data=train, fit_reg=True).set_title("PassengerId")
sns.barplot(x="Pclass", y="Survived", data=train).set_title("Pclass")
sns.regplot(x="Pclass", y="Survived", data=train, fit_reg=True).set_title("Pclass")
sns.regplot(x="SibSp", y="Survived", data=train, fit_reg=True).set_title("SibSp")
sns.regplot(x="Parch", y="Survived", data=train, fit_reg=True).set_title("Parch")
sns.barplot(x="Embarked", y="Survived", data=train).set_title("Embarked")
sns.regplot(x="Fare", y="Survived", data=train, fit_reg=True).set_title("Fare")
logit_reg = smf.glm("Survived~Age+Sex+Pclass+Embarked+Fare", data=train,family=sm.families.Binomial()).fit()

print(logit_reg.summary())

predictions = logit_reg.predict(test)

print(predictions)
y = train["Survived"]

pred_probs =logit_reg.fittedvalues



fpr,sens,th = roc_curve(y,pred_probs)



plt.plot(fpr,sens) #plots the ROC curve

plt.plot([0,1],[0,1],'k') #adds a 1-1 line for reference

plt.xlabel('FPR') #adds an x label.

plt.ylabel('Sensitivity') #adds an y label.



print("Area Under Curve:",auc(fpr,sens))
thresh = np.linspace(0,1,num=100)



misclass = np.repeat(np.NaN,repeats=thresh.size)



for i in range(thresh.size): 

    my_classification = pred_probs>thresh[i] 

    misclass[i] = np.mean(my_classification!=y)

    

thresh = thresh[np.argmin(misclass)]

print("The threshold is:", thresh)
log_predictions = (predictions > thresh) + 0



#For classification matrix

predictions_train = logit_reg.predict(train)

log_predictions_train = (predictions_train > thresh) + 0
confusion_matrix(y,log_predictions_train)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': log_predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")