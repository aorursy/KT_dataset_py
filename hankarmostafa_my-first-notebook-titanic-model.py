

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sb 

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
#for men 

men=train['Survived'][train.Sex=='male']

percentage =sum(men)/len(men)*100

print('the rate of survived men is {}'.format(percentage))

#for women 

women=train['Survived'][train.Sex=='female']

percentage = sum(women)/len(women)*100

print('the rate of survived women is {}'.format(percentage))
# let's visualize percenteges by sex

sb.set_context('poster')

sb.set_style('ticks')

sb.barplot(x='Sex',y='Survived',data=train)     

sb.despine()     

# let's count by sex 



sb.set_context('poster')

sb.set_style('ticks')

sb.countplot(x='Survived',hue='Sex',data=train)  

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# let's visulise how many survived by Class (High, Middle ,Low )

sb.barplot(x='Pclass',y='Survived',hue='Sex',data=train)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#Let's visulize by age 

sb.barplot(x='Survived',y='Age',hue='Sex',data=train)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
from sklearn.model_selection import train_test_split

#target

y = train["Survived"]

#get the features .. only important features 

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



#X_test = pd.get_dummies(test_data[features])

# train our model 

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=600, max_depth=5, random_state=1)

rfc.fit(X_train,y_train)

# predicttions on the model

predictions=rfc.predict(X_test)
#Let's run some metrics 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))

print('\n')

print('Accuracu score is :{}'.format(accuracy_score(y_test,predictions)))
#grab the test set

test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
features = ["Pclass", "Sex", "SibSp", "Parch"]

X_test = pd.get_dummies(test[features])

test_predictions=rfc.predict(X_test)

#save my submission 



submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test_predictions})

submission.to_csv('submission.csv', index=False)

print(" Submission  successfully saved!")
submission.head()