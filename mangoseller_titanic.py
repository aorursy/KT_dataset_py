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
titanic = pd.read_csv('../input/train.csv')

titanic.head(10)
def agegroup(x):

    if x < 1: return '1. less than 1'

    elif x <5: return '2. 1 to 5'

    elif x <10: return '3. 5 to 10'

    elif x <15: return '4. 10 to 15'

    elif x <20: return '5. 15 to 20'

    elif x <25: return '6. 20 to 25'

    elif x <30: return '7. 25 to 30'

    elif x <35: return '8. 30 to 35'

    elif x <40: return '9. 35 to 40'

    elif x <45: return '10. 40 to 45'

    elif x <50: return '11. 45 to 50'

    elif x <55: return '12. 50 to 55'

    elif x <60: return '13. 55 to 60'

    else: return '14. 60+'

    

def agegroup2(x):

    if x < 1: return 0

    elif x <5: return 1

    elif x <10: return 2

    elif x <15: return 3

    elif x <20: return 4

    elif x <25: return 5

    elif x <30: return 6

    elif x <35: return 7

    elif x <40: return 8

    elif x <45: return 9

    elif x <50: return 10

    elif x <55: return 11

    elif x <60: return 12

    else: return 13

    



titanic['Age2'] =  titanic['Age'].apply(agegroup2)
import seaborn as sns
for column in ['Pclass','SibSp','Parch','Age2']:

    

    sns.catplot(x="Sex", y="Survived", hue= column, kind="bar", data=titanic);
tit_heat = titanic.groupby(["SibSp","Parch",'Sex']).agg(['count', np.sum])['Survived']

tit_heat.reset_index(inplace=True)

tit_heat.head()
tit_heat['Survive'] = tit_heat['sum']/tit_heat['count']

tit_heat.head()
female =tit_heat[tit_heat['Sex'] == 'female']

male =tit_heat[tit_heat['Sex'] == 'male']

tit_piv = female.pivot('SibSp','Parch','Survive')

tit_piv2 = male.pivot('SibSp','Parch','Survive')

ax = sns.heatmap(tit_piv,cmap ='Greens')



ax2 = sns.heatmap(tit_piv2,cmap ='Greens')
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
model = SVC()
feat = titanic[['Age2','Pclass','SibSp','Parch','Sex','Embarked']]

feats = pd.get_dummies(feat)

feats.head()



survival = titanic['Survived']



X_train, X_test, y_train, y_test = train_test_split(feats, survival, test_size=0.30,random_state = 23)
feats.head()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
final=pd.read_csv('../input/test.csv')

final['Age2'] = final['Age'].apply(agegroup2)

final.head()

final2 = pd.get_dummies(final[['Age2','Pclass','SibSp','Parch','Sex','Embarked']])

pred = model.predict(final2)

pred2 = pd.DataFrame(pred)

pred2.columns = ['Survived']

pred2.head()

pred3 = pd.concat([final['PassengerId'],pred2], axis =1)
pred3.to_csv('results.csv',index=False)