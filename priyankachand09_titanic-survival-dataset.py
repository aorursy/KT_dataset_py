import pandas as pd
data=pd.read_csv('../input/train.csv')
data.head()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
font={

    'size':18

}
plt.figure(figsize=(12,8))

sns.heatmap(data.corr(),annot=True,linewidths=1,linecolor='w')

plt.xlabel('columns', fontdict=font)

plt.ylabel('columns', fontdict=font)

plt.xticks(rotation=0)

plt.title('Correlations', fontdict=font)

plt.savefig('1.png')
data.info()
data.Sex[data.Sex == 'male'] = 1

data.Sex[data.Sex == 'female'] =0
data.head()
test=data['Sex'].value_counts()
test=pd.Series(test)
sns.set(rc={'figure.figsize':(15, 7)})

test.plot(kind='bar')

plt.savefig('male.png')
data1=data.groupby('Sex')['Survived'].value_counts()
data1
survived_data = pd.DataFrame()
female_survival=data1[0][1]

male_survival=data1[1][1]

female_unsurvival=data1[0][0]

male_unsurvival=data1[1][0]
survived_data['Sex'] = ['male', 'female']

survived_data['Survived'] = [male_survival, female_survival]

survived_data['unSurvived'] = [male_unsurvival, female_unsurvival]

survived_data
survived_data=survived_data.set_index('Sex')
sns.set(rc={'figure.figsize':(15, 7)})

survived_data.plot(kind='bar',stacked=True)

plt.savefig('comparision.png')

plt.xticks(rotation=0)
survived_data
test
survived_data['total']=[test[1],test[0]]
survived_data
new_data = survived_data.apply(lambda x: round(100 * x/survived_data['total']))
new_data
new_data.drop('total',axis=1,inplace=True)
new_data
sns.set(rc={'figure.figsize':(12,6)})

new_data.plot(kind='bar',stacked=True)

ag=pd.crosstab(data['Sex'], data['Survived'])
sns.set(rc={'figure.figsize':(15, 7)})

ag.plot(kind='bar',stacked=True)

plt.savefig('comparision1.png')

plt.xticks(rotation=0)
numerical_data.isnull().sum()
numerical_data=data[['PassengerId','Pclass','Fare','Age','Sex','Parch','SibSp']].copy()
numerical_data['Age']=numerical_data['Age'].fillna(22)
survival_data=data['Survived'].copy()
from sklearn.model_selection import train_test_split 
X,x_test,Y,y_test=train_test_split(numerical_data,survival_data,test_size=0.3,random_state=5)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X,Y)
coefficients=model.coef_
coeff_dict={

    'PassengerId':coefficients[0][0],

    'Pclass':coefficients[0][1],

    'Fare' :coefficients[0][2],

    'Age' :coefficients[0][3],

    'Sex' :coefficients[0][4],

    'Parch' :coefficients[0][5],

    'SibSp' :coefficients[0][6]

}
coeff_dict=pd.Series(coeff_dict)
plt.figure(figsize=(10,5))

coeff_dict.plot(kind='bar')

plt.xticks(rotation=0)
predictions=model.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion= confusion_matrix(predictions,y_test)
confusion
score= accuracy_score(predictions,y_test)
score
test_data=pd.read_csv('../input/test.csv')
test_data.head()
test_data=test_data[['PassengerId','Pclass','Fare','Age','Sex','SibSp','Parch']].copy()
test_data.Sex[test_data.Sex == 'male'] = 1

test_data.Sex[test_data.Sex == 'female'] =0
test_data.head()
test_data=test_data[['PassengerId','Pclass','Fare','Age','Sex','SibSp','Parch']].copy()
test_data['Fare']=test_data['Fare'].fillna(10)
test_data['Age']=test_data['Age'].fillna(22)
test_data.isnull().sum()
test_predictions=model.predict(test_data)
test_predictions[:3]
test_predictions_df={

    'PassengerId':test_data['PassengerId'],

    'Survived': test_predictions

}
test_predictions_df=pd.DataFrame(test_predictions_df)
test_predictions_df.head()
test_predictions_df=test_predictions_df.set_index('PassengerId')
test_predictions_df.to_csv('Submission.csv')