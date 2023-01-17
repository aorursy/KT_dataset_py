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
numerical_data=data[['PassengerId','Pclass','Fare','Age','Sex','Parch','SibSp']].copy()
numerical_data.isnull().sum()
numerical_data['Age']=numerical_data['Age'].fillna(22)
survival_data=data['Survived'].copy()
from sklearn.model_selection import train_test_split 
X,x_test,Y,y_test=train_test_split(numerical_data,survival_data,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X,Y)
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