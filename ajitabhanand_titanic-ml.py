import pandas as pd
data = pd.read_csv('../input/train.csv')
data.head(2)
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
data.info()
plt.figure(figsize=(12,8))

sns.heatmap(data.corr(), annot=True, linecolor='w', linewidths='1')

plt.savefig('ML.png')
training_data = data[['PassengerId', 'Pclass', 'Fare', 'Sex']]
def checker(x):

    if x.lower()=="male":

        return 1

    else:

        return 0
training_data['Sex']=training_data['Sex'].apply(checker)
output_data = data['Survived']
from sklearn.model_selection import train_test_split
X, x_test, Y, y_test = train_test_split(training_data, output_data, test_size=0.3, random_state=42)
X.head()
Y.head()
from sklearn.linear_model import LogisticRegression
model =  LogisticRegression()
model.fit(X,Y)
predictions = model.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion = confusion_matrix(predictions, y_test)
confusion
score = accuracy_score(predictions, y_test)
score
test_data = pd.read_csv('../input/test.csv')
test_data.head(2)
test_data = test_data[['PassengerId', 'Pclass', 'Fare', 'Sex']]
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_data['Sex'] = test_data['Sex'].apply(checker)
test_data.isnull().sum()
test_predictions = model.predict(test_data)
test_predictions_df = {

    'PassengerId': test_data['PassengerId'],

    'Survived': test_predictions

}
test_predictions_df = pd.DataFrame(test_predictions_df)
test_predictions_df = test_predictions_df.set_index('PassengerId')
test_predictions_df.head()
test_predictions_df.to_csv('submission1.csv')
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(n_splits = 3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):

    print(train_index,test_index)
def get_score(model,X,x_test,Y,y_test):

    model.fit(X,Y)

    return model.score(x_test,y_test)
get_score(LogisticRegression(),X,x_test,Y,y_test)