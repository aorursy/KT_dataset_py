import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import scipy.stats as stats
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression as LR
import sklearn.metrics as m

input_folder='/kaggle/input/titanic/'
train_data = pd.read_csv(input_folder+'train.csv')
test_data = pd.read_csv(input_folder+'test.csv')
print(train_data.isnull().sum()*100/train_data.isnull().count())
# About 20% of age is missing, about 77% of cabin is missing, 0.22% of embarked is missing

sns.heatmap(train_data.isnull(),
            annot=True,
            yticklabels=False,
            cbar=False,
            cmap='plasma')
# We need to change sex to a numerical value, otherwise the correlation plot will not include sex
train_data['Sex'] = train_data['Sex'].astype('category')
train_data['Sex'] = train_data['Sex'].cat.codes


sns.heatmap(train_data.corr(),
           annot=True,
           cbar=False)
train_filtered = train_data.filter(['Survived','Pclass','Sex','Age','Fare'])
train_filtered.dropna(how='any',axis='rows', inplace=True)

test_filtered = test_data.filter(['Survived','Pclass','Sex','Age','Fare','PassengerId'])
test_filtered.dropna(how='any',axis='rows', inplace=True)
train_filtered['Sex'] = train_filtered['Sex'].astype('category')
train_filtered['Sex'] = train_filtered['Sex'].cat.codes

test_filtered['Sex'] = test_filtered['Sex'].astype('category')
test_filtered['Sex'] = test_filtered['Sex'].cat.codes
# Survival by pclass
g = sns.jointplot(train_filtered.groupby(['Pclass']).sum().index,
                  train_filtered.groupby(['Pclass']).sum()['Survived']
    )

# Survival by sex
g = sns.jointplot(train_filtered.groupby(['Sex']).sum().index,
                  train_filtered.groupby(['Sex']).sum()['Survived']
    )

# Survival by age
g = sns.jointplot(train_filtered.groupby(['Age']).sum().index,
                  train_filtered.groupby(['Age']).sum()['Survived']
    )

# Survival by fare
g = sns.jointplot(train_filtered.groupby(['Fare']).sum().index,
                  train_filtered.groupby(['Fare']).sum()['Survived']
    )
# Split data into test and train sets based on the train data set
X_train, X_test, y_train, y_test = train_test_split(
    train_filtered.drop('Survived', axis=1), train_filtered['Survived'], test_size=0.33)

# Train model
model = LR().fit(y=y_train,X=X_train)

# Predict results
results = model.predict(X_test)

# Add results to a data frame
res = pd.DataFrame(data=y_test.tolist(),columns=['Survived actual'])
res['Survived predicted'] = results

# Confusion matrix
confmatrix=m.confusion_matrix(res['Survived actual'],res['Survived predicted'])
sns.heatmap(confmatrix,annot=True,fmt='d',cbar=0).set_title('Confusion Matrix')
#   True negatives (tn)     True positives (tp)
#   False negatives (fn)    False positives (fp)

print('Accuracy: '+str(m.accuracy_score(res['Survived actual'],res['Survived predicted']))) # percent of accurate classification
print('Precision: '+str(m.precision_score(res['Survived actual'],res['Survived predicted']))) # tp / (tp + fp), 0 is worst, 1 is best
print('Recall: '+str(m.recall_score(res['Survived actual'],res['Survived predicted']))) # tp / (tp + fn), 0 is worst, 1 is best
# Train model
model = LR().fit(y=train_filtered['Survived'],X=train_filtered.drop('Survived', axis=1))

# Predict survival for the test set
results = model.predict(test_filtered.drop('PassengerId', axis=1))

res = pd.DataFrame(data=results.tolist(), columns = ['Survived'])
res['PassengerId'] = test_filtered['PassengerId'].astype(int)

print(res)
res.to_csv('results.csv', index=False)
