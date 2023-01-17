# The fast guys

from fastai.imports import *

import pandas_profiling

import h2o

from h2o.automl import H2OAutoML

import warnings

warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'retina'

# List of files in the directory

PATH = '../input/titanic/'

list(os.listdir(PATH))
train = pd.read_csv(PATH+'train.csv')

test = pd.read_csv(PATH+'test.csv')

submission = pd.read_csv(PATH+'gender_submission.csv')
%%time

pandas_profiling.ProfileReport(train)
train = train.drop(columns=['PassengerId', 'Ticket'])

test = test.drop(columns=['PassengerId', 'Ticket']);
train['Deck'] = train['Cabin'].astype(str).str[0]

train['Deck'] = train['Deck'].map({'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E':1, 

                                             'F':1, 'G': 1, 'T': 0, 'n' : 0}).astype(int)

test['Deck'] = test['Cabin'].astype(str).str[0]

test['Deck'] = test['Deck'].map({'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E':1, 

                                           'F':1, 'G': 1, 'T': 0, 'n' : 0}).astype(int)

train = train.drop(columns=['Cabin'])

test = test.drop(columns=['Cabin']);
train = train.dropna(subset=['Embarked']).reset_index(drop=True)
missing = test.isnull().sum()

missing = missing[missing > 0]

missing
test['Fare'].hist();

plt.title('Fare Distribution on Test Set')

plt.xlabel('Fare');
test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])
train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
train['Family_Name'] = train['Name'].apply(lambda x : x.split(',')[0])

train['First_Name'] = train['Name'].apply(lambda x : x.split(',')[1])

train['Honorific'] = train['First_Name'].apply(lambda x : x.split('.')[0])

train = train.drop(columns=['Name', 'First_Name'])

test['Family_Name'] = test['Name'].apply(lambda x : x.split(',')[0])

test['First_Name'] = test['Name'].apply(lambda x : x.split(',')[1])

test['Honorific'] = test['First_Name'].apply(lambda x : x.split('.')[0])

test = test.drop(columns=['Name', 'First_Name'])
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

train[['Age', 'Fare']] = sc.fit_transform(train[['Age','Fare']])

test[['Age', 'Fare']] = sc.fit_transform(test[['Age','Fare']])
display('Train Dataset')

display(train.head(3))

display('Test Dataset')

display(test.head(3))
h2o.init()

# Load the DataFrames on H2O format

train_frame = h2o.H2OFrame(train)

test_frame = h2o.H2OFrame(test)

train_frame['Survived'] = train_frame['Survived'].asfactor()

# Features to predict

x = train_frame.columns

y = 'Survived'

x.remove(y)
%%time

# Build and Train the model

model = H2OAutoML(max_models=10, seed=42, nfolds=5)

model.train(x=x, y=y, training_frame=train_frame)

lb = model.leaderboard

lb
m = h2o.get_model(lb[2, "model_id"])

m.varimp_plot()
train_frame_b = train_frame[:, ['Family_Name', 'Honorific', 'Sex', 'Pclass', 'Fare', 'Deck', 'Survived']]

train_frame_b['Survived'] = train_frame_b['Survived'].asfactor()

x_b = train_frame_b.columns

y_b = 'Survived'

x_b.remove(y_b)

model_b = H2OAutoML(max_models=10, seed=42, nfolds=5)

model_b.train(x=x_b, y=y_b, training_frame=train_frame_b)

lb_b = model_b.leaderboard

lb_b
predictions = model.leader.predict(test_frame);

predictions = predictions.as_data_frame()

predictions = predictions.predict

submission_frame = h2o.H2OFrame(submission)

passenger_id = submission_frame['PassengerId'].as_data_frame()

submission_final = pd.concat([passenger_id, predictions], axis=1, ignore_index=False)

submission_final.columns = ['PassengerId', 'Survived']

submission_final.to_csv('submission.csv', index=False)

print('Submission file saved!')