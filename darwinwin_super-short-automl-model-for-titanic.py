import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



# Import H2O AutoML

import h2o

from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
# Read training set

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_train.head()



# Read evaluation set

data_eval = pd.read_csv("/kaggle/input/titanic/test.csv")

data_eval.head()

def feature_engineering(data):



    # Column Pclass

    dummies = pd.get_dummies(data['Pclass'], prefix='Pclass')

    data = pd.concat([data, dummies], axis=1)

    

    # Column Name

    data['Name_Mr'] = data['Name'].str.contains('Mr.').astype(int)

    data['Name_Mrs'] = data['Name'].str.contains('Mrs.').astype(int)

    data['Name_Miss'] = data['Name'].str.contains('Miss.').astype(int)

    data['Name_Master'] = data['Name'].str.contains('Master.').astype(int)

    data['Name_Doctor'] = data['Name'].str.contains('Dr.').astype(int)

    data['Name_Ms'] = data['Name'].str.contains('Ms.').astype(int)

    data['Name_Rev'] = data['Name'].str.contains('Rev.').astype(int)

    data['Name_Major'] = data['Name'].str.contains('Major.').astype(int)

    data['Name_OtherTitle']=((data.Name_Mr==0)&(data.Name_Mrs==0)&(data.Name_Miss==0)&(data.Name_Master==0)&(data.Name_Doctor==0)&(data.Name_Ms==0)&(data.Name_Rev==0)&(data.Name_Major==0)).astype(int)

    

    # Column Sex

    data['Sex'] = data['Sex'].map({'male':0, 'female':1})

    

    # Column Ticket

    data['Ticket_PC'] = data['Ticket'].fillna('').str.contains('PC').astype(int)

    data['Ticket_CA'] = data['Ticket'].fillna('').str.contains('CA').astype(int)

    data['Ticket_A/5'] = data['Ticket'].fillna('').str.contains('A/5').astype(int)

    data['Ticket_A/4'] = data['Ticket'].fillna('').str.contains('A/4').astype(int)

    data['Ticket_PP'] = data['Ticket'].fillna('').str.contains('PP').astype(int)

    data['Ticket_SOTON'] = data['Ticket'].fillna('').str.contains('SOTON').astype(int)

    data['Ticket_STON'] = data['Ticket'].fillna('').str.contains('STON').astype(int)

    data['Ticket_SC/Paris'] = data['Ticket'].fillna('').str.contains('SC/PARIS').astype(int)

    data['Ticket_W/C'] = data['Ticket'].fillna('').str.contains('W/C').astype(int)

    data['Ticket_FCC'] = data['Ticket'].fillna('').str.contains('FCC').astype(int)

    data['Ticket_LINE'] = data['Ticket'].fillna('').str.contains('LINE').astype(int)

    data['Ticket_SOC'] = data['Ticket'].fillna('').str.contains('SOC').astype(int)

    data['Ticket_SC'] = data['Ticket'].fillna('').str.contains('SC').astype(int)

    data['Ticket_C'] = data['Ticket'].fillna('').str.contains('C ').astype(int)

    data['Ticket_Numeric'] = data['Ticket'].str.isnumeric().astype(int)

       

    # Column Cabin

    data['Cabin_A'] = data.Cabin.fillna('').str.contains('A').astype(int)

    data['Cabin_B'] = data.Cabin.fillna('').str.contains('B').astype(int)

    data['Cabin_C'] = data.Cabin.fillna('').str.contains('C').astype(int)

    data['Cabin_D'] = data.Cabin.fillna('').str.contains('D').astype(int)

    data['Cabin_E'] = data.Cabin.fillna('').str.contains('E').astype(int)

    data['Cabin_F'] = data.Cabin.fillna('').str.contains('F').astype(int)

    data['Cabin_G'] = data.Cabin.fillna('').str.contains('G').astype(int)

    

    # Column Embarked

    dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')

    data = pd.concat([data, dummies], axis=1)

    

    # Drop columns

    data.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked', 'Pclass'], inplace=True)

    

    return data
# Cleanse train set

data_train_cleansed = feature_engineering(data_train.copy())
# Train AutoML Model

H2O_train = h2o.H2OFrame(data_train_cleansed)

x =H2O_train.columns

y ='Survived'

x.remove(y)



H2O_train[y] = H2O_train[y].asfactor()



aml = H2OAutoML(max_models=99999, seed=1)

aml.train(x=x, y=y, training_frame=H2O_train)



# Print AutoML leaderboard

aml.leaderboard
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

# model = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# Get train data accuracy

pred = aml.predict(h2o.H2OFrame(data_train_cleansed))

pred = pred.as_data_frame()['predict'].tolist()



accuracy = sum(1 for x,y in zip(np.round(pred),data_train_cleansed.Survived) if x == y) / len(data_train_cleansed.Survived)

print('accuracy:',accuracy)
# Transform test set

data_eval_cleansed = feature_engineering(data_eval.copy())



X_eval = data_eval_cleansed



y_eval_pred = aml.predict(h2o.H2OFrame(X_eval))

y_eval_pred = y_eval_pred.as_data_frame()['predict'].tolist()



output = pd.DataFrame({'PassengerId': data_eval.PassengerId, 'Survived': y_eval_pred})

output.to_csv('my_submission_202002015.csv', index=False)

print("Your submission was successfully saved!")