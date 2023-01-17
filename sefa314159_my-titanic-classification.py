from warnings import filterwarnings

filterwarnings('ignore')

import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

from statistics import mean 
titanic_train = pd.read_csv('../input/titanic/train.csv')

train         = titanic_train.copy()
titanic_test = pd.read_csv('../input/titanic/test.csv')

test         = titanic_test.copy()
dataset = pd.merge(train, test, how = 'outer')
dataset.head()
titanic_train.head()
titanic_test.head()
dataset.info()
train.info()
test.info()
dataset.shape, train.shape, test.shape
dataset.isnull().sum()
train.isnull().sum()
test.isnull().sum()
dataset['Surname'] = 'Surname'
dataset['Title'] = 'Title'
dataset.head()
for counter in range(0, len(dataset)):

    full_name                   = dataset['Name'][counter].split(',')

    surname                     = full_name[0]

    name                        = full_name[1].split('.')

    title                       = name[0]

    dataset['Surname'][counter] = surname

    dataset['Title'][counter]   = title
dataset.head()
dataset.drop(columns = ['PassengerId', 'Survived', 'Name', 'Cabin'], inplace = True)
dataset.head()
train['Surname'] = 'Surname'
train['Title'] = 'Title'
train.head()
for counter in range(0, len(train)):

    full_name                 = train['Name'][counter].split(',')

    surname                   = full_name[0]

    name                      = full_name[1].split('.')

    title                     = name[0]

    train['Surname'][counter] = surname

    train['Title'][counter]   = title
train.drop(columns = ['PassengerId', 'Name', 'Cabin'], inplace = True)
train.head()
train.isnull().sum()
test['Surname'] = 'Surname'
test['Title'] = 'Title'
test.head()
for counter in range(0, len(test)):

    full_name                = test['Name'][counter].split(',')

    surname                  = full_name[0]

    name                     = full_name[1].split('.')

    title                    = name[0]

    test['Surname'][counter] = surname

    test['Title'][counter]   = title
test.drop(columns = ['Name', 'Cabin'], inplace = True)
test.head()
test.isnull().sum()
dataset.isnull().sum()
dataset.Embarked.fillna(-1000, inplace = True)
dataset = dataset[(dataset.Embarked != -1000)]
dataset.isnull().sum()
dataset_surname_unique = list(dataset.Surname.unique())
dataset_surname_unique[:10]
len(dataset_surname_unique)
surname_unique_list = []



for i in range(len(dataset_surname_unique)):

    surname_unique_list.append(i)

    

dataset_surname_number_array = np.array(surname_unique_list)
dataset_surname_number_array[:10]
def binary_encoder(x_array):

    #---------------------------------------------------------------

    binary_list = []

    #---------------------------------------------------------------

    for i in range(0, len(x_array)):

        x_ = f"{x_array[i]:b}"

        str_list = []

        for j in x_:

            str_list.append(int(j))

        binary_list.append(str_list)

    #---------------------------------------------------------------

    lengths = [len(i) for i in binary_list]

    max_length = max(lengths)

    #---------------------------------------------------------------

    for i in range(0, len(binary_list)):

        if len(binary_list[i]) < max_length:

            x_ = binary_list[i]

            x_.reverse()

            for j in range(0, int(max_length - len(binary_list[i]))):

                x_.append(0)

            x_.reverse()

            binary_list[i] = x_

    #---------------------------------------------------------------

    binary_array = np.array(binary_list)      

    return binary_array
surname_binary = binary_encoder(dataset_surname_number_array)
surname_binary[:10]
dataset_title_unique = list(dataset.Title.unique())
dataset_title_unique[:10]
len(dataset_title_unique)
title_unique_list = []



for i in range(len(dataset_title_unique)):

    title_unique_list.append(i)

    

dataset_title_array = np.array(title_unique_list)
dataset_title_array[:10]
title_binary = binary_encoder(dataset_title_array)
title_binary[:10]
dataset_ticket_unique = list(dataset.Ticket.unique())
dataset_ticket_unique[:10]
len(dataset_ticket_unique)
ticket_unique_list = []



for i in range(len(dataset_ticket_unique)):

    ticket_unique_list.append(i)

    

dataset_ticket_array = np.array(ticket_unique_list)
dataset_ticket_array[:10]
ticket_binary = binary_encoder(dataset_ticket_array)
ticket_binary[:10]
max(dataset['Fare']), min(dataset['Fare'])
max(dataset['Age']), min(dataset['Age'])
train['Fare_Seg'] = 0

train['Age_Seg']  = 0
train.head()
test['Fare_Seg'] = 0

test['Age_Seg']  = 0
test.head()
train['Fare'].fillna(-1000, inplace = True)

train['Age'].fillna(-1000, inplace = True)
test['Fare'].fillna(-1000, inplace = True)

test['Age'].fillna(-1000, inplace = True)
for i in range(len(train)):

    if (train['Fare'][i] == -1000):

        train['Fare_Seg'][i] = 0

    if (train['Fare'][i] >= 0 and train['Fare'][i] < 10):

        train['Fare_Seg'][i] = 1

    if (train['Fare'][i] >= 10 and train['Fare'][i] < 30):

        train['Fare_Seg'][i] = 2

    if (train['Fare'][i] >= 30 and train['Fare'][i] < 70):

        train['Fare_Seg'][i] = 3

    if (train['Fare'][i] >= 70 and train['Fare'][i] < 150):

        train['Fare_Seg'][i] = 4

    if (train['Fare'][i] >= 150 and train['Fare'][i] < 250):

        train['Fare_Seg'][i] = 5

    if (train['Fare'][i] >= 250 and train['Fare'][i] < 375):

        train['Fare_Seg'][i] = 6

    if (train['Fare'][i] >= 375):

        train['Fare_Seg'][i] = 7
for i in range(len(test)):

    if (test['Fare'][i] == -1000):

        test['Fare_Seg'][i] = 0

    if (test['Fare'][i] >= 0 and test['Fare'][i] < 10):

        test['Fare_Seg'][i] = 1

    if (test['Fare'][i] >= 10 and test['Fare'][i] < 30):

        test['Fare_Seg'][i] = 2

    if (test['Fare'][i] >= 30 and test['Fare'][i] < 70):

        test['Fare_Seg'][i] = 3

    if (test['Fare'][i] >= 70 and test['Fare'][i] < 150):

        test['Fare_Seg'][i] = 4

    if (test['Fare'][i] >= 150 and test['Fare'][i] < 250):

        test['Fare_Seg'][i] = 5

    if (test['Fare'][i] >= 250 and test['Fare'][i] < 375):

        test['Fare_Seg'][i] = 6

    if (test['Fare'][i] >= 375):

        test['Fare_Seg'][i] = 7
for i in range(len(train)):

    if (train['Age'][i] == -1000):

        train['Age_Seg'][i] = 0

    if (train['Age'][i] >= 0 and train['Age'][i] < 6):

        train['Age_Seg'][i] = 1

    if (train['Age'][i] >= 6 and train['Age'][i] < 15):

        train['Age_Seg'][i] = 2

    if (train['Age'][i] >= 15 and train['Age'][i] < 21):

        train['Age_Seg'][i] = 3

    if (train['Age'][i] >= 21 and train['Age'][i] < 30):

        train['Age_Seg'][i] = 4

    if (train['Age'][i] >= 30 and train['Age'][i] < 45):

        train['Age_Seg'][i] = 5

    if (train['Age'][i] >= 45 and train['Age'][i] < 60):

        train['Age_Seg'][i] = 6

    if (train['Age'][i] >= 60):

        train['Age_Seg'][i] = 7
for i in range(len(test)):

    if (test['Age'][i] == -1000):

        test['Age_Seg'][i] = 0

    if (test['Age'][i] >= 0 and test['Age'][i] < 6):

        test['Age_Seg'][i] = 1

    if (test['Age'][i] >= 6 and test['Age'][i] < 15):

        test['Age_Seg'][i] = 2

    if (test['Age'][i] >= 15 and test['Age'][i] < 21):

        test['Age_Seg'][i] = 3

    if (test['Age'][i] >= 21 and test['Age'][i] < 30):

        test['Age_Seg'][i] = 4

    if (test['Age'][i] >= 30 and test['Age'][i] < 45):

        test['Age_Seg'][i] = 5

    if (test['Age'][i] >= 45 and test['Age'][i] < 60):

        test['Age_Seg'][i] = 6

    if (test['Age'][i] >= 60):

        test['Age_Seg'][i] = 7
train.drop(columns = ['Fare', 'Age'], inplace = True)
train.head()
test.drop(columns = ['Fare', 'Age'], inplace = True)
test.head()
train.isnull().sum()
train.Embarked.fillna(-1000, inplace = True)
train = train[(train.Embarked != -1000)]
train.isnull().sum()
len(dataset_surname_unique)
dataset_surname_unique[:10]
surname_binary[:10]
surnames_lists_train = []
for i in train['Surname']:

    if(i in dataset_surname_unique):

        index = dataset_surname_unique.index(i)

        surnames_lists_train.append(surname_binary[index])

        

surnames_arr_train = np.array(surnames_lists_train)
surnames_arr_train[:10]
len(dataset_surname_unique)
dataset_surname_unique[:10]
surname_binary[:10]
surnames_lists_test = []
for i in test['Surname']:

    if(i in dataset_surname_unique):

        index = dataset_surname_unique.index(i)

        surnames_lists_test.append(surname_binary[index])

        

surnames_arr_test = np.array(surnames_lists_test)
surnames_arr_test[:10]
len(dataset_title_unique)
dataset_title_unique[:10]
title_binary[:10]
title_list_train = []
for i in train['Title']:

    if(i in dataset_title_unique):

        index = dataset_title_unique.index(i)

        title_list_train.append(title_binary[index])

        

title_arr_train = np.array(title_list_train)
title_arr_train[:10]
len(dataset_title_unique)
dataset_title_unique[:10]
title_binary[:10]
title_list_test = []
for i in test['Title']:

    if(i in dataset_title_unique):

        index = dataset_title_unique.index(i)

        title_list_test.append(title_binary[index])

        

title_arr_test = np.array(title_list_test)
title_arr_test[:10]
len(dataset_ticket_unique)
dataset_ticket_unique[:10]
ticket_binary[:10]
ticket_list_train = []
for i in train['Ticket']:

    if(i in dataset_ticket_unique):

        index = dataset_ticket_unique.index(i)

        ticket_list_train.append(ticket_binary[index])

        

ticket_arr_train = np.array(ticket_list_train)
ticket_arr_train[:10]
len(dataset_ticket_unique)
dataset_ticket_unique[:10]
ticket_binary[:10]
ticket_list_test = []
for i in test['Ticket']:

    if(i in dataset_ticket_unique):

        index = dataset_ticket_unique.index(i)

        ticket_list_test.append(ticket_binary[index])

        

ticket_arr_test = np.array(ticket_list_test)
ticket_arr_test[:10]
train.head()
train.drop(columns = ['Ticket', 'Surname', 'Title'], inplace = True)
train.head()
X = train.iloc[:, 1:].values
X[:10]
y = train.iloc[:, 0].values
y[:10]
labelencoder_gender = LabelEncoder()

X[:, 1]             = labelencoder_gender.fit_transform(X[:, 1])
X[:10]
labelencoder_embarked = LabelEncoder()

X[:, 4]               = labelencoder_embarked.fit_transform(X[:, 4])
X[:10]
X_without_embarked = np.delete(X, 4, 1)
X_without_embarked[:10]
X_embarked_binary = binary_encoder(X[:, 4])
X_embarked_binary[:10]
X = np.concatenate([X_without_embarked, X_embarked_binary], axis = 1)
X = np.concatenate([X, ticket_arr_train], axis = 1)
X = np.concatenate([X, surnames_arr_train], axis = 1)
X = np.concatenate([X, title_arr_train], axis = 1)
X[:5]
sc        = StandardScaler()

X_scaled  = sc.fit_transform(X)
X_scaled[:5]
X_scaled.shape
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 12)
lgbm_model = LGBMClassifier(n_estimators      = 10000,

                            learning_rate     = 0.01,

                            subsample         = 0.85,

                            max_depth         = 16,

                            min_child_samples = 32,

                            random_state      = 502,

                            reg_lambda        = 10,

                            reg_alpha         = 3,

                            num_leaves        = 250)



lgbm_model.fit(X_train, y_train)
y_pred_train = lgbm_model.predict(X_train)

accuracy_score(y_train, y_pred_train)
y_pred_test = lgbm_model.predict(X_test)

accuracy_score(y_test, y_pred_test)
splits = [10, 15, 20, 25]



cross_list = []



for i in range(4):

    kfold = KFold(n_splits = splits[i], random_state = 12)

    lgbm_cross_val = cross_val_score(lgbm_model, X_scaled, y, cv = kfold, scoring = 'accuracy')

    cross_list.append(lgbm_cross_val.mean())



cross_list
print(f'Accuracy : %{mean(cross_list) * 100}')
test.head()
test.drop(columns = ['PassengerId', 'Ticket', 'Surname', 'Title'], inplace = True)
test.head()
test_arr = test.values
test_arr[:10]
labelencoder_gender = LabelEncoder()

test_arr[:, 1]      = labelencoder_gender.fit_transform(test_arr[:, 1])
test_arr[:10]
labelencoder_embarked = LabelEncoder()

test_arr[:, 4]        = labelencoder_embarked.fit_transform(test_arr[:, 4])
test_arr[:10]
test_arr_without_embarked = np.delete(test_arr, 4, 1)
test_arr_without_embarked[:10]
test_arr_embarked_binary = binary_encoder(test_arr[:, 4])
test_arr_embarked_binary[:10]
test_arr = np.concatenate([test_arr_without_embarked, test_arr_embarked_binary], axis = 1)
test_arr = np.concatenate([test_arr, ticket_arr_test], axis = 1)
test_arr = np.concatenate([test_arr, surnames_arr_test], axis = 1)
test_arr = np.concatenate([test_arr, title_arr_test], axis = 1)
test_arr[:5]
sc               = StandardScaler()

test_arr_scaled  = sc.fit_transform(test_arr)
test_arr_scaled[:5]
test_arr_scaled.shape
y_pred = lgbm_model.predict(test_arr_scaled)
gender_sub = pd.read_csv('../input/titanic/gender_submission.csv')
gender_sub.head()
y_true = gender_sub.iloc[:, 1].values
accuracy_score(y_true, y_pred)
titanic_test.head()
submission = titanic_test.iloc[:, 0].values
submission[:10]
y_pred[:10]
submission.shape, y_pred.shape
submission = np.reshape(submission, (-1, 1))
submission[:10]
y_pred = np.reshape(y_pred, (-1, 1))
y_pred[:10]
submission = np.concatenate([submission, y_pred], axis = 1)
submission[:10]
sub_df = pd.DataFrame(data = submission, columns = ['PassengerId', 'Survived'])
sub_df.head(5)
sub_df.to_csv('My_Submission.csv', index = False)