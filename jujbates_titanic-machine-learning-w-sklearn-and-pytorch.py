import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import cufflinks as cf

cf.set_config_file(offline=True, world_readable=True, theme='space')



# matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import re

from bs4 import BeautifulSoup



# sklearn

# from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.utils import shuffle



from sklearn.metrics import accuracy_score, log_loss



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression







import os

print(os.listdir("../input"))
def normalizer(data, min_data_scaler, max_data_scaler, min_scaler, max_scaler):

    """

    Normalize the data with Min-Max scaling to a range of [min_scaler, max_scaler]

    :param data: The data to be normalized

    :return: Normalized data

    """

    return min_scaler + (data - min_data_scaler)*(max_scaler-min_scaler)/(max_data_scaler - min_data_scaler)





def name_str_to_word_list(name):

    """"""

    name = re.sub(r"[^a-zA-Z0-9]", " ", name.lower()) # Convert to lower case

    words = name.split() # Split string into words

    return words



def build_name_dict(data, vocab_size = 5000):

    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    name_count = {} # A dict storing the words that appear in the reviews along with how often they occur



    for index, row in data.iterrows():

        full_name = name_str_to_word_list(row.Name)

        for name in full_name:

            if name in name_count:

                name_count[name] += 1

            else:

                name_count[name] = 1

                

    # if vocab_size is larger that actual vocab_size change the size

    if len(name_count.keys()) < vocab_size:

        vocab_size = len(name_count.keys())

    

    # Sort the names found in `data` so that sorted_names[0] is the most frequently appearing name and

    # sorted_names[-1] is the least frequently appearing word.

    sorted_names = None

    sorted_names = [item[0] for item in sorted(name_count.items(), key=lambda x: x[1], reverse=True)]



    name_dict = {} # This is what we are building, a dictionary that translates words into integers

    for idx, name in enumerate(sorted_names[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'

        name_dict[name] = idx + 2                              # 'infrequent' labels

        

    return name_dict



def convert_and_pad(word_dict, sentence, pad=500):

    NOWORD = 0 # We will use 0 to represent the 'no word' category

    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict

    

    working_sentence = [NOWORD] * pad

    sentence = name_str_to_word_list(sentence)

    for word_index, word in enumerate(sentence[:pad]):

        

        if word in word_dict:

            working_sentence[word_index] = word_dict[word]

        else:

            working_sentence[word_index] = INFREQ

            

    return working_sentence, min(len(sentence), pad)



def convert_and_pad_data(word_dict, data, pad=500):

    new_col_name = {}

    new_col_name_len = {}

    for index, row in data.iterrows():

        full_name = row.Name

        p_id = row.PassengerId

        

        row.Name, row.NameLen = convert_and_pad(word_dict, full_name, pad)

        new_col_name[p_id] = row.Name

        new_col_name_len[p_id] = row.NameLen

    data['Name'] = data['PassengerId'].map(new_col_name)

    data['NameLen'] = data['PassengerId'].map(new_col_name_len)



    return data
intput_dir = '../input/'

df_train = pd.read_csv(intput_dir + "train.csv", header = 0, dtype={'Age': np.float64})

df_test = pd.read_csv(intput_dir + "test.csv", header = 0, dtype={'Age': np.float64})

df_test_survive = pd.read_csv(intput_dir + "gender_submission.csv")





df_test.insert(1, 'Survived', df_test_survive.Survived)

df_train.insert(4, 'NameLen', 0)

df_test.insert(4, 'NameLen', 0)



datasets = {

    'train': df_train,

    'test': df_test

}



for k in datasets.keys():

    print("-"*60)

    print("{}: ".format(k))

    datasets[k].info()

datasets['train'].head(10)
survived = datasets['train'][datasets['train']["Survived"] == 1]

died = datasets['train'][datasets['train']["Survived"] == 0]



trace1 = go.Bar(x=['Died', 'Survived'],

                y=datasets['train'].Survived.value_counts(),

                marker=dict(

                     color=['rgb(107, 107, 107)', 'rgb(26, 118, 255)']),

                name = "Survived",

               )



layout = dict(title = 'Survivers of Titanic Passanger',

              xaxis= dict(title= 'Died and Survived Count',zeroline= False),

             )



data = [trace1]

fig = dict(data = data, layout = layout)

iplot(fig)
age_avg = datasets['train']['Age'].mean()

age_std = datasets['train']['Age'].std()

for k in datasets.keys():

    age_null_count = datasets[k]['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    datasets[k]['Age'][np.isnan(datasets[k]['Age'])] = age_null_random_list

    datasets[k]['Age'] = datasets[k]['Age'].astype(int)
print('Age of Youngest Passanger: {}'.format(min(datasets['train'].Age)))

print('Age of Oldest Passanger: {}'.format(max(datasets['train'].Age)))



df = pd.DataFrame({

    'Died': died.Age,

    'Survived': survived.Age,

})

color=['rgb(107, 107, 107)', 'rgb(26, 118, 255)']

df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Ages of Passangers', title='(Survived and Died Count) vs Ages of Passangers', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].Age.describe())
df = pd.DataFrame({

    'Died': died.Pclass,

    'Survived': survived.Pclass,

})

color=['rgb(107, 107, 107)', 'rgb(26, 118, 255)']

df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Social Class of Passangers', title='(Survived and Died Count) vs Social Class of Passangers', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].Pclass.describe())
for k in datasets.keys():

    datasets[k]['FamilySize'] = datasets[k]['SibSp'] + datasets[k]['Parch'] + 1



survived = datasets['train'][datasets['train']["Survived"] == 1]

died = datasets['train'][datasets['train']["Survived"] == 0]


df = pd.DataFrame({

    'Died': died.SibSp,

    'Survived': survived.SibSp,

})

color=['rgb(107, 107, 107)', 'rgb(26, 118, 255)']



df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Sibling and Spouse Count of Passangers', title='(Survived and Died Count) vs Passanger\'s (Sibling and Spouse Count)', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].SibSp.describe())

df = pd.DataFrame({

    'Died': died.Parch,

    'Survived': survived.Parch,

})

color=['rgb(107, 107, 107)', 'rgb(26, 118, 255)']

df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Parents and Children Count of Passangers', title='(Survived and Died Count) vs Passanger\'s Parental and Child Count', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].Parch.describe())
df = pd.DataFrame({

    'Died': died.FamilySize,

    'Survived': survived.FamilySize,

})

color=['rgb(107, 107, 107)', 'rgb(26, 118, 255)']

df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Family Size of Passangers', title='(Survived and Died Count) vs Passanger\'s Family Size', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].Parch.describe())
fare_avg = datasets['train']['Fare'].mean()

fare_std = datasets['train']['Fare'].std()

for k in datasets.keys():

    fare_null_count = datasets[k]['Fare'].isnull().sum()

    fare_null_random_list = np.random.randint(fare_avg - fare_std, fare_avg + fare_std, size=fare_null_count)

    datasets[k]['Fare'][np.isnan(datasets[k]['Fare'])] = fare_null_random_list

df = pd.DataFrame({

    'Died': died.Fare,

    'Survived': survived.Fare,

})

df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Passanger\'s Fare', title='(Family Size) vs Passanger\'s Fare', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].Fare.describe())

for k in datasets.keys():

    datasets[k]['Embarked'] = datasets[k]['Embarked'].fillna('U')



survived = datasets['train'][datasets['train']["Survived"] == 1]

died = datasets['train'][datasets['train']["Survived"] == 0]



df = pd.DataFrame({

    'Died': died.Embarked,

    'Survived': survived.Embarked,

})

df.iplot(kind='histogram', yTitle='Number of Passangers', xTitle='Passanger\'s Fare', title='(Family Size) vs Passanger\'s Fare', filename='cufflinks/multiple-histograms', color=color, theme='space')



print(datasets['train'].Embarked.describe())

name_dict = build_name_dict(datasets['train'])
# print(len(name_dict))

# print(name_dict)



# This is my high for fullname bag of word truncation and padding

# h = 0

# for row in np_test:

#     l = len(name_str_to_word_list(row[3]))

#     if l > h:

#         h = l

# print('h: ', h)
pad = 15 # highest name length

for k in datasets.keys():

    datasets[k] = convert_and_pad_data(name_dict, datasets[k], pad=pad)

    for index, row in datasets[k].iterrows():

        for i in range(1, pad+1):

            datasets[k]['Name_' + str(i)] = row.Name[i-1]
for k in datasets.keys():

    dummy_fields = ['Pclass', 'Sex', 'Embarked']

    for each in dummy_fields:

        dummies = pd.get_dummies(datasets[k][each], prefix=each, drop_first=False)

        for header in list(dummies.columns.values):

            datasets[k][header] = dummies[header]

            

datasets['train'].head(1)
datasets['test']['Embarked_U'] = 0
print(datasets['train'].columns.tolist())

for k in datasets.keys():

    datasets[k] = datasets[k][[

        'PassengerId',

        'Survived',

        'Age',

        'SibSp',

        'Parch',

        'FamilySize',

        'Pclass_1', 'Pclass_2', 'Pclass_3',

        'Fare',

        'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_U',

        'Sex_female', 'Sex_male',

        'Name_1', 'Name_2', 'Name_3', 'Name_4', 'Name_5', 'Name_6', 'Name_7', 'Name_8', 'Name_9', 'Name_10', 'Name_11', 'Name_12', 'Name_13', 'Name_14', 'Name_15',

        'NameLen',

    ]]

datasets['train'].head(1)



# age scalers

age_min = 0

age_max = 90



# sibsp scalers

sibsp_min = 0

sibsp_max = 10



# parch scalers

parch_min = 0

parch_max = 10



# family size scalers

family_size_min = 0

family_size_max = 12



# fare scalers

fare_min = 0

fare_max = 600



# name scalers

name_min = 0

name_max = 1527



# name scalers

name_len_min = 0

name_len_max = 15

for k in datasets.keys():

    datasets[k].Age = normalizer(datasets[k].Age, age_min, age_max, 0, 1)

    datasets[k].SibSp = normalizer(datasets[k].SibSp, sibsp_min, sibsp_max, 0, 1)

    datasets[k].Parch = normalizer(datasets[k].Parch, parch_min, parch_max, 0, 1)

    datasets[k].FamilySize = normalizer(datasets[k].FamilySize, family_size_min, family_size_max, 0, 1)

    datasets[k].Fare = normalizer(datasets[k].Fare, fare_min, fare_max, 0, 1)

    for i in range(1,pad+1):

        datasets[k]['Name_'+str(i)] = normalizer(datasets[k]['Name_'+str(i)], name_min, name_max, 0, 1)

    datasets[k].NameLen = normalizer(datasets[k].NameLen, name_len_min, name_len_max, 0, 1)



datasets['train'].head(1)
for k in datasets.keys():

    if 'PassengerId' in datasets[k]:

        datasets[k] = datasets[k].drop("PassengerId", axis =1)



train = datasets['train'].values

test = datasets['test'].values

classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 100),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()

]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)

print('log: ', log.shape, '\n', log.head())



acc_dict = {}

n_tests = 5

# sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)

X_train, y_train = train[0::, 1::], train[0::, 0]

X_test, y_test = test[0::, 1::], test[0::, 0]



for i in range(n_tests):

    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    X_test, y_test = shuffle(X_test, y_test, random_state=0)

    for clf in classifiers:

        

        name = clf.__class__.__name__

        

        print("%%"*30)

        print(name)

        clf.fit(X_train, y_train)

        test_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, test_predictions)

        if name in acc_dict:

            acc_dict[name] += acc * 100.0

        else:

            acc_dict[name] = acc * 100.0



for clf in acc_dict:

    log = log.append(pd.DataFrame([[clf, acc_dict[clf]/n_tests]], columns=log_cols))





  
trace1 = go.Bar(x=log.Accuracy,

                y=log.Classifier,

                orientation = 'h'

               )



layout = dict(title = 'Classifier Accuracy',

              xaxis= dict(title= 'Accuracy', zeroline= False),

              margin=go.layout.Margin(

                  l=250,

                  r=100,

                  b=100,

                  t=100,

              ),

             )



data = [trace1]

fig = dict(data = data, layout = layout)

iplot(fig)


# import torch

# import torch.utils.data



# # train = np_datasets[0] 

# # test = np_datasets[1]



# # print('train: \n', type(train))

# # print('train: \n', train[0])



# # train_y = torch.from_numpy(train).float().squeeze()

# # train_X = torch.from_numpy(train.drop([0], axis=1).values).long()

# # test_y = torch.from_numpy(test).float().squeeze()

# # test_X = torch.from_numpy(test.drop([0], axis=1).values).long()

# # print('train_y: ', train_y)

# # print('train_X: ', train_X)

# # print('test_y: ', test_y)

# # print('test_X: ', test_X)



# # train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()

# # train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()







# # train = train.values

# # test  = test.values

# import torch



# np_train_y = train.Survived.values

# np_train_X = train.drop('Survived', axis=1).values

# torch_train_y = torch.from_numpy(np_train_y).float().squeeze()

# torch_train_X = torch.from_numpy(np_train_X).float()





# np_test_y = test.Survived.values

# np_test_X = test.drop('Survived', axis=1).values



# torch_test_y = torch.from_numpy(np_test_y).float().squeeze()

# torch_test_X = torch.from_numpy(np_test_X).float()



# from sklearn.model_selection import train_test_split



# x_train, x_val, y_train, y_val = train_test_split(np_train_X, np_train_y, test_size = 0.1)

# torch_train_X.shape
# import torch

# import torch.nn as nn

# import numpy as np

# import matplotlib.pyplot as plt

# # 



# # Hyper-parameters

# _, input_size = torch_train_X.shape 

# output_size = 2

# num_epochs = 10

# learning_rate = 0.01



# # Linear regression model

# model = nn.Linear(input_size, output_size)



# # Loss and optimizer

# criterion = nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  



# # Train the model

# for epoch in range(num_epochs):

    

#     # Convert numpy arrays to torch tensors

#     inputs = torch_train_X 

#     targets = torch_train_y



#     # Forward pass

#     outputs = model(inputs)

#     loss = criterion(outputs, targets)

    

#     # Backward and optimize

#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()

#     if (epoch+1) % 5 == 0:

#         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))







# # # Plot the graph

# # predicted = model(torch.from_numpy(np_test_X).float()).detach().numpy()

# # plt.plot(np_train_X, np_train_y, 'ro', label='Original data')

# # plt.plot(np_test_X, predicted, label='Fitted line')

# # plt.legend()

# # plt.show()



        

# # # Plot the graph

# # predicted = model(np_train_X.detach().numpy()

# # plt.plot(np_train_X, np_train_y, 'ro', label='Original data')

# # plt.plot(np_train_X, predicted, label='Fitted line')

# # plt.legend()

# # plt.show()



# # Save the model checkpoint

# # torch.save(model.state_dict(), 'model.ckpt')

# import torch

# import torch.nn as nn

# import torch.nn.functional as F



# # MultiLayer Perceptron

# class MLP(nn.Module):

    

#     def __init__(self, input_size, output_size=2):

#         super(MLP, self).__init__()

#         self.fc1 = nn.Linear(input_size, 1000)

#         self.fc2 = nn.Linear(1000, output_size)

        

#     def forward(self, x):

#         x = self.fc1(x)

#         x = F.dropout(x, p=0.4)

#         x = F.relu(x)

#         x = self.fc2(x)

#         x = torch.sigmoid(x)  

#         return x



# mlp_net = MLP(30)
# batch_size = 50

# num_epochs = 50

# learning_rate = 0.001

# batch_no = len(x_train) // batch_size
# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(mlp_net.parameters(), lr=learning_rate)
# from sklearn.utils import shuffle

# from torch.autograd import Variable



# for epoch in range(num_epochs):

#     x_train, y_train = shuffle(x_train, y_train)

#     # Mini batch learning

#     for i in range(batch_no):

#         start = i * batch_size

#         end = start + batch_size

#         x_var = Variable(torch.FloatTensor(x_train[start:end]))

#         y_var = Variable(torch.LongTensor(y_train[start:end]))



        

#         optimizer.zero_grad()

#         ypred_var = mlp_net(x_var)

#         loss =criterion(ypred_var, y_var)

#         loss.backward()

#         optimizer.step()

#     if (epoch+1) % 5 == 0:

#         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



# # Evaluate the model

# test_var = Variable(torch.FloatTensor(x_val), requires_grad=True)

# with torch.no_grad():

#     result = mlp_net(test_var)

# values, labels = torch.max(result, 1)

# num_right = np.sum(labels.data.numpy() == y_val)

# print('Accuracy {:.2f}'.format(num_right / len(y_val)))

# # Applying model on the test data

# X_test_var = Variable(torch.FloatTensor(np_test_X), requires_grad=True) 

# with torch.no_grad():

#     test_result = mlp_net(X_test_var)

# print(test_result)



# values, labels = torch.max(test_result, 1)

# survived = labels.data.numpy()