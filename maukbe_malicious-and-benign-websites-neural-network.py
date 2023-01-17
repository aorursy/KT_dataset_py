# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings  

warnings.filterwarnings('ignore')
dataset = pd.read_csv('../input/malicious-and-benign-websites/dataset.csv')

dataset.describe(include='all')
dataset.head()
dataset.drop('URL', axis =1, inplace=True)
# Look for null values 

print(dataset.isnull().sum())
dataset.drop('CONTENT_LENGTH', axis =1, inplace=True)

dataset.dropna(inplace=True)

print(dataset.isnull().sum())
dataset.drop(['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', ], axis =1, inplace=True)
corr = dataset.corr()

corr.style.background_gradient(cmap='coolwarm')
dataset.drop(['TCP_CONVERSATION_EXCHANGE','URL_LENGTH','APP_BYTES','SOURCE_APP_PACKETS','REMOTE_APP_PACKETS','SOURCE_APP_BYTES','REMOTE_APP_BYTES'], axis = 1, inplace=True)

corr = dataset.corr()

corr.style.background_gradient(cmap='coolwarm')
import seaborn as sns

sns.distplot(dataset.loc[dataset['Type'] == 1]['NUMBER_SPECIAL_CHARACTERS'], bins = 50, color='red')

sns.distplot(dataset.loc[dataset['Type'] == 0]['NUMBER_SPECIAL_CHARACTERS'], bins = 50, color='blue')
sns.distplot(dataset.loc[dataset['Type'] == 1]['DIST_REMOTE_TCP_PORT'], bins = 50, color='red')

sns.distplot(dataset.loc[dataset['Type'] == 0]['DIST_REMOTE_TCP_PORT'], bins = 50, color='blue')
print(dataset.loc[dataset['Type'] == 0]['DIST_REMOTE_TCP_PORT'].value_counts())
print(dataset.loc[dataset['Type'] == 1]['DIST_REMOTE_TCP_PORT'].value_counts())
sns.distplot(dataset.loc[dataset['Type'] == 1]['REMOTE_IPS'], bins = 50, color='red')

sns.distplot(dataset.loc[dataset['Type'] == 0]['REMOTE_IPS'], bins = 50, color='blue')
sns.distplot(dataset.loc[dataset['Type'] == 1]['APP_PACKETS'], bins = 50, color='red')

sns.distplot(dataset.loc[dataset['Type'] == 0]['APP_PACKETS'], bins = 50, color='blue')
sns.boxplot(dataset['APP_PACKETS'])
dataset = dataset[((dataset.APP_PACKETS - dataset.APP_PACKETS.mean()) / dataset.APP_PACKETS.std()).abs() < 3]
sns.boxplot(dataset['APP_PACKETS'])
sns.distplot(dataset.loc[dataset['Type'] == 1]['APP_PACKETS'], bins = 50, color='red')

sns.distplot(dataset.loc[dataset['Type'] == 0]['APP_PACKETS'], bins = 50, color='blue')
print(dataset.loc[dataset['Type'] == 1]['DIST_REMOTE_TCP_PORT'].value_counts())
print(dataset.loc[dataset['Type'] == 0]['DIST_REMOTE_TCP_PORT'].value_counts())
sns.distplot(dataset.loc[dataset['Type'] == 1]['DNS_QUERY_TIMES'], bins = 50, color='red')

sns.distplot(dataset.loc[dataset['Type'] == 0]['DNS_QUERY_TIMES'], bins = 50, color='blue')
print(dataset['DNS_QUERY_TIMES'].value_counts())
# Scale data then split

from sklearn import preprocessing

# Separate into train and test as well as features and predictor

X = dataset.drop('Type',axis=1) #Predictors

y = dataset['Type']

X = preprocessing.scale(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
# Method for evaluating results

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculateScores(y_test, predictions):

    accuracy = 100*accuracy_score(y_test, predictions)

    precision = 100*precision_score(y_test, predictions)

    recall = 100*recall_score(y_test, predictions)

    f1 = 100*f1_score(y_test, predictions)

    print (' Accuracy  %.2f%%' % accuracy)

    print (' Precision %.2f%%'% precision)

    print (' Recall    %.2f%%'% recall)

    print (' F1        %.2f%%'% f1)

    print('Confusion Matrix')

    print(confusion_matrix(y_test,predictions))

    return {'Accuracy':accuracy, 'F1': f1}
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(solver='lbfgs')

model = reg.fit(X_train, y_train)

predictions = model.predict(X_test)

scores = calculateScores(y_test, predictions)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=1)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

scores = calculateScores(y_test, predictions)
def predict( X_train, y_train, **kwargs):

    mlp = MLPClassifier(**kwargs, random_state=1)

    mlp.fit(X_train, y_train)

    return mlp.predict(X_test)
def calculateScoresNoOutput(y_test, predictions):

    accuracy = 100*accuracy_score(y_test, predictions)

    precision = 100*precision_score(y_test, predictions)

    recall = 100*recall_score(y_test, predictions)

    f1 = 100*f1_score(y_test, predictions)

    return {'Accuracy':accuracy, 'F1': f1}
# Let's try the different solvers

solvers = ['lbfgs', 'sgd', 'adam']

results = []

for solver in solvers:

    result_dict = calculateScoresNoOutput(y_test, predict(X_train, y_train, solver=solver))

    result_dict['Solver'] = solver

    results.append(result_dict)

df = pd.DataFrame(results, columns = ['Solver','Accuracy', 'F1'])

df
# Generalise attempting different values

def try_different_values(values, column_name, X_train, y_train, **kwargs):

    results = []

    for value in values:

        kwargs[column_name] = value

        result_dict = calculateScoresNoOutput(y_test, predict(X_train, y_train, **kwargs))

        result_dict[column_name] = value

        results.append(result_dict)

    df = pd.DataFrame(results, columns = [column_name,'Accuracy', 'F1'])

    return df
activations = ['identity', 'logistic', 'tanh', 'relu']

try_different_values(activations, 'activation', X_train, y_train, solver='lbfgs')
alphas = []

for i in range(5,40):

     alphas.append(1/(2**i))

alpha_df = try_different_values(alphas, 'alpha', X_train, y_train, solver='lbfgs', activation='logistic')
alpha_df.set_index('alpha', inplace=True)

alpha_df.plot()
print(alpha_df.loc[alpha_df['Accuracy'].idxmax()])

print(alpha_df.loc[alpha_df['F1'].idxmax()])
# Store alpha

alpha= 4.76837158203125e-07
batch_sizes = [2 ** e for e in range(10)]

batch_df = try_different_values(batch_sizes, 'batch_size', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)
batch_df.set_index('batch_size', inplace=True)

batch_df.plot()
layers = []

for i in range (1,25):

    layers+= [(i)]

layers_df = try_different_values(layers, 'hidden_layer_sizes', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)
layers_df.set_index('hidden_layer_sizes', inplace=True)

layers_df.plot()
print(layers_df.loc[layers_df['Accuracy'].idxmax()])

print(layers_df.loc[layers_df['F1'].idxmax()])
layers = []

for i in range (1,15):

    for j in range(1,15):

        layers+= [(i,j)]

layers_df = try_different_values(layers, 'hidden_layer_sizes', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)
layers_df.set_index('hidden_layer_sizes', inplace=True)

layers_df.plot()
#layers_df = layers_df.reset_index()

layers_df = layers_df.reset_index()

print(layers_df.iloc[[layers_df['Accuracy'].idxmax()]])

print(layers_df.iloc[[layers_df['F1'].idxmax()]])
layers = []

for i in range (1,10):

    for j in range(1,10):

        for k in range (1,10):

            layers+= [(i,j,k)]

layers_3_df = try_different_values(layers, 'hidden_layer_sizes', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)
layers_3_df.set_index('hidden_layer_sizes', inplace=True)

layers_3_df.plot()
layers_3_df = layers_3_df.reset_index()

print(layers_3_df.iloc[[layers_3_df['Accuracy'].idxmax()]])

print(layers_3_df.iloc[[layers_3_df['F1'].idxmax()]])