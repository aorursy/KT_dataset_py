from __future__ import print_function

from builtins import range

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

torch.manual_seed(1234)

from sklearn import preprocessing
# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier ,AdaBoostClassifier

from sklearn.model_selection import train_test_split

import lightgbm as lgb

# loading dataset 

training_v2 = pd.read_csv("../input/widsdatathon2020/training_v2.csv")
# creating independent features X and dependant feature Y

y = training_v2['hospital_death']

X = training_v2

X = training_v2.drop('hospital_death',axis = 1)
# Remove Features with more than 75 percent missing values

train_missing = (X.isnull().sum() / len(X)).sort_values(ascending = False)

train_missing = train_missing.index[train_missing > 0.60]

X = X.drop(columns = train_missing)
#Convert categorical variable into dummy/indicator variables.

X = pd.get_dummies(X)
# Imputation transformer for completing missing values.

my_imputer = SimpleImputer()

new_data = pd.DataFrame(my_imputer.fit_transform(X))

new_data.columns = X.columns

X= new_data
# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = X.corr().abs()

corr_matrix.head()

# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()

# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))

#Drop the columns with high correlations

X = X.drop(columns = to_drop)
# Initialize an empty array to hold feature importances

feature_importances = np.zeros(X.shape[1])



# Create the model with several hyperparameters

model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')

for i in range(2):

    

    # Split into training and validation set

    train_features, valid_features, train_y, valid_y = train_test_split(X, y, test_size = 0.25, random_state = i)

    

    # Train using early stopping

    model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)],eval_metric = 'auc', verbose = 200)

    

    # Record the feature importances

    feature_importances += model.feature_importances_

# Make sure to average feature importances! 

feature_importances = feature_importances / 2

feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

# Find the features with zero importance

zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

print('There are %d features with 0.0 importance' % len(zero_features))

# Drop features with zero importance

X = X.drop(columns = zero_features)
# Normalize the data attributes

normalized_X = preprocessing.normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Define training hyperprameters.

batch_size = 60

num_epochs = 50

learning_rate = 0.01

size_hidden= 100



#Calculate some other hyperparameters based on data.  

batch_no = len(X_train) // batch_size  #batches

cols=X_train.shape[1] #Number of columns in input matrix

classes= len(np.unique(y_train))
class Net(nn.Module):

    def __init__(self,cols,size_hidden,classes):

        super(Net, self).__init__()

        #Note that 17 is the number of columns in the input matrix. 

        self.fc1 = nn.Linear(cols, size_hidden)

        #variety of # possible for hidden layer size is arbitrary, but needs to be consistent across layers.  3 is the number of classes in the output (died/survived)

        self.fc2 = nn.Linear(size_hidden, classes)

        

    def forward(self, x):

        x = self.fc1(x)

        x = F.dropout(x, p=0.1)

        x = F.relu(x)

        x = self.fc2(x)

        return F.softmax(x, dim=1)

    

net = Net(cols, size_hidden, classes)
#Adam is a specific flavor of gradient decent which is typically better

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
from sklearn.utils import shuffle

from torch.autograd import Variable

running_loss = 0.0

for epoch in range(num_epochs):

    #Shuffle just mixes up the dataset between epocs

    train_X, train_y = shuffle(X_train, y_train)

    # Mini batch learning

    for i in range(batch_no):

        start = i * batch_size

        end = start + batch_size

        inputs = Variable(torch.FloatTensor(train_X[start:end].values.astype(np.float32)))

        labels = Variable(torch.LongTensor(train_y[start:end].values.astype(np.float32)))

        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        

    print('Epoch {}'.format(epoch+1), "loss: ",running_loss)

    running_loss = 0.0  
import pandas as pd

#This is a little bit tricky to get the resulting prediction.  

def calculate_accuracy(x,y=[]):

    """

    This function will return the accuracy if passed x and y or return predictions if just passed x. 

    """

    # Evaluate the model with the test set. 

    X = Variable(torch.FloatTensor(x))  

    result = net(X) #This outputs the probability for each class.

    _, labels = torch.max(result.data, 1)

    if len(y) != 0:

        num_right = np.sum(labels.data.numpy() == y)

        print('Accuracy {:.2f}'.format(num_right / len(y)), "for a total of ", len(y), "records")

        return pd.DataFrame(data= {'actual': y, 'predicted': labels.data.numpy()})

    else:

        print("returning predictions")

        return labels.data.numpy()

result1=calculate_accuracy(X_train.values.astype(np.float32),y_train.values.astype(np.float32))

result2=calculate_accuracy(X_test.values.astype(np.float32),y_test.values.astype(np.float32))
