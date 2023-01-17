# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np

train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
train_set.head()
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

%matplotlib inline

mpl.RcParams.update({'font.size': 22, 'figure.figsize': (20, 10)})
plt.style.use('ggplot')
green_cmap = sns.light_palette("green", as_cmap=True)
pd.crosstab(train_set.Pclass, train_set.Survived, margins=True)\
.style.background_gradient(cmap=green_cmap)
sns.factorplot('Pclass', 'Survived', data = train_set)
sex_perc = train_set[["Sex", "Survived"]].groupby("Sex", as_index=False).mean()
sns.barplot(x="Sex", y="Survived", data=sex_perc)
sns.factorplot("Sex", "Survived", data=train_set)
age_percent = train_set[["Age", "Survived"]].groupby("Age", as_index=False).mean()
sns.barplot("Age", "Survived", data=age_percent)
# Age smoothing : Choose median among the same title group.
# It is better than choosing median of total data set.
train_set['Title'] = train_set['Name']
for name_string in train_set['Name']:
    train_set['Title'] = train_set['Name'].str.extract('([A-Za-z]+)\.')
    
titles = set(train_set['Title'])
for title in titles:
    smoothed_age = train_set.groupby('Title')['Age'].median()[title]
    train_set['Age'].loc[train_set['Age'].isnull() & (train_set['Title'] == title)] = smoothed_age
# divide age sections.
train_set["Age_sec"] = np.nan
for i in range(10, 0, -1):
    train_set.loc[train_set["Age"] <= i*10, "Age_sec"] = i*10
    
train_set.head()
age_sec_percent = train_set[["Age_sec", "Survived"]].groupby("Age_sec", as_index=False).mean()
sns.barplot("Age_sec", "Survived", data=age_sec_percent)
# According to above barplot, select age segment.
train_set['Age_seg'] = 'Adult'
train_set['Age_seg'].loc[train_set['Age'] > 70] = 'Very Old'
train_set['Age_seg'].loc[train_set['Age'] < 10 ] = 'Very Young'
age_seg_perc = train_set[['Age_seg', 'Sex','Survived']].groupby(['Age_seg', 'Sex'], as_index=False).mean()
sns.barplot('Age_seg', 'Survived', hue='Sex', data=age_seg_perc)
# Test set (It is better to setting after merging with train set.)
test_set['Title'] = test_set['Name']
for name_string in test_set['Name']:
    test_set['Title'] = test_set['Name'].str.extract('([A-Za-z]+)\.')
    
test_titles = set(test_set['Title'])
for title in test_titles:
    smoothed_age = test_set.groupby('Title')['Age'].median()[title]
    if np.isnan(smoothed_age):
        smoothed_age = train_set.groupby('Title')['Age'].median()[title]
    test_set['Age'].loc[test_set['Age'].isnull() & (test_set['Title'] == title)] = smoothed_age

test_set['Age_seg'] = 'Adult'
test_set['Age_seg'].loc[test_set['Age'] > 70] = 'Very Old'
test_set['Age_seg'].loc[test_set['Age'] < 10 ] = 'Very Young'

test_set.describe()
sns.factorplot("Age_sec", "Survived", hue="Pclass", data=train_set)
sns.factorplot("Age_sec", "Survived", hue="Sex", data=train_set)
# I will use this categorical features, so indexed it accoring to the order.
title_list = sorted(list(set(train_set['Title'])))
train_set['Title'] = train_set['Title'].map({
    'Capt': 0, 'Col': 1, 'Countess': 2, 'Don': 3,
    'Dr': 4, 'Jonkheer': 5, 'Lady': 6, 'Major': 7,
    'Master': 8, 'Miss': 9, 'Mlle': 10, 'Mme': 11,
    'Mr': 12, 'Mrs': 13, 'Ms': 14, 'Rev': 15, 'Sir': 16
})

train_set.head()
# Following barplot, we can get the insight about title.
title_prec = train_set[['Title', 'Survived']].groupby('Title', as_index=False).mean()
sns.barplot('Title', 'Survived', data=title_prec)
# Test set (It is better to setting after merging with train set.)
test_title_list = sorted(list(set(test_set['Title'])))

test_set['Title'] = test_set['Title'].map({
    'Capt': 0, 'Col': 1, 'Countess': 2, 'Don': 3,
    'Dr': 4, 'Jonkheer': 5, 'Lady': 6, 'Major': 7,
    'Master': 8, 'Miss': 9, 'Mlle': 10, 'Mme': 11,
    'Mr': 12, 'Mrs': 13, 'Ms': 14, 'Rev': 15, 'Sir': 16
})

# Test set null value: It was 'Miss'.
test_set['Title'] = test_set['Title'].fillna(14)
# SibSp -> Sex could affect!
pd.crosstab([train_set.Survived, train_set.Sex], train_set.SibSp, margins=True)\
.style.background_gradient(cmap=green_cmap)
pd.crosstab([train_set.Survived, train_set.Sex], train_set.Parch, margins=True)\
.style.background_gradient(cmap=green_cmap)
# Total family number
train_set['Num_of_family'] = train_set['SibSp'] + train_set['Parch']

num_of_family_perc = train_set[['Num_of_family', 'Survived']].groupby('Num_of_family', as_index=False).mean()
sns.barplot('Num_of_family', 'Survived', data=num_of_family_perc)
# According to above barplot I will divide the segments.
train_set['Family_size'] = 'None'

train_set['Family_size'].loc[train_set['Num_of_family'] >= 4] = 'Large'
train_set['Family_size'].loc[train_set['Num_of_family'] < 4] = 'Small'
train_set['Family_size'].loc[train_set['Num_of_family'] == 0] = 'None'

train_set['Family_size'] = train_set['Family_size'].map({'None': 0, 'Small': 1, 'Medium': 2, 'Large': 3})

train_set.head()
title_prec = train_set[['Family_size', 'Survived']].groupby('Family_size', as_index=False).mean()
sns.barplot('Family_size', 'Survived', data=title_prec)
# Test set (It is better to setting after merging with train set.)
test_set['Num_of_family'] = test_set['SibSp'] + test_set['Parch']
test_set['Family_size'] = 'None'

test_set['Family_size'].loc[test_set['Num_of_family'] >= 4] = 'Large'
test_set['Family_size'].loc[test_set['Num_of_family'] < 4] = 'Small'
test_set['Family_size'].loc[test_set['Num_of_family'] == 0] = 'None'

test_set['Family_size'] = test_set['Family_size'].map({'None': 0, 'Small': 1, 'Medium': 2, 'Large': 3})

test_set.head()
# I also checked the exsitance of family!
train_set['Has_family'] = 1
train_set['Has_family'].loc[train_set['Num_of_family'] == 0] = 0

has_family_perc = train_set[['Has_family', 'Survived']].groupby('Has_family', as_index=False).mean()
sns.barplot('Has_family', 'Survived', data=has_family_perc)
# Test set (It is better to setting after merging with train set.)
test_set['Has_family'] = 1
test_set['Has_family'].loc[test_set['Num_of_family'] == 0] = 0
fare_percent = train_set[["Fare", "Survived"]].groupby("Fare", as_index=False).mean()
sns.barplot('Fare', 'Survived', data=fare_percent)
# It it better to divide the segments.
train_set['Fare_seg'] = 0

train_set['Fare_seg'].loc[train_set['Fare'] >= 25] = 2
train_set['Fare_seg'].loc[train_set['Fare'] >= 50] = 3
train_set['Fare_seg'].loc[train_set['Fare'] >= 75] = 4
train_set['Fare_seg'].loc[train_set['Fare'] < 25] = 1

train_set.head()
fare_seg_perc = train_set[['Fare_seg', 'Survived']].groupby('Fare_seg', as_index=False).mean()
sns.barplot('Fare_seg', 'Survived', data=fare_seg_perc)
# Test set (It is better to setting after merging with train set.)
test_set["Fare"] = test_set["Fare"].fillna(test_set["Fare"].median())

test_set['Fare_seg'] = 0

test_set['Fare_seg'].loc[test_set['Fare'] >= 25] = 2
test_set['Fare_seg'].loc[test_set['Fare'] >= 50] = 3
test_set['Fare_seg'].loc[test_set['Fare'] >= 75] = 4
test_set['Fare_seg'].loc[test_set['Fare'] < 25] = 1

test_set.head()
embarked_perc = train_set[["Embarked", "Survived"]].groupby("Embarked", as_index=False).mean()

fig, (axes_1, axes_2, axes_3) = plt.subplots(1, 3, figsize=(21, 7))

sns.countplot(x="Embarked", data=train_set, ax=axes_1)
sns.countplot(x="Survived", hue="Embarked", data=train_set, order=[0, 1], ax=axes_2)
sns.barplot(x="Embarked", y="Survived", data=embarked_perc, ax=axes_3)
train_set['Embarked'] = train_set['Embarked'].fillna('S')
# Existance of cabin!
train_set['Is_cabin'] = 0
train_set['Is_cabin'].loc[train_set['Cabin'].notnull()] = 1
cabin_perc = train_set[['Is_cabin', 'Survived']].groupby('Is_cabin', as_index=False).mean()
sns.barplot('Is_cabin', 'Survived', data=cabin_perc)
# Test set (It is better to setting after merging with train set.)
test_set['Is_cabin'] = 0
test_set['Is_cabin'].loc[test_set['Cabin'].notnull()] = 1
train_set['Last_name'] = train_set['Name'].apply(lambda x: str.split(x, ',')[0])
test_set['Last_name'] = test_set['Name'].apply(lambda x: str.split(x, ',')[0])

feat_list = ['PassengerId', 'Fare', 'Last_name', 'Embarked', 'Name', 'Num_of_family', 'SibSp', 'Parch', 'Ticket']
group_1ist = ['Fare', 'Last_name', 'Embarked']

# To infer family : same fare, same last name, same embarked
f_train_set = pd.merge(train_set[feat_list], train_set[['PassengerId', 'Survived']])

f_train_set.head()
f_test_set = test_set[feat_list]
f_test_set.head()
#concat train set and test set
f_merged_df = pd.concat([f_train_set, f_test_set], ignore_index=True)
# We have to check weather our assumption is right or wrong.
from IPython.display import display
cnt = 0
for group, group_df in f_merged_df.groupby(group_1ist):
    if (pd.Series(group_df['Num_of_family']).min() > 0 and len(group_df) > 1):
        display(group_df)
        cnt+=1
        if cnt == 10:
            break
# Beacuse of this, we have to check the tickets.
f_merged_df[(f_merged_df['Last_name'] == 'Andersson') & (f_merged_df['Num_of_family'] == 6)]
# Initial value : 0.5 is reasonable. Because the probabily of the family survived is 0.5 when no information is given.
f_merged_df['Is_family_survived'] = 0.5
# If family survived then 1, if not 0.
for _, group_df in f_merged_df.groupby(group_1ist):
    if (len(group_df) > 1):
        _max = group_df['Survived'].max()
        _min = group_df['Survived'].min()
        for idx, row in group_df.iterrows():
            if (_max == 1):
                f_merged_df['Is_family_survived'].loc[f_merged_df['PassengerId'] == row['PassengerId']] = _max
            elif (_min == 0):
                f_merged_df['Is_family_survived'].loc[f_merged_df['PassengerId'] == row['PassengerId']] = _min
# We also checked the ticket. This is because, there are some cases family had same tickets.
# If family survived then 1, if not 0.
for _, group_df in f_merged_df.groupby('Ticket'):
    if (len(group_df) > 1):
        _max = group_df['Survived'].max()
        _min = group_df['Survived'].min()
        for idx, row in group_df.iterrows():
            if ((row['Is_family_survived'] == 0) | (row['Is_family_survived'] == 0.5)):
                if (_max == 1):
                    f_merged_df['Is_family_survived'].loc[f_merged_df['PassengerId'] == row['PassengerId']] = _max
                elif (_min == 0):
                    f_merged_df['Is_family_survived'].loc[f_merged_df['PassengerId'] == row['PassengerId']] = _min
f_merged_df.head()
train_set['Is_family_survived'] = f_merged_df['Is_family_survived'][:891]

# When I use above code, NaN values occure.
test_set = pd.merge(test_set, f_merged_df[['PassengerId', 'Is_family_survived']][891:]) 
train_set.head()
test_set.head()
temp_df = train_set[['Survived', 'Sex', 'Age_seg', 'Has_family', 'Family_size', 'Fare_seg', 'Embarked', \
                     'Age', 'Is_cabin', 'Title', 'Is_family_survived']]

categorical_features = ['Sex', 'Age_seg', 'Embarked']

selected_df = pd.get_dummies(temp_df, categorical_features)
selected_df.head()
selected_df.describe()
# If you want to deal with figure, axes, use plt
fig = plt.figure()
fig.set_size_inches(13, 13)

correlation = selected_df.corr()
sns.heatmap(correlation, cmap="viridis", annot=True)
# Be careful. Do not use normalizer of sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

labels = selected_df["Survived"]
features = selected_df.drop(['Survived'], axis=1)

scaler.fit(features)
scaled_features = scaler.transform(features)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_features, labels, random_state=20180408, test_size=0.3)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 1. Model fitting
logistic_model = LogisticRegression()
lr_scores = cross_val_score(logistic_model, scaled_features, labels, cv=3)
lr_scores.mean()
from sklearn.ensemble import RandomForestClassifier

# 1. Model fitting
random_forest_model = RandomForestClassifier(max_depth = 5, n_estimators = 150)
rf_scores = cross_val_score(random_forest_model, scaled_features, labels, cv=3)
rf_scores.mean()
from sklearn.svm import SVC

# 1. Model fitting
svm_model = SVC()

svm_scores = cross_val_score(svm_model, scaled_features, labels, cv=3)
svm_scores.mean()
from sklearn.ensemble import GradientBoostingClassifier

gradient_boost_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=5)
gb_scores = cross_val_score(gradient_boost_model, scaled_features, labels, cv=3)

gb_scores.mean()
from sklearn.ensemble import VotingClassifier

en1 = VotingClassifier\
(estimators=[('lm', logistic_model), ('rfm', random_forest_model),\
             ('svm', svm_model), ('gb', gradient_boost_model)], voting='hard')

en_scores = cross_val_score(en1, scaled_features, labels, cv=3)
en_scores.mean()
# For neural network, make test set
test_temp_df = test_set[['Sex', 'Age_seg', 'Has_family', 'Family_size', 'Fare_seg', 'Embarked', \
                         'Age', 'Is_cabin', 'Title', 'Is_family_survived']]

categorical_features = ['Sex', 'Age_seg', 'Embarked']
test_selected_df = pd.get_dummies(test_temp_df, categorical_features)

scaler_test = StandardScaler()

scaler_test.fit(test_selected_df)
scaled_test_features = scaler_test.transform(test_selected_df)
# 단층 신경망
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_hidden_unit = 1024
num_hidden_unit_1 = 1024

num_of_inputs = len(x_train[0])

x = tf.placeholder(tf.float32, [None, num_of_inputs])
t = tf.placeholder(tf.float32, [None, 2])

# First hidden layer
# Activate function : Relu
w_1 = tf.get_variable('w_1', shape=[num_of_inputs, num_hidden_unit], \
                      initializer=tf.keras.initializers.he_normal(seed=None))

# Activate function : Sigmoid, Hyper tangent
# w_1 = tf.get_variable('w_1', shape=[num_of_inputs, num_hidden_unit], \
#                       initializer=tf.contrib.layers.xavier_initializer())

b_1 = tf.Variable(tf.zeros([num_hidden_unit]))
relu_hidden_layer_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)
# sigmoid_hidden_layer_1 = tf.nn.sigmoid(tf.matmul(x, w_1) + b_1)
#Second hidden layer
# Activate function : Relu
w_2 = tf.get_variable('w_2', shape=[num_hidden_unit, num_hidden_unit_1], \
                      initializer=tf.keras.initializers.he_normal(seed=None))

# Activate function : Sigmoid, Hyper tangent
# w_2 = tf.get_variable('w_2', shape=[num_hidden_unit, num_hidden_unit_1], \
#                       initializer=tf.contrib.layers.xavier_initializer())

b_2 = tf.Variable(tf.zeros([num_hidden_unit_1]))
relu_hidden_layer_2 = tf.nn.relu(tf.matmul(relu_hidden_layer_1, w_2) + b_2)
# sigmoid_hidden_layer_2 = tf.nn.sigmoid(tf.matmul(sigmoid_hidden_layer_1, w_2) + b_2)

w_0 = tf.Variable(tf.zeros([num_hidden_unit_1, 2]))
b_0 = tf.Variable(tf.zeros([2]))

logits = tf.matmul(relu_hidden_layer_2, w_0) + b_0
prob = tf.nn.softmax(logits=logits)

loss = -tf.reduce_sum(t*tf.log(prob))
train_step = tf.train.AdamOptimizer().minimize(loss)
num_of_correct = tf.equal(tf.argmax(prob, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(num_of_correct, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
batch_size = 30
cnt = 0
epoch = 5

x_batches = [x_train[i:i + batch_size] for i in range(0, len(x_train), batch_size)]
y_batches = [y_train[i:i + batch_size] for i in range(0, len(y_train), batch_size)]

y_test_onehot = sess.run(tf.one_hot(list(y_test), 2))

for _ in range(epoch):
    for x_batch, y_batch in zip(x_batches, y_batches):
        y_batch_onehot = sess.run(tf.one_hot(list(y_batch), 2))
        sess.run(train_step, feed_dict={x: x_batch, t: y_batch_onehot})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: x_test, t: y_test_onehot})
        cnt+=1
        print('Step: %d, Loss: %f, Accuracy: %f' %(cnt, loss_val, acc_val))

# prediction 
mlp_pred = sess.run(prob, feed_dict={x: scaled_test_features})

mlp_pred = sess.run(tf.argmax(mlp_pred, 1))
sess.close()
