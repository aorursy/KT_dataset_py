# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.model_selection
import sklearn.pipeline  
import sklearn.preprocessing
from subprocess import check_output
import seaborn as sns
import os
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

#display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));
# read data and have a first look at it
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.info()
print('_'*40)
test_df.info()
# missing values
print(train_df.isnull().sum())
print('')
print(test_df.isnull().sum())
# look at the first five rows
#train_df.head()
# look at the first five rows
#test_df.head() 
# describe numerical data
#train_df.describe()
# describe numerical data
#test_df.describe()
# describe object data
#train_df.describe(include=['O'])
# describe object data
#test_df.describe(include=['O'])
# check Pclass - Survived correlation
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# check Sex - Survived correlation
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# check SibSp - Survived correlation
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# check Parch - Survived correlation
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Age histograms depending on Survived
grid = sns.FacetGrid(train_df, col='Survived');
grid.map(plt.hist, 'Age', bins=20);
# Age histograms depending on Survived, Pclass
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
# Survived values depending on Embarked, Sex
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6);
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep');
grid.add_legend();
# Fare depending on Embarked, Survived, Sex
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# combine train and test data for manipulation
data_df = pd.concat((train_df, test_df)).reset_index(drop=True)
print('data_df.shape = ', data_df.shape)
# make categorial feature numerical
data_df['Sex'] = data_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
print(data_df.Embarked.isnull().values.sum(),'missing values')

# most frequent occurence of Embarked value
freq_port = data_df.Embarked.dropna().mode()[0]
print(freq_port,'= most frequent');

# replace na entries with most frequent value of Embarked
data_df['Embarked'] = data_df['Embarked'].fillna(freq_port)
    
data_df[['Embarked', 'Survived']].groupby(['Embarked'],
    as_index=False).mean().sort_values(by='Survived', ascending=False)
# complete missing age entries by using information on Sex, Pclass

# many missing values
print(data_df.Age.isnull().values.sum(), 'missing values')

guess_ages = np.zeros((2,3));

for i in range(0, 2):
    for j in range(0, 3):
        guess_df = data_df[(data_df['Sex'] == i) & 
                           (data_df['Pclass'] == j+1)]['Age'].dropna()

        
        age_mean = guess_df.mean()
        age_std = guess_df.std()
        age_guess = np.random.normal(age_mean, age_std)

        #age_guess = guess_df.median()
        #print(age_guess)

        guess_ages[i,j] = int((age_guess/0.5 + 0.5) * 0.5)


for i in range(0, 2):
    for j in range(0, 3):
        data_df.loc[(data_df.Age.isnull()) & (data_df.Sex == i) & 
                    (data_df.Pclass == j+1),'Age'] = guess_ages[i,j]

data_df['Age'] = data_df['Age'].astype(int)

print(data_df.Age.isnull().values.sum(), 'missing values')

#train_df.head()
# create new feature AgeBand
data_df['AgeBand'] = pd.qcut(data_df['Age'], 5)
data_df[['AgeBand', 'Survived']].groupby(['AgeBand'], 
            as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# Creat AgeBins with ordinals based on the bands in AgeBand
data_df.loc[data_df['Age'] <= 16, 'AgeBin'] = 0
data_df.loc[(data_df['Age'] > 16) & (data_df['Age'] <= 32), 'AgeBin'] = 1
data_df.loc[(data_df['Age'] > 32) & (data_df['Age'] <= 48), 'AgeBin'] = 2
data_df.loc[(data_df['Age'] > 48) & (data_df['Age'] <= 64), 'AgeBin'] = 3
data_df.loc[data_df['Age'] > 64, 'AgeBin']

# drop AgeBand
data_df = data_df.drop(['AgeBand'], axis=1)

print(data_df.AgeBin.isnull().values.sum(), 'missing values')
#data_df.head()
# complete feature Fare

print(data_df.Fare.isnull().values.sum(), 'missing values')
data_df['Fare'].fillna(data_df['Fare'].dropna().median(), inplace=True)
print(data_df.Fare.isnull().values.sum(), 'missing values')

# create feature FareBand
data_df['FareBand'] = pd.qcut(data_df['Fare'], 5)
data_df[['FareBand', 'Survived']].groupby(['FareBand'], 
        as_index=False).mean().sort_values(by='FareBand', ascending=True)
# create feature FareBin by ordinals based on FareBand
data_df.loc[ data_df['Fare'] <= 7.91, 'FareBin'] = 0
data_df.loc[(data_df['Fare'] > 7.91) & (data_df['Fare'] <= 14.454), 'FareBin'] = 1
data_df.loc[(data_df['Fare'] > 14.454) & (data_df['Fare'] <= 31), 'FareBin']   = 2
data_df.loc[ data_df['Fare'] > 31, 'FareBin'] = 3
data_df['FareBin'] = data_df['FareBin'].astype(int)

data_df = data_df.drop(['FareBand'], axis=1)
    
#train_df.head()
# extract title from Name and then create new feature: Title  
data_df['Title'] = data_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Survived'], data_df['Title'])
# correlation of Sex and Title
pd.crosstab(data_df['Sex'], data_df['Title'])
# reduce the number of titles
data_df['Title'] = data_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                             'Jonkheer', 'Dona'], 'Rare')
data_df['Title'] = data_df['Title'].replace('Mlle', 'Miss')
data_df['Title'] = data_df['Title'].replace('Ms', 'Miss')
data_df['Title'] = data_df['Title'].replace('Mme', 'Mrs')

# no missing titles
print('missing titles = ', data_df.Title.isnull().sum())

#data_df.head()
# create new feature FamilySize
data_df['FamilySize'] = data_df['SibSp'] + data_df['Parch'] + 1
data_df[['FamilySize', 'Survived']].groupby(['FamilySize'],
    as_index=False).mean().sort_values(by='Survived', ascending=False)
# create new feature IsAlone
data_df['IsAlone'] = 0
data_df.loc[data_df['FamilySize'] == 1, 'IsAlone'] = 1
data_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# create new feature Age*Pclass
data_df['AgeBin*Pclass'] = data_df.AgeBin * data_df.Pclass
data_df[['AgeBin*Pclass', 'Survived']].groupby(['AgeBin*Pclass'], as_index=False).mean()
# create feature: length of the name
data_df['NameLength'] = data_df['Name'].apply(len)
#data_df[['NameLength', 'Survived']].groupby(['NameLength'], as_index=False).mean()
# create feature that tells whether a passenger had a cabin on the Titanic
data_df['HasCabin'] = data_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
data_df[['HasCabin', 'Survived']].groupby(['HasCabin'], as_index=False).mean()
## drop features
if True:
    drop_ft = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'AgeBin', 'AgeBin*Pclass',
               'FareBin']
    data_df = data_df.drop(drop_ft, axis=1)
    print("data_df = ", data_df.shape)
## use LabelEncoding or dummy variables on all categorial features

cols = data_df.select_dtypes(exclude = [np.number]).columns.values
print('numerical columns:', data_df.select_dtypes(include = 
                                                  [np.number]).columns.values.shape[0])
print('categorial columns:', cols.shape[0])

if True:
    # create dummy variables
    data_df = pd.get_dummies(data_df).copy()
else:
    # create one-hot encodings
    for c in cols:
        lbl = sklearn.preprocessing.LabelEncoder() 
        lbl.fit(list(data_df[c].values)) 
        data_df[c] = lbl.transform(list(data_df[c].values))

# final shape        
print('data_df.shape = ', data_df.shape)
# check data_df
print(data_df.isnull().sum())
data_df.head()
# pearson correlation of features
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features for Train Set', y=1.05, size=15)
sns.heatmap(data_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
## create training, validation, testing sets

# function to normalize data
def normalize_data(data): 
    # scale features using statistics that are robust to outliers
    rs = sklearn.preprocessing.RobustScaler()
    rs.fit(data)
    data = rs.transform(data)
    #x_train_valid = (x_train_valid)/(x_train_valid.max(axis=0));
    #x_test = (x_test)/(x_test.max(axis=0));
    return data

# get accuracy from classes
def get_accuracy(y_target, y_pred):
    y_target_class = get_classes(y_target).reshape(-1,)
    y_pred = get_classes(y_pred).reshape(-1,)
    return np.mean(y_target_class == y_pred)

# get classes from probabilities
def get_classes(y_proba):
    return np.greater(y_proba, 0.5).astype(np.int) 

# create train/validation and test sets as copies from dataframes
x_train_valid = data_df.drop(['Survived'],axis=1)[:train_df.shape[0]].copy().values
y_train_valid = data_df['Survived'][:train_df.shape[0]].copy().values.reshape(-1,)
x_test  = data_df.drop(['Survived'],axis=1)[train_df.shape[0]:].copy().values

# store used features
features_df = pd.DataFrame(data_df.drop(["Survived"], axis=1).columns)
features_df.columns = ['Features']
print(features_df)
print('')

# normalize train, validation, test data
x_train_valid = normalize_data(x_train_valid)
x_test = normalize_data(x_test)
          
print('x_train_valid.shape = ', x_train_valid.shape)
print('y_train_valid.shape = ', y_train_valid.shape)
print('x_test.shape = ', x_test.shape)
## neural network implementation

# parameters for batch function
perm_array_train = np.array([])
index_in_epoch = 0 

# function to get the next mini batch
def next_batch(batch_size, x_train, y_train):
    global index_in_epoch, perm_array_train
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if not len(perm_array_train) == len(x_train):
        perm_array_train = np.arange(len(x_train))
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array_train) # shuffle data
        start = 0 # start next epoch
        index_in_epoch = batch_size # set index to batch size
                
    end = index_in_epoch
    
    x_tr  = x_train[perm_array_train[start:end]]
    y_tr  = y_train[perm_array_train[start:end]].reshape(-1,1)
     
    return x_tr, y_tr


# function to create the graph
def create_nn_graph(num_input_features = 10, num_output_features = 1):

    # reset default graph
    tf.reset_default_graph()

    # parameters of NN architecture
    x_size = num_input_features # number of features
    y_size = num_output_features # output size
    n_n_fc1 = 1024; # number of neurons of first layer
    n_n_fc2 = 1024; # number of neurons of second layer

    # variables for input and output 
    x_data = tf.placeholder('float', shape=[None, x_size])
    y_data = tf.placeholder('float', shape=[None, y_size])

    # 1.layer: fully connected
    W_fc1 = tf.Variable(tf.truncated_normal(shape = [x_size, n_n_fc1], stddev = 0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [n_n_fc1]))  
    h_fc1 = tf.nn.relu(tf.matmul(x_data, W_fc1) + b_fc1)

    # dropout
    tf_keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, tf_keep_prob)

    # 2.layer: fully connected
    W_fc2 = tf.Variable(tf.truncated_normal(shape = [n_n_fc1, n_n_fc2], stddev = 0.1)) 
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [n_n_fc2]))  
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) 

    # dropout
    h_fc2_drop = tf.nn.dropout(h_fc2, tf_keep_prob)

    # 3.layer: fully connected
    W_fc3 = tf.Variable(tf.truncated_normal(shape = [n_n_fc2, y_size], stddev = 0.1)) 
    b_fc3 = tf.Variable(tf.constant(0.1, shape = [y_size]))  
    z_pred = tf.matmul(h_fc2_drop, W_fc3) + b_fc3  

    # cost function
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_data, logits=z_pred));

    # optimisation function
    tf_learn_rate = tf.placeholder(dtype='float', name="tf_learn_rate")
    train_step = tf.train.AdamOptimizer(tf_learn_rate).minimize(cross_entropy)

    # evaluation
    y_pred_proba = tf.cast(tf.nn.sigmoid(z_pred),dtype = tf.float32);
    y_pred_class = tf.cast(tf.greater(y_pred_proba, 0.5),'float')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_data ), 'float'))
 
    # tf tensors
    tf_tensors = {'train_step': train_step, 'cross_entropy': cross_entropy, 
                  'y_pred_proba': y_pred_proba,'accuracy': accuracy,
                  'tf_keep_prob': tf_keep_prob, 'tf_learn_rate': tf_learn_rate,
                  'x_data': x_data, 'y_data': y_data}
    
    return tf_tensors


# function to train the graph
def train_nn_graph(tf_tensors, x_train, y_train, x_valid, y_valid, verbose = False):

    # tf tensors
    train_step = tf_tensors['train_step']
    cross_entropy = tf_tensors['cross_entropy']
    y_pred_proba = tf_tensors['y_pred_proba']
    accuracy = tf_tensors['accuracy']
    tf_keep_prob = tf_tensors['tf_keep_prob']
    tf_learn_rate = tf_tensors['tf_learn_rate']
    x_data = tf_tensors['x_data']
    y_data = tf_tensors['y_data']

    # parameters
    keep_prob = 0.5; # dropout regularization with keeping probability
    learn_rate_range = [0.01,0.005,0.0025,0.001,0.001,0.001,0.00075,0.0005,0.00025,0.0001,
                       0.0001,0.0001,0.0001];
    learn_rate_step = 10 # in terms of epochs 
    batch_size = 10 # batch size
    n_epoch = 10 # number of epochs
    cv_num = 10 # number of cross validations
    n_step = -1;
        
    # start TensorFlow session and initialize global variables
    sess = tf.InteractiveSession() 
    sess.run(tf.global_variables_initializer())  
    
    # training model
    for i in range(int(n_epoch*x_train.shape[0]/batch_size)):

        if i%int(learn_rate_step*x_train.shape[0]/batch_size) == 0:
            n_step += 1;
            learn_rate = learn_rate_range[n_step];
            if verbose:
                print('set learnrate = ', learn_rate)

        # get next batch
        x_batch, y_batch = next_batch(batch_size, x_train, y_train)

        sess.run(train_step, feed_dict={x_data: x_batch, y_data: y_batch, 
                                        tf_keep_prob: keep_prob, tf_learn_rate: learn_rate})

        if verbose and i%int(1.*x_train.shape[0]/batch_size) == 0:
            train_loss = sess.run(cross_entropy,feed_dict={x_data: x_train, 
                                                           y_data: y_train, 
                                                           tf_keep_prob: 1.0})

            train_acc = accuracy.eval(feed_dict={x_data: x_train, 
                                                 y_data: y_train, 
                                                 tf_keep_prob: 1.0})    

            valid_loss = sess.run(cross_entropy,feed_dict={x_data: x_valid, 
                                                           y_data: y_valid, 
                                                           tf_keep_prob: 1.0})

            valid_acc = accuracy.eval(feed_dict={x_data: x_valid, 
                                                 y_data: y_valid, 
                                                 tf_keep_prob: 1.0})      

            print('%.2f epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(
                (i+1)*batch_size/x_train.shape[0], train_loss, valid_loss, train_acc, 
                valid_acc))

    
    # predictions
    y_train_pred_proba = y_pred_proba.eval(feed_dict={x_data: x_train,
                                                      tf_keep_prob: 1.0}).flatten()
    y_valid_pred_proba = y_pred_proba.eval(feed_dict={x_data: x_valid, 
                                                      tf_keep_prob: 1.0}).flatten()
    y_test_pred_proba = y_pred_proba.eval(feed_dict={x_data: x_test, 
                                                     tf_keep_prob: 1.0}).flatten()
    
    sess.close();
    

    return (y_train_pred_proba, y_valid_pred_proba, y_test_pred_proba)
# check neural network

if False:
    
    # store results
    y_train_pred_proba = {}
    y_valid_pred_proba = {}
    train_acc = {}
    valid_acc = {}
    
    # create graph and receive tf tensors
    tf_tensors = create_nn_graph(x_train_valid.shape[1], 1)

    # cross validations
    cv_num = 10
    kfold = sklearn.model_selection.KFold(cv_num, shuffle=True)

    for train_index, valid_index in kfold.split(x_train_valid):

        x_train = x_train_valid[train_index]
        y_train = y_train_valid[train_index]
        x_valid = x_train_valid[valid_index]
        y_valid = y_train_valid[valid_index]

        # train NN
        (y_train_pred_proba['nn'], 
         y_valid_pred_proba['nn'],
         y_test_pred['nn']) = train_nn_graph(tf_tensors, x_train,  
                                             y_train.reshape(-1,1), x_valid, 
                                             y_valid.reshape(-1,1), False)
        
        # compute accuracy
        train_acc['nn'] = get_accuracy(y_train_pred_proba['nn'], y_train)
        valid_acc['nn'] = get_accuracy(y_valid_pred_proba['nn'], y_valid)
        
        # loss
        print('nn: train/valid accuracy = %.4f/%.4f'%(train_acc['nn'], valid_acc['nn']))
## base models

# base models
logreg = sklearn.linear_model.LogisticRegression()
extra_trees = sklearn.ensemble.ExtraTreesClassifier(max_depth = 4,n_estimators=10)
gradient_boost = sklearn.ensemble.GradientBoostingClassifier(max_depth = 4,n_estimators=10)
random_forest = sklearn.ensemble.RandomForestClassifier(max_depth = 4,n_estimators=10)
decision_tree = sklearn.tree.DecisionTreeClassifier(max_depth = 4)
gaussianNB = sklearn.naive_bayes.GaussianNB()
ada_boost = sklearn.ensemble.AdaBoostClassifier(n_estimators=10)

# store models in dictionary
base_models = {'logreg': logreg, 'extra_trees': extra_trees, 'ada_boost': ada_boost, 
               'gradient_boost': gradient_boost, 'random_forest': random_forest, 
               'decision_tree': decision_tree, 'gaussianNB': gaussianNB}

# choose models for out-of-folds predictions
take_models = ['logreg', 'extra_trees', 'gradient_boost', 'gaussianNB', 
               'random_forest', 'decision_tree', 'nn', 'ada_boost']

#take_models = ['adaboost']

# train data for meta model
train_acc = {}
valid_acc = {}
y_test_pred_proba = {}
y_train_pred_proba = {}
y_valid_pred_proba = {}

for mn in take_models:
    train_acc[mn] = 0
    valid_acc[mn] = 0
    y_test_pred_proba[mn] = 0

# cross validations
cv_num = 10
kfold = sklearn.model_selection.KFold(cv_num, shuffle=True)

print('Training {} models with {}-fold cross validation'.format(len(take_models), cv_num))
print('')

# make out-of-folds predictions from base models
for i,(train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
    
    print('{}. training of models in progress'.format(i+1))
    x_train = x_train_valid[train_index]
    y_train = y_train_valid[train_index]
    x_valid = x_train_valid[valid_index]
    y_valid = y_train_valid[valid_index]

    for mn in take_models:
        
        if mn == 'nn':
            # create graph and receive tf tensors
            tf_tensors = create_nn_graph(x_train_valid.shape[1], 1)
            
            # train neural network
            params = train_nn_graph(tf_tensors, x_train, y_train.reshape(-1,1),
                                    x_valid, y_valid.reshape(-1,1), False) 
     
            # save results
            train_acc['nn'] += get_accuracy(params[0], y_train)
            valid_acc['nn'] += get_accuracy(params[1], y_valid)
            y_test_pred_proba['nn'] += params[2]
        
        else:
            # create cloned model from base models
            model = sklearn.base.clone(base_models[mn])
            model.fit(x_train, y_train)
            
            # save results
            train_acc[mn] += get_accuracy(model.predict_proba(x_train)[:,1], y_train)
            valid_acc[mn] += get_accuracy(model.predict_proba(x_valid)[:,1], y_valid)
            y_test_pred_proba[mn] += model.predict_proba(x_test)[:,1]

print('')

# store and print results
for mn in take_models:
    
    train_acc[mn] /= cv_num
    valid_acc[mn] /= cv_num
    y_test_pred_proba[mn] /= cv_num
        
    print(mn,'train/valid accuracy = %.3f/%.3f'%(train_acc[mn], valid_acc[mn]))

# average rmse over the following models
take_model_avg = ['logreg', 'random_forest', 'decision_tree']

train_acc['majority_vote'] = 0
valid_acc['majority_vote'] = 0
y_test_pred_proba['majority_vote'] = 0

for mn in take_model_avg:
    train_acc['majority_vote'] += train_acc[mn] / len(take_model_avg)
    valid_acc['majority_vote'] += valid_acc[mn] / len(take_model_avg)
    y_test_pred_proba['majority_vote'] += y_test_pred_proba[mn] / len(take_model_avg)
    
print('')
print('Average models:', take_model_avg)
print('Soft majority vote: train/valid accuracy = %.3f/%.3f'%(train_acc['majority_vote'], 
                                                              valid_acc['majority_vote']))

# indices of features
features_df.Features
## Look at feature importances using trees

for mn in ['random_forest', 'extra_trees']:
    
    model = sklearn.base.clone(base_models[mn])
    model.fit(x_train_valid, y_train_valid)
        
    ft_importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(ft_importances)[::-1]

    # plot feature importances 
    plt.figure(figsize=(20,10))
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), ft_importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(x_train.shape[1]), features_df.Features.loc[indices].values)
    plt.xlim([-1, x_train.shape[1]])
    plt.show()
## summarize the results

models_summary = pd.DataFrame({'Model':[],'Train Acc':[],'Valid Acc':[]})

for i,mn in enumerate(train_acc.keys()):
    models_summary.loc[i,'Model'] = mn
    models_summary.loc[i,'Train Acc'] = train_acc[mn]
    models_summary.loc[i,'Valid Acc'] = valid_acc[mn]
    
models_summary.sort_values(by='Valid Acc', ascending=False)
## correlation map of test predictions of the base models

y_test_pred_df = pd.DataFrame({})
for key in y_test_pred_proba.keys():
    y_test_pred_df[key] = get_classes(y_test_pred_proba[key])

corrmat = y_test_pred_df.corr()
corrmat
plt.subplots(figsize=(10,5))
plt.title('Correlation of Test Predictions')
sns.heatmap(corrmat, vmax=1, square=True)
y_test_pred_df.describe()
## Ensemble learning

# choose base models for ensemble learning
#take_models = ['logreg','decision_tree','gradient_boost','extra_trees','nn',
#               'random_forest','gaussianNB','ada_boost']

take_models = ['logreg', 'random_forest', 'gradient_boost', 'ada_boost', 'nn']

# cross validations
cv_num = 10
kfold = sklearn.model_selection.KFold(cv_num, shuffle=True)

print('Ensemble learning with', len(take_models),'models')
print('Pasting with', cv_num, 'out-of-fold predictions')
print('')

# make out-of-folds predictions from base models
for i, (train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
    
    # train and validation sets
    x_train = x_train_valid[train_index]
    y_train = y_train_valid[train_index]
    x_valid = x_train_valid[valid_index]
    y_valid = y_train_valid[valid_index]
    
    for mn in take_models:
        
        if mn == 'nn':
            
            # create graph and receive tf tensors
            tf_tensors = create_nn_graph(x_train_valid.shape[1],1)

            # train neural network
            params = train_nn_graph(tf_tensors, x_train, y_train.reshape(-1,1),
                                    x_valid, y_valid.reshape(-1,1), False) 

            # save predictions
            y_train_pred_proba[mn,i] = params[0].reshape(-1,)
            y_valid_pred_proba[mn,i] = params[1].reshape(-1,) 
            y_test_pred_proba[mn,i] = params[2].reshape(-1,)

        else:
            
            # create cloned model from base models
            model = sklearn.base.clone(base_models[mn])
            model.fit(x_train, y_train)

            # save predictions
            y_train_pred_proba[mn,i] = model.predict_proba(x_train)[:,1]
            y_valid_pred_proba[mn,i] = model.predict_proba(x_valid)[:,1]
            y_test_pred_proba[mn,i] = model.predict_proba(x_test)[:,1]

        print(i, mn,': train/valid accuracy = %.3f/%.3f'%(
            get_accuracy(y_train_pred_proba[mn,i], y_train),
            get_accuracy(y_valid_pred_proba[mn,i], y_valid)))
    
    # training targets for the meta model
    # - y_train_meta = all instances appearing in y_valid for each fold
    if i==0: 
        y_train_meta = y_valid
    else:
        y_train_meta = np.concatenate([y_train_meta, y_valid]) 

if False:
    # 1. possibility to create training and test set for meta model
    # - x_train_meta = concatenate the average of the base 
    #                  model predictions for each of the 10 out-of-fold predictions
    # - x_test_meta = average test set predictions of all models
    for i in range(cv_num):
        tmp = 0
        for mn in take_models:
            tmp += y_valid_pred_proba[mn,i] / len(take_models)
            x_test_meta += y_test_pred_proba[mn,i] / (len(take_models)*cv_num)

        x_train_meta = np.concatenate([x_train_meta, tmp])
    
if True:
    # 2. possibility to create training and test set for meta model
    # - x_train_meta = concatenate the base model predictions
    # - x_test_meta = average test set predictions for each model
    for i in range(cv_num):
        
        for j,mn in enumerate(take_models):
            if j==0:
                tmp1 = y_valid_pred_proba[mn,i].reshape(-1,1)
                tmp2 = y_test_pred_proba[mn,i].reshape(-1,1)
            else:
                tmp1 = np.concatenate([tmp1, y_valid_pred_proba[mn,i].reshape(-1,1)], axis=1)
                tmp2 = np.concatenate([tmp2, y_test_pred_proba[mn,i].reshape(-1,1)], axis=1)

        if i==0:
            x_train_meta = tmp1
            x_test_meta = tmp2 / cv_num 
        else:
            x_train_meta = np.concatenate([x_train_meta, tmp1], axis=0)
            x_test_meta += tmp2 / cv_num 

# Ensemble method: soft majority vote = averaged test set predictions over all models
print('')
print('Ensemble method: soft majority vote of all models')

y_valid_pred_proba['majority_vote'] = np.mean(x_train_meta, axis=1)
y_test_pred_proba['majority_vote'] = x_test_meta

print('Soft majority vote: valid accuracy = %.3f'%get_accuracy(y_train_meta,
                                            y_valid_pred_proba['majority_vote']))
## Train the meta model

# choose meta model
take_meta_model = 'logreg'

# Ensemble method: stacked generalisation
print('')
print('Ensemble method: stacking of base models using the meta-model', take_meta_model)
print('x_train_meta.shape =', x_train_meta.shape)
print('y_train_meta.shape =', y_train_meta.shape)
print('x_test_meta.shape =', x_test_meta.shape)

model = sklearn.base.clone(base_models[take_meta_model]) 
model.fit(x_train_meta, y_train_meta)
y_train_pred_proba['meta_model'] = model.predict_proba(x_train_meta)[:,1]
y_test_pred_proba['meta_model'] = model.predict_proba(x_test_meta)[:,1]

print('Meta model: train accuracy = %.3f'%get_accuracy(y_train_meta, 
                                                       y_train_pred_proba['meta_model']))
# choose prediction
y_test_submit = get_classes(y_test_pred_proba['meta_model'])
y_test_submit.mean()
# submit prediction
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_test_submit
    })

submission.to_csv('submission.csv', index=False)
submission.head()