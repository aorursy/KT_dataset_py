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
# Load the raw training data

df_raw_train = pd.read_csv('../input/titanic/train.csv', header=0)

# Make a copy of df_raw_train

df_train = df_raw_train.copy(deep=True)



# Load the raw testing data

df_raw_test = pd.read_csv('../input/titanic/test.csv' , header=0)



# Make a copy of df_raw_test

df_test = df_raw_test.copy(deep=True)
# Print the dimension of df_train

pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
pd.DataFrame([[df_test.shape[0],df_test.shape[1]]],columns = ['#rows','# colums'])
df_train.head()
df_test.head()
target = 'Survived'
from sklearn.model_selection import train_test_split

df_train , df_valid = train_test_split(df_train , train_size = 0.8 ,random_state = 42 , stratify = df_train[target])



df_train , df_valid = df_train.reset_index(drop = True) ,df_valid.reset_index(drop=True)

pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])
df = pd.concat([df_train,df_valid, df_test] , sort = False)
pd.DataFrame([[df.shape[0], df.shape[1]]], columns=['# rows', '# columns'])
def id_checker(df):

    """

    The identifier checker



    Parameters

    ----------

    df : dataframe

    

    Returns

    ----------

    The dataframe of identifiers

    """

    

    # Get the identifiers

    df_id = df[[var for var in df.columns 

                if df[var].nunique(dropna=True) == df[var].notnull().sum()]]

                

    return df_id
# Call id_checker on df

df_id = id_checker(df)



# Print the first 5 rows of df_id

df_id.head()
import numpy as np



# Remove the identifiers from df_train

df_train = df_train.drop(columns=np.intersect1d(df_id.columns, df_train.columns))



# Remove the identifiers from df_valid

df_valid = df_valid.drop(columns=np.intersect1d(df_id.columns, df_valid.columns))



# Remove the identifiers from df_test

df_test = df_test.drop(columns=np.intersect1d(df_id.columns, df_test.columns))
# Combine df_train, df_valid and df_test

df = pd.concat([df_train, df_valid, df_test], sort=False)
def nan_checker(df):

    """

    The NaN checker



    Parameters

    ----------

    df : dataframe

    

    Returns

    ----------

    The dataframe of variables with NaN, their proportion of NaN and dtype

    """

    

    # Get the variables with NaN, their proportion of NaN and dtype

    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]

                           for var in df.columns if df[var].isna().sum() > 0],

                          columns=['var', 'proportion', 'dtype'])

    

    # Sort df_nan in accending order of the proportion of NaN

    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    

    return df_nan

                          

    
# Call nan_checker on df

df_nan = nan_checker(df)



# Print df_nan

df_nan
# Print the unique dtype of the variables with NaN

pd.DataFrame(df_nan['dtype'].unique(), columns=['dtype'])
# Get the variables with missing values, their proportion of missing values and dtype

df_miss = df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)



# Print df_miss

df_miss
# Remove rows with missing values from df_train

df_remove_train = df_train.dropna(subset=np.intersect1d(df_miss['var'], 

                                                        df_train.columns),

                                 inplace=False)



# Remove rows with missing values from df_valid

df_remove_valid = df_valid.dropna(subset=np.intersect1d(df_miss['var'], 

                                                        df_valid.columns),

                                 inplace=False)



# Remove rows with missing values from df_test

df_remove_test = df_test.dropna(subset=np.intersect1d(df_miss['var'], 

                                                      df_test.columns),

                               inplace=False)
# Print the dimension of df_remove_train

pd.DataFrame([[df_remove_train.shape[0], df_remove_train.shape[1]]], columns=['# rows', '# columns'])
# Print the dimension of df_remove_valid

pd.DataFrame([[df_remove_valid.shape[0], df_remove_valid.shape[1]]], columns=['# rows', '# columns'])
# Print the dimension of df_remove_test

pd.DataFrame([[df_remove_test.shape[0], df_remove_test.shape[1]]], columns=['# rows', '# columns'])
from sklearn.impute import SimpleImputer

si = SimpleImputer(missing_values=np.nan , strategy = 'mean')



from sklearn.impute import SimpleImputer



# The SimpleImputer

si = SimpleImputer(missing_values=np.nan, strategy='mean')



# Make a copy of df_train, df_valid and df_test

df_impute_train = df.iloc[:df_train.shape[0], :].copy(deep=True)

df_impute_valid = df.iloc[df_train.shape[0]:df_train.shape[0] + df_valid.shape[0], :].copy(deep=True)

df_impute_test = df.iloc[df_train.shape[0] + df_valid.shape[0]:, :].copy(deep=True)



# Impute the variables with missing values in df_impute_train, df_impute_valid and df_impute_test 

df_impute_train[df_miss['var']] = si.fit_transform(df_impute_train[df_miss['var']])

df_impute_valid[df_miss['var']] = si.transform(df_impute_valid[df_miss['var']])

df_impute_test[df_miss['var']] = si.transform(df_impute_test[df_miss['var']])
# Print the first 10 rows of df_impute_train

df_impute_train.head(10)
# Combine df_impute_train, df_impute_valid and df_impute_test

df = pd.concat([df_impute_train, df_impute_valid, df_impute_test], sort=False)



# Print the unique dtype of variables in df

pd.DataFrame(df.dtypes.unique(), columns=['dtype'])
def cat_var_checker(df):

    """

    The categorical variable checker



    Parameters

    ----------

    df: the dataframe

    

    Returns

    ----------

    The dataframe of categorical variables and their number of unique value

    """

    

    # Get the dataframe of categorical variables and their number of unique value

    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]

                           for var in df.columns if df[var].dtype == 'object'],

                          columns=['var', 'nunique'])

    

    # Sort df_cat in accending order of the number of unique value

    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    

    return df_cat
# Call cat_var_checker on df

df_cat = cat_var_checker(df)



# Print the dataframe

df_cat
#drop names

df = df.drop('Name', 1)

df = df.drop('Ticket', 1)

# Since 77 percent of values are missing we will drop cabin

df = df.drop('Cabin',1)



df.head()
df_cat

df_cat = df_cat.drop(df.index[0:3])
df_cat
# One-hot-encode the categorical features in the combined data

df = pd.get_dummies(df, columns=np.setdiff1d(df_cat['var'], [target]))



# Print the first 5 rows of df

df.head()
from sklearn.preprocessing import LabelEncoder



# The LabelEncoder

le = LabelEncoder()



# Encode the categorical target in the combined data

df[target] = le.fit_transform(df[target])



# Print the first 5 rows of df

df.head()
df.head()
# Separating the training data

df_train = df.iloc[:df_impute_train.shape[0], :]



# Separating the validation data

df_valid = df.iloc[df_impute_train.shape[0]:df_impute_train.shape[0] + df_impute_valid.shape[0], :]



# Separating the testing data

df_test = df.iloc[df_impute_train.shape[0] + df_impute_valid.shape[0]:, :]
# Print the dimension of df_remove_train

pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])
pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])
df_test.head()
df_test = df_test.drop('Survived', 1)

df_test.head()
df_valid.head()
#normalize



features = np.setdiff1d(df_train.columns, [target])
from sklearn.preprocessing import MinMaxScaler



# The MinMaxScaler

mms = MinMaxScaler()



# Normalize the training data

df_train[features] = mms.fit_transform(df_train[features])



# Normalize the validation data

df_valid[features] = mms.transform(df_valid[features])



# Normalize the testing data

df_test[features] = mms.transform(df_test[features])
# Call nan_checker on df

df_train.describe()
# Get the feature matrix

X_train = df_train[features].to_numpy()

X_valid = df_valid[features].to_numpy()

X_test = df_test[features].to_numpy()



# Get the target vector

y_train = df_train[target].astype(df_raw_train[target].dtype).to_numpy()

y_valid = df_valid[target].astype(df_raw_train[target].dtype).to_numpy()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(class_weight = 'balanced' ,random_state =42)



# Train the logistic regression model on the training data

lr.fit(X_train, y_train)
# Get the prediction on the validation data using lr

y_valid_pred = lr.predict(X_valid)
from sklearn.metrics import accuracy_score



# Get the accuracy

accuracy = accuracy_score(y_valid, y_valid_pred)



# Print the accuracy

pd.DataFrame([accuracy], columns=['accuracy'])
from sklearn.metrics import precision_recall_fscore_support



# Get the precision, recall, f-score and support

precision, recall, fscore, support = precision_recall_fscore_support(y_valid, y_valid_pred, average='micro')



# Print the precision, recall and f-score

pd.DataFrame([[precision, recall, fscore]], columns=['precision', 'recall', 'f-score'])
import os



# Make directory

directory = os.path.dirname('./figure/')

if not os.path.exists(directory):

    os.makedirs(directory)
import matplotlib.pyplot as plt

%matplotlib inline 



# Get the sizes

plt.rc('font', size=20)

plt.rc('axes', titlesize=20)

plt.rc('axes', labelsize=20)

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

plt.rc('legend', fontsize=20)

plt.rc('figure', titlesize=20)
from sklearn.metrics import plot_roc_curve



# For each unique class

for class_ in np.unique(y_train):

    # The LogisticRegression

    lr_class = LogisticRegression()

    

    # Train the logistic regression model on the training data

    lr_class.fit(X_train, (y_train == class_).astype(int))

    

    # Create a figure

    fig = plt.figure(figsize=(5, 4))



    # Create the axes

    ax = plt.axes()

    

    # Set title

    ax.set_title('class ' + str(class_))

    

    # Plot the ROC

    plot_roc_curve(lr_class, X_valid, (y_valid == class_).astype(int), ax=ax, name='', linewidth=2, color='green')



    # Save and show the figure

    plt.tight_layout()

    plt.savefig('./figure/roc' + '_' + str(class_) + '.pdf')

    plt.show()
from sklearn.metrics import plot_confusion_matrix



# Create a figure and axes

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))



# Plot the confusion matrix

plot_confusion_matrix(lr,

                      X_valid,

                      y_valid,

                      normalize='true',

                      display_labels=np.unique(y_valid),

                      values_format='.2f',

                      cmap=plt.cm.Blues,

                      ax=ax)



# Save and show the figure

plt.tight_layout()

plt.savefig('./figure/confusion_matrix.pdf')

plt.show()
from sklearn.base import BaseEstimator, ClassifierMixin



class LogisticRegression_BGD(BaseEstimator, ClassifierMixin):

    """The logistic regression model using batch gradient descent"""

    

    def __init__(self,

                 penalty='l2',

                 tol=0.0001,

                 C=1,

                 random_state=42,

                 max_iter=100, 

                 l1_ratio=0,

                 eta=0.01):

        

        # The type of regularization

        self.penalty = penalty

        

        # The tolerance for stopping criteria.

        self.tol = tol

        

        # The inverse of regularization strength

        self.C = C

        

        # The random state

        self.random_state = random_state

        

        # The maximum number of iterations

        self.max_iter = max_iter

        

        # The ElasticNet mixing parameter

        self.l1_ratio = l1_ratio

        

        # The learning rate

        self.eta = eta



    def fit(self, X, y):

        """

        The fit function

        

        Parameters

        ----------

        X : the feature matrix

        y : the target vector

        """

        

        # Get the unique classes of the target

        self.classes = np.unique(y)

        

        # The feature matrix with x0 (the dummy feature)

        Z = np.hstack((np.ones((X.shape[0], 1)), X))

        

        # The one-hot-encoded target matrix

        Y = pd.get_dummies(y).to_numpy()

        

        # The random number generator

        self.rgen = np.random.RandomState(seed=self.random_state)

        

        # Initialize the weight matrix

        self.w = self.rgen.normal(loc=0.0, scale=0.01, size=(Z.shape[1], len(self.classes)))

        

        # Initialize the costs

        self.costs = []



        # For each iteration

        for i in range(self.max_iter):

            # Get the net_input (the predicted value of the target)

            net_input = self.net_input(Z)

            

            # Get the activation (softmax)

            activation = self.activation(net_input)

            

            # Get the error

            error = Y - activation

                                    

            # Get the mean squared error (mse)

            mse = (error ** 2).sum() / Z.shape[0]

                                    

            # Update the weight using batch gradient descent

            self.w += self.eta * Z.T.dot(error) / Z.shape[0]

            

            # Get the weight matrix for regularization

            w_reg = np.append(np.zeros((1, self.w.shape[1])), self.w[1:], axis=0)

            

            # Update the weight using regulariztion

            if self.penalty == 'l1':

                self.w -= self.eta * np.sign(w_reg) / self.C

            elif self.penalty == 'l2':

                self.w -= self.eta * w_reg / self.C

            elif self.penalty == 'elasticnet':

                self.w -= self.eta * (self.l1_ratio * np.sign(w_reg) 

                                      + (1 - self.l1_ratio) * w_reg) / self.C                

                           

            # Update the costs

            self.costs.append(mse)

            

            # The stopping criteria

            if i > 0 and abs(self.costs[i] - self.costs[i - 1]) < self.tol:

                break



    def net_input(self, Z):

        """

        Get the net input

        

        Parameters

        ----------

        Z : The feature matrix with x0 (the dummy feature)

        

        Returns

        ----------

        The net input (the predicted value of the target)

        """

        

        return Z.dot(self.w)

    

    def activation(self, net_input):

        """

        Get the activation (softmax)

        

        Parameters

        ----------

        net_input : the net input

        

        Returns

        ----------

        The activation

       

        """

        

        # Get the exponent of the net input (using the minum max trick)

        net_input_exp = np.exp(net_input - np.max(net_input, axis=1).reshape(-1, 1))

        

        return net_input_exp / np.sum(net_input_exp, axis=1).reshape(-1, 1)

    

    def predict_proba(self, X):

        """

        The predict probability function

        

        Parameters

        ----------

        X : the feature matrix

        

        Returns

        ----------

        The predicted probability of each class (i.e., the activation)

        """

        

        # The feature matrix with x0 (the dummy feature)

        Z = np.hstack((np.ones((X.shape[0], 1)), X))

            

        # Get the net_input (the predicted value of the target)

        net_input = self.net_input(Z)



        return self.activation(net_input)

    

    def predict(self, X):

        """

        The predict class function

        

        Parameters

        ----------

        X : the feature matrix

        

        Returns

        ----------

        The predicted class of the target

        """

    

        return np.argmax(self.predict_proba(X), axis=1)
# The LinearRegression_BGD

lr_bgd = LogisticRegression_BGD(C=100, eta=1)



# Train the logistic regression model on the training data

lr_bgd.fit(X_train, y_train)
# The line plot

plt.plot(range(len(lr_bgd.costs)), lr_bgd.costs, color='red', lw=3, alpha=0.6)  



# Set x-axis

plt.xlabel('Epoch', fontsize=20)

plt.xticks(fontsize=20)



# Set y-axis

plt.ylabel('Cost', fontsize=20)

plt.yticks(fontsize=20)



# Save and show the figure

plt.tight_layout()

plt.savefig('./figure/lr_bgd_cost.pdf')

plt.show()
# Get the prediction on the validation data using lr

y_valid_pred = lr_bgd.predict(X_valid)



# Get the accuracy

accuracy = accuracy_score(y_valid, y_valid_pred)



# Print the accuracy

pd.DataFrame([accuracy], columns=['accuracy'])
from sklearn.base import BaseEstimator, ClassifierMixin



class LogisticRegression_Newton(BaseEstimator, ClassifierMixin):

    """The logistic regression model using Newton's method"""

    

    def __init__(self, random_state=42):

        # The random state

        self.random_state = random_state



    def fit(self, X, y):

        """

        The fit function

        

        Parameters

        ----------

        X : the feature matrix

        y : the target vector

        """

        

        # Get the unique classes of the target

        self.classes = np.unique(y)

        

        # The feature matrix with x0 (the dummy feature)

        Z = np.hstack((np.ones((X.shape[0], 1)), X))

                

        # The one-hot-encoded target matrix

        Y = pd.get_dummies(y).to_numpy()

        

        # The random number generator

        self.rgen = np.random.RandomState(seed=self.random_state)

        

        # Initialize the weight matrix

        self.w = self.rgen.normal(loc=0.0, scale=0.01, size=(Z.shape[1], len(self.classes)))

        

        # Get the net_input

        net_input = self.net_input(Z)



        # Get the activation

        activation = self.activation(net_input)



        # Get the error

        error = Y - activation



        # For the kth class

        for k in range(len(self.classes)):

            # Get the hessian

            hessian = self.hessian(Z, activation, k)



            # Update the weight using Newton's method

            self.w[:, k] += np.linalg.pinv(hessian).dot(Z.T).dot(error[:, k].reshape(-1, 1)).reshape(-1)           



    def net_input(self, Z):

        """

        Get the net input

        

        Parameters

        ----------

        Z : The feature matrix with x0 (the dummy feature)

        

        Returns

        ----------

        The net input (the predicted value of the target)

        """

        

        return Z.dot(self.w)

    

    def activation(self, net_input):

        """

        Get the activation (softmax)

        

        Parameters

        ----------

        net_input : the net input

        

        Returns

        ----------

        The activation

       

        """

        

        # Get the exponent of the net input (using the minum max trick)

        net_input_exp = np.exp(net_input - np.max(net_input, axis=1).reshape(-1, 1))

        

        return net_input_exp / np.sum(net_input_exp, axis=1).reshape(-1, 1)

    

    def hessian(self, Z, activation, k):

        """

        Get the hessian (second-order derivative of the objective function)

        

        Parameters

        ----------

        Z : The feature matrix with x0 (the dummy feature)

        activation : The activation

        k : The kth class

        

        Returns

        ----------

        The hessian

       

        """

        

        # Initialize the hessian matrix

        H = np.zeros((Z.shape[1], Z.shape[1]))

        

        # Get activation * (1 - activation)

        prod = activation[:, k] * (1 - activation[:, k])

                

        # Update H

        for i in range(H.shape[0]):

            for j in range(H.shape[1]):

                H[i, j] = np.sum(prod * Z[:, j] * Z[:, i])

                

        return H

    

    def predict_proba(self, X):

        """

        The predict probability function

        

        Parameters

        ----------

        X : the feature matrix

        

        Returns

        ----------

        The predicted probability of each class (i.e., the activation)

        """

        

        # The feature matrix with x0 (the dummy feature)

        Z = np.hstack((np.ones((X.shape[0], 1)), X))

            

        # Get the net_input

        net_input = self.net_input(Z)



        return self.activation(net_input)

    

    def predict(self, X):

        """

        The predict class function

        

        Parameters

        ----------

        X : the feature matrix

        

        Returns

        ----------

        The predicted class of the target

        """

    

        return np.argmax(self.predict_proba(X), axis=1)
# The LogisticRegression_Newton

lr_newton = LogisticRegression_Newton()



# Train the logistic regression model on the training data

lr_newton.fit(X_train, y_train)
# Get the prediction on the validation data using lr

y_valid_pred = lr_newton.predict(X_valid)



# Get the accuracy

accuracy = accuracy_score(y_valid, y_valid_pred)



# Print the accuracy

pd.DataFrame([accuracy], columns=['accuracy'])
models = {'lr': LogisticRegression(class_weight='balanced', random_state=42)}
param_grids = {}
from sklearn.pipeline import Pipeline



pipes = {}



for acronym, model in models.items():

    pipes[acronym] = Pipeline([('model', model)])
from sklearn.model_selection import PredefinedSplit



# Combine the feature matrix in the training and validation data

X_train_valid = np.vstack((X_train, X_valid))



# Combine the target vector in the training and validation data

y_train_valid = np.append(y_train, y_valid)



# Get the indices of training and validation data

train_valid_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_valid.shape[0], 0))



# The PredefinedSplit

ps = PredefinedSplit(train_valid_idxs)
# The grids for C

C_grids = [10 ** i for i in range(-2, 3)]



# The grids for tol

tol_grids = [10 ** i for i in range(-6, -1)]



# Update param_grids

param_grids['lr'] = [{'model__C': C_grids,

                      'model__tol': tol_grids}]
import os



# Make directory

directory = os.path.dirname('./cv_results/')

if not os.path.exists(directory):

    os.makedirs(directory)
from sklearn.model_selection import GridSearchCV



# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV

best_score_param_estimator_gs = []



for acronym in pipes.keys():

    # GridSearchCV

    gs = GridSearchCV(estimator=pipes[acronym],

                      param_grid=param_grids[acronym],

                      scoring='f1_micro',

                      n_jobs=2,

                      cv=ps,

                      return_train_score=True)

        

    # Fit the pipeline

    gs = gs.fit(X_train_valid, y_train_valid)

    

    # Update best_score_param_estimator_gs

    best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

    

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'

    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    

    # Get the important columns in cv_results

    important_columns = ['rank_test_score',

                         'mean_test_score', 

                         'std_test_score', 

                         'mean_train_score', 

                         'std_train_score',

                         'mean_fit_time', 

                         'std_fit_time',                        

                         'mean_score_time', 

                         'std_score_time']

    

    # Move the important columns ahead

    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]



    # Write cv_results file

    cv_results.to_csv(path_or_buf='./cv_results/' + acronym + '.csv', index=False)
# Sort best_score_param_estimator_gs in descending order of the best_score_

best_score_param_estimator_gs = sorted(best_score_param_estimator_gs, key=lambda x : x[0], reverse=True)



# Print best_score_param_estimator_gs

pd.DataFrame(best_score_param_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])
# Get the best_score, best_param and best_estimator obtained by GridSearchCV

best_score_gs, best_param_gs, best_estimator_gs = best_score_param_estimator_gs[0]
# Get the prediction on the testing data using best_model

y_test_pred = best_estimator_gs.predict(X_test)



# Transform y_test_pred back to the original class

y_test_pred = le.inverse_transform(y_test_pred)



# Get the submission dataframe

df_submit = pd.DataFrame(np.hstack((np.arange(1, y_test_pred.shape[0] + 1).reshape(-1, 1), y_test_pred.reshape(-1, 1))),

                         columns=['PassengerId', 'Survived'])



passenger = list(df_raw_test['PassengerId'])



df_submit.insert(1,'PassengerIds',passenger)



df_submit = df_submit.drop(["PassengerId"], axis = 1)



# Generate the submission file

df_submit.to_csv('submission.csv', index=False)