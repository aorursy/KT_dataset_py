import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline
#Configure plots
rc={'savefig.dpi': 75, 'figure.figsize': [12,8], 'lines.linewidth': 2.0, 'lines.markersize': 8, 'axes.labelsize': 18,\
   'axes.titlesize': 18, 'font.size': 18, 'legend.fontsize': 16}

sns.set(style='dark',rc=rc)
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.keys()
train_data.shape
test_data.shape
train_data.columns[train_data.isnull().any()]
test_data.columns[test_data.isnull().any()]
sns.countplot(x='label', data=train_data, color="#99c2ff")
labels_train = train_data.iloc[:,0]
features_train = train_data.iloc[:,1:]
labels_train.head()
labels_train.shape
features_train.shape
#cross-validation method
cv = StratifiedShuffleSplit(n_splits = 3, random_state = 0, 
                                train_size = 0.70)
#Function to perform cross-validation
def nn_cross_validation(alpha, units, max_iter):
    mlp = MLPClassifier(solver='lbfgs', activation = 'logistic', random_state=0, 
                    max_iter = max_iter, alpha = alpha, hidden_layer_sizes = (units,))
    
    pipe = Pipeline([('scaling', StandardScaler()),
                 ('clf', mlp)])
    
    scores = cross_validate(pipe, features_train, labels_train, cv=cv, scoring = 'accuracy')

    return scores
#Function to test the different configurations and plot the result
def test_options(alpha, units, max_iter, parameter):    
    mean_train = []
    std_train = []
    mean_test = []
    std_test = []
    option = 0
    
    if parameter == 'alpha':
        options = alpha
        var1, var2, var3 = option, units, max_iter
    elif parameter == 'max_iter':
        options = max_iter
        var1, var2, var3 = alpha, units, option
    else:
        options = units
        var1, var2, var3 = alpha, option, max_iter

    for option in options:
        
        if parameter == 'alpha':
            var1 = option
        elif parameter == 'max_iter':
            var3 = option
        else:
            var2 = option
            
        CV_scores = nn_cross_validation(var1, var2, var3)
        mean_train = np.append(mean_train, CV_scores['train_score'].mean())
        std_train = np.append(std_train, CV_scores['train_score'].std())
        mean_test = np.append(mean_test, CV_scores['test_score'].mean())
        std_test = np.append(std_test, CV_scores['test_score'].std())

    plt.figure()
    plt.errorbar(options, mean_train, std_train, linestyle='None', marker='*', c = 'b')
    plt.errorbar(options, mean_test, std_test, linestyle='None', marker='*', c = 'r') 

    plt.ylabel('Accuracy')
    plt.legend(['Train data', 'Test data'])
alphas = [0.0001, 0.1, 1, 10, 100]

test_options(alphas, 50, 50, 'alpha')

plt.xscale('log', basex = 10)
plt.xlabel('Regularization term (alpha)')
max_iter = [50, 100, 150]

test_options(10, 50, max_iter, 'max_iter')

plt.xlabel('Maximum number of iterations')
units = [50, 100, 150]

test_options(10, units, 100, 'units')

plt.xlabel('Number of units in the hidden layer')
scaler = StandardScaler()

features_transf = scaler.fit_transform(features_train)
mlp = MLPClassifier(solver='lbfgs', activation = 'logistic', random_state=0, 
                    max_iter = 100, alpha = 10, hidden_layer_sizes = (100,))

mlp.fit(features_transf, labels_train)
mlp.score(features_transf, labels_train)
test_data_transf = scaler.transform(test_data)
pred = mlp.predict(test_data_transf)
unique, counts = np.unique(pred, return_counts=True)
result = dict(zip(unique, counts))

plt.bar(result.keys(), result.values())
output = {}

output['Label'] = pred
output = pd.DataFrame(output)
output.index+=1
output.to_csv('output.csv', index_label = 'ImageID')