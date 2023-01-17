# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
print(data.info())
print(data.describe())
data.dtypes
from sklearn.model_selection import train_test_split
dtypes_col = data.columns
dtypes_type_old = data.dtypes
dtypes_type = ['int16', 'bool', 'category', 'object', 'category', 'float32', 'int8', 'int8', 'object', 'float32', 'object', 'category']
optimized_dtypes = dict(zip(dtypes_col, dtypes_type))
optimized_dtypes
data_optimized = pd.read_csv("/kaggle/input/titanic/train.csv", dtype = optimized_dtypes)
test_optimized = pd.read_csv("/kaggle/input/titanic/test.csv", dtype = optimized_dtypes)
data_optimized.head()
train, validate = train_test_split(data_optimized, test_size=0.2)

combined = {'train':train,
            'valid':validate,
            'test':test_optimized,}

display(data_optimized.info())
display(combined)
data_optimized.isnull().sum()
combined_cleaned = {}
ct = 0
for i, data in combined.items():
    ct+=1
#     print(i)
    combined_cleaned[i] = data.drop('Cabin', 1).copy()
display(combined_cleaned)
train_num = combined_cleaned['train'].select_dtypes(include = ['float32', 'int16', 'int8', 'bool'])

colormap = plt.cm.cubehelix_r
plt.figure(figsize=(12,9))

plt.title('Person correlation of numeric features',)
sns.heatmap(train_num.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
def survived_percent(categories,column):
    survived_list = []
    for c in categories.dropna():
        count = combined_cleaned["train"][combined_cleaned["train"][column] == c][column].count()
        survived = combined_cleaned["train"][combined_cleaned["train"][column] == c]["Survived"].sum()/count
        survived_list.append(survived)
    return survived_list
category_features_list = ['Sex', 'Embarked', 'Pclass']
category_features = {}

for x in category_features_list:
    unique_values = combined_cleaned['train'][x].unique().dropna()
    survived = survived_percent(unique_values, x)
    category_features[x] = [unique_values, survived]
fig, axs = plt.subplots(1, 3, figsize = (18,4), sharey = True)
cb_dark_blue = (0/255, 107/255, 164/255)
cb_orange = (255/255, 128/255, 14/255)
cb_grey = (89/255, 89/255, 89/255)
color = [cb_dark_blue, cb_orange, cb_grey]

font_dict = {'fontsize':20,
            'fontweight':'bold',
            'color':'white'}

for i, cat in enumerate(category_features.keys()):
    number_categories = len(category_features[cat][0])
    axs[i].bar(range(number_categories), category_features[cat][1], color=color[:number_categories])
    axs[i].set_title("Survival rate "+cat, fontweight = 'bold')
    for j, indx in enumerate(category_features[cat][1]):
        label_text = category_features[cat][0][j]
        x = j
        y = indx
        axs[i].annotate(label_text, xy = (x-0.15 ,y/2), **font_dict )
for i in range(3):
    axs[i].tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    axs[i].patch.set_visible(False)

for i,data in combined_cleaned.items():
    data["Embarked"].fillna(value="S",inplace=True)
    mean_Fare = data["Fare"].mean()
    data["Fare"].fillna(value=mean_Fare,inplace=True)
# filling NaN in age
fig, ax = plt.subplots( figsize=(6,4))
x = combined_cleaned["train"]["Age"].dropna()
hist, bins = np.histogram( x,bins=15)

# histogram
ax.hist(x, normed=True)
ax.set_title('Age histogram')
plt.show()
from random import choices

bin_centers = 0.5*(bins[:len(bins)-1]+bins[1:])
probabilities = hist/hist.sum()

#dictionary with random numbers from existing age distribution
for i,data in combined_cleaned.items():
    data["Age_rand"] = data["Age"].apply(lambda v: np.random.choice(bin_centers, p=probabilities))
    Age_null_list   = data[data["Age"].isnull()].index
    
    data.loc[Age_null_list,"Age"] = data.loc[Age_null_list,"Age_rand"]
from sklearn import preprocessing,tree
from sklearn.model_selection import GridSearchCV

tree_data = {}
tree_data_category = {}

for i,data in combined_cleaned.items():
    tree_data[i] = data.select_dtypes(include=['float32','int16','int8']).copy()
    tree_data_category[i] = data.select_dtypes(include=['category'])

    for column in tree_data_category[i].columns:
        le = preprocessing.LabelEncoder()
        le.fit(data[column])
        tree_data[i][column+"_encoded"] = le.transform(data[column])
param_grid = {'min_samples_leaf':np.arange(20,50,5),
              'min_samples_split':np.arange(20,50,5),
              'max_depth':np.arange(3,6),
              'min_weight_fraction_leaf':np.arange(0,0.4,0.1),
              'criterion':['gini','entropy']}
clf = tree.DecisionTreeClassifier()
tree_search = GridSearchCV(clf, param_grid, scoring='average_precision')

X =  tree_data["train"].drop("PassengerId",axis=1)
Y = combined_cleaned["train"]["Survived"]
tree_search.fit(X,Y)

print("Tree best parameters :",tree_search.best_params_)
print("Tree best estimator :",tree_search.best_estimator_ )
print("Tree best score :",tree_search.best_score_ )
tree_best_parameters = tree_search.best_params_
tree_optimized = tree.DecisionTreeClassifier(**tree_best_parameters)
tree_optimized.fit(X,Y)

train_columns = list(tree_data["train"].columns)
train_columns.remove("PassengerId")
fig, ax = plt.subplots( figsize=(6,4))
ax.bar(range(len(X.columns)),tree_optimized.feature_importances_ )
plt.xticks(range(len(X.columns)),X.columns,rotation=90)
ax.set_title("Feature importance")
plt.show()
import graphviz 

dot_data = tree.export_graphviz(tree_optimized, 
                                out_file=None,
                                filled=True, 
                                rounded=True,  
                                special_characters=True,
                               feature_names = train_columns) 
graph = graphviz.Source(dot_data)
graph
test_without_PId = tree_data["test"].drop("PassengerId",axis=1)
prediction_values = tree_optimized.predict(test_without_PId).astype(int)
prediction = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],
                           "Survived":prediction_values})

prediction.head()
prediction.to_csv("Titanic_tree_prediction.csv",index=False)
from sklearn.metrics import confusion_matrix

evaluation = {}
cm = {}


valid_without_PId = tree_data["valid"].drop("PassengerId",axis=1)
evaluation["tree"] = tree_optimized.predict(valid_without_PId).astype(int)
survival_from_data = combined_cleaned["valid"]["Survived"].astype(int)

print(survival_from_data.value_counts())

cm["tree"] = confusion_matrix(survival_from_data, evaluation["tree"])
cm["tree"] = cm["tree"].astype('float') / cm["tree"].sum(axis=1)[:, np.newaxis]

cm["tree"]
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
gnb = GaussianNB()
gnb.fit(X, Y)
prediction_values_NaiveBayes = gnb.predict(test_without_PId).astype(int)
prediction_NaiveBayes = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],"Survived":prediction_values_NaiveBayes})
import itertools

def plot_confusion_matrix(cm, classes,title='Confusion matrix', cmap=plt.cm.Blues):
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
evaluation["NB"] = gnb.predict(valid_without_PId).astype(int)

cm["NB"] = confusion_matrix(survival_from_data, evaluation["NB"])
cm["NB"] = cm["NB"].astype('float') / cm["NB"].sum(axis=1)[:, np.newaxis]

cm["NB"]

plot_confusion_matrix(cm["NB"], classes=["No","Yes"],title='Normalized confusion matrix')
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold
import itertools
import time
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context('poster')
df = pd.read_csv('../input/titanic/train.csv', usecols=['Survived','Pclass','Sex','Age','Fare'])
df = df.fillna(df.mean())
df = df.round()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
sum(df.Survived)/len(df)
fares = np.array(list(df.Fare))
classes = np.array(list(df.Pclass))

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(classes,fares,'o',alpha=0.2)
ax.set_xlabel('Class', fontsize=20)
ax.set_ylabel('Fare', fontsize=20)
ax.set_xticks([1,2,3])
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()
from scipy.stats import pearsonr
pearsonr(fares,classes)[0]
df.groupby(['Pclass', 'Sex'])['Survived'].mean()
df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack().plot(kind='bar',figsize=(13,6), fontsize=20, color=['r','b']);
plt.ylabel('Survival Rate',fontsize=20)
plt.title('Survival Rates aboard Titanic',fontsize=20);
plt.xlabel('Class',fontsize=20);
plt.xticks([0,1,2],rotation=0);
df.head()
# first OHE the gender feature
df = pd.get_dummies(df, columns=['Sex'])
X = df[['Pclass','Age','Sex_female','Sex_male','Fare']].to_numpy()
Y = df[['Survived']].to_numpy()
# normalize age and fare
X[:,1] = (X[:,1] - X[:,1].min())/(X[:,1].max() - X[:,1].min())
X[:,4] = (X[:,4] - X[:,4].min())/(X[:,4].max() - X[:,4].min())
df.head()
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, perplexity=15, n_iter=2000, n_iter_without_progress=1000)
matrix_2d = tsne.fit_transform(X)
colors = df.Survived.values
colors = ['G' if i==1 else 'R' for i in colors]
df_tsne = pd.DataFrame(matrix_2d)
df_tsne['Survived'] = df['Survived']
df_tsne['color'] = colors
df_tsne.columns = ['x','y', 'Survived', 'color']
# rearrange columns
cols = ['Survived','color','x','y']
df_tsne = df_tsne[cols]
# show the 2D coordinates of the TSNE output
df_tsne.head()
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(df_tsne[df_tsne.Survived==1].x.values, df_tsne[df_tsne.Survived==1].y.values,
           c='green', s=10, alpha=0.5, label='Survived')
ax.scatter(df_tsne[df_tsne.Survived==0].x.values, df_tsne[df_tsne.Survived==0].y.values,
           c='red', s=10, alpha=0.5, label='Died')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend()
plt.show();
from sklearn.model_selection import train_test_split

data = df.to_numpy()

data[:,2] = (data[:,2] - data[:,2].min())/(data[:,2].max() - data[:,2].min())
data[:,3] = (data[:,3] - data[:,3].min())/(data[:,3].max() - data[:,3].min())
X_train, X_test = train_test_split(data, test_size=0.1)
Y_train = X_train[:,0] # first column is class
Y_train = np.reshape(Y_train, newshape=(len(Y_train),1)) # reshape to a columns vector
X_train = X_train[:,1:] # select all columns but class
Y_test = X_test[:,0]
Y_test = np.reshape(Y_test, newshape=(len(Y_test),1)) # reshape to a columns vector
X_test = X_test[:,1:]
def sigmoid(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def relu(x, deriv=False):
    if deriv == True:
        x[x<0] = 0.01
        x[x>0] = 1.
        return x
    x[x<0] = 0.01*x[x<0]
    return x

def predict(x, w0, w1, b1, b2):
    A = np.dot(x,w0) + b1 # mXN X NxH +1xH ~ mxH
    layer_1 = relu(A)
    B = np.dot(layer_1,w1) + b2 # mxH X Hx1 ~ mx1 (preds)
    layer_2 = B
    return (sigmoid(layer_2) > 0.5).astype(int)
def get_batch(x,y,i,batchSize=32):
    """
    Function that returns a minibatch of a dataset
    """
    return x[i:i+batchSize],y[i:i+batchSize]
alpha, hidden_size, drop_rate, batch_size = (0.04,32,0.5,32)
# randomly initialise synapses
syn0 = 2*np.random.random((X_train.shape[1],hidden_size)) - 1 
syn1 = 2*np.random.random((hidden_size,1)) - 1
# randomly initialise biases
b1 = np.random.randn(hidden_size)
b2 = np.random.randn(1) 
avg_err = []


for epoch in range(2000):
    err = []

    for i in range(int(X_train.shape[0]/batch_size)):

        x,y = get_batch(X_train,Y_train,i,batch_size)

        # Forward
        layer_0 = x
#         print("np.dot", np.dot(layer_0,syn0))
#         print("b1", b1)
        A = np.dot(layer_0,syn0) + b1 
        layer_1 = relu(A)
        # drop out to reduce overfitting
        layer_1 *= np.random.binomial([np.ones((len(x),hidden_size))],1-drop_rate)[0] * (1/(1-drop_rate))
        
        B = np.dot(layer_1,syn1) + b2 
        layer_2 = sigmoid(B)

        # Backprop
        layer_2_error = layer_2 - y 
        layer_2_delta = layer_2_error * sigmoid(layer_2,deriv=True) 

        layer_1_error = np.dot(layer_2_delta,syn1.T) 
        layer_1_delta = layer_1_error * relu(layer_1,deriv=True) 

        # update weights
        syn1 -= alpha*np.dot(layer_1.T,layer_2_delta) 
        syn0 -= alpha*np.dot(layer_0.T,layer_1_delta) 

        # update biases
        m = len(y)
        b2 -= alpha * (1.0 / m) * np.sum(layer_2_delta)
        b1 -= alpha * (1.0 / m) * np.sum(layer_1_delta)

        err.append(layer_2_error)

    avg_err.append(np.mean( np.abs(err) ))
    if epoch%500 == 0:
        print("Epoch: %d, Error: %.8f" % (epoch, np.mean( np.abs(err) )))
# accuracy ของ train set
100*(1-np.sum(np.abs(predict(X_train, syn0, syn1, b1, b2) - Y_train))/len(X_train))
# accuracy ของ test set
100*(1-np.sum(np.abs(predict(X_test, syn0, syn1, b1, b2) - Y_test))/len(Y_test))
fig,ax = plt.subplots(figsize=(15,8))
ax.plot(np.arange(len(avg_err)), np.array(avg_err))
ax.set_xlabel('Iteration', fontsize=18)
ax.set_ylabel('Mean Error', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50,n_jobs=1,max_depth=10)
model = clf.fit(X_train,Y_train.T[0])
# accurency train
100*model.score(X_train,Y_train.T[0])
# accurency test
100*(1-np.sum(np.abs(model.predict(X_test) - Y_test.T[0]))/len(Y_test))
