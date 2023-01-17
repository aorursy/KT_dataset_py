import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



file_path = '../input/Immunotherapy.xlsx'

data = pd.read_excel(file_path) 



data.describe().T
#check null values

cols_with_missing = [col for col in data.columns

                     if data[col].isnull().any()]
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
sns.countplot(data['Result_of_Treatment'],label="Sum")

data['Result_of_Treatment'].value_counts()

# We can use SMOTE algorithm for oversampling
pd.concat([data.groupby('Result_of_Treatment').mean(), data.groupby('Result_of_Treatment').mean().diff().dropna()]).T
fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(2, 2, 1)

ax = sns.distplot( data[data['Result_of_Treatment'] == 0]['age'] , color="skyblue", label="0", ax = ax)

ax = sns.distplot( data[data['Result_of_Treatment'] == 1]['age'] , color="red", label="1", ax = ax)

ax.legend()

ax = fig.add_subplot(2, 2, 2)

ax = sns.distplot( data[data['Result_of_Treatment'] == 0]['Time'] , color="skyblue", label="0")

ax = sns.distplot( data[data['Result_of_Treatment'] == 1]['Time'] , color="red", label="1")

ax.legend()

ax = fig.add_subplot(2, 2, 3)

ax = sns.distplot( data[data['Result_of_Treatment'] == 0]['Number_of_Warts'] , color="skyblue", label="0")

ax = sns.distplot( data[data['Result_of_Treatment'] == 1]['Number_of_Warts'] , color="red", label="1")

ax.legend()

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax = fig.add_subplot(2, 2, 4)

ax = sns.distplot( data[data['Result_of_Treatment'] == 0]['Area'] , color="skyblue", label="0", ax = ax)

ax = sns.distplot( data[data['Result_of_Treatment'] == 1]['Area'] , color="red", label="1", ax = ax)

ax.legend()

plt.show()
ax = sns.distplot( data[data['Result_of_Treatment'] == 0]['age'] , color="skyblue", label="0")

ax = sns.distplot( data[data['Result_of_Treatment'] == 1]['age'] , color="red", label="1")

ax.legend()
ax = sns.distplot( data[data['Result_of_Treatment'] == 0]['Time'] , color="skyblue", label="0")

ax = sns.distplot( data[data['Result_of_Treatment'] == 1]['Time'] , color="red", label="1")

ax.legend()
sns.catplot(x="Result_of_Treatment", y="Time", kind = 'violin', palette="pastel", inner="stick", data=data);

sns.catplot(x="Result_of_Treatment", y="age", kind = 'violin', palette="pastel", inner="stick", data=data);

sns.catplot(x="Type", y="Time", hue= 'Result_of_Treatment', kind = 'violin', palette="pastel", inner="stick", split=True, data=data);
corr = data.corr()

corr.T
mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(corr, mask=mask, annot = True, 

                     vmin =-1, vmax =1, center = 0, 

                     xticklabels=corr.columns, cmap="RdBu_r")    
sns.scatterplot(x=data['Time'], y=data['age'], hue=data['Result_of_Treatment'])
# 2D KDE plot

sns.jointplot(x=data['Time'], y=data['age'], kind="kde")
fig = plt.figure()

fig.subplots_adjust(hspace=1, wspace=.4)

ax = fig.add_subplot(1, 2, 1)

ax = sns.scatterplot(x=data['Time'], y=data['age'], hue=data['Result_of_Treatment'], ax = ax)

ax = fig.add_subplot(1, 2, 2)

successed = data.loc[data.Result_of_Treatment == 1]

failed = data.loc[data.Result_of_Treatment == 0]

ax = sns.kdeplot(failed.Time, failed.age,label ="0",cmap="Blues", ax=ax)

ax = sns.kdeplot(successed.Time, successed.age,label ="1", cmap="Reds", ax = ax)

ax.legend()

plt.show()

successed = data.loc[data.Result_of_Treatment == 1]

failed = data.loc[data.Result_of_Treatment == 0]

ax = sns.kdeplot(failed.Time, failed.age,label ="0",

                 cmap="Blues")

ax = sns.kdeplot(successed.Time, successed.age,label ="1",

                 cmap="Reds")

ax.legend()
c = sns.FacetGrid(data, col="Type", hue="Result_of_Treatment")

c.map(plt.scatter, "Time", "age", alpha=.7)

c.add_legend();
from sklearn.model_selection import train_test_split



X = data.loc[:, data.columns != 'Result_of_Treatment']

y = data.loc[:, 'Result_of_Treatment']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

#train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=1)
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

import graphviz

from IPython.display import Image  



decisionTreeModel = DecisionTreeClassifier(random_state=1)

decisionTreeModel.fit(train_X, train_y) 



dot_data = tree.export_graphviz(decisionTreeModel, out_file=None, 

    feature_names = train_X.columns.values.tolist(),  

    class_names= ['0','1'],  

    filled=True, 

    rounded=True

    )  

graph = graphviz.Source(dot_data) 

from IPython.display import SVG

Image(graph.pipe(format='png'))
from sklearn.model_selection import cross_validate

from sklearn import metrics



def scoringModel(model, train, y):

    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

    scores = cross_validate(model, train, y, scoring=scoring, cv=5)

    return scores



def scoring(model):

    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

    scores = cross_validate(model, train_X, train_y, scoring=scoring, cv=5)

    for metric_name in scores.keys():

        print('%s : %f' % (metric_name, scores[metric_name].mean()))

    return scores



scoreTreeModel = scoringModel(decisionTreeModel,train_X, train_y)

#scoreTreeModel = scoring(decisionTreeModel)



cnf_matrix = metrics.confusion_matrix(test_y, decisionTreeModel.predict(test_X))

cnf_matrix

from sklearn.linear_model import LogisticRegression



LR = LogisticRegression()

scoreLR = scoring(LR)



data = {'Model' : ['Decision tree', 'Logistic regression']}



for metric_name in scoreTreeModel.keys():

    data[metric_name] = [scoreTreeModel[metric_name].mean(), scoreLR[metric_name].mean()]



df = pd.DataFrame(data) 

df.T
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(train_X)

x_train_norm = std_scale.transform(train_X)

training_norm_X = pd.DataFrame(x_train_norm, index=train_X.index, columns=train_X.columns) 

print(training_norm_X.head())



x_test_norm = std_scale.transform(test_X)

testing_norm_X = pd.DataFrame(x_test_norm, index=test_X.index, columns=test_X.columns) 

print(testing_norm_X.head())
training_norm_X[["age","Time",'Type',"induration_diameter"]]
scoreTree = scoringModel(decisionTreeModel, training_norm_X, train_y)

scoreLR = scoringModel(LR, training_norm_X, train_y)



data = {'Model' : ['Decision tree', 'Logistic regression']}

for metric_name in scoreTree.keys():

    data[metric_name] = [scoreTree[metric_name].mean(), scoreLR[metric_name].mean()]



df = pd.DataFrame(data) 

df.T
scoreTree = scoringModel(decisionTreeModel, training_norm_X[["age","Time",'Type',"induration_diameter"]], train_y)

scoreLR = scoringModel(LR, training_norm_X[["age","Time",'Type',"induration_diameter"]], train_y)



data = {'Model' : ['Decision tree', 'Logistic regression']}



for metric_name in scoreTree.keys():

    data[metric_name] = [scoreTree[metric_name].mean(), scoreLR[metric_name].mean()]



df = pd.DataFrame(data) 

df.T
from imblearn.over_sampling import SMOTE



os = SMOTE(random_state=0)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



columns = training_norm_X.columns

os_data_X,os_data_y=os.fit_sample(training_norm_X, train_y)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns )

os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])



# we can Check the numbers of our data

print("length of oversampled data is ",len(os_data_X))

print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))

print("Number of subscription",len(os_data_y[os_data_y['y']==1]))

print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))

print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
import statsmodels.api as sm

logit_model=sm.Logit(train_y,train_X)

result=logit_model.fit()

print(result.summary2())
def sigmoid(z):

  return 1.0 / (1 + np.exp(-z))



def predict(features, weights):

  '''

  Returns 1D array of probabilities

  that the class label == 1

  '''

  z = np.dot(features, weights)

  return sigmoid(z)



def cost_function(features, labels, weights):

    '''

    Using Mean Absolute Error



    Features:(100,3)

    Labels: (100,1)

    Weights:(3,1)

    Returns 1D matrix of predictions

    Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)

    '''

    observations = len(labels)



    predictions = predict(features, weights)



    #Take the error when label=1

    class1_cost = -labels*np.log(predictions)



    #Take the error when label=0

    class2_cost = (1-labels)*np.log(1-predictions)



    #Take the sum of both costs

    cost = class1_cost - class2_cost



    #Take the average cost

    cost = cost.sum() / observations



    return cost



def update_weights(features, labels, weights, lr):

    '''

    Vectorized Gradient Descent



    Features:(200, 3)

    Labels: (200, 1)

    Weights:(3, 1)

    '''

    N = len(features)

    #1 - Get Predictions

    predictions = predict(features, weights)

    #2 Transpose features from (200, 3) to (3, 200)

    gradient = np.dot(features.T,  predictions - labels)

    #3 Take the average cost derivative for each feature

    gradient /= N

    #4 - Multiply the gradient by our learning rate

    gradient *= lr

    #5 - Subtract from our weights to minimize cost

    weights -= gradient



    return weights

def decision_boundary(prob):

  return 1 if prob >= .5 else 0



def classify(predictions):

  '''

  input  - N element array of predictions between 0 and 1

  output - N element array of 0s (False) and 1s (True)

  '''

  _decision_boundary = np.vectorize(decision_boundary)

  return _decision_boundary(predictions).flatten()



def train(features, labels, weights, lr, iters):

    cost_history = []



    for i in range(iters):

        #print (features)

        #print (weights)

        

        weights = update_weights(features, labels, weights, lr)



        #Calculate error for auditing purposes

        cost = cost_function(features, labels, weights)

        cost_history.append(cost)



        # Log Progress

        if i % 100 == 0:

            print ("iter: {}  cost: {}".format(i, cost))



    return weights, cost_history



#CALCULATION

np.random.seed(0)

weights, cost_history = train(np.hstack((np.ones((train_y.count(),1)), train_X[['Time','age']].to_numpy())), 

      train_y.to_numpy(), 

      np.random.uniform(low=-1.0, high=1.0, size=3), 

      0.001, 1000)

possiblities = predict(np.hstack((np.ones((test_y.count(),1)),test_X[['Time','age']].to_numpy())), weights)

predictions = classify(possiblities)

possiblities
def accuracy(predicted_labels, actual_labels):

    diff = predicted_labels - actual_labels

    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

accuracy(predictions, test_y)
def plot_decision_boundary(trues, falses):

    fig = plt.figure()

    ax = fig.add_subplot(111)



    no_of_preds = len(trues) + len(falses)



    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')

    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')



    plt.legend(loc='upper right');

    ax.set_title("Decision Boundary")

    ax.set_xlabel('N/2')

    ax.set_ylabel('Predicted Probability')

    plt.axhline(.5, color='black')

    plt.show()



logit_model=sm.Logit(train_y,train_X[['Time','age']])

result=logit_model.fit()

print(result.summary2())
#import numpy as np

#import matplotlib.pyplot as plt

from sklearn import linear_model

from scipy.special import expit



# General a toy dataset:s it's just a straight line with some Gaussian noise:

def drawSigmoidFunction(X, y) :

    # Fit the classifier

    clf = linear_model.LogisticRegression(C=1e5)

    clf.fit(X, y)



    # and plot the result

    plt.figure(1, figsize=(8, 6))

    plt.clf()

    plt.scatter(X.ravel(), y, color='black')

    X_test = np.linspace(-5, 10, 300)



    loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()

    plt.plot(X_test, loss, color='red', linewidth=3)



    #ols = linear_model.LinearRegression()

    #ols.fit(X, y)

    #plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)

    

    plt.axhline(.5, color='.5')



    plt.ylabel('y')

    plt.xlabel('X')

    plt.xticks(range(-5, 10))

    plt.yticks([0, 0.5, 1])

    plt.ylim(-.25, 1.25)

    plt.xlim(-4, 10)

    plt.legend(('Logistic Regression Model', 'Decision Boundary'),

               loc="lower right", fontsize='small')

    plt.tight_layout()

    plt.show()

    



    

xmin, xmax = -5, 5

n_samples = 100

np.random.seed(0)

X = np.random.normal(size=n_samples)

y = (X > 0).astype(np.float)

X[X > 0] *= 4

X += .3 * np.random.normal(size=n_samples)



X = X[:, np.newaxis]

print(X.ravel())

#drawSigmoidFunction(train_X.to_numpy(), train_y.to_numpy())

drawSigmoidFunction(X, y)



import numpy.ma as ma



z = np.dot(np.hstack((np.ones((train_y.count(),1)),train_X[['Time','age']].to_numpy())), weights)

predictions = predict(np.hstack((np.ones((train_y.count(),1)),train_X[['Time','age']].to_numpy())), weights)



mz = ma.masked_array(z, mask=train_y.to_numpy())

mpredictions = ma.masked_array(predictions, mask=train_y.to_numpy())



plt.scatter(z, predictions, color='black', zorder=20)

plt.scatter(mz, mpredictions, color='green', zorder=20)



X_test = np.linspace(-2, 4, 300)

clf = linear_model.LogisticRegression(C=1e5)

clf.fit(X, y)



loss = expit(X_test * clf.coef_*.15 + clf.intercept_ +1.6).ravel()

plt.plot(X_test, loss, color='red', linewidth=3)



#ols = linear_model.LinearRegression()

#ols.fit(X, y)

#plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)



plt.axhline(.5, color='.5')



plt.ylabel('y')

plt.xlabel('X')

plt.xticks(range(-2, 4))

plt.yticks([0, 0.5, 1])

plt.ylim(-.25, 1.25)

plt.xlim(-2, 4)

plt.legend(('Logistic Regression Model', 'Decision Boundary', '1 | 1/1-e^(-h(x_i))', '0 | 1/1-e^(-h(x_i))'),

           loc="lower right", fontsize='small')

plt.tight_layout()

plt.show()
plt.plot(cost_history)

plt.xlabel('epoch')

plt.ylabel('cost')