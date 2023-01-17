import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os

# Any results you write to the current directory are saved as output.
# Read some arff data
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

org_data = read_data("../input/kiwhs-comp-1-complete/train.arff")

test_data = pd.read_csv('../input/kiwhs-comp-1-complete/test.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
np_test = test_data[['X','Y']].as_matrix()

data = [(x,y,c) for x,y,c in org_data]
data = np.array(data)
# Chosen Model Algorithm
from sklearn import naive_bayes

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

# Compare
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()
def get_score(predictions,labels):
    correct = 0
    total = 0
    for i in range(0, len(labels)):
        total = total + 1
        if predictions[i] == labels[i]:
            correct = correct + 1
    return correct/total
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)
def find_best_param():
    global train_x
    global train_y
    best_b = 0
    best_score = 0
    b = 0.01
    while b <= 30:
        bnb = naive_bayes.BernoulliNB(binarize = b)
        bnb.fit(train_x,train_y)

        predictions = bnb.predict(train_x)

        score = get_score(predictions,train_y)

        if score >= best_score:
            best_score = score
            best_b = b
        b = b + 0.01
    bnb = naive_bayes.BernoulliNB(binarize = best_b)
    bnb.fit(train_x,train_y)
    return bnb
bnb = find_best_param()
predictions = bnb.predict(test_x)
print("Plotting with binarize = %f   Correct:%f"%(bnb.binarize,get_score(predictions,test_y)))
plot_decision_boundary(bnb,test_x,test_y)
## Eine einfache Funktion, die die Klassifizierungen der Testdaten plottet
def plot_test_data(data,pred,title):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(0, len(data)):
        if pred[i] == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    plt.plot(x1, y1, 'o')
    plt.plot(x2, y2, 'o')
    plt.title(title)
test_data_labels = []
for i in range(0,200):
    test_data_labels.append(-1)
for i in range(200,400):
    test_data_labels.append(1)

test_predictions = bnb.predict(np_test)
print("Plotting with binarize = %f   Correct:%f"%(bnb.binarize,get_score(test_predictions,test_data_labels)))
plot_test_data(np_test, test_predictions,"Classified Test Data")
org_data = read_data("../input/skewed-data/train-skewed.arff")

test_data = pd.read_csv('../input/skewed-data/test-skewed.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
np_test = test_data[['X','Y']].as_matrix()

data = [(x,y,c) for x,y,c in org_data]
data = np.array(data)

train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)

bnb = find_best_param()
predictions = bnb.predict(test_x)
print("Plotting with binarize = %f   Correct:%f"%(bnb.binarize,get_score(predictions,test_y)))
plot_decision_boundary(bnb,test_x,test_y)
print("Vorhandene Testdaten: ",len(np_test))
test_predictions = bnb.predict(np_test)
print("Plotting with binarize = %f   Correct:%f"%(bnb.binarize,get_score(predictions,test_y)))
plot_test_data(np_test, test_predictions,"Classified Skewed Test Data")