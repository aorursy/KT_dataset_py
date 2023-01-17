
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# import arff # muss man dann "zu Fuss" laden, dann geht auch
# data = [(x,y,int(c)) for x,y,c in arff.load('../input/kiwhs-comp-1-complete/train.arff')]
# von Herrn Drewniok
import random

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

# Lesen der Daten
org_data = read_data("../input/train.arff")
print('datapoints:', len(org_data))

## For later use
test_data = pd.read_csv('../input/test.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
np_test = test_data[['X','Y']].as_matrix()

# Data is a list of lists, damit wir es mit Mengenoperatoren splitten
# können, müssen die inneren Listen Tupel sein (könnten wir gleich
# so anlegen, aber wir wollen den Einlesecode nicht verändern)
data = [(x,y,c) for x,y,c in org_data] # Jetzt haben wir eine Liste von Tupeln
## Warum brauchen wir das gleich?
## Weil Tupel "immutable" sind und deshalb "stabil" gehasht werden können
# Und das ist eine Voraussetzung, um eine Mengendifferenz effizient
# implementieren zu können (und also erforderlich für die Anwendung
# einer Mengendifferenz in Python)

# Nicht, dass sie jetzt denken, sie hätten das wissen müssen...
# man kann natürlich auch ohne Mengen arbeiten und "anders" splitten

## Erzeugen von Trainings- und Validierungsset "zu Fuss"
## Wird unten mit scikit gemacht
# random.seed(2) # Fixer Startpunkt für das Seeding des Random number generators,
# stellt Wiederholbarkeit sicher (falls man die haben möchte ;)
# validation = random.sample(data, 100) 
# train = set(data) - set(validation)
# print('train set:', len(train))
# print('validation set:', len(validation))

print(len(data),"Daten wurden eingelesen")
# Daten zeigen
import matplotlib.pyplot as plt
def plot_datapoints(data):
    x1,y1 = zip(*[(x,y) for x,y,c in data if c == 1])
    x2,y2 = zip(*[(x,y) for x,y,c in data if c == -1])
    plt.plot(x1, y1, 'o')
    plt.plot(x2, y2, 'o')

# For demonstration purposes
# Idee hier ist wie folgt: Um plot in 2D anzuwenden, braucht man
# zwei "Listen" x und y, in denen an der Position i die Daten zu einem
# zu zeichnenden Punkt auftauchen.  Diese Daten kann man wie folgt
# erzeugen:  x erzeugt man als mittels np.arange()
# über einen bestimmten Bereich als "Vector" von x-Koordinaten, die man
# beim Zeichnen verwenden möchte (plot füllt selbst die Lücken mit einer Gerade
# an sich bräuchte man also nur zwei Punkte...)
# y erhält man durch Anwendung der Geradengleichung - oder, allgemeiner, einer FUNKTION, die von
# x abhängig ist - und weiteren Parametern. Mit Numpy kann man aus einer Funktion, die
# für ein Element (kann mehr als eine Dimension haben) geschrieben wurde, eine
# vetorized Function bauen, die, ähnlich wie Pythons map, auf einen vector von
# x'sen angewendet werden kann und eine Vector mit y's erzeugt.
# Das tun wir hier. Wir schreiben also erst unsere allgemeine "Element"-Funktion,
# dann vectorisieren wir eine speziell parametrisierte Funktion (die wir mit partial aus functools erzeugen),
# bei der also die Parameter m und b an konkrete Werte gebunden wurden und dann vectorisieren wir die
# noch. Am ende können wir dann die Vectorisierung unmittelbar auf x anwenden und das Ganza als
# y an plot übergeben. Richtig cool wird das selbstverständlich erst,
# wenn man die so entstandenen vectorisierten Funktionen wiederholt auf verschiedene x anwenden
# möchte. Dann spart man sich die explizite Iteration über x, die man jedesmal wieder
# "zu Fuss" mit veränderten Parametern programmieren würde und kann direkt die Funktion auf Vektoren anwenden.

def line_func(x,m,b):
    return m*x + b

from functools import partial
# test_line = partial(line_func,1,1)

def plot_line(line,color='black'):
    m,b = line
    lf = partial(line_func,m=m,b=b)
    lf = np.vectorize(lf)
    x = np.arange(-3,4)
    plt.plot(x,lf(x),'-',lw=2,color=color)
    
# def plot_line(line, color='black'):
#    m,b = line
#    points_x = range(-3,4)
#    points_y = [m*x+b for x in points_x]
#    plt.plot(points_x, points_y, '-', lw=2, color=color)
    
plot_datapoints(data)
plot_line((-1,0.4)) ## Good guess for demonstation purposes 
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

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
#%matplotlib inline = show plots in Jupyter Notebook browser
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
data = np.array(data)
#split train and test data with function defaults
#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)
print(len(train_x),len(test_x))
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

## IGNORE THIS!
def true_accuracy(classes):
    right = 0
    for i,c in enumerate(classes):
        if i < 200:
            if c == -1: right += 1
        else:
            if c == 1: right += 1
    return right/len(classes)
        
# Make sure data are an np.array
data = np.array(data)

# As not everything is up-to-date below, we disable scikit warnings
# DON'T TO THIS IN SERIOUS EXPERIMENTS
import warnings
warnings.filterwarnings(action='ignore', category=Warning)

#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['ID', 'MLA Name', 'True Accuracy', 'MLA Test Accuracy', '#correct','MLA Parameters','MLA Train Accuracy Mean', 'MLA Validate Accuracy Mean', 'MLA Validate Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

#create table to compare MLA predictions
MLA_predict = pd.DataFrame(columns=['Target'])

#index through MLA and save performance to table
row_index = 0
for alg in MLA:    
    # Remeber initial ID (not necessary)
    MLA_compare.loc[row_index, 'ID'] = row_index
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
       
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    print("Computing scores for ",MLA_name, end="")
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, train_x, train_y, cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Validate Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Validate Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    
    #save MLA predictions - see section 6 for usage
    alg.fit(train_x, train_y)
    MLA_predict[MLA_name] = alg.predict(test_x)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = metrics.accuracy_score(test_y,MLA_predict[MLA_name])
    print(" -->",MLA_compare.loc[row_index, 'MLA Test Accuracy'],end="")
    MLA_compare.loc[row_index, '#correct'] = metrics.accuracy_score(test_y,MLA_predict[MLA_name],normalize=False)
    alg.fit(data[:,0:2], data[:,2])
    MLA_compare.loc[row_index, 'True Accuracy'] = true_accuracy(alg.predict(np_test))
    print(" [",MLA_compare.loc[row_index, 'True Accuracy'],"]")
    plot_decision_boundary(alg,train_x,train_y)
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['#correct'], ascending = False, inplace = True)
MLA_compare

# barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

#correlation heatmap
def correlation_heatmap(df,title=""):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title(title, y=1.05, size=15)
        
correlation_heatmap(MLA_predict,title='Heatmap of Correlations on Validation data between classifiers')
    

test = pd.read_csv('../input/test.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])

# Pick the best classifier from above
# MLA_compare.head()
best_classifier_data = MLA_compare.iloc(0)[13] # Access the first row
best_classifier = MLA[best_classifier_data['ID']] # give us the instantiated algo

print("Using ",best_classifier_data['MLA Name']," for submission: \n\nStart of submission file:\n")

data = np.array(data)

# Re-fit to ALL data (perhaps better to start new?)
best_classifier.fit(data[:,0:2], data[:,2])

# Makes it easier (for me) to fead the classifier
np_test = test[['X','Y']].as_matrix()

test['Category (String)'] = best_classifier.predict(np_test).astype(int)
test.drop(['X', 'Y'], axis=1, inplace=True) # X und Y brauchen wir nicht
test.to_csv('submission.csv')
print(test[0:6])
print("\nSubmission file completed!")
