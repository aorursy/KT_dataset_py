## username: premkamal
## challenge 1
## InMobi ML Academy (IMA)

## load required libraries
import numpy as np
import pandas as pd
from pandas import tools
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

inDebugMode = False

def load_dataset():
    global iris_train
    iris_train = pd.read_csv("../input/iris_train.csv")
    global iris_eval 
    iris_eval = pd.read_csv("../input/iris_test.csv")
    global iris_test
    iris_test = pd.read_csv("../input/iris_actual.csv")

def summarize_data():
    ## prelim checks on how the data looks
    print(iris_train.head(2));
    # -> data structure
    # ->    Sepal.Length  Sepal.Width    Petal.Length  Petal.Width Species
    # 0           5.1            3.5           1.4          0.2     setosa
    # 1           4.9            3.0           1.4          0.2     setosa
    print(iris_train.shape)
    print('\n')
    # -> (120,5) 
    print(iris_train.describe())
    # -> Mean, min, max data shows some clear separation of buckets based on the 4 features
    print(iris_eval.head(5))
    print(iris_test.head(5))

def visualize_data():
    ## visualize data
    print('Understanding data from plots and graphs')
    # box-whisker plot
    iris_train.plot(kind='box', subplots=True, layout=(2,2), sharex = False, sharey = False, title='box-whisker')
    plt.tight_layout()
    plt.savefig('/code/output/box_whisker_plot.png')
    
    # histogram plot
    iris_train.plot(kind='hist', subplots=True, layout=(2,2), sharex = False, sharey = False, title='histogram')
    plt.tight_layout()
    plt.savefig('/code/output/histogram_plot.png')
    
    # scatter plot
    iris_train.plot(kind='hist', subplots=True, layout=(2,2), sharex = False, sharey = False, title='scatter')
    plt.tight_layout()
    plt.savefig('/code/output/scatter_plot.png')
    
    # kde plot
    iris_train.plot(kind='kde', subplots=True, layout=(2,2), sharex = False, sharey = False, title='scatter')
    plt.tight_layout()
    plt.savefig('/code/output/kde_plot.png')
    
    # pie plot
    iris_train.plot(kind='pie', subplots=True, layout=(2,2), sharex = False, sharey = False, title='pie')
    plt.tight_layout()
    plt.savefig('/code/output/pie_plot.png')
    
    # density plot
    iris_train.plot(kind='density', subplots=True, layout=(2,2), sharex = False, sharey = False, title='density')
    plt.tight_layout()
    plt.savefig('/code/output/density_plot.png')

    # -> Inferences
    # -----------------
    # --> Based on leaf length, data forms 2 clusters
    # --> Sepal width feature has a normal distribution, nothing looks fishy
    # --> Width and length vs density graphs have 2 spikes

def evaluate_algorithms():
    global seed
    kfold_seed = 10
    kfold_splits = 7
    
    global scoring_criteria
    scoring_criteria = 'accuracy'
    # Setting accuracy as the metric of evaluation of estimators
    # Other trials = 'precision', 'recall', 'average_precision'
    
    # separate out features and supervised variable
    data_values = iris_train.values
    global features
    global supervised
    features = data_values[:,0:4]
    supervised = data_values[:,4]
    print(features)
    print(supervised)
    
    # Applying common known models to check for the best performance
    results =[]
    names = []
    models = []
    # trying linear model
    models.append(('log_reg', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('SVM', SVC()))
    # models.append(('KNN', KNeighborsClassifier()))
    for model_name, model in models:
        kfold = model_selection.KFold(n_splits = kfold_splits, shuffle = True, random_state = kfold_seed)
        result = model_selection.cross_val_score(model, features, supervised, cv = kfold, scoring = scoring_criteria)
        results.append(result)
        names.append(model_name)
        print ("%s: Mean: %f, Deviation: %f", model_name, result.mean(), result.std())
    ## Data comparision results
    # -------------------------
    # LOG_REG: Mean: 0.949580, Deviation: 0.049000
    # LDA: Mean: 0.966387, Deviation: 0.029110
    # SVM: Mean: 0.966853, Deviation: 0.028726
    # LIN_SVM: Mean: 0.949580, Deviation: 0.049000
    # KNN: Mean: 0.950047, Deviation: 0.037494
    # Best Performer -> SVM, minimum deviation

def make_prediction():
    test_values = iris_eval.values
    validation_values = iris_test.values
    test_feature_data = test_values[:,1:5]
    svm = SVC()
    svm.fit(features, supervised)
    predictions = svm.predict(test_feature_data)
    print(predictions)
    
    global iris_pred
    raw_data = []
    cnt = 0;
    for prediction in predictions:
        raw_data.append([test_values[cnt,0].astype(int), prediction])
        cnt += 1
    iris_pred = pd.DataFrame(raw_data, columns =['id','Species'])
    print(iris_pred.to_string(index = False))
    print(classification_report(validation_values[:,5], predictions))
    print(accuracy_score(validation_values[:,5], predictions))
    print(confusion_matrix(validation_values[:,5], predictions))
    
    iris_pred.to_csv("iris_prediction.csv", index = False)

def main() :
    load_dataset()
    summarize_data()
    if inDebugMode:
        visualize_data()
    evaluate_algorithms()
    make_prediction()

main()

# EOF
