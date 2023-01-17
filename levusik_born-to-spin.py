import folium



# Change these to your city

latitude_def  = 51.509865

longitude_def = -0.118092



m = folium.Map(location=[latitude_def, longitude_def])

folium.Circle(radius=10000,

        location=[latitude_def, longitude_def],

        popup='PSR B1919+21',

        color='blue',

        fill=True,).add_to(m)

m
# utils 

import os, sys

import pandas as pd

import numpy as np



# machine learning

import sklearn as sk

import statsmodels as sms 



# plotting

import plotly_express

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



plt.style.use('ggplot')



data_folder = '/kaggle/input/predicting-a-pulsar-star/'

os.listdir(data_folder)
df = pd.read_csv(os.path.join(data_folder,"pulsar_stars.csv"))

df.head()
df.describe()
names = ['meanIP', 'stdIP', 'kurtIP', 'skewIP', 'meanDS', 'stdDS', 'kurtDS', 'skewDS', 'target']

df.columns = names
plt.style.use('ggplot')

plt.figure(figsize=(8,8))



def plotHist(df):

    plt.title("Percent of pulsars {} %".format( sum(df == 1) / len(df) * 100))

    plt.bar(['not pulsar', 'pulsar'], [sum(df == 0), sum(df == 1)])

    

plotHist(df['target'])
sns.set(style="ticks", color_codes=True)

sns.pairplot(df, hue='target', markers=['o', 's'], vars =names[:-1])

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



X_features, labels = df[names[:-1]], df[names[-1]]

X_features_scaled, Y_features_scaled = df[names[:-1]].copy(), df[names[-1]].copy()

print("All        {} : {}".format(X_features.shape, labels.shape))
scaler = StandardScaler()



# stratification by class and shuffling are default 

X_train, X_test, Y_train, Y_test = train_test_split(X_features, labels, test_size = 0.1, random_state = 42)

X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, test_size=2/9, random_state=42)



# X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = train_test_split(X_features_scaled, Y_features_scaled, test_size = 0.1, random_state = 42)

# X_train_scaled, X_val_scaled, Y_train_scaled, Y_val_scaled  = train_test_split(X_train_scaled, Y_train_scaled, test_size=2/9, random_state=42)



X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.fit_transform(X_val)

X_test_scaled = scaler.fit_transform(X_test)



print("Training   {} : {}".format(X_train.shape, Y_train.shape))

print("Validation {}  : {}".format(X_val.shape, Y_val.shape))

print("Testing    {}  : {}".format(X_test.shape, Y_test.shape))
from itertools import product



# Here's poor implementation of grid searching through parameters and picking the best 

# Implemented because i couldn't find a way ( maybe it's simple ) to call sklearn GridSearchCV 

# and evaluate classifier from that on validation set

def iterDicts(options):

    values = [options[key] for key in options.keys()]

    return [dict(zip(options.keys(), it)) for it in product(*values)]



def GridSearchAndCrossValidate(Classifier, options, X_train_scaled, Y_train_scaled, 

                               X_val_scaled, Y_val_scaled, accuracy_metric, verbose=False):

    best_acc = 0

    best_classifier = None



    for option in iterDicts(options):

        classifier = Classifier(**option)

        classifier.fit(X_train_scaled, Y_train_scaled)

        predictions = classifier.predict(X_val_scaled)

        accuracy = accuracy_metric(predictions, Y_val_scaled)

    

        if accuracy > best_acc:

            best_acc = accuracy

            best_classifier = classifier

    

        if verbose:

            print("for {} val acc is {}".format(option, accuracy))

    if verbose:

        print("-"*32)

        print("Final best accuracy {}\n for {}".format(best_acc, best_classifier))

        print("-"*32)

    

    return best_classifier, best_acc
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, roc_auc_score, f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from time import time



class model():

    def __init__(self, Model, plotName, options):

        self.model = Model

        self.plotName = plotName

        self.options = options

        

# not really pretty solution.

# this should be in some sort of config file ( json or whatever) 

models = [ model(SVC, "SVC", {"kernel": ['rbf', 'poly'], "C": [0.5, 1, 2, 4, 6, 8, 10], 'random_state':[42], 'probability':[True]}),

           model(DecisionTreeClassifier, "Tree",  {'min_samples_split' : [2, 4, 6, 8, 10, 15, 20, 25] , "criterion" : ["gini", 'entropy'], 'random_state' : [42]}),

           model(RandomForestClassifier, "R. forest", {'criterion' : ['gini', 'entropy'], "max_depth": [5, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] , 'n_estimators': [10, 50, 100, 250] ,'random_state' : [42], 'n_jobs': [-1]}),

           model(AdaBoostClassifier, "AdaBoost", {'n_estimators' : [10, 100, 500], 'random_state' : [42]}),

           model(BaggingClassifier, "Bagging", {'n_estimators' : [2, 5, 10, 15], 'n_jobs' : [-1], 'random_state' : [42]}),

           model(MLPClassifier, "neural network", {"hidden_layer_sizes" : [(64, 1), (64, 2), (64, 3)], 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'max_iter' : [500], 'random_state' : [42]}),

           model(LGBMClassifier, "lightGBM", {'boosting_type' : ['gbdt', 'dart', 'goss'], "num_leaves" : [16, 32, 64 ], 'max_bin' : [50, 100, 200, 400], 'learning_rate' : [0.005, 0.01, 0.1], 'num_iterations' : [100, 200], 'random_state' : [42]}),

           model(XGBClassifier, "xgboost", {"booster" : ["gbtree", 'gblinear'], 'n_jobs' : [-1],  'max_depth' : np.arange(3, 11,1), 'learning_rate' : [0.01, 0.05, 0.1, 0.2], 'random_state' : [42]})]





def plot_sklearn_models(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):



    accuracies = {}

    plt.style.use('ggplot')

    plt.figure(figsize=(len(models) * 2, len(models) * 5)) 

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    cnt = 1

    

    for model in models:



        t = time()

        # perform grid Search

        classifier, acc = GridSearchAndCrossValidate(model.model, model.options, X_train, Y_train, 

                                             X_valid, Y_valid, f1_score, verbose=False)

        print("Trained {} in {} seconds ".format(model.plotName, time() - t))

    

        # plot confusion matrix 

        plt.subplot(len(models), 3, cnt)

        test_predictions = classifier.predict(X_test)

        

        # our accuracy metric for test data

        acc = f1_score(test_predictions, Y_test)

        

        # calculate confustion matrix from test data

        conf = confusion_matrix(test_predictions,Y_test) 

        plt.title("Conf matrix {0}, F1 {1: .3f}".format(model.plotName, acc))

        sns.heatmap(conf, annot=True, annot_kws={"size": 16})



    

        # plot precision recall curve 

        plt.subplot(len(models), 3, cnt+1)

        

        # get probabilities of classes from test data from clasifier 

        # ( for SVM probability parameter must be set to True)

        probs = classifier.predict_proba(X_test)[:, 1]

        

        # create precision recall curve for testing data

        precision, recall, _ = precision_recall_curve(Y_test, probs)

        plt.title("Precision Recall curve")

        plt.step(precision, recall)

        plt.xlabel("precision")

        plt.ylabel("recall")

        plt.fill_between(precision, recall, alpha=.2, step = 'pre')

        

        plt.subplot(len(models), 3, cnt+2)        

        # calculate area under curve from ROC

        roc_score = roc_auc_score(Y_test, probs)

        plt.title("AUC-ROC {0:.3f}".format(roc_score))

        

        # Roc Curve

        fpr, tpr, threshs = roc_curve(Y_test, probs)

        plt.plot(fpr, tpr)

        plt.plot([[0, 0], [1, 1]], linestyle='dashed')

        cnt += 3

        

        accuracies[model.plotName] = {"F1" : acc, "roc_auc" : roc_score, "classifier" : classifier} 

        

    plt.show()

    return accuracies

    

sample_accuracies = plot_sklearn_models(X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test)
sample_accuracies
s = StandardScaler()

features_scaled = s.fit_transform(X_features)
from sklearn.decomposition import PCA, KernelPCA

import plotly_express as pe



string_labels = ((labels.values).copy()).reshape(-1, 1).astype("str")

string_labels[string_labels == '0'] = "not pulsar"

string_labels[string_labels == '1'] = "pulsar"



def get_lower_dim_space(algorithm, options, features, labels):

    transformer = algorithm(**options)

    latent_space = transformer.fit_transform(features)

    all_data     = np.concatenate([latent_space, labels.reshape(-1, 1)], axis=1)

    

    if latent_space.shape[1] == 2:

        return pd.DataFrame(all_data, columns=['x', 'y', 'label']) 

    else:

        return pd.DataFrame(all_data, columns=['x', 'y', 'z', 'label'])

        

df = get_lower_dim_space(KernelPCA, {'kernel':"linear", "n_components" : 2}, features_scaled, string_labels)

pe.scatter(df, "x", "y", "label")
df = get_lower_dim_space(KernelPCA, {'kernel':"linear", "n_components" : 3}, features_scaled, string_labels)

pe.scatter_3d(df, "x", "y", "z", "label")
from umap import UMAP

df = get_lower_dim_space(UMAP, { "n_components" : 2, 'n_neighbors':15}, features_scaled, string_labels)

pe.scatter(df, "x", "y", "label")
df = get_lower_dim_space(UMAP, { "n_components" : 3, "n_neighbors": 15}, features_scaled, string_labels)

pe.scatter_3d(df, "x", "y", "z", "label")