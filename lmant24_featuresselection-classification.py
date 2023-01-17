!pip install pydotplus
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



import seaborn as sns



from sklearn.preprocessing import StandardScaler



# Load libraries

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



# PCA

from sklearn import decomposition



# Classification

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

import matplotlib.table as tbl



from sklearn import svm, datasets

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



import math #for NaN 

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv',  encoding ='latin1')



# Our last column is just an error in the data reading. Dropping it

df = df.drop(['id', 'Unnamed: 32'], axis=1)



df.head(5)
df.shape
df.describe()
print("Features del Dataset")

df.info()
print("Null values:")

df.isnull().sum()
Y = np.array(df['diagnosis'])



unique, counts = np.unique(Y, return_counts=True)

diagnosis_count = dict(zip(unique, counts))

labels = diagnosis_count.keys()

sizes = diagnosis_count.values()



fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=( "#008000", "#B30000"))

ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

ax.set_title('Diagnosis')



plt.show()

print(diagnosis_count)
sns.set(font_scale=0.8)



correlation = df.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, vmax=1, vmin=-1, square=True,annot=True,cmap='RdYlGn', fmt='.1f')



plt.title('Correlation between different features')

plt.show()
features = [f for f in df.columns if f not in ['diagnosis']]



i = 0

t0 = df[df['diagnosis'] == 'M']

t1 = df[df['diagnosis'] == 'B']



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(6,5,figsize=(16,24))



for feature in features:

    i += 1

    plt.subplot(6,5,i)

    sns.distplot (t0[feature], label="Malignant", color="#B30000")

    sns.distplot (t1[feature], label="Benign", color = "#008000")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=8)

plt.show()
import pandas as pd

from collections import defaultdict

import re

import matplotlib.pyplot as plt

import seaborn as sns



# TODO: add pathvariable



# main method: clean

sns.set(font_scale=0.8)



class DataCleaner:

    def __init__(self, df, filterStrongCorr = False, classlabel="diagnosis"):

        self.df = df

        self.classlabel_ = classlabel

        self.id = df.index

        self.filterStrongCorr = filterStrongCorr

        

        self.signalSelected = []

        

        self.minDistinct = 100

        self.valueHighCorr = 0.95

    

    def clean(self):

        candidates = self.df.keys()

        candidates = self.filterDiscrete(df, candidates)

        

        for signal in candidates:

            if not signal in self.signalSelected:

                self.signalSelected.append(signal)

       

        if self.filterStrongCorr == True:

            signalToDrop = self.filterCorrSignal()

            for signal in signalToDrop:

                self.signalSelected.remove(signal)

        

        print("Filtered correlation matrix:")

        df_out = self.df[self.signalSelected]

        correlation = df_out.corr()

        plt.figure(figsize=(15,15))

        sns.heatmap(correlation, vmax=1, vmin=-1, square=True,annot=True,cmap='RdYlGn', fmt='.1f')        

        

        return self.df[self.signalSelected]

    

    def filterDiscrete (self, df, features):

        candidates = []

        candidates.append(self.classlabel_)

        

        for feature in features: 

            distinct = len(set(df[feature]))

            if distinct >= self.minDistinct:

                candidates.append(feature)

        return candidates



    def filterCorrSignal(self):

        

        correlation = df.corr()

        #plt.figure(figsize=(15,15))

        #sns.heatmap(correlation, vmax=1, vmin=-1, square=True,annot=True,cmap='RdYlGn', fmt='.1f')



        #plt.title('Correlation between different features')

        #plt.show()



        dropForCorr = []



        for index, row in correlation.iterrows():

            if row.name not in dropForCorr:

                for i in row.index:

                    if(row[i] > self.valueHighCorr):

                        if (row.name != i and i not in dropForCorr):

                            dropForCorr.append(i)

        

        print("Eliminate le seguenti features per correlazione: " + str(dropForCorr))

        return dropForCorr
cleaner = DataCleaner(df, filterStrongCorr=True)

df1  = cleaner.clean()

print("Features after cleaning:" + str(df1.columns))
import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from numpy import matlib

class KBestFeaturesSelection:

    def __init__(self, df, classlabel="diagnosis"):

        self.df = df

        self.classlabel_ = classlabel

        

        #self.alpha = alpha

    

    def select(self):

        X = self.df.drop([self.classlabel_], axis = 1) #independent columns

        

        y = pd.factorize(self.df[self.classlabel_])[0].astype(np.uint16)



        #apply SelectKBest class to extract top 10 best features

        bestfeatures = SelectKBest(score_func=chi2, k="all")

        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)

        dfcolumns = pd.DataFrame(X.columns)

        #concat two dataframes for better visualization 

        featureScores = pd.concat([dfcolumns,dfscores],axis=1)

        featureScores.columns = ['Specs','Score']  #naming the dataframe columns

        

        #print(np.cumsum(featureScores.nlargest(19,'Score')["Score"]))

        plt.plot(range(1,20),np.cumsum(featureScores.nlargest(19,'Score')["Score"]))

        plt.xticks(range(1,20))



        

        

        kneeIndex, kneeValue = self.findKnee(np.cumsum(featureScores.nlargest(19,'Score')["Score"]).values)

        plt.axhline(y = kneeValue, color='k', linestyle='--', label = 'Knee value')

        plt.axvline(x = kneeIndex, color='g', linestyle='--', label = 'Knee')

        plt.show()

        

        featureSelected = featureScores.nlargest(kneeIndex,'Score')["Specs"].values

        

        #featureSelected = featureSelected.put(1, "diagnosis")

        df_out = self.df[list(featureSelected)]

        df_out[self.classlabel_] = self.df[self.classlabel_]

        

        print("First three important features and their scores")

        print(featureScores.nlargest(5,'Score'))  #print 19 best features

        

        return df_out

        

    def findKnee (self, values):

        #get coordinates of all the points

        nPoints = len(values)

        allCoord = np.vstack((range(nPoints), values)).T

        #np.array([range(nPoints), values])



        # get the first point

        firstPoint = allCoord[0]

        # get vector between first and last point - this is the line

        lineVec = allCoord[-1] - allCoord[0]

        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))



        # find the distance from each point to the line:

        # vector between all points and first point

        vecFromFirst = allCoord - firstPoint





        scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)

        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)

        vecToLine = vecFromFirst - vecFromFirstParallel



        # distance to line is the norm of vecToLine

        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))



        # knee/elbow is the point with max distance value

        idxOfBestPoint = np.argmax(distToLine)



        print("Knee value: " + str(values[idxOfBestPoint]))

        return idxOfBestPoint+1, values[idxOfBestPoint]
import warnings

warnings.filterwarnings("ignore")



selector = KBestFeaturesSelection(df1)

df2_1 = selector.select()
import pandas as pd

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt



class RFFeaturesSelection:

    def __init__(self, df, classlabel="diagnosis"):

        self.df = df

        self.classlabel_ = classlabel

        self.feat_importances = []

        

    def getImportance(self):

        X = self.df.drop([self.classlabel_], axis = 1) #independent columns

        

        y = pd.factorize(self.df[self.classlabel_])[0].astype(np.uint16)

        

        model = ExtraTreesClassifier()

        model.fit(X,y)

        #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

        

        #plot graph of feature importances for better visualization

        feat_importances = pd.Series(model.feature_importances_, index=X.columns)

        feat_importances.nlargest(len(self.df.columns)-2).plot(kind='barh')

        #feat_importances.plot(kind='barh')



        

        #print(type(feat_importances.nlargest(len(self.df.columns)-2).keys()[:5]))

        self.feat_importances = list(feat_importances.nlargest(len(self.df.columns)-2).keys())



        plt.show()

    

    def select(self, k):

        if self.feat_importances:

            df_out = self.df[self.feat_importances[0:k]]

            df_out[self.classlabel_] = self.df[self.classlabel_]

            return df_out
selector2 = RFFeaturesSelection(df1)

selector2.getImportance()

df2_2 = selector2.select(10)
class PerformanceEvaluator:

    def f1score(self, p,r):

        return 0 if p == 0 or r == 0 else (2 * p * r) / (p + r)



    def performance(self, clf1, X, Y , Id, name=None, userScale = False, plot=True):

        id_sequence = []   

        y_pred_all = []

        y_true_all = []

        

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, Y):

            clf = clf1



            if userScale == True:

                scaler = StandardScaler()

                scaler.fit(X[train_index])

                X[train_index] = scaler.transform(X[train_index])



            clf.fit(X[train_index], Y[train_index])



            if userScale == True:

                X[test_index] = scaler.transform(X[test_index])





            y_pred = clf.predict(X[test_index])

            y_true = Y[test_index]



            y_pred_all += list(y_pred)

            y_true_all += list(y_true)

            id_sequence += list(Id[test_index])



        C = confusion_matrix (y_true_all, y_pred_all) 

        labels = [ str(s) for s in sorted(list(set(y_true_all)))]

        accuracy = round(float(C.diagonal().sum()) /C.sum(),4)





        metrics = {'precision': {}, 'recall': {}, 'f1-score': {}, 'accuracy': accuracy}

        stats = {"M" : {}, "B" : {}}





        for i,label in enumerate(labels): 

            try:

                p = float(C[i][i]) / C.transpose()[i].sum()

                if math.isnan(p):

                    p=0

            except:

                p = 0

            try:

                r = float(C[i][i]) / C[i].sum()

                if math.isnan(r):

                    r=0

            except:

                r = 0



            stats[label]['precision'] = round(p,4)

            stats[label]['recall'] = round(r,4)

            stats[label]["f1-score"] = round(self.f1score(p,r),4)



        if (plot):

            

            fig, ax =plt.subplots(1,2, figsize=(7,4))

            ax[0].set_title('Matrice di correlazione')

            ax[1].set_title('Performance')

            

            df_cm = pd.DataFrame(C, index = [i for i in labels], columns = [i for i in labels])

            sns.heatmap(df_cm, annot=True, ax=ax[0], cmap="Blues", fmt="d", cbar_kws={"orientation": "horizontal"})

            

            df_cm = pd.DataFrame(stats)

            sns.heatmap(df_cm, annot=True, ax=ax[1], cbar=False, cmap="RdYlGn", vmin = 0, vmax = 1)



            fig.tight_layout()

            fig.show()



            print ('Accuracy: ' + str(accuracy) + ' %')

        return stats
import numpy as np

from sklearn import svm

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from itertools import product

from sklearn.model_selection import learning_curve



class SVMClassifier:

    def __init__(self, df, evaluator):



        self.Xnp = df.drop("diagnosis", axis = 1).values

        self.Ynp = df["diagnosis"]

        self.Id = df.index

        

        #self.columns = columns

        self.evaluator = evaluator

        

        self.svm_params = {'C': 10**np.linspace(-3,3,8),

            'random_state': [42],

            'kernel': ['linear', 'poly','rbf','sigmoid'],

            'degree': [2],

            'max_iter': [50000],

            'class_weight':[{"M":1, "B":10}]}

    

        self.done = False

        

    def paramTuning(self):

        clf_svm=svm.SVC()

        self.clf=GridSearchCV(clf_svm,self.svm_params, verbose = 1, scoring = "f1_micro",cv= 10)

        self.clf.fit(self.Xnp, self.Ynp)

        self.done = True

        

    def bestPerformances(self):

        if not self.done:

            self.paramTuning()

        print(self.clf.best_estimator_,)

        

        #----------------------------

        # learning_curve

        #----------------------------

        

        stats = self.evaluator.performance(self.clf.best_estimator_, self.Xnp, self.Ynp, self.Id, "SVM", userScale=True, plot=True)

        return stats

    

    







evaluator = PerformanceEvaluator()

clf = SVMClassifier(df2_1, evaluator)

stats = clf.bestPerformances()
evaluator = PerformanceEvaluator()

clf = SVMClassifier(df2_2, evaluator)

stats = clf.bestPerformances()
import numpy as np

from sklearn.tree import DecisionTreeClassifier

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from itertools import product



from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

from collections import defaultdict



class DTClassifier:

    def __init__(self, df, evaluator):

        self.df = df

        self.Xnp = df.drop("diagnosis", axis = 1).values

        self.Ynp = df["diagnosis"]

        self.Id = df.index

        

        #self.columns = columns

        self.evaluator = evaluator

        

        self.dtree_parameters={'min_impurity_decrease': np.linspace(0,.5,11), 

            'min_samples_split' : range(10,200,20),

            'max_depth': range(1,10,2), 

            'min_samples_leaf':range(1,20,2),

            'max_leaf_nodes': range(10,110,10),

            'random_state': [42]}

    

        self.done = False

        

    def paramTuning(self):

        clf_dt=DecisionTreeClassifier()

        self.clf = GridSearchCV(clf_dt, self.dtree_parameters, verbose = 1, scoring = "f1_micro", n_jobs=12, cv= 2)

        self.clf.fit(self.Xnp, self.Ynp)

        self.done = True

        

    def bestPerformances(self):

        if not self.done:

            self.paramTuning()

        

        stats = self.evaluator.performance(self.clf.best_estimator_, self.Xnp, self.Ynp, self.Id, "Decision Tree", userScale=True, plot=True)

        return stats
evaluator = PerformanceEvaluator()

clf = DTClassifier(df2_1, evaluator)

stats = clf.bestPerformances()
import pydotplus



feature_cols = df2_1.columns

feature_cols = feature_cols.drop("diagnosis")

decoder = np.vectorize(lambda x: x.encode('UTF-8'))

cols = decoder(feature_cols)



dot_data = StringIO()

export_graphviz(clf.clf.best_estimator_, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True, class_names=['Benign', 'Malign'], feature_names = cols)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  



nodes = graph.get_node_list()



colors =  ('darkolivegreen2', 'indianred2')



for node in nodes:

    if node.get_name() not in ('node', 'edge'):

        values = clf.clf.best_estimator_.tree_.value[int(node.get_name())][0]        

        arr = np.array(values)



        result = np.where(arr == np.amax(arr))[0]





        if result[0] == [0]:

            node.set_fillcolor(colors[0])

        elif result[0] == [1]:

            node.set_fillcolor(colors[1])



Image(graph.create_png())

evaluator = PerformanceEvaluator()

clf = DTClassifier(df2_2, evaluator)

stats = clf.bestPerformances()
feature_cols = df2_2.columns

feature_cols = feature_cols.drop("diagnosis")

decoder = np.vectorize(lambda x: x.encode('UTF-8'))

cols = decoder(feature_cols)



dot_data = StringIO()

export_graphviz(clf.clf.best_estimator_, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True, class_names=['Benign', 'Malign'], feature_names = cols)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  



nodes = graph.get_node_list()



colors =  ('darkolivegreen2', 'indianred2')



for node in nodes:

    if node.get_name() not in ('node', 'edge'):

        values = clf.clf.best_estimator_.tree_.value[int(node.get_name())][0]        

        arr = np.array(values)



        result = np.where(arr == np.amax(arr))[0]





        if result[0] == [0]:

            node.set_fillcolor(colors[0])

        elif result[0] == [1]:

            node.set_fillcolor(colors[1])



Image(graph.create_png())

import numpy as np

from sklearn.ensemble import RandomForestClassifier 

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from itertools import product



class RFClassifier:

    def __init__(self, df, evaluator):



        self.Xnp = df.drop("diagnosis", axis = 1).values

        self.Ynp = df["diagnosis"]

        self.Id = df.index

        

        #self.columns = columns

        self.evaluator = evaluator

        

        self.rf_params = {'n_estimators': [200, 600, 1000, 1400, 1600],

                          'max_features': ['auto', 'sqrt'],

                          'max_depth': range(1,12),

                          'min_samples_split': [2, 10, 30],

                          'min_samples_leaf': [1, 2, 4],

                          'bootstrap': [True, False],

                          'random_state' : [42]}

    

        self.done = False

        

    def paramTuning(self):

        clf_rf=RandomForestClassifier()

        self.clf=GridSearchCV(clf_rf, self.rf_params, verbose = 1, scoring = "f1_micro", n_jobs=12, cv= 2)

        self.clf.fit(self.Xnp, self.Ynp)

        self.done = True

        

    def bestPerformances(self):

        if not self.done:

            self.paramTuning()

        

        stats = self.evaluator.performance(self.clf.best_estimator_, self.Xnp, self.Ynp, self.Id, "RF", userScale=True, plot=True)

        return stats
evaluator = PerformanceEvaluator()

clf = RFClassifier(df2_1, evaluator)

stats = clf.bestPerformances()
evaluator = PerformanceEvaluator()

clf = RFClassifier(df2_2, evaluator)

stats = clf.bestPerformances()
def findKnee (values):

    nPoints = len(values)

    allCoord = np.vstack((range(nPoints), values)).T



    firstPoint = allCoord[0]



    lineVec = allCoord[-1] - allCoord[0]

    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))



    vecFromFirst = allCoord - firstPoint



    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)

    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)

    vecToLine = vecFromFirst - vecFromFirstParallel



    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))



    idxOfBestPoint = np.argmax(distToLine)



    return idxOfBestPoint+1, values[idxOfBestPoint]
X = df.drop("diagnosis", axis = 1).values

Y = df["diagnosis"]



# Standardize: standardize dataset (mean 0 and variance 1)

scal = StandardScaler()

X_std = scal.fit_transform(X)



#PCA

df_total_scaled = pd.DataFrame(X_std)

pca = decomposition.PCA().fit(df_total_scaled)
cumsum = np.zeros(len(df.columns)-1)



cumsum += np.cumsum(pca.explained_variance_ratio_)*100

d = [n for n in range(1,len(cumsum)+1)]



kneeIndex, kneeValue = findKnee(cumsum)



plt.figure(figsize=(15, 7))

plt.plot(d,cumsum, color = 'red',label='cumulative explained variance')

plt.title('Cumulative Explained Variance as a Function of the Number of Components')

plt.ylabel('Cumulative Explained variance')

plt.xlabel('Principal components')

plt.axhline(y = kneeValue, color='k', linestyle='--', label = 'Knee value')

plt.bar(x=d,height=pca.explained_variance_ratio_*100)

plt.xticks(range(1,len(pca.explained_variance_ratio_)+1))

plt.axvline(x = kneeIndex, color='g', linestyle='--', label = 'Knee')

plt.legend(loc='best')

plt.show()
class PCA_PerformanceEvaluator:

    def f1score(self, p,r):

        return 0 if p == 0 or r == 0 else (2 * p * r) / (p + r)



    def performance(self, clf1, X, Y , Id, name=None, userScale = False, plot=True):

        id_sequence = []   

        y_pred_all = []

        y_true_all = []

        

        scaler = StandardScaler()

        scaler.fit(X)

        X1 = scaler.transform(X)

        pca = decomposition.PCA(n_components=7)

        pca.fit(X1)

        

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, Y):

            clf = clf1



            X_pca_train=[]

            X_pca_test=[]

            

            if userScale == True:

                

                scaler = StandardScaler()

                scaler.fit(X[train_index])

                X[train_index] = scaler.transform(X[train_index])

                

  

                X_pca_train = pca.transform(X[train_index])

                



            clf.fit(X_pca_train, Y[train_index])



            if userScale == True:

                X[test_index] = scaler.transform(X[test_index])

                X_pca_test = pca.transform(X[test_index])



            y_pred = clf.predict(X_pca_test)

            y_true = Y[test_index]



            y_pred_all += list(y_pred)

            y_true_all += list(y_true)

            id_sequence += list(Id[test_index])



        C = confusion_matrix (y_true_all, y_pred_all) 

        labels = [ str(s) for s in sorted(list(set(y_true_all)))]

        accuracy = round(float(C.diagonal().sum()) /C.sum(),4)





        metrics = {'precision': {}, 'recall': {}, 'f1-score': {}, 'accuracy': accuracy}

        stats = {"M" : {}, "B" : {}}





        for i,label in enumerate(labels): 

            try:

                p = float(C[i][i]) / C.transpose()[i].sum()

                if math.isnan(p):

                    p=0

            except:

                p = 0

            try:

                r = float(C[i][i]) / C[i].sum()

                if math.isnan(r):

                    r=0

            except:

                r = 0



            stats[label]['precision'] = round(p,4)

            stats[label]['recall'] = round(r,4)

            stats[label]["f1-score"] = round(self.f1score(p,r),4)



        if (plot):

            

            fig, ax =plt.subplots(1,2, figsize=(7,4))

            ax[0].set_title('Matrice di correlazione')

            ax[1].set_title('Performance')

            

            df_cm = pd.DataFrame(C, index = [i for i in labels], columns = [i for i in labels])

            sns.heatmap(df_cm, annot=True, ax=ax[0], cmap="Blues", fmt="d", cbar_kws={"orientation": "horizontal"})

            

            df_cm = pd.DataFrame(stats)

            sns.heatmap(df_cm, annot=True, ax=ax[1], cbar=False, cmap="RdYlGn", vmin = 0, vmax = 1)



            fig.tight_layout()

            fig.show()



            print ('Accuracy: ' + str(accuracy) + ' %')

        return stats
evaluator2 = PCA_PerformanceEvaluator()

clf = RFClassifier(df1, evaluator2)

stats = clf.bestPerformances()
import warnings

warnings.filterwarnings("ignore")



evaluator3 = PCA_PerformanceEvaluator()

clf = SVMClassifier(df1, evaluator3)

stats = clf.bestPerformances()
evaluator3 = PCA_PerformanceEvaluator()

clf = DTClassifier(df1, evaluator3)

stats = clf.bestPerformances()
""" plt.xlabel("Training examples")

        plt.ylabel("Score")

        train_sizes, train_scores, test_scores = learning_curve(self.clf.best_estimator_, self.Xnp, self.Ynp, train_sizes=np.linspace(0.1, 1, 25), cv=10, scoring = "f1_micro")

        train_scores_mean = np.mean(train_scores, axis=1)

        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)

        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()



        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                         train_scores_mean + train_scores_std, alpha=0.1,

                         color="r")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

                 label="Training score")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

                 label="Cross-validation score")



        plt.legend(loc="best")"""