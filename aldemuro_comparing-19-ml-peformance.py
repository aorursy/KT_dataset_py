#sklearn

from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve

from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



#load package

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#from math import sqrt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
#upload the data

df=pd.read_csv('../input/mushrooms.csv',encoding='ISO-8859-1')

df.head()
#check missing value

df.isnull().sum()
df.select_dtypes(include=['object']).head()
#Check the data type

df.dtypes
#it is shown in previous section that all of the columns type are Object (string), thus i has to be encode to

#integer type



from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

for col in df.columns:

    df[col] = enc.fit_transform(df[col])

 

df.head()
df.dtypes
#spliting the data

x = df.drop("class", axis=1)

y = df["class"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)
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

    linear_model. RidgeClassifierCV(),

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

    

    ]

MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)





row_index = 0

for alg in MLA:

    

    

    predicted = alg.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    #roc_auc_rf = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    #MLA_compare.loc[row_index, 'Square root mean error'] = sqrt(mean_squared_error(y_test,predicted))

    MLA_compare.loc[row_index, 'MLA Accuracy'] = round(alg.score(x_train, y_train), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Accuracy'], ascending = False, inplace = True)    

MLA_compare
index = 1

for alg in MLA:

    

    

    predicted = alg.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
