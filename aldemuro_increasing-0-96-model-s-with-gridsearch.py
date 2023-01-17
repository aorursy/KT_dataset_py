import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.cross_validation import train_test_split

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.metrics import precision_score, recall_score, auc,roc_curve

from sklearn import model_selection



import warnings

warnings.filterwarnings('ignore')

data=pd.read_csv('../input/data.csv',encoding='ISO-8859-1')

data.head()
data.diagnosis = pd.factorize(data.diagnosis)[0]
data.head()
# y includes our labels and x includes our features

y = data.diagnosis                          # M or B 

list = ['Unnamed: 32','id','diagnosis']

x = data.drop(list,axis = 1 )

x.head()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

x_1 = x.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

x_1.head()
#correlation map after deleting similar column

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
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

    

    alg.fit(x_train, y_train)

    predicted = alg.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
#base model

tunealg = ensemble.GradientBoostingClassifier()

tunealg.fit(x_train, y_train)



print('BEFORE tuning Parameters: ', tunealg.get_params())

print("BEFORE tuning Training w/bin set score: {:.2f}". format(tunealg.score(x_train, y_train))) 

print("BEFORE tuning Test w/bin set score: {:.2f}". format(tunealg.score(x_test, y_test)))

print('-'*10)





#tune parameters. you can set the value parameter as much as you want as long as within it's requirement.

param_grid = {#'criterion': 'friedman_mse', 

              #'init': None, 

              'learning_rate': [0.1,0.2,0.3,0.4,0.5], 

              #'loss': 'deviance', 

              'max_depth': [1, 2,3,4],

              #'max_features': None, 

              #'max_leaf_nodes': None, 

              #'min_impurity_decrease': 0.0, 

              #'min_impurity_split': None, 

              #'min_samples_leaf': 1, 

              #'min_samples_split': 2, 

              #'min_weight_fraction_leaf': 0.0, 

              'n_estimators': [10,15,25,35,45,100], 

              #'presort': 'auto', 

              #'random_state': None, 

              #'subsample': 1.0, 

              #'verbose': 0, 

              'warm_start': [True, False]

              

             }





tune_model = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = 5)

tune_model.fit (x_train, y_train)



print('AFTER tuning Parameters: ', tune_model.best_params_)

print("AFTER tuning Training w/bin set score: {:.2f}". format(tune_model.score(x_train, y_train))) 

print("AFTER tuning Test w/bin set score: {:.2f}". format(tune_model.score(x_test, y_test)))

print('-'*10)