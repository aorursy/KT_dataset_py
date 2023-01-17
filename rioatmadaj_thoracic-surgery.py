%matplotlib inline 

import pandas as pd 

import numpy as np 



# Logistic Regression 

from sklearn.linear_model import LogisticRegression

# KNN 

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve



# Alias 

from pandas.core.series import Series

from pandas.core.frame import DataFrame

from typing import Dict, List

# Graph 

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap

import random 

plt.rcParams["figure.figsize"] = (20,10)

plt.rcParams["font.size"] = 15
def ml_model(clf, df: DataFrame, feature_cols: List[str], response_vector: str) -> dict:

    """

    This is a generic ML model engine

    :clf: given the ML classifiers

    :df: given the dataframe 

    :feature_cols: given the independent variables

    :response_vector: given the dependent variables 

    :return: a dictionary independent and dependent variables

    """

    random.seed(10000)

    X = df[feature_cols]

    y = df[response_vector]

    # Note: In R implementation use train_test_split becuase it relies on fewer assumptions for feature selection. 

    X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=random.randint(100,150)) 

    ml_clf = clf.fit(X_train,y_train)

    y_pred_class = ml_clf.predict(X_test)

    return {

        'X_train': X_train,

        'X_test': X_test,

        'y_train': y_train, 

        'y_test': y_test,

        'y_pred_class': y_pred_class

    }
def get_accuracy(y_pred_class: Series, y_test: Series) -> str:

    """

    This function will compute the prediction accuracy and rounded in 2 decimal points 

    :y_pred_class: given the prediction class 

    :y_test: given the dependent test vector 

    :return: a string of an accuracy score

    """

    return f"{accuracy_score(y_pred_class, y_test) * 100:.2f} %"
def calculate_cm(y_test: Series, y_pred_class: Series) -> Dict:

    """

    This function compute the confusion matrix given the actual response vector and predicted class vector

    :y_test: given the actual response vector    

    :y_pred_class: given the prediction class vector 

    :return: a dictionary of confusion matrix attributes

    """

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel() 

    total: int = tn + fp + fn + tp 

    return {

        'accuracy': f"{(tp+tn)/total:.2f}",

        'sensitifity': f"{tp/float(tp+fn):.2f}",

        'specificity': f"{tn/ float(tn+fp):.2f}", 

        'error_rate': f"{(fp+fn)/total:.2f}",

        'false_postive_rate': f"{fp/(tn+fp):.2f}"

    }
def plot_cm(y_test:Series, y_pred_class: Series):

    """

    This function will plot the confusion matrix using seaborn heatmaps 

    :y_test: given the actual response vector

    :y_pred_class: given the prediction class vector

    :return: 

    """

    sns.heatmap(confusion_matrix(y_test, y_pred_class), yticklabels=['dead','survived'], xticklabels=['dead','survived'])
surgery = pd.read_csv("../input/binary-classifier-data.csv")

surgery.index = np.arange(1,len(surgery) + 1 )
surgery.head(15)
surgery.describe() 
surgery.corr() 
surgery.cov() 
# Correlation Graphs 

sns.heatmap( surgery.corr() )

plt.title("Variable Correlations")
sns.distplot(surgery.y)

plt.title("Surgery Risk1")

plt.grid(True)
sns.boxplot(x=surgery.label, y=surgery.y)

plt.title("Surgery Risk1")
# Logistic Regression Graphs 

sns.lmplot(x='y', y='label', data=surgery, ci=None, aspect=3.5, logistic=True)

plt.ylabel("Risk after surgery")

plt.grid(True)
feature_cols: List[str] = ['y']

response_vector: str = 'label'

logreg = LogisticRegression(C=1e25)

ml_results: Dict = ml_model(clf=logreg, df=surgery, feature_cols=feature_cols, response_vector=response_vector)
surgery['yvar_logreg_prob'] =  logreg.predict_proba(surgery[feature_cols])[:,1]
sns.distplot(surgery.yvar_logreg_prob)

plt.title("Logistic Regression prediction of survival one year after surgery")

plt.xlabel("Logistic prediction of y variable")

plt.grid(True)
calculate_cm(ml_results.get('y_test'), ml_results.get("y_pred_class"))
plt.scatter(surgery.y, surgery.label)

plt.plot(surgery.y, surgery.yvar_logreg_prob, color='red')

plt.title("Using y variable")

plt.grid(True)
plot_cm(ml_results.get('y_test'), ml_results.get("y_pred_class"))
feature_cols: List[str] = ['x']

response_vector: str = 'label'

logreg = LogisticRegression(C=1e25)

ml_results: Dict = ml_model(clf=logreg, df=surgery, feature_cols=feature_cols, response_vector=response_vector)
calculate_cm(ml_results.get('y_test'), ml_results.get("y_pred_class"))
sns.distplot(surgery.x)

plt.title("Surgery Risk2")

plt.grid(True)
surgery['xvar_logreg_prob'] =  logreg.predict_proba(surgery[feature_cols])[:,1]
surgery[surgery.columns.tolist()[-2:]].plot(kind='hist')

plt.xlabel("Predicted probabilities of survival one year after surgery")

plt.grid(True)
feature_cols: List[str] = ['x','y']

response_vector: str = 'label'

logreg = LogisticRegression(C=1e25)

ml_results: Dict = ml_model(clf=logreg, df=surgery, feature_cols=feature_cols, response_vector=response_vector)
print(f"[+] Logistic regression intercept: {str(logreg.intercept_[0])}")

# Negative coefficients decreases the probability

print( '\n'.join( list( map(lambda x: f"[+] A 1 unit increase in {x[0]} associated with a \033[42m{x[-1]:.6f}\033[0m decrease in the log-odds of surgery survival", list( zip( feature_cols, logreg.coef_[0] ) ) ) ) )) 
calculate_cm(ml_results.get('y_test'), ml_results.get("y_pred_class"))
### Logistic Regressions with different K-folds

X_train, y_train, = ml_results.get('X_train'), ml_results.get('y_train')

k_folds_logreg: List[float] = []

for k_fold in range(2,151):

    k_folds_logreg.append( cross_val_score(LogisticRegression(C=1e25) , X_train, y_train, cv=k_fold, scoring='accuracy' ).mean() ) 
df_folds = pd.DataFrame( { 'K_Folds' : k_folds_logreg }, index=range(2,151) )

df_folds[ df_folds['K_Folds'] == df_folds.K_Folds.max() ]  
# Maximum accuracy, when K-folds = 117 

pd.DataFrame( { 'K_Folds' : k_folds_logreg }, index=range(2,151) ).plot() 

plt.title("Improving accuracy of Logistic Regression using cross validations")

plt.ylabel("Accuracy")

plt.xticks(range(0,155,5))

plt.grid(True) 

color_map = ListedColormap(["#F92C10", "#45F910"])

surgery.plot(kind='scatter', x='x', y='y', c='label', colormap=color_map)

plt.title("Survial after surgery")

plt.ylabel("Risk1")

plt.xlabel("Risk2")
# Find the best n_neighbor 

feature_cols = ['x', 'y']

response_vector = 'label'

pred_results: List = []

for neighbor in range(1,100):

    knn = KNeighborsClassifier(n_neighbors=neighbor)

    results: Dict = ml_model(clf=knn, df=surgery, feature_cols=feature_cols, response_vector=response_vector)

    pred_results.append(calculate_cm(results.get('y_test'), results.get("y_pred_class")))

    results = {}
neighbor_model = pd.DataFrame.from_dict(pred_results)

neighbor_model.index = range(1,100)

neighbor_model = neighbor_model.astype(float)
neighbor_model.astype(float).describe()
# Find n_neighbor with the lowest error rate 

print( f"[+] The lowest error rate:{ list(map(lambda x: str(x) , neighbor_model[ neighbor_model['error_rate'] == neighbor_model.error_rate.min() ].index.tolist() ))}")

# Find n_neighbor with the best prediction accuracy 

print( f"[+] The best prediction accuracy: { list(map(lambda x: str(x) , neighbor_model[ neighbor_model['accuracy'] == neighbor_model.accuracy.max() ].index.tolist() ))}")

# Find the best sensitivity score 

print( f"[+] The best prediction sensitivity: { list(map(lambda x: str(x) , neighbor_model[ neighbor_model['sensitifity'] == neighbor_model.sensitifity.max() ].index.tolist() ))}")

# Find the best specificity score 

print( f"[+] The best prediction specificity: { list(map(lambda x: str(x) , neighbor_model[ neighbor_model['specificity'] == neighbor_model.specificity.max() ].index.tolist() ))}")
# When the n_neighbors with the range of 3 to 36, the KNN algorithm has the highest prediction accuracy

neighbor_model.accuracy.plot() 

plt.title("Prediction accuracy using K Nearest Neighbor")

plt.xlabel("Number of neighbors")

plt.xticks(range(0,110,10))

plt.grid(True)
# Error Rate 

# When the n_neighbors with the range of 3 to 54, the KNN algorithm has the lowest misclasification rate

neighbor_model.error_rate.plot() 

plt.title("Misclassification Rate using K Nearest Neighbor")

plt.xlabel("Number of neighbors")

plt.xticks(range(0,110,10))

plt.grid(True)
# How often is the prediction correct when the actual value is positive

neighbor_model.sensitifity.plot() 

plt.title("Sensitifity Rate using K Nearest Neighbor")

plt.xlabel("Number of neighbors")

plt.xticks(range(0,110,10))

plt.grid(True)
# How often is the prediction correct when the actual value is positive

neighbor_model.specificity.plot() 

plt.title("Specificity Rate using K Nearest Neighbor")

plt.xlabel("Number of neighbors")

plt.xticks(range(0,110,10))

plt.grid(True)
# neighbor_model[neighbor_model.index == 54]

knn = KNeighborsClassifier(n_neighbors=54)

knn_predict: Dict = ml_model(clf=knn, df=surgery, feature_cols=feature_cols, response_vector=response_vector )

X_train, X_test, y_train, y_test, y_pred = knn_predict.get('X_train'), knn_predict.get('X_test'), knn_predict.get('y_train'), knn_predict.get('y_test'), knn_predict.get('y_pred_class')
# Confusion Matrix where n_neighbor = 54

calculate_cm(y_test,y_pred)
surgery['knn_pred_prob'] = knn.predict_proba(surgery[['x','y']])[:,1]

sns.distplot( pd.DataFrame( knn.predict_proba(X_test)[:,1] ) )

plt.title("KNN survival prediction one year after surgery")
color_map = ListedColormap(["#E8F910", "#3E91FA"])

surgery.plot(kind='scatter', x='y', y='knn_pred_prob', c='label', colormap=color_map)

plt.title("KNN classification (k=54): Survial after surgery")
knn_pred_proba = knn.predict_proba(X_test)[:,1]

logreg_pred_proba = logreg.predict_proba(X_test)[:,1]
pd.DataFrame( { "Prediction" : knn_pred_proba, "Actual": y_test }).hist(column='Prediction', by='Actual', sharex=True, sharey=True)
pd.DataFrame(data={ 'KNN':knn_pred_proba, 'Logistic Regression': logreg_pred_proba}).hist(sharex=True, sharey=True )
# Prediction accuracy KNN vs Logistic Regression 

ml_results: Dict = ml_model(clf=logreg, df=surgery, feature_cols=['x', 'y'], response_vector=response_vector)

logreg_accuracy: int = eval(calculate_cm(ml_results.get('y_test'), ml_results.get('y_pred_class') ).get('accuracy'))

knn_accuracy: int = eval(calculate_cm(y_test,y_pred).get('accuracy') )
for clf, attr in {'KNN Classifier': (knn_pred_proba, knn_accuracy) , 'Logistic Regression Classifier': (logreg_pred_proba, logreg_accuracy)}.items():

    pred_prob, accuracy = attr

    fp, tp, thresholds = roc_curve(y_test, pred_prob)

    roc_auc = auc(fp, tp)

    plt.plot(fp, tp, label=f"[+] Classifier: {clf}\n[+] AUC: {roc_auc:.3f}\n[+] Accuracy: {accuracy*100:.2f}%\n")



plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.xlabel("True Positive Rate")

plt.ylabel("False Positive Rate")

plt.xlim([-0.005, 1.0])

plt.ylim([0.0, 1.0])

plt.show() 
plot_cm(y_test=y_test,y_pred_class=y_pred)