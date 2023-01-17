%matplotlib inline 

import pandas as pd 

import numpy as np 



# Classfiers 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

# Train, Test

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline



# Graphs

import seaborn as sns 

import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap

plt.rcParams['figure.figsize'] = (20,10)

plt.rcParams['font.size'] = 15



# Miscs

import re 

from typing import Dict, List

# Aliases

from pandas.core.series import Series

from pandas.core.frame import DataFrame

np.random.seed(10000) # initial seed
def format_to_percent(col: Series) -> Series:

    """

    Helper function to conver the series vector into percentage format 

    :col: given the series columns

    :return: series of percentage 

    """

    return f"{col * 100:.3f}%"
def remove_duplicate_feature_matrix(cols: List[str]) -> List[List]:

    """

    Helper fuctions to remove duplicate feature matrix 

    :cols: given the feature matrix columns 

    :return: a list of list of unique feature matrix 

    """

    unique_str_cols = sorted(set(list(map(lambda col: ' '.join(col), cols))))

    return list(map(lambda str_cols: str_cols.split(" "), unique_str_cols))
def get_random_feature_cols(cols: List[str]) -> List[str] :

    """

    Helper function to generate feature cols to aid in feature engineering

    :cols: a list of feature_matrix 

    :return: a list of random feature engineering

    """

    lower_bound: int = np.random.randint(1, 17) 

    upper_bound: int = np.random.randint(0,lower_bound)

    return cols[upper_bound:lower_bound]
def pred_accuracy(y_pred_class: Series, y_test: Series) -> float:

    """

    Helper function to calculate the accuracy of the predictions.

    Given the predection class and response vectors 

    :y_pred_class: given the prediction class 

    :y_test: given the response vector 

    :return: a float of prediction score 

    """

    if not y_pred_class and not y_test:

        raise AttributeError("Must supply parameters y_pred_class and y_test")

    

    return accuracy_score(y_pred_class, y_test)
def clf_model(clf, df: DataFrame, feature_matrix: List[str] , response_vector: str) -> Dict:

    """

    Helper function to fit ML model in order to find the best features 

    :clf: given the ML classfier 

    :df: given the dataframe 

    :feature_matrix: given the vector feature maxtrix

    :response_vector: given an index string of the response vector

    :return: a dictionary attributes of the ML predictions

    """

    

    X =  df[feature_matrix]

    y =  df[response_vector]

    

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=np.random.randint(100,150))

    clf_ml = clf.fit(X_train, y_train) 

    y_pred_class = clf_ml.predict(X_test)

    

    return {

        "X_train": X_train, 

        "X_test": X_test, 

        "y_train": y_train, 

        "y_test": y_test,

        "y_pred_class": y_pred_class

    }
def map_colors(**colors) -> List[str]:

    """

    Helper function to map hex colors

    :colors: kwargs color arguments

    :return: a list of map colors 

    """

    color_codes: List[str] = list(colors.values())

    

    if not "#" in ", ".join(color_codes):

        raise ValueError("Must be in hex format.")

        

    return color_codes
def map_to_bool(attr: Series) -> Series:

    """

    This function will map the given vector boolean representation into 1 = True, 0 = False

    :attr: given the vector boolean representation to be converted

    :return: a vector Series of 1 and 0 

    """

    return np.where(attr == "T", 1, 0 )
def map_col_names(cols: List[str]) -> List[str]:

    """

    Helper function to map column names 

    :cols: given the column names

    :return: 

    """

    return list(zip(range(len(cols)), cols))
def calculate_cm(y_test: Series, y_pred_class: Series) -> Dict:

    """

    Helper function to calculate the confusion matrix attributes:

        1. Specificity 

        2. Sensitivity 

        3. Error Rate 

        4. False Posite 

        5. Accuracy 

        

    :y_test: given the response vector 

    :y_pred: given the prediction class vector 

    :return: a dictionary of confusion matrix attributes 

    """

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel() 

    total: int = tn +  fp +  fn +  tp

    

    accuracy: float = (tp+tn)/float(total)

    error_rate: float = (fp+fn)/float(total)

    false_positive: float = (fp/float(tn+fp))

    sensitivity: float = tp/float(fn+tp)

    specificity: float = tn / float(tn+fp)

        

    return {

        "accuracy": accuracy, 

        "error_rate": error_rate, 

        "false_positive": false_positive, 

        "sensitivity": sensitivity, 

        "specificity": specificity, 

        

    }
# get column names 

raw_data: List[str] = open("../input/ThoraricSurgery.arff.txt",'r').read().split("\n") 

columns: List[str] = [ col.split(" ")[1] for col in raw_data if "attr" in col ]

# add missing column names 

col_names: [str] = [columns[0]] + ['PREP4', 'PREP5'] + columns[1:13] + ['AGE'] + [columns[-1]] 
surgery = pd.DataFrame.from_dict( [ x.split("\t") for x in raw_data[21:] ] )

surgery.columns = col_names 

surgery.index = np.arange(1, len(surgery) + 1 )

surgery.dropna(inplace=True) 



surgery['AGE'] = surgery['AGE'].astype(int)
surgery.head(25) 
pre_col: List[str] = [ col for col in col_names if "PRE" in col and col != "PRE14"] + [col_names[-1]]

map_pre_surgery = surgery[pre_col].astype(str).apply(map_to_bool) 

for col in pre_col:

    surgery[col] = map_pre_surgery[col]
# Map Diagnosis and Size of the original tumour

surgery['DGN'] = pd.factorize(surgery['DGN'])[0]

surgery['PRE14'] = pd.factorize(surgery['PRE14'])[0]
surgery.head(15)
surgery.describe() 
surgery.corr() 
sns.heatmap(surgery.corr() )
# Variable relationshipt

pd.plotting.scatter_matrix(surgery, figsize=(100,100))
surgery.cov()
sns.distplot( surgery.AGE.astype(int) )

plt.title('Surgery Age distributions')

plt.xticks(range(0,100,10))

plt.grid(True)
# Group by 1 year survial period 

surgery.groupby('Risk1Yr').agg(['mean'])['AGE'].plot()

plt.title("1 Year survival period")

plt.xlabel("1 Year Survail period")

plt.ylabel("Mean Age")

plt.grid(True)
relation_two = ListedColormap( map_colors(col_1="#1050F9", col_2="#F9F510") ) 

surgery.plot(kind='scatter', x='AGE', y='DGN', c='Risk1Yr', colormap=relation_two)

plt.xlabel("Age")

plt.xticks(range(0,100,10))
relation_two = ListedColormap( map_colors(col_1="#10F9F5", col_2="#F92910") ) 

surgery.plot(kind='scatter', x='AGE', y='PRE14', c='Risk1Yr', colormap=relation_two) # Size of original tumours 
# generate pseudo random feature matrix 

random_cols: List = []

for index in range(1,2000):

    random_cols.append( get_random_feature_cols(surgery.columns.tolist() ) )

unique_feature_matrix = remove_duplicate_feature_matrix(random_cols)
# Tunned with different features matrix and find the best fit 

tuned_results: List = []

for col in random_cols:

    logreg = LogisticRegression(C=1e25)

    results: Dict = clf_model(logreg, surgery, col, 'Risk1Yr')

    y_test, y_pred_class = results.get("y_test") , results.get("y_pred_class")

    cm_calculation : Dict = calculate_cm(y_test, y_pred_class)

    cm_calculation['feature_matrix'] = col 

    cm_calculation['auc_score'] = roc_auc_score(results.get('y_test'), results.get('y_pred_class'))

    tuned_results.append(cm_calculation)
tuned_logreg = pd.DataFrame.from_dict( tuned_results )

tuned_logreg['number_features'] = tuned_logreg['feature_matrix'].apply(lambda x : len(x))
tuned_logreg.groupby('error_rate')['accuracy'].plot(kind='hist') 

plt.title("ERROR RATE: Auto tuned feature matrix")

plt.legend()

plt.xlim([0.7, 1.0])

plt.grid(True)

plt.show() 
# Pick 1 

tuned_logreg[ tuned_logreg['auc_score'] == tuned_logreg['auc_score'].astype(float).max() ]
tuned_logreg[ (tuned_logreg['error_rate'] == tuned_logreg.error_rate.min() ) & (tuned_logreg['sensitivity'] > 0.0) ]
# Find the best fit, by error rate 

list_tuned_logreg: List[List] = tuned_logreg[ (tuned_logreg['error_rate'] == tuned_logreg.error_rate.min() ) & (tuned_logreg['sensitivity'] > 0.0) ]['feature_matrix'].tolist() 
# another possible fit 

list_tuned_logreg.extend( tuned_logreg[ (tuned_logreg['error_rate'] < 0.5 ) & (tuned_logreg['sensitivity'] > 0.1) ]['feature_matrix'].tolist() )   

list_tuned_logreg.extend(tuned_logreg[ tuned_logreg['auc_score'] == tuned_logreg['auc_score'].astype(float).max() ]['feature_matrix'].tolist())
# Find the best model 

for index,col in enumerate(list_tuned_logreg, 1):

    print(f"{index}. {col}")

    logreg=LogisticRegression(C=1e25)

    clf_results: Dict = clf_model(clf=logreg,df=surgery,feature_matrix=col, response_vector='Risk1Yr')

    y_pred_class, y_test, X_test = clf_results.get("y_pred_class"), clf_results.get("y_test"), clf_results.get("X_test")

    accuracy_prediction: float = accuracy_score(y_pred_class, y_test)



    pred_probability = logreg.predict_proba(X_test)[:,1]

    fp, tp, thresholds = roc_curve(y_test, pred_probability)

    roc_auc = auc(fp, tp) 

    plt.plot(fp, tp, label=f"[+]FEATURE MATRIX: {col}\n[+] AUC: {roc_auc:.3f}\n[+] Accuracy Prediction: {accuracy_prediction * 100:.2f} %\n")



plt.title("Nth Feature Matrix comparission")

plt.legend(loc='upper left')

plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.show()
# use the best model

cv_results: List = []

feature_matrix = ['DGN', 'PREP4', 'PREP5', 'PRE6', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE14', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32']

X = surgery[feature_matrix]

y = surgery['Risk1Yr']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=123)

for k_fold in range(2,150):

    cv_results.append( cross_val_score(LogisticRegression(C=1e25),X_train, y_train, cv=k_fold, scoring='accuracy').mean() ) 
highest_kfolds: tuple = max( list( zip(range(2,150), cv_results) ) )

print(f"[+] Logistic Regression has an accuracy of {highest_kfolds[-1] * 100: .3f} % with number of K-folds equals to {highest_kfolds[0]}")
plt.plot(cv_results)

plt.xticks(range(0,150,10))

plt.xlabel("Number of K-folds")

plt.ylabel("Prediction Accuracy")
pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=1e15))

scaled_accuracy: float = cross_val_score(pipe, X_train, y_train, cv=149, scoring='accuracy').mean() 
print( f"[+] Scaled Accuracy: {round( scaled_accuracy * 100, 3 )} %" )
null_accuracy: float = round(( float(y_test.value_counts().head(1)) / float( len(y_test) ) )  * 100, 3)

# Compare with Null Accuracy 

print( f"[+] NULL Accuracy:  {null_accuracy} %")