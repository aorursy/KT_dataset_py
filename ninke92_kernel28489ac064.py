from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error

import pandas as pd
data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

Num_val = {'Yes':1, 'No':0}

data['Attrition'] = data['Attrition'].apply(lambda x: Num_val[x])

import seaborn as sns

from matplotlib import pyplot as plt

corr = data.corr()

plt.figure(figsize=(17,17))



ax = sns.heatmap(

    corr, vmin=-1, vmax = 1, center = 0, cmap=sns.diverging_palette(2,220,n=200),

    square = True)

ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation = 45,

    horizontalalignment='right')
data = data.drop([], axis = 1)
import seaborn as sns

from matplotlib import pyplot as plt

corr = data.corr()

plt.figure(figsize=(17,17))



ax = sns.heatmap(corr, vmin=-1, vmax = 1, center = 0, cmap=sns.diverging_palette(2,220,n=200),square = True)

ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation = 45,

    horizontalalignment='right')
dependent = data.Attrition

data = data.drop(['Attrition'], axis = 1)
data.describe()
print('nr columns: '+str(len(data.columns)))

print('nr rows:' + str(len(data)))
categorical_data = data.select_dtypes(include='object')

categorical_data
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression



# Function for comparing different approaches

def score_dataset_random_forest(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



def score_dataset_XG_boost(X_train, X_valid, y_train, y_valid):

    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

    model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



def score_dataset_logistic(X_train, X_valid, y_train, y_valid):

    model = LogisticRegression(verbose = 3)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
from sklearn.model_selection import train_test_split



X_train, X_valid_full, y_train, y_valid_full = train_test_split(data, dependent, train_size=0.8, test_size=0.2,

                                                                random_state=0)



X_valid, X_test, y_valid, y_test = train_test_split(X_valid_full, y_valid_full, train_size=0.5, test_size=0.5,

                                                                random_state=0)
categorical_cols = [cname for cname in X_train.columns if

                    X_train[cname].nunique() < 10 and 

                    X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if 

                X_train[cname].dtype in ['int64', 'float64']]



numerical_transformer = SimpleImputer(strategy = 'constant')



categorical_transformer = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



model = LogisticRegression(max_iter = 1000, verbose = 3)



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                             ('model', model)

                             ])



my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_valid, preds)

print('MAE: ', score)
my_pipeline.score(X_train,y_train)
my_pipeline.score(X_test,y_test)
from sklearn.metrics import classification_report



predictions=my_pipeline.predict(X_test)

print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0])

                                  , range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")    

        plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')#Generating the Confusion Matrix



    

plt.figure()

cm = np.array([[252, 1], [31, 10]])

plot_confusion_matrix(confusion_matrix(y_test,predictions), 

                      classes=[0,1], normalize=True, title='Normalized Confusion Matrix')

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

import pylab as pl



y_roc = np.array(y_test)

fpr, tpr, thresholds = roc_curve(y_roc, my_pipeline.decision_function(X_test))

roc_auc = auc(fpr, tpr)

pl.clf()

pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

pl.plot([0, 1], [0, 1], 'k--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.0])

pl.xlabel('False Positive Rate')

pl.ylabel('True Positive Rate')

pl.legend(loc="lower right")

pl.show() # Output shown below