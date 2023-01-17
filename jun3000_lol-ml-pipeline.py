import pandas as pd
import numpy as np
import seaborn as sb
data = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
data.head()
data = data.drop(columns=['gameId','redWardsPlaced', 'redWardsDestroyed', 'redFirstBlood', 'redKills',
       'redDeaths', 'redAssists', 'redEliteMonsters', 'redDragons',
       'redHeralds', 'redTowersDestroyed', 'redTotalGold',
       'redTotalExperience', 'redTotalMinionsKilled',
       'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff','redAvgLevel',
       'redCSPerMin', 'redGoldPerMin']) #remove game ID
data = data.rename(columns={'blueWins':'target'})
data.head()
data2 = data.drop(columns=['target'])
pearsoncorr = data2.corr(method='pearson')
sb.heatmap(pearsoncorr)
#change 0 to blue win, 1 to red win
data['target'].replace([0,1],[1,0],inplace=True)
data.head()
data['target'].value_counts()
#0 = blue wins
#1 = red wins
#data set is pretty balanced in terms of win and loses for each team
print(data.dtypes)
dataTypeSeries = data.columns.to_series().groupby(data.dtypes).groups #set df.types as variable dataTypeSeries
 
print('Data type of each column of Dataframe :')
print(dataTypeSeries)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn import linear_model, decomposition, datasets
from sklearn.preprocessing import scale
#Pipleine for imputing, OHE, preprocessing
numeric_features = ['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',
       'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
       'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueTotalGold',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueAvgLevel', 'blueCSPerMin', 'blueGoldPerMin']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

#categorical_features = []

#categorical_transformer = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])
        #('cat', categorical_transformer, categorical_features)])
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
classifiers = [
    #LogisticRegression(),
    #DecisionTreeClassifier(),
    RandomForestClassifier(),
    #AdaBoostClassifier(),
    #XGBClassifier(),
    #SVC(kernel='linear', 
            #class_weight='balanced', # penalize
            #probability=True),
    #GaussianNB()
    #MLPClassifier(),
    #KNeighborsClassifier()
]
clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators = 100))),
                        ('clf', None)])
# y are the values we want to predict
y = np.array(data['target'])
# Remove the labels from the features
# axis 1 refers to the columns
X = data.drop('target', axis = 1)
# Saving feature names for later use
X_list = list(X.columns)
X_list
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
def get_transformer_feature_names(columnTransformer):
  output_features = []

  for name, pipe, features in columnTransformer.transformers_:
    if name!='remainder':
      for i in pipe:
        trans_features = []
        if hasattr(i, 'categories_'):
          trans_features.extend(i.get_feature_names(features))
        else:
          trans_features = features
      output_features.extend(trans_features)

  return np.array(output_features)

roc_things = []
precision_recall_things = []
for classifier in classifiers:
    clf.set_params(clf=classifier).fit(X_train, y_train)
    classifier_name = classifier.__class__.__name__
    print(str(classifier))
    print('~Model Score: %.3f' % clf.score(X_test, y_test))

    y_score = clf.predict_proba(X_test)[:,1]

    y_pred = clf.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_things.append((fpr, tpr, '{} AUC: {:.3f}'.format(classifier_name, roc_auc)))
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    precision_recall_things.append((recall, precision, thresholds, '{} AUC: {:.3f}'.format(classifier_name, pr_auc)))
    #plot_precision_recall_curve(clf, X_test, y_test)
     
    feature_names = get_transformer_feature_names(clf.named_steps['preprocessor'])

    try:
      importances = classifier.feature_importances_
      indices = np.argsort(importances)[::-1]
      print('~Feature Ranking:')
    
      for f in range (X_test.shape[1]):
        print ('{}. {} {} ({:.3f})'.format(f + 1, feature_names[indices[f]], indices[f], importances[indices[f]]))
    except:
      pass
    
    scores = cross_val_score(clf, X, y, cv=5)
    print('~Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    
    titles_options = [("Confusion Matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    
    print('~Confusion Matrix:''\n',
    confusion_matrix(y_test, y_pred))
    print('~Classification Report:''\n',
    classification_report(y_test, y_pred))
   
    print('~Average Precision Score: {:.3f}'.format(average_precision_score(y_test, y_score)))
    print('~roc_auc_score: {:.3f}'.format(roc_auc))
    print('~precision-recall AUC: {:.3f}'.format(pr_auc))
    print()
#ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
roc_plt = plt.figure()
lw = 4
for roc_thing in roc_things:
    fpr, tpr, label = roc_thing
    plt.plot(fpr, tpr, lw=lw, label=label)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
#Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a 
#predictive model using different probability thresholds.
pr_plt = plt.figure()
for pr_thing in precision_recall_things:
    recall, precision, _, label = pr_thing
    plt.plot(recall, precision, lw=lw, label=label)
ratio = y_test[y_test].shape[0] / y_test.shape[0]
plt.hlines(y=ratio, xmin=0, xmax=1, color='navy', lw=lw, linestyle='--')
plt.vlines(x=ratio, ymin=0, ymax=1, color='navy', lw=lw, linestyle='--')
plt.title('Precision-recall plot')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
from scipy.stats import hmean
import numpy.ma as ma

recall, precision, thresholds, _ = precision_recall_things[0]

a = np.column_stack((recall,precision))

a = ma.masked_less_equal(a, 0)
a = ma.mask_rows(a)
f1 = hmean(a,axis=1)

threshold_that_maximizes_f1 = thresholds[np.argmax(f1)]
print('threshold that optimizes f1: {}'.format(threshold_that_maximizes_f1))