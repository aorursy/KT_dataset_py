!pip install --upgrade xgbfir
import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sbn

import sys

from datetime import datetime

import category_encoders as ce

from scipy.stats import chi2_contingency

from statsmodels.stats.multitest import fdrcorrection, multipletests

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, precision_recall_curve

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import xgboost as xgb

import xgbfir

import string

import joblib

from category_encoders import TargetEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import CategoricalNB
print("python", sys.version)

for module in np, pd, mpl, sbn, ce, sk, xgb, joblib:

    print(module.__name__, module.__version__)
np.random.seed(0)
df = pd.read_csv('../input/cat-in-the-dat-ii/train.csv').drop('id',axis=1)

test = pd.read_csv(r'../input/cat-in-the-dat-ii/test.csv', index_col=0)

sample_submission = pd.read_csv(r'../input/cat-in-the-dat-ii/sample_submission.csv', index_col=0)
sample_submission.head()
df.head().T
test.head()
print('Shape of the training set: ', df.shape)

print('Shape of the test set: ', test.shape)
df.info()
(df.isna().sum(axis=1)==0).sum()/df.shape[0]
(test.isna().sum(axis=1)==0).sum()/test.shape[0]
df.isna().sum(axis=1).value_counts()
cont_table = pd.crosstab(df.nom_1, df.nom_2, normalize='index')

cont_table
plt.matshow(cont_table, cmap=plt.cm.gray)
def cochran_criterion(crosstab):

  criterion = True

  E = []

  N = df.shape[0]

  for O_i in crosstab.sum(axis=1):

    for O_j in crosstab.sum():

      if O_i*O_j/N==0:

        critetion = False

      E.append(O_i*O_j/N > 4)

  criterion = criterion & (np.mean(E)>0.8)

  return criterion
def chi2_test(X,Y):

  crosstab = pd.crosstab(X, Y)

  criterion = cochran_criterion(crosstab)

  chi2, p = chi2_contingency(crosstab)[:2]

  return [criterion, chi2, p]
start = datetime.now()



df_total = pd.concat([df.drop('target', axis=1), test])

chi2_missingness = pd.DataFrame(columns=['Cochran_criterion', 'Chi2', 'p_value'])



for col1 in df_total.columns:

  for col2 in df_total.columns.drop(col1):

    missingness_indicator = df_total[col1].isna()

    other_variable = df_total[col2]

    chi2_missingness.loc[col1 + '_' + col2] = chi2_test(missingness_indicator, other_variable)



for col in df.columns.drop('target'):

  missingness_indicator = df[col].isna()

  chi2_missingness.loc[col + '_target'] = chi2_test(missingness_indicator, df.target)



runtime = datetime.now() - start

print('Runtime ', runtime)
chi2_missingness.sort_values('p_value', inplace=True)

reject, pvalue_corrected = fdrcorrection(pvals=chi2_missingness.p_value, alpha=0.05, method='indep', is_sorted=True)

chi2_missingness['p_value_corrected'] = pvalue_corrected

chi2_missingness['BH_reject'] = chi2_missingness['Cochran_criterion'] & reject

chi2_missingness.to_csv('chi2_tests_missingness')
chi2_missingness.head()
significant_tests = chi2_missingness[chi2_missingness['BH_reject']==True]

print('Number of statistically significant tests: ', significant_tests.shape[0])
def predict_missingness(col):



  X = df_total.drop(col, axis=1)

  y = df_total[col].isna()

  cols = X.columns.tolist()



  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

  X_train.reset_index(drop=True, inplace=True)

  y_train.reset_index(drop=True, inplace=True)

  X_train_te = X_train.copy()



  # Target encoding

  for index_fit, index_transform in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(X_train, y_train):

    target_enc = TargetEncoder(cols = cols, smoothing=0.5, min_samples_leaf=5)

    target_enc.fit(X_train.iloc[index_fit,:], y_train.iloc[index_fit])

    X_train_te.loc[index_transform,:] = target_enc.transform(X_train.iloc[index_transform, :])

    

  X_train_te = X_train_te.astype(float)



  target_enc = TargetEncoder(cols = cols, smoothing=0.5, min_samples_leaf=5)

  target_enc.fit(X_train, y_train)

  X_val_te = target_enc.transform(X_val)



  weight = sum(y_train==0)/sum(y_train==1)



  # Gradient boosted trees

  gbt = xgb.XGBClassifier(n_estimators=3000, random_state=0, objective='binary:logistic', scale_pos_weight=weight,

                          learning_rate=0.2, max_depth=4, subsample=0.8, colsample_bytree = 0.8, min_child_weight=1000)

  

  gbt.fit(X_train_te, y_train, eval_set= [(X_train_te, y_train), (X_val_te, y_val)], eval_metric=['auc'], early_stopping_rounds=50, 

          verbose=False)



  # Performance

  y_pred = gbt.predict_proba(X_train_te)

  train_score = roc_auc_score(y_train, y_pred[:,1])

  y_pred = gbt.predict_proba(X_val_te)

  val_score = roc_auc_score(y_val, y_pred[:,1])



  return(train_score, val_score)
start = datetime.now()



pred_missingness = pd.DataFrame(columns=['Feature', 'train_score', 'val_score'])



for col in df_total.columns:

  train_score, val_score = predict_missingness(col)

  pred_missingness.loc[col] = [col, train_score, val_score]



pred_missingness.sort_values('val_score', ascending=False, inplace=True)

pred_missingness.to_csv('pred_missingness.csv', index=False)



runtime = datetime.now() - start

print('Runtime ', runtime)
pred_missingness.head()
def predict_feature(col):



  df_new = df_total[df_total[col].isna()==False].astype(str) # We remove the NaN of the variable to predict.



  X = df_new.drop(col, axis=1)

  y = df_new[col]



  le = LabelEncoder() # The label encoding of the variable that we want to predict is needed to use this classifier.

  y = le.fit_transform(y)



  train_scores = []

  val_scores = []



  for index_train, index_val in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(X, y):

          

      X_train = X.iloc[index_train]

      y_train = y[index_train]

      X_valid = X.iloc[index_val]

      y_valid = y[index_val]



      # Ordinal encoding.

      oe = OrdinalEncoder()

      X_train_oe = oe.fit_transform(X_train.astype(str))

      X_val_oe = oe.transform(X_valid.astype(str))



      X_train_oe += 2 # we add 2 to avoid negative values resulting from the ordinal encoding 

      X_val_oe += 2   # (the classifier only accept positive values)



      # Training.

      nb = CategoricalNB()

      nb.fit(X_train_oe, y_train)



      # We stock the results.

      if np.max(y)==1:  

        y_pred = nb.predict_proba(X_train_oe)

        train_score = roc_auc_score(y_train, y_pred[:,1])

        train_scores.append(train_score)



        y_pred = nb.predict_proba(X_val_oe)

        val_score = roc_auc_score(y_valid, y_pred[:,1])

        val_scores.append(val_score)

      else:

        y_pred = nb.predict_proba(X_train_oe)

        train_score = roc_auc_score(y_train, y_pred, multi_class='ovr') 

        train_scores.append(train_score)                                



        y_pred = nb.predict_proba(X_val_oe)

        val_score = roc_auc_score(y_valid, y_pred, multi_class='ovr')

        val_scores.append(val_score)



  return(train_scores, val_scores)
start = datetime.now()



pred_features = pd.DataFrame(columns=['Feature', 'train_score', 'val_score', 'std_val_score'])



high_cardinality_features = ['nom_' + str(i) for i in range(5,10)] + ['ord_5']

for col in df_total.columns.drop(high_cardinality_features):

  train_score, val_score = predict_feature(col)

  pred_features.loc[col] = [col, np.mean(train_score), np.mean(val_score), np.std(val_score)]



pred_features.sort_values('val_score', ascending=False, inplace=True)

pred_features.to_csv('pred_features.csv', index=False)



runtime = datetime.now() - start

print('Runtime ', runtime)
pred_features.head()
df.bin_4.fillna('NaN').value_counts()
df.nom_5.fillna('NaN').value_counts()
def plot_hist_bar(col, dtype="numeric", dataset=df, xlabels=True, histogram=True):



    """ Function to plot a barplot and optionally an histogram of one chosen feature from the data.



        Parameters

        ----------

        col : String

            The feature whose the function will create graphics from.

        dtype : String (default="numeric")

            If "str", then the feature will be treated as a string feature and so will be ordinal encoded before plotting

            the charts. Otherwise, the feature is considered to be already numeric.

        dataset : DataFrame (default=df, namely the training set)

            The dataframe from containing the variables used.

        xlabels : Boolean (default=True)

            Whether to keep the xlabels.

        histogram : Boolean (default=True)

            Whether to plot an histogram and a barplot or just a barplot.

        

        Display

        -------

        

        - Optionally an histogram given the distribution of the feature col.

        - A barplot providing the target frequency for each value of the feature col.

        - A dictionary mapping the encoded values on the x axis of the precedent charts to their original values. For all 

        of them, the missing values of the feature col are filled with the value -1 and represented with a red bar.

        

        """



    if (dtype=='str'):

      map_categories = {cat:i for i,cat in enumerate(df.dropna()[col].unique())}

    else:

      map_categories = {}

  

    map_categories[np.nan] = -1

    dataset = dataset.replace({col: map_categories})

    dataset.loc[col] = dataset[col].astype(int)



    fig = plt.figure()

    palette = dict((value,'grey') if (value != -1) else (value,'red') for value in dataset[col].unique())



    if histogram==True:



      plt.subplot(1,3,1)

      data = dataset[col].value_counts().reset_index()

      data.columns = ['Values','Count']

      ax1 = sbn.barplot(x = 'Values', y = 'Count', data = data, palette=palette)

      plt.title('Histogram')

      if xlabels==False:

        frame1 = plt.gca()

        frame1.axes.xaxis.set_ticklabels([])



      plt.subplot(1,3,3)

      ax2 = sbn.barplot(x = col, y = 'target', data = dataset, palette=palette)

      ax2.set(xlabel='Values', ylabel='Target frequency')

      if xlabels==False:

        frame2 = plt.gca()

        frame2.axes.xaxis.set_ticklabels([])

      plt.title('Barplot')



    else:

      ax = sbn.barplot(x = col, y = 'target', data = dataset, palette=palette)

      ax.set(xlabel='Values', ylabel='Target frequency')

      if xlabels==False:

        frame = plt.gca()

        frame.axes.xaxis.set_ticklabels([])



    fig.suptitle("Feature: " + col, fontsize=16)

    plt.show()

  

    if xlabels==True:

      print('\033[30m'' Relation between the numbers on the x axis and the categories they represent:', '\n', map_categories)
plot_hist_bar('ord_0')
# Creation of a dictionnary to encode the features.

replace_letters =  {j:i for i,j in enumerate(list(string.ascii_uppercase + string.ascii_lowercase))}



# We plot a barplot representing the target frequency of each modality of ord_3 after encoding.

# For more clarity, the x-axis is made invisible. The red bar corresponds to the NaN.

col ='ord_3'

plot_hist_bar(col, dataset = df.replace({col:replace_letters}), xlabels=False, histogram=False)
# Same chart for ord_4.

col = 'ord_4'

plot_hist_bar(col, dataset = df.replace({col:replace_letters}), xlabels=False, histogram=False)
df_new=df.copy()

# We fill the NaN with 0 and keep only the first character of ord_5.

df_new.ord_5 = df_new.ord_5.fillna('0').astype(str).str[0] 

df_new.ord_5 = df_new.ord_5.replace({'0':np.nan}) # We transform back the 0 to NaN. 

#This trick allowed to conserve the NaN even when keeping only the first character.

plot_hist_bar(col, dataset = df_new.replace({col:replace_letters}), xlabels=False, histogram=False)
def wrap_around_unit_circle(n, with_nan=False, title=None):



    x = np.cos(2*np.pi*np.arange(1,n+1)/n)

    y = np.sin(2*np.pi*np.arange(1,n+1)/n)

    

    unit_circle = plt.Circle((0, 0), 1, color='grey', fill=False)

    

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=10, color='black', marker='x')

    ax.set_xlim((-1.1,1.1))

    ax.add_artist(unit_circle)

    

    for i in range(1,n+1):

        if (i==n)&(with_nan==True):

            ax.annotate("NaN", (x[i-1]+0.02, y[i-1]), color='red')

        else:

            ax.annotate(i, (x[i-1]+0.02, y[i-1]), color='black')



    if title!=None:

        plt.title(title, color='white')



wrap_around_unit_circle(7, title="Wrapping of the days around the unit circle")
wrap_around_unit_circle(8, with_nan=True, title="Wrapping of the days (including the missing ones) around the unit circle")
wrap_around_unit_circle(13, with_nan=True, title="Wrapping of the months (including the missing ones) around the unit circle")
def preprocessing_1(df_train, df_test):



  """ Function to preprocess the data.



        Parameters

        ----------

        df_train : DataFrame, shape = (600000, 24)

            Training set to preprocess, containing the features plus the target.

        df_test : DataFrame, shape = (400000, 23)

            Test set to preprocess.

        

        Returns

        -------

        X_train_encoded :  DataFrame, shape = (600000, 44)

            Training set preprocessed, without the target variable.

        y_train :  Series, shape = (600000,)

            Training's target variable.

        X_test_encoded :  DataFrame, shape = (400000, 44)

            Test set preprocessed.



        """



  # Extraction of the target from the training set.    

  y_train = df_train['target'].copy()

  df = pd.concat([df_train.drop('target', axis=1), df_test])



  # Filling of the missing values.

  df = df.fillna(value={'bin_0': df.bin_0.mode()[0], 'bin_1':df.bin_1.mode()[0], 'bin_2':df.bin_2.mode()[0], 'bin_3':df.bin_3.mode()[0], 

                        'bin_4':df.bin_4.mode()[0], 'nom_0':df.nom_0.mode()[0], 'nom_1':'nan', 'nom_2':'nan', 'nom_3':'nan', 'nom_4':'nan',

                        'nom_5':'nan', 'nom_6':'nan', 'nom_7':'nan', 'nom_8':'nan', 'nom_9':'nan', 'ord_0':df.ord_0.median(), 'ord_5':'0',

                        'day':8, 'month':13})



  # Dropping of the second letter of ord_5.

  df['ord_5'] = df['ord_5'].astype(str).str[0] # If we had not filled the NaN with zero, they would have resulted in the "n" value when taking the first letter.

  

  # Creation of a dictionnary for replacing the letters of ord_3, ord_4 and ord_5.

  replace_letters = {j:i for i,j in enumerate(list(string.ascii_uppercase + string.ascii_lowercase))}

  replace_letters['0'] = np.nan # This trick enables to later fill the NaN with the median of the encoded non-missing data.



  # Encoding of bin_3, bin_4, ord_1, ord_2, ord_3, ord_4 and ord_5.

  df = df.replace({'bin_3':{'F':0, 'T':1}, 'bin_4':{'N':0, 'Y':1},'ord_1': {'Novice':1, 'Contributor':2, 'Expert':3, 'Master':4, 'Grandmaster':5},

                   'ord_2':{'Freezing': 1, 'Cold':2, 'Warm':3, 'Hot':4, 'Boiling Hot': 5, 'Lava Hot':6},'ord_3': replace_letters, 'ord_4':replace_letters,

                   'ord_5':replace_letters})

  

  # The missing values of ord_1, ord_2, ord_3, ord_4 and ord_5 are imputed with the median of the values after encoding. 

  df = df.fillna(value={'ord_1':df['ord_1'].median(), 'ord_2':df['ord_2'].median(), 'ord_3':df['ord_3'].median(), 'ord_4':df['ord_4'].median(), 'ord_5':df['ord_5'].median()})



  # One hot encoding.

  one_hot_cols = ['nom_0','nom_1','nom_2','nom_3','nom_4']

  X_cat = df[one_hot_cols].astype(str).copy()

  one_hot_enc = OneHotEncoder()

  X_cat_1hot = one_hot_enc.fit_transform(X_cat)

  X_cat_1hot = pd.DataFrame(X_cat_1hot.toarray(), index=X_cat.index, columns = one_hot_enc.get_feature_names())

  X_cat_1hot.drop(['x'+str(i)+'_' + X_cat['nom_' + str(i)].mode()[0] for i in range(5)], axis=1, inplace=True)

  X = pd.concat([df.drop(one_hot_cols, axis=1), X_cat_1hot], axis=1)



  # Two-dimensional projection of day and month.

  X['sin_day'] = np.sin(2*np.pi*X.day/8)

  X['cos_day'] = np.cos(2*np.pi*X.day/8)

  X['sin_month'] = np.sin(2*np.pi*X.month/13)

  X['cos_month'] = np.cos(2*np.pi*X.month/13)

  X.drop(['day', 'month'], axis=1, inplace=True)



  # To apply a target encoding, we need to separate the training set from the test set.

  X_train = X.iloc[:600000]

  X_test = X.iloc[600000:1000000]



  # Target encoding.

  target_enc_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

  X_train_encoded = X_train.copy()



  for index_fit, index_transform in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(X_train, y_train):

    target_enc = TargetEncoder(cols = target_enc_cols, smoothing=0.5)

    target_enc.fit(X_train.iloc[index_fit,:], y_train.iloc[index_fit])

    X_train_encoded.loc[index_transform,:] = target_enc.transform(X_train.iloc[index_transform, :])

  

  target_enc = TargetEncoder(cols = target_enc_cols, smoothing=0.5)

  target_enc.fit(X_train, y_train)

  X_test_encoded = target_enc.transform(X_test)



  # Data types management.

  X_train_encoded.loc[:,target_enc_cols] = X_train_encoded[target_enc_cols].astype(float)

  X_train_encoded.loc[:, X_train_encoded.columns.drop(target_enc_cols).to_list()] = X_train_encoded.loc[:, X_train_encoded.columns.drop(target_enc_cols).to_list()].astype(float)

  X_test_encoded.loc[:,'ord_3'] = X_test_encoded.loc[:,'ord_3'].astype(float)

  X_test_encoded.loc[:,'ord_4'] = X_test_encoded.loc[:,'ord_4'].astype(float)

  

  return(X_train_encoded, y_train, X_test_encoded)
start = datetime.now()

X, y, X_test = preprocessing_1(df, test)

end = datetime.now()

print('Duration: ', end-start)
print('X shape :', X.shape)

print('y shape :', y.shape)

print('X_test shape :', X_test.shape)
X.head().T
# Saving of the dataframes.

X.to_csv('X_1st_strategy.csv', index=False)

y.to_csv('y.csv', index=False, header=True)

X_test.to_csv('X_test_1st_strategy.csv', index=False)
# Function to plot the ROC curve of all predictions on validation sets.

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, linewidth=2)

    plt.plot([0,1],[0,1],'k--')

    plt.axis([0,1,0,1])

    plt.xlabel('Taux de faux positifs')

    plt.ylabel('Taux de vrais positifs')

    plt.suptitle('ROC curve of all predictions on validation sets', color="white")

    plt.show()
# Precision and recall as a function of the decision threshold.

def plot_precision_recall_vs_threshold(y, y_pred):

    precisions, recalls, thresholds = precision_recall_curve(y, y_pred)

    plt.plot(thresholds, precisions[:-1],'b--', label='Precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')

    plt.xlabel('Thresholds')

    plt.legend(loc='center left')

    plt.ylim([0,1])

    plt.suptitle('Precision and recall versus thresholds of all predictions on validation sets', color="white")

    plt.show()
def display_scores(train_scores, val_scores, y_validations, y_predictions):



    # Printing of the scores.

    print('Training scores: ', train_scores)

    print('Mean training score: ', np.mean(train_scores))

    print('Standard deviation of the training scores: ', np.std(train_scores), '\n')



    print('Validation scores: ', val_scores)

    print('Mean validation score: ', np.mean(val_scores))

    print('Standard deviation of the validation scores: ', np.std(val_scores), '\n\n')



    # Precision-Recall versus decision thresholds.

    y_valid = np.concatenate(tuple(y_validations))

    y_pred = np.concatenate(tuple(y_predictions))



    plot_precision_recall_vs_threshold(y_valid, y_pred)



    print('\n')



    # ROC curve.

    fpr, tpr, threshold = roc_curve(y_valid, y_pred)

    plot_roc_curve(fpr,tpr)
start = datetime.now()



train_scores = []

val_scores = []

y_validations = []

y_predictions = []



for index_train, index_val in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(X, y):

        

    X_train = X.iloc[index_train]

    y_train = y.iloc[index_train]

    X_valid = X.iloc[index_val]

    y_valid = y.iloc[index_val]



    weights = sum(y_train.values==0)/sum(y_train.values==1)



    

    xgb_gbrf = xgb.XGBClassifier(n_estimators=3000, random_state=0, objective='binary:logistic', scale_pos_weight=weights, 

                                    learning_rate=0.15, max_depth=2, subsample=0.7, min_child_weight=500,  

                                    colsample_bytree = 0.2, reg_lambda = 3.5, reg_alpha=1.5, num_parallel_tree= 5)

    

    xgb_gbrf.fit(X_train, y_train, eval_set= [(X_train, y_train), (X_valid, y_valid)], eval_metric=['auc'], 

                    early_stopping_rounds=100, verbose=False)

    

    # We stock the results.

    y_pred = xgb_gbrf.predict_proba(X_train)

    train_score = roc_auc_score(y_train, y_pred[:,1])

    train_scores.append(train_score)



    y_pred = xgb_gbrf.predict_proba(X_valid)

    val_score = roc_auc_score(y_valid, y_pred[:,1])

    val_scores.append(val_score)

    

    y_validations.append(y_valid)

    y_predictions.append(y_pred[:,1])

    

    

display_scores(train_scores, val_scores, y_validations, y_predictions)



runtime = datetime.now() - start

print('\n\nRuntime ', runtime)
xgb.plot_importance(xgb_gbrf, importance_type='weight', max_num_features=15)
xgb.plot_importance(xgb_gbrf, importance_type='cover', max_num_features=15)
xgb.plot_importance(xgb_gbrf, importance_type='gain', max_num_features=15)
gbrf_feature_importances = pd.DataFrame({'Predictors': X.columns, 'Normalized gains': xgb_gbrf.feature_importances_}) 

gbrf_feature_importances.sort_values('Normalized gains',ascending=False, inplace=True)

gbrf_feature_importances.reset_index(inplace=True, drop=True)

gbrf_feature_importances.to_csv('fi_gbrf_1st_strategy')
gbrf_feature_importances
np.mean([749,738,759,943,901])
#### Modelisation with XGBoost Gradient Boosting trees Classifier.



weight = sum(y.values==0)/sum(y.values==1)





xgb_gbrf = xgb.XGBClassifier(n_estimators=818, random_state=0, objective='binary:logistic', scale_pos_weight=weight, 

                             learning_rate=0.15, max_depth=2, subsample=0.7, min_child_weight=500,  colsample_bytree = 0.2,

                             reg_lambda = 3.5, reg_alpha=1.5, num_parallel_tree= 5)
xgb_gbrf.fit(X, y, eval_set= [(X, y)], eval_metric=['auc'], verbose=False)
joblib.dump(xgb_gbrf, 'model_gbrf_1st_strategy')
y_test = xgb_gbrf.predict_proba(X_test)[:,1]

output = pd.DataFrame({'id':sample_submission.index, 'target':y_test})

output
output.describe()
output.to_csv('output_gbrf_1st_strategy.csv', index=False)
# Creation of the training set used with the logistic regression.

df_total = pd.concat([df.drop('target', axis=1), test])



one_hot_cols = ['day','month']

X_cat = df_total[one_hot_cols].astype(str).copy()



one_hot_enc = OneHotEncoder()

X_cat_1hot = one_hot_enc.fit_transform(X_cat)

X_cat_1hot = pd.DataFrame(X_cat_1hot.toarray(), index=X_cat.index, 

                          columns = pd.Series(one_hot_enc.get_feature_names()).str.replace('x0', 'day').str.replace('x1', 'month'))

X_cat_1hot.drop(['day_' + X_cat['day'].mode()[0], 'month_' + X_cat['month'].mode()[0]], axis=1, inplace=True)



X_lr_total = pd.concat([X, X_test])

X_lr_total = pd.concat([X_lr_total.drop(['sin_day', 'cos_day', 'sin_month', 'cos_month'], axis=1), X_cat_1hot], axis=1)



X_lr_total['bin_2_nan'] = df_total.bin_2.isna().astype(int)

X_lr_total['bin_4_nan'] = df_total.bin_4.isna().astype(int)



X_lr = X_lr_total.iloc[:600000]

X_test_lr = X_lr_total.iloc[600000:1000000]
X_lr.shape
X_lr.head().T
start = datetime.now()



X_train, X_valid, y_train, y_valid = train_test_split(X_lr, y, test_size=0.2, stratify=y, random_state=42)

weight = sum(y_train.values==0)/sum(y_train.values==1)

    

gbrf = xgb.XGBClassifier(n_estimators=3000, random_state=0, objective='binary:logistic', scale_pos_weight=weight, 

                                learning_rate=0.10, max_depth=4, subsample=0.7, min_child_weight=500,  

                                colsample_bytree = 0.2, reg_lambda = 3.5, reg_alpha=1.5, num_parallel_tree= 5)

    

gbrf.fit(X_train, y_train, eval_set= [(X_train, y_train), (X_valid, y_valid)], eval_metric=['auc'], 

                    early_stopping_rounds=100, verbose=False)

    

y_pred = gbrf.predict_proba(X_train)

print("Train AUROC = ", roc_auc_score(y_train, y_pred[:,1]))



y_pred = gbrf.predict_proba(X_valid)

print("Validation AUROC = ", roc_auc_score(y_valid, y_pred[:,1]))



runtime = datetime.now() - start

print('\n\nRuntime ', runtime)
joblib.dump(gbrf, 'model_gbrf_2nd_strategy')
xgbfir.saveXgbFI(gbrf, feature_names=X_lr.columns, MaxInteractionDepth=3, OutputXlsxFile='fi_gbrf_2nd_strategy.xlsx')
xls = pd.ExcelFile('fi_gbrf_2nd_strategy.xlsx')
fi_depth_1 = pd.read_excel(xls, 'Interaction Depth 1')

fi_depth_2 = pd.read_excel(xls, 'Interaction Depth 2')

fi_depth_3 = pd.read_excel(xls, 'Interaction Depth 3')
fi_depth_1[:10]
fi_depth_2[:10]
fi_depth_3[:10]
X_lr_total['int_1'] = X_lr_total['ord_0'] * X_lr_total['ord_3'] * X_lr_total['ord_5']



X_lr_total.drop(['bin_3', 'x1_Square', 'x2_Cat', 'x4_Oboe', 'day_5.0'], axis=1, inplace=True)



X_lr = X_lr_total.iloc[:600000]

X_test_lr = X_lr_total.iloc[600000:1000000]
start = datetime.now()



train_scores = []

val_scores = []

y_validations = []

y_predictions = []



for index_train, index_val in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(X_lr, y):

    X_train = X_lr.iloc[index_train]

    y_train = y.iloc[index_train]

    X_valid = X_lr.iloc[index_val]

    y_valid = y.iloc[index_val]

    

    # Logistic regression.

    lr = LogisticRegression(random_state=0, max_iter=10000, fit_intercept=True, class_weight='balanced', 

                            penalty='l2', C=0.4, verbose=0)



    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_valid_scaled = scaler.transform(X_valid)



    lr.fit(X_train_scaled, y_train)



    # We stock the results.

    y_pred_train = lr.predict_proba(X_train_scaled)[:,1]

    train_score = roc_auc_score(y_train, y_pred_train)

    train_scores.append(train_score)



    y_pred_valid = lr.predict_proba(X_valid_scaled)[:,1]

    val_score = roc_auc_score(y_valid, y_pred_valid)

    val_scores.append(val_score)

    

    y_validations.append(y_valid)

    y_predictions.append(y_pred_valid)

    

display_scores(train_scores, val_scores, y_validations, y_predictions)



runtime = datetime.now() - start

print('\n\nRuntime ', runtime)
lr_coefficients = pd.DataFrame({'Predictors': X_lr.columns.to_list() + ['Intercept'], 

                                'Coefficients': list(lr.coef_.reshape(-1)) + list(lr.intercept_),

                                'Direction': np.sign(list(lr.coef_.reshape(-1)) + list(lr.intercept_)),

                                'Importances': np.abs(list(lr.coef_.reshape(-1)) + list(lr.intercept_))})



lr_coefficients.sort_values('Importances',ascending=False, inplace=True)

lr_coefficients.reset_index(inplace=True, drop=True)

lr_coefficients.to_csv('coefficients_lr_2nd_strategy')
lr_coefficients[:50]
lr_coefficients[50:]
# Full training.

lr = LogisticRegression(random_state=0, max_iter=10000, fit_intercept=True, class_weight='balanced', penalty='l2', 

                        C=0.4, verbose=0)



scaler = StandardScaler()

X_lr_scaled = scaler.fit_transform(X_lr)

X_test_lr_scaled = scaler.transform(X_test_lr)



lr.fit(X_lr_scaled, y)
joblib.dump(lr,'model_lr_2nd_strategy')
#### Submissions.

y_test = lr.predict_proba(X_test_lr_scaled)[:,1]

output = pd.DataFrame({'id':sample_submission.index, 'target':y_test})

output
output.describe()
output.to_csv('output_lr_2nd_strategy.csv', index=False)
start = datetime.now()



chi2_target = pd.DataFrame(columns=['Cochran_criterion', 'Chi2', 'p_value'])

for col in df.columns.drop('target'):

  feature = df[col]

  chi2_target.loc[col + '_target'] = chi2_test(feature, df.target)



chi2_target['rejected'] = chi2_target['Cochran_criterion'] & (chi2_target['p_value']<0.05)

chi2_target.sort_values('p_value', ascending=True, inplace=True)

chi2_target.to_csv('chi2_target_tests.csv', index=False)

print(chi2_target, '\n')



runtime = datetime.now() - start

print('Runtime ', runtime)
for col in ['nom_'+str(i) for i in range(5,10)]:

    rare_categories = df[col].value_counts().iloc[np.where(df[col].value_counts()<100)].index

    print("The predictor " + col + " has ", rare_categories.size, " rare categories among its ", df[col].unique().size,

         " categories.")
for col in df.columns.drop('target'):

  unknowns = set(test[col].astype(str).unique()) - set(df[col].astype(str).unique())

  print('The predictor ', col, ' contains ', len(unknowns), ' value present only in the test set.')
start = datetime.now()



train_scores = []

val_scores = []

y_validations = []

y_predictions = []



X_nb = df.drop(['bin_3', 'nom_6', 'target'], axis=1).astype(str)



for index_train, index_val in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(X_nb, y):

        

    X_train = X_nb.iloc[index_train].copy()

    y_train = y.iloc[index_train].copy()

    X_valid = X_nb.iloc[index_val].copy()

    y_valid = y.iloc[index_val].copy()



    for col in ['nom_5', 'nom_7', 'nom_8', 'nom_9']:

      rare_categories = X_train[col].value_counts().iloc[np.where(X_train[col].value_counts()<100)].index

      X_train.loc[:,col] = X_train.loc[:,col].replace(rare_categories, 'nan')

      X_valid.loc[:,col] = X_valid.loc[:,col].replace(rare_categories, 'nan')



    # Ordinal encoding.

    oe = OrdinalEncoder()

    X_train_oe = oe.fit_transform(X_train)

    X_val_oe = oe.transform(X_valid)



    X_train_oe += 2 # we add 2 to avoid negative values resulting from the ordinal encoding

    X_val_oe += 2



    # Training.

    nb = CategoricalNB(alpha = 3.4)

    nb.fit(X_train_oe, y_train)



    # We stock the results.

    y_pred = nb.predict_proba(X_train_oe)

    train_score = roc_auc_score(y_train, y_pred[:,1])

    train_scores.append(train_score)



    y_pred = nb.predict_proba(X_val_oe)

    val_score = roc_auc_score(y_valid, y_pred[:,1])

    val_scores.append(val_score)



    y_validations.append(y_valid)

    y_predictions.append(y_pred[:,1])



display_scores(train_scores, val_scores, y_validations, y_predictions)



runtime = datetime.now() - start

print('\n\nRuntime ', runtime)
# Full preprocessing

oe = OrdinalEncoder()



X_nb = df.drop(['bin_3', 'nom_6', 'target'], axis=1).astype(str)

X_test_nb = test.drop(['bin_3', 'nom_6'], axis=1).astype(str)



for col in ['nom_5', 'nom_7', 'nom_8', 'nom_9']:

  rare_categories = X_nb[col].value_counts().iloc[np.where(X_nb[col].value_counts()<100)].index

  X_nb[col] = X_nb[col].replace(rare_categories, 'nan')

  X_test_nb[col] = X_test_nb[col].replace(rare_categories, 'nan')



X_oe = oe.fit_transform(X_nb)

X_test_oe = oe.transform(X_test_nb)

  

X_oe += 2 # we add 2 to avoid negative values resulting from the ordinal encoding

X_test_oe += 2
start = datetime.now()



# Full training

nb = CategoricalNB(alpha = 3.4) 

nb.fit(X_oe, df.target)



y_pred = nb.predict_proba(X_oe)

train_score = roc_auc_score(df.target, y_pred[:,1])

print('AUROC on the entire training set: ', train_score)



runtime = datetime.now() - start

print('\n\nRuntime ', runtime)
joblib.dump(nb, 'model_nb_3rd_strategy')
y_test = nb.predict_proba(X_test_oe)[:,1]

output = pd.DataFrame({'id':sample_submission.index, 'target':y_test})

output
output.describe()
output.to_csv('output_nb_3rd_strategy.csv', index=False)
output_gbrf = pd.read_csv('output_gbrf_1st_strategy.csv')

output_lr = pd.read_csv('output_lr_2nd_strategy.csv')

output_nb = pd.read_csv('output_nb_3rd_strategy.csv')
mean = output_nb

mean.target = (output_gbrf.target * 0.78560 + output_lr.target * 0.78552 + output_nb.target * 0.78477)/(0.78560 + 0.78552 + 0.78477)
mean.to_csv('ouput_ensemble.csv', index=False)