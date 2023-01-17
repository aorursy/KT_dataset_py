import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



def auc_scorer(clf, X, y, model): # Helper function to plot the ROC curve

    if model=='RF':

        fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1])

    elif model=='SVM':

        fpr, tpr, _ = roc_curve(y, clf.decision_function(X))

    roc_auc = auc(fpr, tpr)



    plt.figure()    # Plot the ROC curve

    plt.plot(fpr, tpr, label='ROC curve from '+model+' model (area = %0.3f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")

    plt.show()



    return fpr,tpr,roc_auc



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/kidney_disease.csv')
# Map text to 1/0 and do some cleaning

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)
# Further cleaning

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

df.drop('id',axis=1,inplace=True)
df.head()
df2 = df.dropna(axis=0)

df2['class'].value_counts()
corr_df = df2.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_df, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlations between different predictors')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,:-1], df2['class'], 

                                                    test_size = 0.33, random_state=44,

                                                   stratify= df2['class'] )

print(X_train.shape)

print(X_test.shape)
y_train.value_counts()
tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],

                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}]

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,scoring='f1')

clf.fit(X_train, y_train)



print("Detailed classification report:")

y_true, lr_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, lr_pred))



confusion = confusion_matrix(y_test, lr_pred)

print('Confusion Matrix:')

print(confusion)



# Determine the false positive and true positive rates

fpr,tpr,roc_auc = auc_scorer(clf, X_test, y_test, 'RF')



print('Best parameters:')

print(clf.best_params_)

clf_best = clf.best_estimator_

plt.figure(figsize=(12,3))

features = X_test.columns.values.tolist()

importance = clf_best.feature_importances_.tolist()

feature_series = pd.Series(data=importance,index=features)

feature_series.plot.bar()

plt.title('Feature Importance')
list_to_fill = X_test.columns[feature_series>0]

print(list_to_fill)
# Are there correlation in missing values?

corr_df = pd.isnull(df).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_df, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
df2 = df.dropna(axis=0)

no_na = df2.index.tolist()

some_na = df.drop(no_na).apply(lambda x: pd.to_numeric(x,errors='coerce'))

some_na = some_na.fillna(0) # Fill up all Nan by zero.



X_test = some_na.iloc[:,:-1]

y_test = some_na['class']

y_true = y_test

lr_pred = clf_best.predict(X_test)

print(classification_report(y_true, lr_pred))



confusion = confusion_matrix(y_test, lr_pred)

print('Confusion Matrix:')

print(confusion)



print('Accuracy: %3f' % accuracy_score(y_true, lr_pred))

# Determine the false positive and true positive rates

fpr,tpr,roc_auc = auc_scorer(clf_best, X_test, y_test, 'RF')

 
