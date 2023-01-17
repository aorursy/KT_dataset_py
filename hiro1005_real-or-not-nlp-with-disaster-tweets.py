# Data file

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier 

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', header=0)

train
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', header=0)

test.head(10)
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission.head()
train_mid = train.copy()

train_mid['train_or_test'] = 'train'



test_mid = test.copy()

test_mid['train_or_test'] = 'test'



test_mid['target'] = 9



alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True)



print('The size of the train data:' + str(train.shape))

print('The size of the test data:' + str(test.shape))

print('The size of the submission data:' + str(submission.shape))

print('The size of the alldata data:' + str(alldata.shape))
# Missing Data

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

    list_type = df[na_col_list].dtypes.sort_values(ascending=False)

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'Missing data', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['Missing data'], ascending=False)



Missing_table(alldata)
# Delete data seems to be unnecessary

del alldata['location']



train = alldata.query('train_or_test == "train"')

test = alldata.query('train_or_test == "test"')



# Index reset

test = test.reset_index()

# Delete unnecessary index column

del test["index"]

test



train_feature = train["text"].values

train_target = train["target"].values

submission_id = test['id'].values



test_feature = test["text"].values

test_feature
from sklearn import feature_extraction

count_vectorizer = feature_extraction.text.CountVectorizer()

train_feature = count_vectorizer.fit_transform(train_feature)

test_feature = count_vectorizer.transform(test_feature)



print(train_feature[0].todense().shape)

print(test_feature[0].todense())



X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
# RandomForest==============



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

print('='*20)

print('RandomForestClassifier')

print(f'accuracy of train set: {rf.score(X_train, y_train)}')

print(f'accuracy of test set: {rf.score(X_test, y_test)}')



rf_prediction = rf.predict(test_feature)

rf_prediction
# SVC==============



svc = SVC(verbose=True, random_state=0)

svc.fit(X_train, y_train)

print('='*20)

print('SVC')

print(f'accuracy of train set: {svc.score(X_train, y_train)}')

print(f'accuracy of test set: {svc.score(X_test, y_test)}')



svc_prediction = svc.predict(test_feature)

svc_prediction
# Save model

import pickle

filename = 'model.sav'

pickle.dump(rf, open(filename, 'wb'))



# Load model

filename = 'model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 0 = Positive, 1 = Negative



# Calculate y_pred using X_test

y_pred = rf.predict(X_test)



print('Accurary data(y_test) : ' + str(y_test))

print('Expectation data(y_pred) : ' + str(y_pred))

print('\n======Creating Matrix======')

print(confusion_matrix(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)

print(type(matrix))



class_names = ["Positive","Negative"]

df = pd.DataFrame(matrix, index=class_names, columns=class_names)

df
print('Accuracy_score:{:.3f}'.format(accuracy_score(y_test, y_pred)))

print('Precision_score:{:.3f}'.format(precision_score(y_test, y_pred)))

print('Recall_score:{:.3f}'.format(recall_score(y_test, y_pred)))

print('F1_score:{:.3f}'.format(f1_score(y_test, y_pred)))

print(matrix.flatten())

tn, fp, fn, tp = matrix.flatten()



print('真陽性 TP (True-Positive ポジティブに分類すべきアイテムを正しくポジティブに分類できた件数) : ' + str(tp))

print('真陰性 TN (True-Negative ネガティブに分類すべきアイテムを正しくネガティブに分類できた件数) : ' + str(tn))

print('偽陽性 FP (False-Positive ネガティブに分類すべきアイテムを誤ってポジティブに分類した件数) : ' + str(fp))

print('偽陰性 FN (False-Negative ポジティブに分類すべきアイテムを誤ってネガティブに分類した件数) : ' + str(fn))



# TP, TN, FP, FN を用いて、識別精度を評価するための指標が検出できる

print('\n正解率(Accuracy) (TP+TN)/(TP+TN+FP+FN): ' + str((tp + tn) / (tp + tn + fp + fn)) + '   (約' + str(round(((tp + tn) / (tp + tn + fp + fn))*100)) + '%)')

print('適合率(Precision) TP/(TP+FP): ' + str(tp/(tp + fp)) + '   (約' + str(round(((tp/(tp + fp)))*100)) + '%) ※適合率は精度と言う事もある')

print('検出率(Recall) TP/(TP+FN): ' + str(tp/(tp+fn)) + '   (約' + str(round((tp/(tp+fn))*100)) + '%) ※真陽性率 (TPR)、感度 (Sensitivity) とも呼ばれる')

print('F値(F-measure,F-score) 2*(precision*recall)/(precision+recall) ※精度 (Precision) と検出率 (Recall) をバランス良く持ち合わせているかを示す指標です。つまり、精度は高くても、検出率が低いモデルでないか、逆に、検出率は高くても、精度が低くなっていないか、といった評価を示します')



# Visualize matrix by seaborn heatmap() method

import seaborn as sns

sns.heatmap(matrix)

plt.savefig('sklearn_confusion_matrix.png')
from sklearn.metrics import classification_report

import pprint

print(classification_report(y_test, y_pred, target_names=['Positive', 'Negative']))
repo = classification_report(y_test, y_pred, output_dict=True)

pprint.pprint(repo)
print(repo['0'])

print(repo['0']['precision'])

print(type(repo['0']['precision']))
df = pd.DataFrame(repo)

df
from sklearn.metrics import plot_confusion_matrix

np.set_printoptions(precision=2)

titles_options = [

        ("Confusion matrix, without normalization", None),

        ("Normalized confusion matrix: true", 'true'),

        ("Normalized confusion matrix: pred", 'pred'),

        ("Normalized confusion matrix: all", 'all'),

    ]



fig = plt.figure(figsize=(10, 10), facecolor="w")

fig.subplots_adjust(hspace=0.2, wspace=0.4)

i = 0

for title, normalize in titles_options:

    i += 1

    ax = fig.add_subplot(2, 2, i)

    disp = plot_confusion_matrix(

                        rf,

                        X_test,

                        y_test,

                        display_labels=class_names,

                        cmap=plt.cm.Blues,

                        normalize=normalize,

                        ax=ax,

                    )



    disp.ax_.set_title(title)



    print(title)

    print(disp.confusion_matrix)

plt.show()
from sklearn.metrics import roc_curve, auc

# AUC Score

Y_score = rf.predict_proba(X_test)

Y_score = Y_score[:,1]



#print(Y_score)

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=Y_score)



# Drawing ROC curve

print('AUC score %0.3f' % auc(fpr, tpr))

plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc(fpr, tpr), color='red')

plt.plot([0, 1], [0, 1], color='black', linestyle='--')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="best")

plt.show()