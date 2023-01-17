## modelling libraries
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

## preprocessing libraries
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
import numpy as np 
import itertools
import os 

## visualization libraries
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

print ("all libraries imported successfully")
## read dataset
train_path = "../input/train.csv"
train_df = pd.read_csv(train_path)
train_copy = train_df.copy()

test_path = "../input/test.csv"
test_df = pd.DataFrame()
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)

print ("dataset loaded")
## separate predictors and target variables
_target = "author"
Y = train_df[_target]
distinct_Y = Y.value_counts().index

## separate the id column
_id = "id"
if _id == "": ## if id is not present, create a dummy 
    _id = "id"
    train_df[_id] = 1
    test_df[_id] = 1
if _id not in list(test_df.columns):
    test_df[_id] = 1
    
## drop the target and id columns
train_df = train_df.drop([_target, _id], axis=1)
test_id = test_df[_id]
test_df = test_df.drop([_id], axis=1)
## flag variables (used by bot to write the relevant code)
textcol = "text"
tag = "doc"
## snapshot of train and test
train_df.head()
tar_dist = dict(Counter(Y.values))

xx = list(tar_dist.keys())
yy = list(tar_dist.values())

plt.figure(figsize=(5,3))
sns.set(style="whitegrid")
ax = sns.barplot(x=xx, y=yy, palette="rocket")
ax.set_title('Distribution of Target')
ax.set_ylabel('count');
ax.set_xlabel(_target);
if tag == "doc":
    txts = []
    for i, y in enumerate(distinct_Y):
        txt = " ".join(train_copy[train_copy[_target] == y]["text"]).lower()
        txts.append(txt)

    for j, text in enumerate(txts):
        wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)
        wc.generate(text)
        plt.figure(figsize=(9,8))
        plt.axis("off")
        plt.title("Most frequent words - " + distinct_Y[j], fontsize=20)
        plt.imshow(wc.recolor(colormap= 'cool' , random_state=17), alpha=0.95)
        plt.show()
mcount = train_df.isna().sum() 
xx = mcount.index 
yy = mcount.values

missing_cols = 0
for each in yy:
    if each > 0:
        missing_cols += 1
print ("there are " + str(missing_cols) + " columns in the dataset having missing values")

if missing_cols > 0:
    plt.figure(figsize=(12,5))
    sns.set(style="whitegrid")
    ax = sns.barplot(x=xx, y=yy, palette="gist_rainbow")
    ax.set_title('Number of Missing Values')
    ax.set_ylabel('Number of Columns');
## find categorical columns in the dataset 
num_cols = train_df._get_numeric_data().columns
cat_cols = list(set(train_df.columns) - set(num_cols))

print ("There are " + str(len(num_cols)) + " numerical columns in the dataset")
print ("There are " + str(len(cat_cols)) + " object type columns in the dataset")
get_corr = False
corr = train_df.corr()
if len(corr) > 0:
    get_corr = True
    colormap = plt.cm.BrBG
    plt.figure(figsize=(10,10));
    plt.title('Pearson Correlation of Features', y=1.05, size=15);
    sns.heatmap(corr, linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
else:
    print ("No variables available for correlation")
columns = train_df.columns
num_cols = train_df._get_numeric_data().columns
cat_cols = list(set(columns) - set(num_cols))
    
if tag == "doc":
    print ("No columns available for label encoding")
elif len(cat_cols) > 0:
    for col in cat_cols: 
        le = LabelEncoder()
        
        if col in list(test_df.columns):
            le.fit(list(train_df[col].values) + list(test_df[col].values))
        else:
            le.fit(list(train_df[col].values))
        
        train_df[col] = le.transform(list(train_df[col].values))
        try:
            test_df[col] = le.transform(list(test_df[col].values))
        except:
            pass
        
## label encode the target variable (if object type)
if Y.dtype.name == "object":
    le = LabelEncoder()
    Y = le.fit_transform(Y.values)
if tag == "doc":
    train_df[textcol] = train_df[textcol].fillna("")
    if textcol in test_df:
        test_df[textcol] = test_df[textcol].fillna("")
else:
    ## for numerical columns, replace the missing values by mean
    train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].mean())
    try:
        test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].mean())
    except:
        pass 
    
    ## for categorical columns, replace the missing values by mode
    train_df[cat_cols] = train_df[cat_cols].fillna(train_df[cat_cols].mode())    
    try:
        test_df[cat_cols] = test_df[cat_cols].fillna(test_df[cat_cols].mode())
    except:
        pass
print ("Treated missing values in the dataset")
if tag == "doc":
    tfidf = TfidfVectorizer(min_df=3,  max_features=None, analyzer='word', 
                            token_pattern=r'\w{1,}', stop_words = 'english')
    tfidf.fit(list(train_df[textcol].values))
    xtrain = tfidf.transform(train_df[textcol].values) 
    if textcol in test_df.columns:
        xtest = tfidf.transform(test_df[textcol].values)
else:
    xtrain = train_df
    xtest = test_df
if tag != "doc":
    print ("Lets plot the dataset distributions after preprocessing step ... ")
    ## pair plots
    sns.pairplot(train_df, palette="cool")
    
    ## distributions
    columns=train_df.columns
    plt.subplots(figsize=(18,15))
    length=len(columns)
    for i,j in itertools.zip_longest(columns,range(length)):
        plt.subplot((length/2),3,j+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.5)
        train_df[i].hist(bins=20, edgecolor='white')
        plt.title(i)
    plt.show()
X_train, X_valid, y_train, y_valid = train_test_split(xtrain, Y, test_size=0.20, random_state=2018)
X_train, X_valid, y_train, y_valid = train_test_split(xtrain, Y, test_size=0.20, random_state=2018)
model1 = LogisticRegression()
model1.fit(X_train, y_train)
valp = model1.predict(X_valid)

def generate_auc(y_valid, valp, model_name):
    auc_scr = roc_auc_score(y_valid, valp)
    print('The AUC for ' +model_name+ ' is :', auc_scr)

    fpr, tpr, thresholds = roc_curve(y_valid, valp)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'purple', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'upper left')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if len(distinct_Y) == 2:
    generate_auc(y_valid, valp, model_name="logistic regression")
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,5));
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_valid, valp)
np.set_printoptions(precision=2)

plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=distinct_Y, title='Confusion matrix Validation Set')
plt.show()
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
valp = model2.predict(X_valid)

if len(distinct_Y) == 2:
    generate_auc(y_valid,valp, model_name="decision tree classifier")
cnf_matrix = confusion_matrix(y_valid, valp)
np.set_printoptions(precision=2)

plt.figure(figsize=(6,5));
plot_confusion_matrix(cnf_matrix, classes=distinct_Y, title='Confusion matrix Validation Set');
plt.show();
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
valp = model3.predict(X_valid)

if len(distinct_Y) == 2:
    generate_auc(y_valid,valp, model_name="random forest classifier")
cnf_matrix = confusion_matrix(y_valid, valp)
np.set_printoptions(precision=2)

plt.figure(figsize=(6,5));
plot_confusion_matrix(cnf_matrix, classes=distinct_Y, title='Confusion matrix Validation Set');
plt.show();
model4 = ExtraTreesClassifier()
model4.fit(X_train, y_train)
valp = model4.predict(X_valid)

if len(distinct_Y) == 2:
    generate_auc(y_valid,valp, model_name="extratrees classifier")
cnf_matrix = confusion_matrix(y_valid, valp)
np.set_printoptions(precision=2)

plt.figure(figsize=(6,5));
plot_confusion_matrix(cnf_matrix, classes=distinct_Y, title='Confusion matrix Validation Set');
plt.show();
model5 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.01)
model5.fit(X_train, y_train)
valp = model5.predict(X_valid)

if len(distinct_Y) == 2:
    generate_auc(y_valid,valp, model_name="xgboost")
cnf_matrix = confusion_matrix(y_valid, valp)
np.set_printoptions(precision=2)

plt.figure(figsize=(6,5))
plot_confusion_matrix(cnf_matrix, classes=distinct_Y, title='Confusion matrix Validation Set')
plt.show()
plt.figure(figsize=(12,8))
xgb.plot_importance(model5, max_num_features=10);
models = [model1, model2, model3, model4, model5]
preds = np.zeros(shape=(xtest.shape[0],))
if xtest.shape[0] == 0:
    print ("this is a dataset kernel, no test data for predictions")
else:
    for model in models:
        pred = model.predict(xtest)/ len(models)
        preds += pred
    print (preds[:100])
if xtest.shape[0] == 0:
    print ("This is a dataset kernel, no need to create a submission file :)")
else:
    pred = model5.predict(xtest)
    sub = pd.DataFrame()
    sub[_id] = test_id
    sub[_target] = pred
    sub.to_csv("baseline_submission.csv", index=False)
    print ("Submission File Generated, here is the snapshot: ")
    print (sub.head(10))