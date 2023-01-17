import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from scipy.sparse import hstack

from sklearn.preprocessing import normalize

import time, re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

import sklearn.metrics as metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename)) 
codestarttime = time.clock()

# Read training variant csv.

trainingV = pd.read_csv(r'/kaggle/input/msk-redefining-cancer-treatment/training_variants',encoding = 'utf-8')
trainingV.head()
# Read training text csv.

trainingT = pd.read_csv(r'/kaggle/input/msk-redefining-cancer-treatment/training_text',sep='\|\|', header = None, skiprows = 1, names = ['ID','Text'],encoding = 'utf-8')

trainingT.head()
#Merging both Data Frames

trainData = trainingV.merge(trainingT,how= 'inner')
trainData.head()
# Re-ordering columns

trainData = trainData.reindex(columns=['ID','Gene','Variation','Text','Class'])
trainData.head()
trainData.isnull().sum()
trainData.shape
trainData = trainData[~trainData.Text.isnull()]
trainData.shape
trainData.info()
df = trainData.groupby('Class').Gene.describe()

df = df.reset_index()

df
plt.figure(figsize=(15,5))

sns.barplot(x = 'Class',y = 'count',data= df)

plt.title('Count of Gene in Each Class')
# Install WordCloud for plotting the most common words in each class

!pip install wordcloud

#Function to remove any special character, any extra spaces in the Text column

def text_preprocessing(total_text):

    if type(total_text) is not int:

        string = ""

        # replace every special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))

        # replace multiple spaces with single space

        total_text = re.sub('\s+', ' ', total_text)

        # converting all the chars into lower-case.

        total_text = total_text.lower()



        for word in total_text.split():

            string += word + " "



        return string
#text processing stage.

start_time = time.clock()

trainData.Text = trainData.Text.apply(text_preprocessing)

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")

trainData.Text.head()
# Get top n words in the Text

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer(stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    string = ''

    words = [string + x[0] for x in words_freq[:n]]

    return ' '.join(words)
# Plot Word Cloud for the top N words in the class

def plot_wordCloud(df,Class):

    df = df[df.Class == Class]

    text = df.Text

    common2500Words = get_top_n_words(text,2500)

    wordcloud = WordCloud(background_color="white").generate(common2500Words)

    plt.figure(figsize= (15,5))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()
plot_wordCloud(df = trainData,Class = 1)
plot_wordCloud(df = trainData,Class = 2)
plot_wordCloud(df = trainData,Class = 3)
plot_wordCloud(df = trainData,Class = 4)
plot_wordCloud(df = trainData,Class = 5)
plot_wordCloud(df = trainData,Class = 6)
plot_wordCloud(df = trainData,Class = 7)
plot_wordCloud(df = trainData,Class = 8)
plot_wordCloud(df = trainData,Class = 9)
topWords = get_top_n_words(trainData.Text, n=100)

topWords
def get_top_words(corpus, n=None):

    vec = CountVectorizer(stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)         

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words = [x[0] for x in words_freq[:n]]

    count = [x[1] for x in words_freq[:n]]

    return words,count
def plot_topwords(df,Class):

    df = df[df.Class == Class]

    text = df.Text

    Words,Count = get_top_words(text,10)

    plt.figure(figsize= (12,5))

    sns.barplot(Count,Words)

    plt.title('Class =' + str(Class))

    plt.xlabel("Count of Words")

    plt.show()
plot_topwords(df = trainData,Class = 1)
plot_topwords(df = trainData,Class = 2)
plot_topwords(df = trainData,Class = 3)
plot_topwords(df = trainData,Class = 4)
plot_topwords(df = trainData,Class = 5)
plot_topwords(df = trainData,Class = 6)
plot_topwords(df = trainData,Class = 7)
plot_topwords(df = trainData,Class = 8)
plot_topwords(df = trainData,Class = 9)
trainData['Number of Words'] = trainData.Text.apply(lambda x: len(x.split()))

trainData.head()
plt.figure(figsize=(12, 8))

sns.distplot(trainData['Number of Words'])

plt.xlabel('Number of words in text', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title("Frequency of number of words", fontsize=15)

plt.show()
df = trainData[trainData['Number of Words'] < 30000]

plt.figure(figsize=(12,8))

sns.boxplot(y= 'Number of Words' ,x='Class', data= df )

plt.xlabel('Class', fontsize=12)

plt.ylabel('Text - Number of words', fontsize=12)

plt.show()
X = trainData.drop(columns=['ID','Class'])

Y = trainData.Class

x_train,x_test, y_train, y_test = train_test_split(X,Y,train_size = 0.7, random_state = 100)
x_train.shape
y_train.shape
geneCV = CountVectorizer()

xtrain_gene_feature = geneCV.fit_transform(x_train.Gene)

xtest_gene_feature = geneCV.transform(x_test.Gene)
variationCV = CountVectorizer()

xtrain_variation_feature = variationCV.fit_transform(x_train.Variation)

xtest_variation_feature = variationCV.transform(x_test.Variation)
textCV = CountVectorizer(stop_words= 'english',min_df= 5 )

xtrain_text_feature = textCV.fit_transform(x_train.Text)

xtest_text_feature = textCV.transform(x_test.Text)

xtrain_text_feature =  normalize(xtrain_text_feature, axis=0)

xtest_text_feature =  normalize(xtest_text_feature, axis=0)
train_gene_var_text = hstack((xtrain_gene_feature,xtrain_variation_feature,xtrain_text_feature)).tocsr()

test_gene_var_text = hstack((xtest_gene_feature,xtest_variation_feature,xtest_text_feature)).tocsr()
train_gene_var_text.shape
def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):

    clf.fit(train_x, train_y)

#     sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

#     sig_clf.fit(train_x, train_y)

    pred_y = clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    print("Log loss :",log_loss(test_y, clf.predict_proba(test_x)))

    # calculating the number of data points that are misclassified

    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_y))/test_y.shape[0])

    plot_confusion_matrix(test_y, pred_y)
def plot_confusion_matrix(y_test,pred_y):

    plt.figure(figsize=(20,7))

    labels = [1,2,3,4,5,6,7,8,9]

    confuMatrix = confusion_matrix(y_test,pred_y)

    sns.heatmap(confuMatrix,annot= True,cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')
# We build a Logistic regression with default Parameters and check the perfomance

LR = LogisticRegression()

LR.fit(train_gene_var_text,y_train)
#predict log-loss for train data

predict_y = LR.predict_proba(train_gene_var_text)

print("The train log loss is:",log_loss(y_train, predict_y, labels=LR.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = LR.predict_proba(test_gene_var_text)

print("The test log loss is:",log_loss(y_test, predict_y, labels=LR.classes_, eps=1e-15))

#We build logistic regression and find out best parameters(alpha and penalty) with Grid search and 10 fold CV on train data.



#Make a dict of our parameters

alpha = [10 ** x for x in range(-6, 3)]

params = {'C':alpha,'penalty':['l2']}



#Build Logistic regression

clfLR = LogisticRegression(class_weight='balanced',multi_class='multinomial',solver='newton-cg',n_jobs= -1)

#Grid Search with 10 fold CV

random = GridSearchCV(clfLR,param_grid=params,n_jobs= -1,cv=10)

random.fit(train_gene_var_text,y_train)
best_alpha = random.best_params_['C']

best_penalty = random.best_params_['penalty']

print('The best value for Cost, C is ', best_alpha)
#build logistic regression with best hyper-parameters(alpha and penalty)

#instead of using one vs rest we use multinomial which performs better compared to ovr in this case.

#multinomial does not support linear solver so we use newton-cg as our optimization problem solver.

clfLR = LogisticRegression(class_weight='balanced', C=best_alpha, penalty=best_penalty,multi_class='multinomial',solver='newton-cg')

clfLR.fit(train_gene_var_text, y_train)
#predict log-loss for train data

predict_y = clfLR.predict_proba(train_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The train log loss is:",log_loss(y_train, predict_y, labels=LR.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = clfLR.predict_proba(test_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The test log loss is:",log_loss(y_test, predict_y, labels=LR.classes_, eps=1e-15))
sig_clfLR = CalibratedClassifierCV(clfLR, method="sigmoid")

sig_clfLR.fit(train_gene_var_text, y_train)
#predict log-loss for train data

predict_y = sig_clfLR.predict_proba(train_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The train log loss is:",log_loss(y_train, predict_y, labels=clfLR.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = sig_clfLR.predict_proba(test_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The test log loss is:",log_loss(y_test, predict_y, labels=clfLR.classes_, eps=1e-15))
predict_and_plot_confusion_matrix(train_gene_var_text,y_train,test_gene_var_text,y_test,clf = sig_clfLR)
# Running the random forest with default parameters.

rfc = RandomForestClassifier()

rfc.fit(train_gene_var_text,y_train)
#predict log-loss for train data

predict_y = rfc.predict_proba(train_gene_var_text)

print("The train log loss is:",log_loss(y_train, predict_y, labels=rfc.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = rfc.predict_proba(test_gene_var_text)

print("The test log loss is:",log_loss(y_test, predict_y, labels=rfc.classes_, eps=1e-15))
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 200),

    'n_estimators': [100,200, 300], 

    'max_features': [5, 10]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)



grid_search.fit(train_gene_var_text,y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# Building the Random Forest Classifier with best parameters

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=4,

                             min_samples_leaf=100, 

                             min_samples_split=200,

                             max_features=5,

                             n_estimators=100)

rfc.fit(train_gene_var_text,y_train)
#predict log-loss for train data

predict_y = rfc.predict_proba(train_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The train log loss is:",log_loss(y_train, predict_y, labels=rfc.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = rfc.predict_proba(test_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The test log loss is:",log_loss(y_test, predict_y, labels=rfc.classes_, eps=1e-15))
# to avoid rounding error while multiplying probabilites we use log-probability estimates.

#Probability calibration with sigmoid regression.

sig_clfRFC = CalibratedClassifierCV(rfc, method="sigmoid")

sig_clfRFC.fit(train_gene_var_text, y_train)
#predict log-loss for train data

predict_y = sig_clfRFC.predict_proba(train_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The train log loss is:",log_loss(y_train, predict_y, labels=rfc.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = sig_clfRFC.predict_proba(test_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The test log loss is:",log_loss(y_test, predict_y, labels=rfc.classes_, eps=1e-15))
predict_and_plot_confusion_matrix(train_gene_var_text,y_train,test_gene_var_text,y_test,clf = sig_clfRFC)
# Build Naive Bayes with default parameters

mnb = MultinomialNB()



mnb.fit(train_gene_var_text,y_train)

#predict log-loss for train data

predict_y = mnb.predict_proba(train_gene_var_text)

print( "The train log loss is:",log_loss(y_train, predict_y, labels=mnb.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = mnb.predict_proba(test_gene_var_text)

print("The test log loss is:",log_loss(y_test, predict_y, labels=mnb.classes_, eps=1e-15))


# We build Multinomial NB and find out best parameters(alpha) with grid search and 10 fold CV on train data.



#Make a dict of our parameters

alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

params = {'alpha':alpha}



#build multinomial NB

clf = MultinomialNB()

#Grid Search with 10 fold CV

random = GridSearchCV(clf,param_grid=params,cv=10,return_train_score=True,n_jobs=2)

random.fit(train_gene_var_text, y_train)
best_alpha = random.best_params_['alpha']

print("The best value for aplha is ", best_alpha)
# to avoid rounding error while multiplying probabilites we use log-probability estimates.

#Probability calibration with sigmoid regression.

sig_clfMNB = CalibratedClassifierCV(mnb, method="sigmoid")

sig_clfMNB.fit(train_gene_var_text, y_train)
#predict log-loss for train data

predict_y = sig_clfMNB.predict_proba(train_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The train log loss is:",log_loss(y_train, predict_y, labels=mnb.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = sig_clfMNB.predict_proba(test_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The test log loss is:",log_loss(y_test, predict_y, labels=mnb.classes_, eps=1e-15))
predict_and_plot_confusion_matrix(train_gene_var_text,y_train,test_gene_var_text,y_test,clf = sig_clfMNB)
# Build a Linear SVC model with default parameters

from sklearn.svm import LinearSVC

linearsvc = LinearSVC()

linearsvc.fit(train_gene_var_text,y_train)

# specify range of parameters (C) as a list

params = {"C": [0.1, 1, 10, 100]}



model = LinearSVC()



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV(estimator = model, param_grid = params,                          

                        cv = 10, 

                        verbose = 1,

                        n_jobs = -1,

                       return_train_score=True)   

# fit the model - it will fit 5 folds across all values of C

model_cv.fit(train_gene_var_text,y_train)  
best_C = model_cv.best_params_['C']

print("The best value for C is ", best_C)
sig_clfLinearSVC = CalibratedClassifierCV(LinearSVC(C = 0.1), method="sigmoid")

sig_clfLinearSVC.fit(train_gene_var_text, y_train)
#predict log-loss for train data

predict_y = sig_clfLinearSVC.predict_proba(train_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The train log loss is:",log_loss(y_train, predict_y, labels=linearsvc.classes_, eps=1e-15))



#predict log-loss for test data

predict_y = sig_clfLinearSVC.predict_proba(test_gene_var_text)

print('For values of best alpha = ', best_alpha,'penalty',best_penalty, "The test log loss is:",log_loss(y_test, predict_y, labels=linearsvc.classes_, eps=1e-15))
predict_and_plot_confusion_matrix(train_gene_var_text,y_train,test_gene_var_text,y_test,clf = sig_clfLinearSVC)
# Read  Test Variant files

testV = pd.read_csv(r'/kaggle/input/msk-redefining-cancer-treatment/test_variants',encoding = 'utf-8')

testV.head()
# Read  Test Text files

testT = pd.read_csv(r'/kaggle/input/msk-redefining-cancer-treatment/test_text',sep='\|\|', header = None, skiprows = 1, names = ['ID','Text'],encoding = 'utf-8')

testT.head()
testData = pd.merge(testV,testT)

testData.head()
#text processing stage.

start_time = time.clock()

testData.Text = testData.Text.apply(text_preprocessing)

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")

testData_gene_feature = geneCV.transform(testData.Gene)
testData_variation_feature = variationCV.transform(testData.Variation)
testData_text_feature = textCV.transform(testData.Text.astype(str))
testData_text_feature =  normalize(testData_text_feature, axis=0)
testData_gene_var_text = hstack((testData_gene_feature,testData_variation_feature,testData_text_feature)).tocsr()
testData_gene_var_text.shape
final_pred = sig_clfLR.predict(testData_gene_var_text)
final_pred
testData['predicted_class'] = final_pred
testData.head()
submission_df = pd.get_dummies(testData['predicted_class'],prefix= 'class',prefix_sep= ' ')
submission_df.reset_index(inplace= True)

submission_df.rename(columns={'index':'ID'},inplace= True)
submission_df.to_csv('submission.csv', index=False)
codeendtime = time.clock()

print('Code execution took: ', str((codeendtime - codestarttime)/60), 'mins')