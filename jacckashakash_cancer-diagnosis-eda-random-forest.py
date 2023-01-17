import pandas as pd

import matplotlib.pyplot as plt

import re

import time

import warnings

import numpy as np

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import SMOTE

from collections import Counter

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

#from sklearn.cross_validation import StratifiedKFold 

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")



from mlxtend.classifier import StackingClassifier



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')

print('Number of data points : ', data.shape[0])

print('Number of features : ', data.shape[1])

print('Features : ', data.columns.values)

#y_true=data['Class'].values

#data=data.drop('Class', axis=1)



data.head()

test_variant = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/test_variants.zip')

test_text = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/test_text.zip', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","TEXT"])
# note the seprator in this file

data_text =pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_text.zip',sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

print('Number of data points : ', data_text.shape[0])

print('Number of features : ', data_text.shape[1])

print('Features : ', data_text.columns.values)

data_text.head()
# loading stop words from nltk library

#import nltk

#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))





def nlp_preprocessing(total_text, index, column):

    if type(total_text) is not int:

        string = ""

        # replace every special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ', total_text)

        # converting all the chars into lower-case.

        total_text = total_text.lower()

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                string += word + " "

        

        return string

#start_time = time.clock()

for index, row in data_text.iterrows():

    if type(row['TEXT']) is str:

        data_text['TEXT'][index]=nlp_preprocessing(row['TEXT'], index, 'TEXT')

    else:

        print("there is no text description for id:",index)

#print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")

#start_time = time.clock()

for index1, row1 in test_text.iterrows():

    if type(row1['TEXT']) is str:

        test_text['TEXT'][index1]=nlp_preprocessing(row1['TEXT'], index1, 'TEXT')

    else:

        print("there is no text description for id:",index1)

#print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
#merging both gene_variations and text data based on ID

d=pd.merge(test_variant,test_text,on='ID',how='left')

test_index = d['ID'].values

print(d.shape)

result = pd.merge(data, data_text,on='ID', how='left')

second=d.shape[0]



akku_data=result.copy()

#akku_data['Class']=y_true

y_true=result['Class'].values

result=result.drop('Class', axis=1)

print(akku_data.shape)

first=result.shape[0]

print(first)

print(second)

result = np.concatenate((result, d), axis=0)

result = pd.DataFrame(result)

result.columns= ["ID", "Gene", "Variation", "TEXT"]

print(result.head())

print(result.shape)

#for i in range(result['TEXT'].shape[0]):

 #   nlp_preprocessing(result['TEXT'][i],i,'TEXT')
result[result.isnull().any(axis=1)]

result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']



akku_data[akku_data.isnull().any(axis=1)]

akku_data.loc[akku_data['TEXT'].isnull(),'TEXT'] = akku_data['Gene'] +' '+akku_data['Variation']
y_true1 = akku_data['Class'].values

#result.drop('Class', axis=1)

result.Gene = result.Gene.str.replace('\s+', '_')

result.Variation = result.Variation.str.replace('\s+', '_')



# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

X_train, test_df, y_train, y_test = train_test_split(akku_data, y_true1, test_size=0.2)

# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]

train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)



print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in test data:', test_df.shape[0])

print('Number of data points in cross validation data:', cv_df.shape[0])
# it returns a dict, keys as class labels and values as the number of data points in that class

train_class_distribution = train_df['Class'].value_counts().sort_values()

test_class_distribution = test_df['Class'].value_counts().sort_values()

cv_class_distribution = cv_df['Class'].value_counts().sort_values()



my_colors = 'rgbkymc'

train_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of yi in train data')

plt.grid()

plt.show()



# ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

# -(train_class_distribution.values): the minus sign will give us in decreasing order

sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')



    

print('-'*80)

my_colors = 'rgbkymc'

test_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of yi in test data')

plt.grid()

plt.show()



# ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

# -(train_class_distribution.values): the minus sign will give us in decreasing order

sorted_yi = np.argsort(-test_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')



print('-'*80)

my_colors = 'rgbkymc'

cv_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of yi in cross validation data')

plt.grid()

plt.show()



# ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

# -(train_class_distribution.values): the minus sign will give us in decreasing order

sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3), '%)')

# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)



    B =(C/C.sum(axis=0))

    

    

    labels = [1,2,3,4,5,6,7,8,9]

 

    print("-"*20, "Confusion matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing B in heatmap format

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
unique_genes = train_df['Gene'].value_counts()

print('Number of Unique Genes :', unique_genes.shape[0])

# the top 10 genes that occured most

print(unique_genes.head(10))

s = sum(unique_genes.values);

h = unique_genes.values/s;

plt.plot(h, label="Histrogram of Genes")

plt.xlabel('Index of a Gene')

plt.ylabel('Number of Occurances')

plt.legend()

plt.grid()

plt.show()

c = np.cumsum(h)

plt.plot(c,label='Cumulative distribution of Genes')

plt.grid()

plt.legend()

plt.show()


# one-hot encoding of Gene feature.

print(first)

gene_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(result['Gene'][0:first])

print(len(result['Gene'][0:first]))

test_gene_feature_onehotCoding = gene_vectorizer.transform(result['Gene'][first:])

#cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
alpha = [10 ** x for x in range(-5, 1)] 





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feature_onehotCoding[0:len(y_train)], y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_gene_feature_onehotCoding[0:len(y_train)], y_train)

    predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding[0:len(y_cv)])

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_gene_feature_onehotCoding[0:len(y_train)], y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_gene_feature_onehotCoding[0:len(y_train)], y_train)



predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding[0:len(y_train)])

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

#predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding[0:len(y_cv)])

#print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding[0:len(y_test)])

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

#plot_confusion_matrix(y_test, predict_y)
unique_variations = train_df['Variation'].value_counts()

print('Number of Unique Variations :', unique_variations.shape[0])

# the top 10 variations that occured most

print(unique_variations.head(10))

s = sum(unique_variations.values);

h = unique_variations.values/s;

plt.plot(h, label="Histrogram of Variations")

plt.xlabel('Index of a Variation')

plt.ylabel('Number of Occurances')

plt.legend()

plt.grid()

plt.show()

c = np.cumsum(h)

print(c)

plt.plot(c,label='Cumulative distribution of Variations')

plt.grid()

plt.legend()

plt.show()
# one-hot encoding of variation feature.

variation_vectorizer = CountVectorizer()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(result['Variation'][0:first])

test_variation_feature_onehotCoding = variation_vectorizer.transform(result['Variation'][first:])

#cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
alpha = [10 ** x for x in range(-5, 1)]



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_variation_feature_onehotCoding[0:len(y_train)], y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_variation_feature_onehotCoding[0:len(y_train)], y_train)

    predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding[0:len(y_cv)])

    

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_variation_feature_onehotCoding[0:len(y_train)], y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_variation_feature_onehotCoding[0:len(y_train)], y_train)



predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding[0:len(y_train)])

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

#predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

#print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding[0:len(y_test)])

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

# cls_text is a data frame

# for every row in data fram consider the 'TEXT'

# split the words by space

# make a dict with those words

# increment its count whenever we see that word



def extract_dictionary_paddle(cls_text):

    dictionary = defaultdict(int)

    for index, row in cls_text.iterrows():

        for word in row['TEXT'].split():

            dictionary[word] +=1

    return dictionary



##########response coding function

import math

#https://stackoverflow.com/a/1602964

def get_text_responsecoding(df):

    text_feature_responseCoding = np.zeros((df.shape[0],9))

    for i in range(0,9):

        row_index = 0

        for index, row in df.iterrows():

            sum_prob = 0

            for word in row['TEXT'].split():

                sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))

            text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['TEXT'].split()))

            row_index += 1

    return text_feature_responseCoding
text_vectorizer = CountVectorizer(min_df=3)

#print(result['TEXT'].shape[0])

train_text_feature_onehotCoding = text_vectorizer.fit_transform(result['TEXT'][0:first])

# getting all the feature names (words)

train_text_features= text_vectorizer.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))





print("Total number of unique words in train data :", len(train_text_features))



######################                          Normalization                   ################

# don't forget to normalize every feature

train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

test_text_feature_onehotCoding = text_vectorizer.transform(result['TEXT'][first:])

# don't forget to normalize every feature

test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

#cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])

# don't forget to normalize every feature

#cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
# Train a Logistic regression+Calibration model using text features whicha re on-hot encoded

alpha = [10 ** x for x in range(-5, 1)]





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_text_feature_onehotCoding[0:len(y_train)], y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_text_feature_onehotCoding[0:len(y_train)], y_train)

    predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding[0:len(y_cv)])

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_text_feature_onehotCoding[0:len(y_train)], y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_text_feature_onehotCoding[0:len(y_train)], y_train)



predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding[0:len(y_train)])

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

#predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)

#print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding[0:len(y_test)])

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

# = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

print(train_x_onehotCoding.shape)

train_y = np.array(list(y_true))

print(train_x_onehotCoding.shape)

test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

print(test_x_onehotCoding.shape)

test_y = np.array(list(y_true))



#cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

#cv_y = np.array(list(cv_df['Class']))
def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    pred_y = sig_clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))

    # calculating the number of data points that are misclassified

    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_y))/test_y.shape[0])

    plot_confusion_matrix(test_y, pred_y)
alpha = [100,200,500,1000,2000]

max_depth = [5, 10]

cv_log_error_array = []



for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_onehotCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_onehotCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(test_x_onehotCoding[0:len(train_y)])

        cv_log_error_array.append(log_loss(train_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(train_y, sig_clf_probs)) 





best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(test_x_onehotCoding[0:len(y_train)])



print(len(test_index))





print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

#predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

#print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print(len(predict_y))

submission = pd.DataFrame(predict_y)

submission['id'] = test_index

submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']

submission.to_csv("submission_all.csv",index=False)

submission.head()

#('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))









############             best aplha  -hyper parameter

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)



#confusion matrix

#predict_and_plot_confusion_matrix(train_y,test_x_onehotCoding,test_y,pre clf)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)

pred_y = sig_clf.predict(test_x_onehotCoding)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

#print("Log loss :",log_loss(train_y[0:len(sig_clf.predict_proba(test_x_onehotCoding))], sig_clf.predict_proba(test_x_onehotCoding)))

    # calculating the number of data points that are misclassified

#print("Number of mis-classified points :", np.count_nonzero((pred_y- train_y))/train_y.shape[0])

print(pred_y.shape)

print(train_y.shape)

plot_confusion_matrix(train_y, pred_y[0:first])