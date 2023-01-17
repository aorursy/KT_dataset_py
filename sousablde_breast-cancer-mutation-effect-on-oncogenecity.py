import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

import math

import re

import time

import warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from nltk.corpus import stopwords



from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, accuracy_score, log_loss

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB, GaussianNB



from imblearn.over_sampling import SMOTE



from collections import Counter, defaultdict



from scipy.sparse import hstack



from sklearn.ensemble import RandomForestClassifier



from mlxtend.classifier import StackingClassifier
#training_variants is a comma separated file

training_variants_df = pd.read_csv('../input/training_variants')



#training_text is separated by // delimeter

training_text_df = pd.read_csv('../input/training_text', sep = "\|\|", engine = "python", names = 

                              ["ID", "TEXT"], skiprows = 1)
#checking the first rows for training variants

training_variants_df.head()
#checking the shape of the data for training_Variants

training_variants_df.shape
#getting info from the file

training_variants_df.info()
#checking the first rows for training_text

training_text_df.head()
#getting info from training_text

training_text_df.info()
training_variants_df.Class.unique()
#removing the stop words from the text, I will be using the natural language toolkit to help me

stop_words = set(stopwords.words('english'))
#create a function to preprocess it

def data_preprocess(input_text, ind, col):

    #remove int values from the text data

    if type(input_text) is not int:

        string = ""

        #replacing special characters with space

        input_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(input_text))

        #replacing multiple spaces with single space

        input_text = re.sub('\s+', ' ', str(input_text))

        #make all lower case

        input_text = input_text.lower()

        

        for word in input_text.split():

            #keep everything but the stop words

            if not word in stop_words:

                string += word + " "

        

        training_text_df[col][ind] = string
for index, row, in training_text_df.iterrows():

    if type(row['TEXT']) is str:

        data_preprocess(row['TEXT'], index, 'TEXT')
training_merged = pd.merge(training_variants_df, training_text_df,on="ID",how = 'left')

training_merged.head()

#since I am checking along the columns I will pass axis = 1

training_merged[training_merged.isnull().any(axis = 1)]
training_merged.loc[training_merged['TEXT'].isnull(), 'TEXT'] = training_merged['Gene'] + ' ' + training_merged['Variation']
training_merged[training_merged.isnull().any(axis = 1)]
#making sure the columns gene and variation have no spaces in them, replacing them with underscore

y_real = training_merged['Class'].values

training_merged.Gene = training_merged.Gene.str.replace('\s+', '_')

training_merged.Variation = training_merged.Variation.str.replace('\s+', '_')
#splitting data into test set

X_train, x_test, y_train, y_test = train_test_split(training_merged, y_real, stratify = y_real, test_size = 0.2)



#splitting the data into training set

X_train, x_crossval, y_train, y_crossval = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.2)

print('Data points in X_train: ', X_train.shape[0])

print('Data points in x_test: ', x_test.shape[0])

print('Data points in cross-val: ', x_crossval.shape[0])
#checking how data was distributed

train_class_distributed = X_train['Class'].value_counts().sort_index()

test_class_distributed = x_test['Class'].value_counts().sort_index()

crossval_class_distributed = x_crossval['Class'].value_counts().sort_index()
for a, b, c in zip(train_class_distributed, test_class_distributed, crossval_class_distributed):

    print(a, b, c)
#visualizing the distribution

train_class_distributed.plot(kind = 'bar')



plt.xlabel('Class')

plt.ylabel('Data Points per Class')

plt.title('Y distribution in train')

plt.show()
#checking the distribution per percentage of data in the classes

sorted_y = np.argsort(-train_class_distributed.values)

for i in sorted_y:

    print('Number of data points in class', i+1, ':',train_class_distributed.values[i], '(', np.round((train_class_distributed.values[i]/X_train.shape[0]*100), 3), '%)')



n_data_points = []

n_class = []

percent_data = []

for i in sorted_y:

    n_class.append(i+1)

    n_data_points.append(train_class_distributed.values[i])

    percent_data.append(np.round((train_class_distributed.values[i]/X_train.shape[0]*100), 3))

#plotting the results for the distribution

figureObject, axesObject = plt.subplots()

axesObject.pie(percent_data, labels = n_class, autopct='%1.1f', startangle=90)

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

wedges, texts = plt.pie(percent_data, shadow=True, startangle=90)

axesObject.axis('equal')

plt.tight_layout()

plt.show()
#checking the distribution per percentage of data in the classes, trying to get fancy with my graphs, failing but still feeling pretty proud lol

#need to come back to this after I am not saturated of looking at it

sorted_y = np.argsort(-train_class_distributed.values)

for i in sorted_y:

    print('Number of data points in class', i+1, ':',train_class_distributed.values[i], '(', np.round((train_class_distributed.values[i]/X_train.shape[0]*100), 3), '%)')



n_data_points = []

n_class = []

percent_data = []

for i in sorted_y:

    n_class.append(i+1)

    n_data_points.append(train_class_distributed.values[i])

    percent_data.append(np.round((train_class_distributed.values[i]/X_train.shape[0]*100), 3))





figureObject, axesObject = plt.subplots()

axesObject.pie(percent_data, labels = n_class, autopct='%1.1f', startangle=90)

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

wedges, texts = plt.pie(percent_data, shadow=True, startangle=90)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    plt.annotate(percent_data[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)

#I need to generate 9 random numbers (because we have 9 classes) and their sum should total 9 (again, 9 classes) so 

#the overall probability can total 1

test_len = x_test.shape[0]

crossval_len = x_crossval.shape[0]
#creating an output array with the same size as the data I am using for cross validation

y_predicted_crossval = np.zeros((crossval_len, 9))



for i in range (crossval_len):

    rand_probab = np.random.rand(1,9)

    y_predicted_crossval[i] = ((rand_probab/sum(sum(rand_probab)))[0])



print ("Log loss on the cross validation data using random model", log_loss(y_crossval, y_predicted_crossval, eps = 1e-15))
#checking for the error in the test set

#creating an output array again

y_predicted_test = np.zeros((test_len, 9))



for i in range(test_len):

    rand_probab = np.random.rand(1,9)

    y_predicted_test[i] = ((rand_probab/sum(sum(rand_probab)))[0])

    

print ("Log loss on the test data using random model", log_loss(y_test, y_predicted_test, eps = 1e-15))
#using argmax to find the maximum probability 

y_predicted = np.argmax(y_predicted_test, axis = 1)
#check the output, should be 600 and something values

y_predicted
#the index on the previous output seemed to start at 0, let's correct it so it matches the classes proposed on the 

#problem statement

y_predicted = y_predicted + 1

y_predicted
C = confusion_matrix(y_test, y_predicted)
labels = [1,2,3,4,5,6,7,8,9]

plt.figure(figsize = (20,7))

sb.heatmap(C, annot = True, cmap = 'YlGnBu', fmt = '.3f', xticklabels = labels, yticklabels = labels)

plt.xlabel('Predicted Classes')

plt.ylabel('Actual Classes')

plt.show()
B = (C/C.sum(axis = 0))
plt.figure(figsize = (20,7))

sb.heatmap(B, annot = True, cmap = 'YlGnBu', fmt = '.3f', xticklabels = labels, yticklabels = labels)

plt.xlabel('Predicted Classes')

plt.ylabel('Actual Classes')

plt.show()
A =(((C.T)/(C.sum(axis=1))).T)
plt.figure(figsize=(20,7))

sb.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
unique_genes = X_train['Gene'].value_counts()

print('Number of Unique Genes: ', unique_genes.shape[0])

print (unique_genes.head(10))
s = sum(unique_genes.values)

h = (unique_genes.values/s)

c = np.cumsum(h)

plt.plot(c, label = "Cumulative distribution of genes")

#plt.ylim(0, 1.0)

#plt.xlim(0, 200)

plt.grid()

plt.legend()

plt.show()
#one-hot encoding of the Gene feature

gene_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(X_train['Gene'])

test_gene_feature_onehotCoding = gene_vectorizer.transform(x_test['Gene'])

crossval_gene_feature_onehotCoding = gene_vectorizer.transform(x_crossval['Gene'])



train_gene_feature_onehotCoding.shape
#name of the columns after one-hot encoding

gene_vectorizer.get_feature_names()
# get_gv_fea_dict: Get Gene variation Feature Dictionary

def get_gv_fea_dict(alpha, feature, df):



    value_count = X_train[feature].value_counts()

    

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation

    gv_dict = dict()

    

    # denominator will contain the number of time that particular feature occured in whole data

    for i, denominator in value_count.items():

        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class

        # vec is 9 diamensional vector

        vec = []

        for k in range(1,10):



            cls_cnt = X_train.loc[(X_train['Class']==k) & (X_train[feature]==i)]

            

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data

            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))



        # we are adding the gene/variation to the dict as key and vec as value

        gv_dict[i]=vec

    return gv_dict
# Get Gene variation feature

def get_gv_feature(alpha, feature, df):

    gv_dict = get_gv_fea_dict(alpha, feature, df)

    # value_count is similar in get_gv_fea_dict

    value_count = X_train[feature].value_counts()

    

    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data

    gv_fea = []

    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea

    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea

    for index, row in df.iterrows():

        if row[feature] in dict(value_count).keys():

            gv_fea.append(gv_dict[row[feature]])

        else:

            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])



    return gv_fea
#response-coding of the Gene feature

# alpha is used for laplace smoothing

alpha = 1

# train gene feature

train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", X_train))

# test gene feature

test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", x_test))

# cross validation gene feature

cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", x_crossval))
train_gene_feature_responseCoding.shape
# We need a hyperparemeter for SGD classifier.

alpha = [10 ** x for x in range(-5, 1)]
#now to implementing a logistic regression using an sgd classifier and the calibrated classifier

#I will be using Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.



cv_log_error_array = []

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feature_onehotCoding, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_gene_feature_onehotCoding, y_train)

    y_predicted = sig_clf.predict_proba(crossval_gene_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

    
#I will make a visual check for the alphas

fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
#now to use the best alpha to compute the log-loss

best_alpha = np.argmin(cv_log_error_array)





clf = SGDClassifier(alpha = alpha[best_alpha], penalty = 'l2', loss = 'log', random_state = 42)

clf.fit(train_gene_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method = "sigmoid")

sig_clf.fit(train_gene_feature_onehotCoding, y_train)



y_predicted = sig_clf.predict_proba(train_gene_feature_onehotCoding)

print('best alpha = ', alpha[best_alpha], " train log loss :",log_loss(y_train, y_predicted, labels=clf.classes_, eps=1e-15))

y_predicted = sig_clf.predict_proba(crossval_gene_feature_onehotCoding)

print('best alpha = ', alpha[best_alpha], " cross validation log loss :",log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

y_predicted = sig_clf.predict_proba(test_gene_feature_onehotCoding)

print('best alpha = ', alpha[best_alpha], " test log loss :",log_loss(y_test, y_predicted, labels=clf.classes_, eps=1e-15))

test_coverage=x_test[x_test['Gene'].isin(list(set(X_train['Gene'])))].shape[0]

cv_coverage=x_crossval[x_crossval['Gene'].isin(list(set(X_train['Gene'])))].shape[0]



print('1. In test data',test_coverage, 'out of',x_test.shape[0], ":",(test_coverage/x_test.shape[0])*100)

print('2. In cross validation data',cv_coverage, 'out of ',x_crossval.shape[0],":" ,(cv_coverage/x_crossval.shape[0])*100)
from wordcloud import WordCloud

#from matplotlib_venn import venn

import matplotlib_venn as venn2

from matplotlib_venn import venn2

#from matplotlib_venn_wordcloud import venn3_wordcloud

train_items = pd.unique(X_train['Gene'])

test_items = pd.unique(x_test['Gene'])

crossval_items = pd.unique(x_crossval['Gene'])



in_train_test = set(test_items).intersection(set(train_items))

in_train_crossval = set(crossval_items).intersection(set(train_items))

in_test_crossval = set(crossval_items).intersection(set(test_items))



only_train = set(train_items).difference(set(test_items))

only_train2 = set(train_items).difference(set(crossval_items))

exclusive_train = set(only_train).difference(set(only_train2))



only_test = set(test_items).difference(set(train_items))

only_test2 = set(test_items).difference(set(crossval_items))

exclusive_test = set(only_test).difference(set(only_test2))



only_crossval = set(crossval_items).difference(set(train_items))

only_crossval2 = set(crossval_items).difference(set(test_items))

exclusive_crossval = set(only_crossval).difference(set(only_crossval2))



#quick print out to check if things make sense

print("Items in train: ", len(train_items), " Overlap train-test: ", len(in_train_test), " Overlap train-crossval: ",

      len(in_train_crossval), " Exclusive to train: ", len(exclusive_train))

figure, axes = plt.subplots(1, 3, figsize = (15,15), squeeze = False)

#overlap between train and test

v = venn2(subsets = (len(train_items), len(in_train_test), len(test_items)), 

      set_labels=('Train', 'Test'), ax = axes[0][0]);





#overlap between train and crossval

v2 = venn2(subsets = (len(train_items), len(in_train_crossval), len(crossval_items)), 

      set_labels=('Train', 'Crossval'), ax = axes[0][1]);



#overlap between test and crossval

v2 = venn2(subsets = (len(test_items), len(in_test_crossval), len(crossval_items)), 

      set_labels=('Test', 'Crossval'), ax = axes[0][2]);



plt.tight_layout()

plt.show()



print('Unique genes in train: ', len(only_train), ' Unique genes in test: ', len(only_test))

print('Unique genes in train: ', len(only_train2), ' Unique genes in crossval: ', len(only_crossval))
unique_variations = X_train['Variation'].value_counts()

print('Number of variations: ', unique_variations.shape[0])



print(unique_variations.head(10))

#most abundant variations visualization

unique_variations.head(10).plot(kind = 'bar')



plt.xlabel('Type of variation')

plt.ylabel('Abundance of variation')

plt.title('Top 10 Gene Variations found in the dataset')

plt.show()

#I will take a further look at the gene variation classifications

unique_variations.head(30)
#Looking at the distribution of the variations

s = sum(unique_variations.values)

h = unique_variations.values/s

c = np.cumsum(h)

print(c)

plt.plot(c,label = 'Cumulative distribution of variations')

plt.legend()

plt.show()

variation_vectorizer = CountVectorizer()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(X_train['Variation'])

test_variation_feature_onehotCoding = variation_vectorizer.transform(x_test['Variation'])

cv_variation_feature_onehotCoding = variation_vectorizer.transform(x_crossval['Variation'])



train_variation_feature_onehotCoding.shape
# alpha is used for laplace smoothing

alpha = 1

# train variation feature

train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", X_train))

# test variation feature

test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", x_test))

# cross validation variation feature

cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", x_crossval))



train_variation_feature_responseCoding.shape
# We need a hyperparemeter for SGD classifier.

alpha = [10 ** x for x in range(-5, 1)]
cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_variation_feature_onehotCoding, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_variation_feature_onehotCoding, y_train)

    y_predicted = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

    

    cv_log_error_array.append(log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))
# pltotting to select best alpha

fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_variation_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_variation_feature_onehotCoding, y_train)



y_predicted = sig_clf.predict_proba(train_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, y_predicted, labels=clf.classes_, eps=1e-15))

y_predicted = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

y_predicted = sig_clf.predict_proba(test_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, y_predicted, labels=clf.classes_, eps=1e-15))
test_coverage=x_test[x_test['Variation'].isin(list(set(X_train['Variation'])))].shape[0]

cv_coverage=x_crossval[x_crossval['Variation'].isin(list(set(X_train['Variation'])))].shape[0]
print('1. In test data',test_coverage, 'out of',x_test.shape[0], ":",(test_coverage/x_test.shape[0])*100)

print('2. In cross validation data',cv_coverage, 'out of ',x_crossval.shape[0],":" ,(cv_coverage/x_crossval.shape[0])*100)
#starting with a function to count each word in the text

def extract_dictionary_paddle(cls_text):

    dictionary = defaultdict(int)

    for index, row in cls_text.iterrows():

        for word in row['TEXT'].split():

            dictionary[word] +=1

    return dictionary
#test if a particular key was already found in the dictionary



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
# building a CountVectorizer with all the words that occured minimum 3 times in train data



text_vectorizer = CountVectorizer(min_df=3)

train_text_feature_onehotCoding = text_vectorizer.fit_transform(X_train['TEXT'])



# getting all the feature names (words)

train_text_features= text_vectorizer.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))





print("Total number of unique words in train data :", len(train_text_features))
dict_list = []

# dict_list =[] contains 9 dictoinaries each corresponds to a class



for i in range(1,10):

    cls_text = X_train[X_train['Class']==i]

    # build a word dict based on the words in that class

    dict_list.append(extract_dictionary_paddle(cls_text))

    # append it to dict_list



# dict_list[i] is build on i'th  class text data

# total_dict is buid on whole training text data



total_dict = extract_dictionary_paddle(X_train)
confuse_array = []

for i in train_text_features:

    ratios = []

    max_val = -1

    for j in range(0,9):

        ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))

    confuse_array.append(ratios)

confuse_array = np.array(confuse_array)
from collections import Counter

word_could_dict=Counter(text_fea_dict)

wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)



plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#response coding the text features



train_text_feature_responseCoding  = get_text_responsecoding(X_train)

test_text_feature_responseCoding  = get_text_responsecoding(x_test)

cv_text_feature_responseCoding  = get_text_responsecoding(x_crossval)
#converting row values so they can have a total sum of 1



train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T

test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T

cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T
#normalizing all my features



train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)





test_text_feature_onehotCoding = text_vectorizer.transform(x_test['TEXT'])

test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)





cv_text_feature_onehotCoding = text_vectorizer.transform(x_crossval['TEXT'])

cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
#we will sort the dictionaries by value (or at least their representation, dictionaries are orderless)



sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))

sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))
#and now to plot the frequencies

#c = Counter(sorted_text_occur)



labels, values = zip(*Counter(sorted_text_occur).items())



indexes = np.arange(len(labels))

width = 1



plt.bar(indexes, values, width)

plt.xticks(indexes + width * 0.5, labels)

plt.show()
cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_text_feature_onehotCoding, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_text_feature_onehotCoding, y_train)

    y_predicted = sig_clf.predict_proba(cv_text_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

#and to proceed to the visualization of the cross-validation error for each alpha:



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')



for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

    

plt.title("Cross Validation Error per alpha")

plt.xlabel("Alpha ")

plt.ylabel("Error measurement")

plt.show()
#using the best alpha:



best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_text_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_text_feature_onehotCoding, y_train)



y_predicted = sig_clf.predict_proba(train_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, y_predicted, labels=clf.classes_, eps=1e-15))

y_predicted = sig_clf.predict_proba(cv_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_crossval, y_predicted, labels=clf.classes_, eps=1e-15))

y_predicted = sig_clf.predict_proba(test_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, y_predicted, labels=clf.classes_, eps=1e-15))

#now to check how much overlap there is between text data



def get_intersec_text(df):

    df_text_vec = CountVectorizer(min_df=3)

    df_text_fea = df_text_vec.fit_transform(df['TEXT'])

    df_text_features = df_text_vec.get_feature_names()



    df_text_fea_counts = df_text_fea.sum(axis=0).A1

    df_text_fea_dict = dict(zip(list(df_text_features),df_text_fea_counts))

    len1 = len(set(df_text_features))

    len2 = len(set(train_text_features) & set(df_text_features))

    return len1,len2
len1,len2 = get_intersec_text(x_test)

print(np.round((len2/len1)*100, 3), "% of words from test that appear in the train set")

len1,len2 = get_intersec_text(x_crossval)

print(np.round((len2/len1)*100, 3), "% of words from Cross Validation that appear in the train set")
# creating a function to return the log-loss

def report_log_loss(train_x, train_y, test_x, test_y,  clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    sig_clf_probs = sig_clf.predict_proba(test_x)

    return log_loss(test_y, sig_clf_probs, eps=1e-15)
#function to plot the confusion matrix for y and y^

#Y hat (written ŷ ) is the predicted value of y (the dependent variable) in a regression equation. 

#It can also be considered to be the average value of the response variable.



def plot_confusion_matrix(test_y, y_predicted):

    C = confusion_matrix(test_y, y_predicted)

    

    A =(((C.T)/(C.sum(axis=1))).T)

    

    B =(C/C.sum(axis=0)) 

    labels = [1,2,3,4,5,6,7,8,9]

    # representing A in heatmap format

    print("-"*20, "Confusion matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sb.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sb.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing B in heatmap format

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sb.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

def predict_and_plot_confusion_matrix(train_x, train_y, test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    predicted_y = sig_clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))

    # calculating the number of data points that are misclassified

    print("Number of mis-classified points :", np.count_nonzero((predicted_y- test_y))/test_y.shape[0])

    plot_confusion_matrix(test_y, predicted_y)

    
def mis_class_datapoints(train_x, train_y, test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    predicted_y = sig_clf.predict(test_x)

    log_loss(test_y, sig_clf.predict_proba(test_x))

    m_class = np.count_nonzero((predicted_y- test_y))/test_y.shape[0]

    return m_class
# this function will be used just for naive bayes

# for the given indices, we will print the name of the features



def get_impfeature_names(indices, text, gene, var, no_features):

    gene_count_vec = CountVectorizer()

    var_count_vec = CountVectorizer()

    text_count_vec = CountVectorizer(min_df=3)

    

    gene_vec = gene_count_vec.fit(X_train['Gene'])

    var_vec  = var_count_vec.fit(X_train['Variation'])

    text_vec = text_count_vec.fit(X_train['TEXT'])

    

    fea1_len = len(gene_vec.get_feature_names())

    fea2_len = len(var_count_vec.get_feature_names())

    

    word_present = 0

    for i,v in enumerate(indices):

        if (v < fea1_len):

            word = gene_vec.get_feature_names()[v]

            yes_no = True if word == gene else False

            if yes_no:

                word_present += 1

                print(i, "Gene feature [{}] present in test data point [{}]".format(word,yes_no))

        elif (v < fea1_len+fea2_len):

            word = var_vec.get_feature_names()[v-(fea1_len)]

            yes_no = True if word == var else False

            if yes_no:

                word_present += 1

                print(i, "variation feature [{}] present in test data point [{}]".format(word,yes_no))

        else:

            word = text_vec.get_feature_names()[v-(fea1_len+fea2_len)]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

                print(i, "Text feature [{}] present in test data point [{}]".format(word,yes_no))



    print("Out of the top ",no_features," features ", word_present, "are present in query point")
# merging gene, variance and text features



train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

cv_gene_var_onehotCoding = hstack((crossval_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(X_train['Class']))



test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

test_y = np.array(list(x_test['Class']))



cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

crossval_y = np.array(list(x_crossval['Class']))





train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))

test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))

cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))



train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))

test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))

cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))

print("One hot encoding features :")

print("(number of data points * number of features) in train data = ", train_x_onehotCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_onehotCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding.shape)
print(" Response encoding features :")

print("(number of data points * number of features) in train data = ", train_x_responseCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_responseCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_responseCoding.shape)
alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = MultinomialNB(alpha=i) # as is requested for the classification

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(crossval_y, sig_clf_probs)) 
fig, ax = plt.subplots()

ax.plot(np.log10(alpha), cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))

plt.grid()

plt.xticks(np.log10(alpha))

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
best_alpha = np.argmin(cv_log_error_array)

nb_alpha = alpha[best_alpha]



clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)





predict_y = sig_clf.predict_proba(train_x_onehotCoding)

nb_train_ll_OH = (log_loss(train_y, predict_y, labels=clf.classes_, eps=1e-15))

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(train_y, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

nb_cv_ll_OH = (log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15))

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

nb_test_ll_OH = (log_loss(test_y, predict_y, labels=clf.classes_, eps=1e-15))

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(test_y, predict_y, labels=clf.classes_, eps=1e-15))

clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)

sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

# to avoid rounding error while multiplying probabilites we use log-probability estimates

print("Log Loss :",log_loss(crossval_y, sig_clf_probs))

nb_misclass_OH = np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- crossval_y))/crossval_y.shape[0]

print("# missclassified points :", np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- crossval_y))/crossval_y.shape[0])

plot_confusion_matrix(crossval_y, sig_clf.predict(cv_x_onehotCoding.toarray()))
test_point_index = 1

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
test_point_index = 50

no_feature = 50

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
alpha = [5, 11, 15, 21, 31, 41, 51, 99]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(train_x_responseCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_responseCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

    cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(crossval_y, sig_clf_probs))
fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
best_alpha = np.argmin(cv_log_error_array)

knn_alpha = alpha[best_alpha]

clf = KNeighborsClassifier(n_neighbors = alpha[best_alpha])

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

knn_train_ll_RE = log_loss(train_y, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(train_y, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(cv_x_responseCoding)

knn_cv_ll_RE = log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(test_x_responseCoding)

knn_test_ll_RE = log_loss(test_y, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(test_y, predict_y, labels=clf.classes_, eps=1e-15))

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

predict_and_plot_confusion_matrix(train_x_responseCoding, train_y, cv_x_responseCoding, crossval_y, clf)

knn_misclass = mis_class_datapoints(train_x_responseCoding, train_y, cv_x_responseCoding, crossval_y, clf)
# Lets look at few test points

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



test_point_index = 1

predicted_cls = sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Actual Class :", test_y[test_point_index])

neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), alpha[best_alpha])

print("The ",alpha[best_alpha]," nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])

print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



test_point_index = 100



predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Actual Class :", test_y[test_point_index])

neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), alpha[best_alpha])

print("k value for knn is",alpha[best_alpha],"and the nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])

print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(crossval_y, sig_clf_probs)) 

fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
best_alpha = np.argmin(cv_log_error_array)

LR_bal_alpha = alpha[best_alpha]

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predicted_y = sig_clf.predict_proba(train_x_onehotCoding)

LR_bal_train_ll_OH = log_loss(train_y, predicted_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(train_y, predicted_y, labels=clf.classes_, eps=1e-15))



predicted_y = sig_clf.predict_proba(cv_x_onehotCoding)

LR_bal_cv_ll_OH = log_loss(crossval_y, predicted_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(crossval_y, predicted_y, labels=clf.classes_, eps=1e-15))



predicted_y = sig_clf.predict_proba(test_x_onehotCoding)

LR_bal_test_ll_OH = log_loss(test_y, predicted_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(test_y, predicted_y, labels=clf.classes_, eps=1e-15))
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, crossval_y, clf)

LR_misclass = mis_class_datapoints(train_x_onehotCoding, train_y, cv_x_onehotCoding, crossval_y, clf)
def get_imp_feature_names(text, indices, removed_ind = []):

    word_present = 0

    tabulate_list = []

    incresingorder_ind = 0

    for i in indices:

        if i < train_gene_feature_onehotCoding.shape[1]:

            tabulate_list.append([incresingorder_ind, "Gene", "Yes"])

        elif i< 18:

            tabulate_list.append([incresingorder_ind,"Variation", "Yes"])

        if ((i > 17) & (i not in removed_ind)) :

            word = train_text_features[i]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

            tabulate_list.append([incresingorder_ind,train_text_features[i], yes_no])

        incresingorder_ind += 1

    print(word_present, "most important features are present in our query point")

    print("-"*50)

    print("The features that are most importent of the ",predicted_cls[0]," class:")

    print (tabulate(tabulate_list, headers=["Index",'Feature name', 'Present or Not']))
# from tabulate import tabulate

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
alpha = [10 ** x for x in range(-6, 1)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(crossval_y, sig_clf_probs)) 

fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
best_alpha = np.argmin(cv_log_error_array)

LR_notbal_alpha = alpha[best_alpha]

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predicted_y = sig_clf.predict_proba(train_x_onehotCoding)

LR_notbal_train_OH = log_loss(train_y, predicted_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(train_y, predicted_y, labels=clf.classes_, eps=1e-15))



predicted_y = sig_clf.predict_proba(cv_x_onehotCoding)

LR_notbal_cv_OH = log_loss(crossval_y, predicted_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(crossval_y, predicted_y, labels=clf.classes_, eps=1e-15))



predicted_y = sig_clf.predict_proba(test_x_onehotCoding)

LR_notbal_test_OH = log_loss(test_y, predicted_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(test_y, predicted_y, labels=clf.classes_, eps=1e-15))
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, crossval_y, clf)

LR_notbal_misclass = mis_class_datapoints(train_x_onehotCoding, train_y, cv_x_onehotCoding, crossval_y, clf)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
alpha = [10 ** x for x in range(-5, 3)]

cv_log_error_array = []

for i in alpha:

    print("for C =", i)

#     clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')

    clf = SGDClassifier( class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(crossval_y, sig_clf_probs)) 

fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()
best_alpha = np.argmin(cv_log_error_array)

SVM_alpha = alpha[best_alpha]

#clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)



clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

SVM_train_OH = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",

      log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

SVM_cv_OH = log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",

      log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

SVM_test_OH = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",

      log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42,class_weight='balanced')

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,crossval_y, clf)

SVM_misclass = mis_class_datapoints(train_x_onehotCoding, train_y,cv_x_onehotCoding,crossval_y, clf)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

# test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
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

        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

        cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(crossval_y, sig_clf_probs)) 
best_alpha = np.argmin(cv_log_error_array)

RF_alpha_OH = alpha[int(best_alpha/2)]

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)],

                             random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

RF_train_OH = log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",

      log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

RF_cv_OH = log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",

      log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15))



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

RF_test_OH = log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",

      log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)],

                             random_state=42, n_jobs=-1)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,crossval_y, clf)

RF_OH_misclass = mis_class_datapoints(train_x_onehotCoding, train_y,cv_x_onehotCoding,crossval_y, clf)
# test_point_index = 10

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



test_point_index = 1

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.feature_importances_)

print("-"*50)

get_impfeature_names(indices[:no_feature], x_test['TEXT'].iloc[test_point_index],x_test['Gene'].iloc[test_point_index],x_test['Variation'].iloc[test_point_index], no_feature)
alpha = [10,50,100,200,500,1000]

max_depth = [2,3,5,10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_responseCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_responseCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

        cv_log_error_array.append(log_loss(crossval_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(crossval_y, sig_clf_probs)) 





best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The cross validation log loss is:",log_loss(crossval_y, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
alpha = [10,50,100,200,500,1000]

max_depth = [2,3,5,10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_responseCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_responseCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 





best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = RandomForestClassifier(max_depth=max_depth[int(best_alpha%2)], n_estimators=alpha[int(best_alpha/2)],

                             criterion='gini', max_features='auto', random_state=42)

predict_and_plot_confusion_matrix(train_x_responseCoding, train_y,cv_x_responseCoding,crossval_y, clf)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini',

                             max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)





test_point_index = 1

no_feature = 27

predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index].reshape(1,-1)),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.feature_importances_)

print("-"*50)

for i in indices:

    if i<9:

        print("Gene is important feature")

    elif i<18:

        print("Variation is important feature")

    else:

        print("Text is important feature")
from sklearn.linear_model import LogisticRegression

clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=0)

clf1.fit(train_x_onehotCoding, train_y)

sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")



clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=0)

clf2.fit(train_x_onehotCoding, train_y)

sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")





clf3 = MultinomialNB(alpha=0.001)

clf3.fit(train_x_onehotCoding, train_y)

sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



sig_clf1.fit(train_x_onehotCoding, train_y)

print("Logistic Regression :  Log Loss: %0.2f" % (log_loss(crossval_y, sig_clf1.predict_proba(cv_x_onehotCoding))))

sig_clf2.fit(train_x_onehotCoding, train_y)

print("Support vector machines : Log Loss: %0.2f" % (log_loss(crossval_y, sig_clf2.predict_proba(cv_x_onehotCoding))))

sig_clf3.fit(train_x_onehotCoding, train_y)

print("Naive Bayes : Log Loss: %0.2f" % (log_loss(crossval_y, sig_clf3.predict_proba(cv_x_onehotCoding))))

print("-"*50)

alpha = [0.0001,0.001,0.01,0.1,1,10] 

best_alpha = 999

for i in alpha:

    lr = LogisticRegression(C=i)

    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

    sclf.fit(train_x_onehotCoding, train_y)

    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(crossval_y, sclf.predict_proba(cv_x_onehotCoding))))

    log_error =log_loss(crossval_y, sclf.predict_proba(cv_x_onehotCoding))

    if best_alpha > log_error:

        best_alpha = log_error
lr = LogisticRegression(C=0.1)

sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

sclf.fit(train_x_onehotCoding, train_y)



log_error = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))

Stack_train = log_error

print("Log loss (train) on the stacking classifier :",log_error)



log_error = log_loss(crossval_y, sclf.predict_proba(cv_x_onehotCoding))

Stack_cv = log_error

print("Log loss (CV) on the stacking classifier :",log_error)



log_error = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))

Stack_test = log_error

print("Log loss (test) on the stacking classifier :",log_error)



Stack_misclass = np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0]

print("Number of missclassified point :", np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])

plot_confusion_matrix(test_y, sclf.predict(test_x_onehotCoding))
from sklearn.ensemble import VotingClassifier

vclf = VotingClassifier(estimators=[('lr', sig_clf1), ('svc', sig_clf2), ('rf', sig_clf3)], voting='soft')

vclf.fit(train_x_onehotCoding, train_y)

Max_voting_train = log_loss(train_y, vclf.predict_proba(train_x_onehotCoding))

print("Log loss (train) on the VotingClassifier :", log_loss(train_y, vclf.predict_proba(train_x_onehotCoding)))



Max_voting_cv = log_loss(crossval_y, vclf.predict_proba(cv_x_onehotCoding))

print("Log loss (CV) on the VotingClassifier :", log_loss(crossval_y, vclf.predict_proba(cv_x_onehotCoding)))



Max_voting_test = log_loss(test_y, vclf.predict_proba(test_x_onehotCoding))

print("Log loss (test) on the VotingClassifier :", log_loss(test_y, vclf.predict_proba(test_x_onehotCoding)))



Max_voting_misclass = np.count_nonzero((vclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0]

print("Number of missclassified point :", np.count_nonzero((vclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])

plot_confusion_matrix(test_y, vclf.predict(test_x_onehotCoding))
summary_table = {'Model' : ['Naive Bayes-OH', 'KNN-RE', 'Log Reg-Bal', 'Log Reg-NBal', 'Linear SVM', 'Rando Forest-OH', 'Stacking Model', 

                           'Maximum v class'], 

                 'Best-a' : [nb_alpha, knn_alpha, LR_bal_alpha, LR_notbal_alpha, SVM_alpha, RF_alpha_OH, 'not applied', 'not applied' ], 

                 'Train LL' : [nb_train_ll_OH, knn_train_ll_RE, LR_bal_train_ll_OH, LR_notbal_train_OH, SVM_train_OH, RF_train_OH, Stack_train, Max_voting_train],

                'Test LL' : [nb_test_ll_OH, knn_test_ll_RE, LR_bal_test_ll_OH, LR_notbal_test_OH, SVM_test_OH, RF_test_OH, Stack_test, Max_voting_test], 

                 'CV LL' : [nb_cv_ll_OH, knn_cv_ll_RE, LR_bal_cv_ll_OH, LR_notbal_cv_OH, SVM_cv_OH, RF_cv_OH, Stack_cv, Max_voting_cv], 

                 'Misclassified' : [nb_misclass_OH, knn_misclass, LR_misclass, LR_notbal_misclass, SVM_misclass, RF_OH_misclass, Stack_misclass, Max_voting_misclass]}  



sum_tab_df = pd.DataFrame(summary_table)



print(sum_tab_df)