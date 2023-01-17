from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file

# irishtimes-date-text.csv has 1425460 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/irishtimes-date-text.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'irishtimes-date-text.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
nRowsRead = 1000 # specify 'None' if want to read whole file

# w3-latnigrin-text.csv has 616037 rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('../input/w3-latnigrin-text.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'w3-latnigrin-text.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
print(df1.describe())

print("----------------------------")

print(df1.groupby('headline_category').size())

print("----------------------------")

df1_numpy=df1.to_numpy()

print("df1_numpy.shape= ",df1_numpy.shape)



#Separate X & y

X=df1_numpy[:,2]

y=df1_numpy[:,1]

print ("X.shape=",X.shape)

print ("y.shape=",y.shape)



# Check uniqueness y

unique_elements_y, counts_elements_y= np.unique(y[:], return_counts=True)

print("y unique_elements:")

print(unique_elements_y)

print("count y unique_elements:")

print(counts_elements_y)



#Apply One hot encoding to y

import category_encoders as ce

from sklearn.preprocessing import OneHotEncoder

ohe =  ce.OneHotEncoder(handle_unknown='ignore')

y_ohe=ohe.fit_transform(y)

print("y.shape (after OHE)= ",y_ohe.shape)

print("y= ")

print(y[540:546])



from sklearn.model_selection import train_test_split

X_train, X_test, y_ohe_train, y_ohe_test = train_test_split(X, y_ohe, test_size=0.2, random_state=0)



#Check if the train and test data has the same dist

unique_elements_y_train, counts_elements_y_train=np.unique(y_ohe_train, axis=0, return_counts=True)

unique_elements_y_test, counts_elements_y_test=np.unique(y_ohe_test, axis=0, return_counts=True)



print("y_train distribution:")

print(unique_elements_y_train)

print(counts_elements_y_train)

counts_elements_y_train_dist=[]

for i in range(counts_elements_y_train.shape[0]):

    b=counts_elements_y_train[i]/y_ohe.shape[0]*100.0

    #counts_elements_y_train_dist.append(counts_elements_y_train[i]/sum(counts_elements_y_train)*100.0)

    counts_elements_y_train_dist.append(b)

print(counts_elements_y_train_dist)

print("sum= ",sum(counts_elements_y_train_dist))



print("y_test distribution:")

print(unique_elements_y_test)

print(counts_elements_y_test)

counts_elements_y_test_dist=[]

for i in range(counts_elements_y_test.shape[0]):

    b2=counts_elements_y_test[i]/y_ohe.shape[0]*100.0

    #counts_elements_y_test_dist.append(counts_elements_y_test[i]/sum(counts_elements_y_test)*100.0)

    counts_elements_y_test_dist.append(b2)

print(counts_elements_y_test_dist)

print("sum= ",sum(counts_elements_y_test_dist))

print("----------------------------")

import category_encoders as ce

from sklearn.preprocessing import LabelEncoder



ce_ord = ce.OrdinalEncoder(y)

y_ce_ord=ce_ord.fit_transform(y)

y_ce_ord=y_ce_ord.to_numpy()        # change from panda dataframe to numpy array

print("y_ce_ord_numpy.shape= ",y_ce_ord.shape)

print("y_ce_ord_numpy type= ",type(y_ce_ord))



X_ce_ord_train, X_ce_ord_test, y_ce_ord_train, y_ce_ord_test = train_test_split(X, y_ce_ord, test_size=0.2, random_state=0)



#Check if the train and test data has the same dist

unique_elements_y_train, counts_elements_y_train=np.unique(y_ce_ord_train, axis=0, return_counts=True)

unique_elements_y_test, counts_elements_y_test=np.unique(y_ce_ord_test, axis=0, return_counts=True)



print("y_ce_ord_train distribution:")

print(unique_elements_y_train)

print(counts_elements_y_train)

counts_elements_y_train_dist=[]

for i in range(counts_elements_y_train.shape[0]):

    b=counts_elements_y_train[i]/y_ce_ord_train.shape[0]*100.0

    #counts_elements_y_train_dist.append(counts_elements_y_train[i]/sum(counts_elements_y_train)*100.0)

    counts_elements_y_train_dist.append(b)

print(counts_elements_y_train_dist)

print("sum= ",sum(counts_elements_y_train_dist))



print("y_ce_ord_test distribution:")

print(unique_elements_y_test)

print(counts_elements_y_test)

counts_elements_y_test_dist=[]

for i in range(counts_elements_y_test.shape[0]):

    b2=counts_elements_y_test[i]/y_ce_ord_test.shape[0]*100.0

    #counts_elements_y_test_dist.append(counts_elements_y_test[i]/sum(counts_elements_y_test)*100.0)

    counts_elements_y_test_dist.append(b2)

print(counts_elements_y_test_dist)

print("sum= ",sum(counts_elements_y_test_dist))

print("----------------------------")

#Apply pre-processing to text data

import nltk

import re

from sklearn.datasets import load_files

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



print("X_train initial =")

print(X_train[0:3])

'''

stemmer = WordNetLemmatizer()

documents = []



for sen in range(0, len(X)):    #use Regex Expressions from Python re library to perform different preprocessing tasks

    # Remove all the special characters

    document = re.sub(r'\W', ' ', str(X[sen]))



    # remove all single characters

    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    

    # Remove single characters from the start

    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    

    # Substituting multiple spaces with single space

    document = re.sub(r'\s+', ' ', document, flags=re.I)

    

    # Removing prefixed 'b'

    document = re.sub(r'^b\s+', '', document)

    

    # Converting to Lowercase

    document = document.lower()

    

    # Lemmatization

    document = document.split()



    document = [stemmer.lemmatize(word) for word in document]

    document = ' '.join(document)

    

    documents.append(document)



# Bag of Words

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

X = vectorizer.fit_transform(documents).toarray()

print("X.shape after Bag of Words= ",X.shape)

print(X[0:3])

#max_features only need 151 columns, but print out shows that the transformation failed

'''



count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(X)



# transform the training and validation data using count vectorizer object

X_train_count =  count_vect.transform(X_train)

X_test_count =  count_vect.transform(X_test)

X_ce_ord_train_count =  count_vect.transform(X_ce_ord_train)

X_ce_ord_test_count =  count_vect.transform(X_ce_ord_test)



print("X_train_count.shape after CountVectorizer= ",X_train_count.shape)

print(X_train_count)

print("X_test_count.shape after CountVectorizer= ",X_test_count.shape)

print(X_test_count)

#max_features only need 151 columns #(1000, 2781)

print("----------------------------")
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(X)

X_train_tfidf =  tfidf_vect.transform(X_train)

X_test_tfidf =  tfidf_vect.transform(X_test)

X_ce_ord_train_tfidf =  tfidf_vect.transform(X_ce_ord_train)

X_ce_ord_test_tfidf =  tfidf_vect.transform(X_ce_ord_test)



print("X_train_tfidf= ",X_train_tfidf.shape)     #(800, 2781)

print("X_test_tfidf= ",X_test_tfidf.shape)     #(200, 2781)

print(X_train_tfidf)

print("---------------------------")



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(X)

X_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)

X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

X_ce_ord_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_ce_ord_train)

X_ce_ord_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_ce_ord_test)



print("X_train_tfidf_ngram= ",X_train_tfidf_ngram.shape)     #(800, 5000)

print("X_test_tfidf_ngram= ",X_test_tfidf_ngram.shape)     #(200, 5000)

print(X_train_tfidf_ngram)

print("---------------------------")



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(X)

X_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 

X_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_test)

X_ce_ord_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_ce_ord_train) 

X_ce_ord_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_ce_ord_test)



print("X_train_tfidf_ngram_chars= ",X_train_tfidf_ngram_chars.shape)     #(800, 4791)

print("X_test_tfidf_ngram_chars= ",X_test_tfidf_ngram_chars.shape)     #(200, 4791)

print(X_train_tfidf_ngram_chars)

print("---------------------------")
#Training & Evaluation-------------------------------------------------

#Random Forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



classifier_RF01 = RandomForestClassifier(n_estimators=1000, random_state=0)

classifier_RF01.fit(X_train_count, y_ohe_train)



y_pred = classifier_RF01.predict(X_test_count)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Vector count-----------------------------")

print("Classification report")

print(classification_report(y_ohe_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ohe_test, y_pred))   #52 %



classifier_RF02 = RandomForestClassifier(n_estimators=1000, random_state=0)

classifier_RF02.fit(X_train_tfidf, y_ohe_train)



y_pred = classifier_RF01.predict(X_test_tfidf)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Word-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ohe_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ohe_test, y_pred))   #52 %



classifier_RF03 = RandomForestClassifier(n_estimators=1000, random_state=0)

classifier_RF03.fit(X_train_tfidf_ngram, y_ohe_train)



y_pred = classifier_RF03.predict(X_test_tfidf_ngram)



print("N gram-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ohe_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ohe_test, y_pred))     # 6.5%



classifier_RF04 = RandomForestClassifier(n_estimators=1000, random_state=0)

classifier_RF04.fit(X_train_tfidf_ngram_chars, y_ohe_train)



y_pred = classifier_RF04.predict(X_test_tfidf_ngram_chars)



print("N gram chars-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ohe_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ohe_test, y_pred))    # 42%

# Naive Bayes (use X & y of ce_ordinal )

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm



classifier_NB01 = naive_bayes.MultinomialNB()

#classifier_NB01 = naive_bayes.GaussianNB()



y_ce_ord_train=y_ce_ord_train.ravel() #flatten from column wise to array wise



classifier_NB01.fit(X_train_count.toarray(), y_ce_ord_train)



y_pred = classifier_NB01.predict(X_test_count.toarray())

print("Vector count-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))   #58 %



classifier_NB02 = naive_bayes.MultinomialNB()

#classifier_NB02 = naive_bayes.GaussianNB()

classifier_NB02.fit(X_ce_ord_train_tfidf, y_ce_ord_train)



y_pred = classifier_NB02.predict(X_ce_ord_test_tfidf.toarray())



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Word-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))   #52 %



classifier_NB03 = naive_bayes.MultinomialNB()

classifier_NB03.fit(X_train_tfidf_ngram, y_ce_ord_train)



y_pred = classifier_NB03.predict(X_ce_ord_test_tfidf_ngram)



print("N gram-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))     # 52%



classifier_NB04 = naive_bayes.MultinomialNB()

classifier_NB04.fit(X_ce_ord_train_tfidf_ngram_chars, y_ce_ord_train)



y_pred = classifier_NB04.predict(X_ce_ord_test_tfidf_ngram_chars)



print("N gram chars-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))    # 42%

# Logistics Regression (use X & y of ce_ordinal)



classifier_LogR01 = linear_model.LogisticRegression()

classifier_LogR01.fit(X_ce_ord_train_count, y_ce_ord_train)



y_pred = classifier_LogR01.predict(X_ce_ord_test_count)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Vector count--------------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))   #56 %



classifier_LogR02 = linear_model.LogisticRegression()

classifier_LogR02.fit(X_ce_ord_train_tfidf, y_ce_ord_train)



y_pred = classifier_LogR02.predict(X_ce_ord_test_tfidf)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Word-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))   #52 %



classifier_LogR03 = linear_model.LogisticRegression()

classifier_LogR03.fit(X_ce_ord_train_tfidf_ngram, y_ce_ord_train)



y_pred = classifier_LogR03.predict(X_ce_ord_test_tfidf_ngram)



print("N gram-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))     # 52%



classifier_LogR04 = linear_model.LogisticRegression()

classifier_LogR04.fit(X_ce_ord_train_tfidf_ngram_chars, y_ce_ord_train)



y_pred = classifier_LogR04.predict(X_ce_ord_test_tfidf_ngram_chars)



print("N gram chars-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score")

print(accuracy_score(y_ce_ord_test, y_pred))    # 52%

# SVM (use X & y of ce_ordinal)



classifier_SVM01 = svm.SVC()

classifier_SVM01.fit(X_ce_ord_train_count, y_ce_ord_train)



y_pred = classifier_SVM01.predict(X_ce_ord_test_count)

y_pred2 = classifier_SVM01.predict(X_ce_ord_train_count)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Vector count--------------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))   #52 %



classifier_SVM02 = svm.SVC()

classifier_SVM02.fit(X_ce_ord_train_tfidf, y_ce_ord_train)



y_pred = classifier_SVM02.predict(X_ce_ord_test_tfidf)

y_pred2 = classifier_SVM02.predict(X_ce_ord_train_tfidf)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Word-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))   #52 %



classifier_SVM03 = svm.SVC()

classifier_SVM03.fit(X_ce_ord_train_tfidf_ngram, y_ce_ord_train)



y_pred = classifier_SVM03.predict(X_ce_ord_test_tfidf_ngram)

y_pred2 = classifier_SVM03.predict(X_ce_ord_train_tfidf_ngram)



print("N gram-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))   #52 %



classifier_SVM04 = svm.SVC()

classifier_SVM04.fit(X_ce_ord_train_tfidf_ngram_chars, y_ce_ord_train)



y_pred = classifier_SVM04.predict(X_ce_ord_test_tfidf_ngram_chars)

y_pred2 = classifier_SVM04.predict(X_ce_ord_train_tfidf_ngram_chars)



print("N gram chars-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))   #52 %

# Extreme Gradient Boosting (use X & y of ce_ordinal)



import xgboost



classifier_XGB01 = xgboost.XGBClassifier()

classifier_XGB01.fit(X_ce_ord_train_count, y_ce_ord_train)



y_pred = classifier_XGB01.predict(X_ce_ord_test_count)

y_pred2 = classifier_XGB01.predict(X_ce_ord_train_count)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Vector count--------------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))   #52 %



classifier_XGB02 = xgboost.XGBClassifier()

classifier_XGB02.fit(X_ce_ord_train_tfidf, y_ce_ord_train)



y_pred = classifier_XGB02.predict(X_ce_ord_test_tfidf)

y_pred2 = classifier_XGB02.predict(X_ce_ord_train_tfidf)



#print("Confusion Matrix")

#print(confusion_matrix(y_test,y_pred)) #too many features to be viewed

print("Word-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))    #52 %



classifier_XGB03 = xgboost.XGBClassifier()

classifier_XGB03.fit(X_ce_ord_train_tfidf_ngram, y_ce_ord_train)



y_pred = classifier_XGB03.predict(X_ce_ord_test_tfidf_ngram)

y_pred2 = classifier_XGB03.predict(X_ce_ord_train_tfidf_ngram)



print("N gram-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))    #52 %





classifier_XGB04 = xgboost.XGBClassifier()

classifier_XGB04.fit(X_ce_ord_train_tfidf_ngram_chars, y_ce_ord_train)



y_pred = classifier_XGB04.predict(X_ce_ord_test_tfidf_ngram_chars)

y_pred2 = classifier_XGB04.predict(X_ce_ord_train_tfidf_ngram_chars)



print("N gram chars-tfidf-----------------------------")

print("Classification report")

print(classification_report(y_ce_ord_test,y_pred))

print("Accuracy Score (Ein vs Eout)")

print(accuracy_score(y_ce_ord_train, y_pred2)," vs ", accuracy_score(y_ce_ord_test, y_pred))   #52 %



#Shallow NN

from keras import layers, models, optimizers

from keras.models import Sequential

from keras.layers import Dense

from keras.metrics import categorical_accuracy



input_size=X_ce_ord_train_tfidf_ngram.shape[1]



net = Sequential()

net.add(Dense(100, activation='relu' , input_shape=(input_size,)))

net.add(Dense(1, activation='sigmoid' ))

net.compile(loss='binary_crossentropy' , optimizer=optimizers.Adam(), metrics=['accuracy'] )

net.summary()

net.fit(X_ce_ord_train_tfidf_ngram, y_ce_ord_train, epochs=10, verbose=0)

print("training error: " + str(net.evaluate(X_ce_ord_train_tfidf_ngram, y_ce_ord_train, verbose=0)))

print("test error: " + str(net.evaluate(X_ce_ord_test_tfidf_ngram, y_ce_ord_test, verbose=0)))
