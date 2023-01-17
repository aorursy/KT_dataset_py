# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading all required packages

# If any of it fails, do not panic. Just install it using "pip3 install <package_name>" or by using conda install package_name



import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import time

import warnings



from nltk.corpus import stopwords

from imblearn.over_sampling import SMOTE

from collections import Counter

from collections import Counter, defaultdict

from scipy.sparse import hstack

from mlxtend.classifier import StackingClassifier



from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import normalized_mutual_info_score

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection

from sklearn.model_selection import StratifiedKFold 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



warnings.filterwarnings("ignore")
# First, look at everything.

from subprocess import check_output

print(check_output(['ls', '../input/msk-redefining-cancer-treatment/']).decode("utf8"))
#  Pick a Dataset you might be interested in.

#  Say, all airline-safety files...

import zipfile



dataset = 'training_variants'

dataset2 = 'training_text'



# Will unzip the files so that you can see them..

with zipfile.ZipFile('../input/msk-redefining-cancer-treatment/'+dataset+'.zip','r') as z:

    z.extractall(".")



# Will unzip the files so that you can see them..

with zipfile.ZipFile('../input/msk-redefining-cancer-treatment/'+dataset2+'.zip','r') as y:

    y.extractall('.')

    

from subprocess import check_output

print(check_output(['ls', 'training_variants']).decode("utf8"))

print(check_output(['ls', 'training_text']).decode("utf8"))
# Select and read the files.

data_variants = pd.read_csv(dataset)

data_text = pd.read_csv(dataset2, sep="\|\|", engine='python', names=['ID','TEXT'], skiprows=1)
data_variants.head()
data_variants.info()
data_variants.describe()
data_variants.shape
data_text.head()
data_text.info()
data_text.describe()
data_text.columns
data_text.shape
data_variants.Class.unique()
# We would like to remove all stop words like a, is, an, the...

# so we are collecting all of them from nltk library

stop_words = set(stopwords.words('english'))
def data_text_preprocess(total_text, ind, col):

    # Remove int values from text data as that might not be important

    if type(total_text) is not int:

        string = ''

        # replacing all special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))

        # replacing multiple spaces with single space

        total_text = re.sub('\s', ' ', str(total_text))

        # bring whole text to same lower-case scale

        total_text = total_text.lower()

        

        for word in total_text.split():

            # if word is not a stop word then retain that word from text

            if not word in stop_words:

                string += word + ''

                

        data_text[col][ind] = string
# Below code will take some time because it's huge text, so run it and have a cup of coffee

for index, row  in data_text.iterrows():

    if type(row['TEXT']) is str:

        data_text_preprocess(row['TEXT'], index, 'TEXT')
# merging both gene_variatiions and text data based on ID

result = pd.merge(data_variants, data_text, on='ID', how='left')

result.head()
result[result.isnull().any(axis=1)]
result.loc[result.TEXT.isnull(),'TEXT'] = result['Gene']+ ' '+result['Variation']
result[result.isnull().any(axis=1)]
y_true = result['Class'].values

result.Gene = result.Gene.str.replace('\s+', '_')

result.Variation = result.Variation.str.replace('\s+', '_')
# Splitting the data into train and test set

X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)

# Split the train data now into train validation and cross validation

train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in test data:', test_df.shape[0])

print('Number of data points in cross validation data:', cv_df.shape[0])
train_class_distribution = train_df['Class'].value_counts().sort_index()

test_class_distribution = test_df['Class'].value_counts().sort_index()

cv_class_distribution = cv_df['Class'].value_counts().sort_index()
train_class_distribution
my_colors = 'rgbkymc'

train_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Number of Data Points per Class')

plt.title('Distribution of yi in Train Data')

plt.grid()

plt.show()
sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')
test_data_len = test_df.shape[0]

cv_data_len = cv_df.shape[0]
# We create an output array that has exactly same size as the CV data

cv_predicted_y = np.zeros((cv_data_len, 9))

for i in range(cv_data_len):

    rand_probs = np. random.rand(1,9)

    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print('Log loss on Cross Validation Data using Random Model', log_loss(y_cv, cv_predicted_y, eps=1e-15))    
# Test-Set error

# We create an output array that has exactly same size as test data

test_predicted_y = np.zeros((test_data_len, 9))

for i in range(test_data_len):

    rand_probs = np. random.rand(1,9)

    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print('Log loss on Test Data using Random Model', log_loss(y_test, test_predicted_y, eps=1e-15))    
# Let's get the index of max probability

predicted_y = np.argmax(test_predicted_y, axis=1)
# Let's see the output. These will be 665 values present in test_dataset

predicted_y
predicted_y = predicted_y + 1

predicted_y
C = confusion_matrix(y_test, predicted_y)
labels = [1,2,3,4,5,6,7,8,9]

plt.figure(figsize=(20,7))

sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
B =(C/C.sum(axis=0))
plt.figure(figsize=(20,7))

sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
A =(((C.T)/(C.sum(axis=1))).T)
plt.figure(figsize=(20,7))

sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
unique_genes = train_df['Gene'].value_counts()

print('Number of Unique Genes:', unique_genes.shape[0])

# the top 10 genes that occured most

print(unique_genes.head(10))
unique_genes.shape[0]
s = sum(unique_genes.values)

h = unique_genes.values / s

c = np.cumsum(h)

plt.plot(c, label='Cumulative Distribution of Genes')

plt.grid()

plt.legend()

plt.show()
# One-hot encoding of Gene feature

gene_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding = gene_vectorizer.fit_transform(test_df['Gene'])

cv_gene_feature_onehotCoding = gene_vectorizer.fit_transform(cv_df['Gene'])
train_gene_feature_onehotCoding.shape
# code for response coding with Laplace smoothing.

# alpha : used for laplace smoothing

# feature: ['gene', 'variation']

# df: ['train_df', 'test_df', 'cv_df']

# algorithm

# ----------

# Consider all unique values and the number of occurances of given feature in train data dataframe

# build a vector (1*9) , the first element = (number of times it occured in class1 + 10*alpha / number of time it occurred in total data+90*alpha)

# gv_dict is like a look up table, for every gene it store a (1*9) representation of it

# for a value of feature in df:

# if it is in train data:

# we add the vector that was stored in 'gv_dict' look up table to 'gv_fea'

# if it is not there is train:

# we add [1/9, 1/9, 1/9, 1/9,1/9, 1/9, 1/9, 1/9, 1/9] to 'gv_fea'

# return 'gv_fea'

# ----------------------



# get_gv_fea_dict: Get Gene varaition Feature Dict

def get_gv_fea_dict(alpha, feature, df):

    # value_count: it contains a dict like

    # print(train_df['Gene'].value_counts())

    # output:

    #        {BRCA1      174

    #         TP53       106

    #         EGFR        86

    #         BRCA2       75

    #         PTEN        69

    #         KIT         61

    #         BRAF        60

    #         ERBB2       47

    #         PDGFRA      46

    #         ...}

    # print(train_df['Variation'].value_counts())

    # output:

    # {

    # Truncating_Mutations                     63

    # Deletion                                 43

    # Amplification                            43

    # Fusions                                  22

    # Overexpression                            3

    # E17K                                      3

    # Q61L                                      3

    # S222D                                     2

    # P130S                                     2

    # ...

    # }

    value_count = train_df[feature].value_counts()

    

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation

    gv_dict = dict()

    

    # denominator will contain the number of time that particular feature occured in whole data

    for i, denominator in value_count.items():

        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class

        # vec is 9 diamensional vector

        vec = []

        for k in range(1,10):

            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])

            #         ID   Gene             Variation  Class  

            # 2470  2470  BRCA1                S1715C      1   

            # 2486  2486  BRCA1                S1841R      1   

            # 2614  2614  BRCA1                   M1R      1   

            # 2432  2432  BRCA1                L1657P      1   

            # 2567  2567  BRCA1                T1685A      1   

            # 2583  2583  BRCA1                E1660G      1   

            # 2634  2634  BRCA1                W1718L      1   

            # cls_cnt.shape[0] will return the number of rows



            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]

            

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data

            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))



        # we are adding the gene/variation to the dict as key and vec as value

        gv_dict[i]=vec

    return gv_dict



# Get Gene variation feature

def get_gv_feature(alpha, feature, df):

    # print(gv_dict)

    #     {'BRCA1': [0.20075757575757575, 0.03787878787878788, 0.068181818181818177, 0.13636363636363635, 0.25, 0.19318181818181818, 0.03787878787878788, 0.03787878787878788, 0.03787878787878788], 

    #      'TP53': [0.32142857142857145, 0.061224489795918366, 0.061224489795918366, 0.27040816326530615, 0.061224489795918366, 0.066326530612244902, 0.051020408163265307, 0.051020408163265307, 0.056122448979591837], 

    #      'EGFR': [0.056818181818181816, 0.21590909090909091, 0.0625, 0.068181818181818177, 0.068181818181818177, 0.0625, 0.34659090909090912, 0.0625, 0.056818181818181816], 

    #      'BRCA2': [0.13333333333333333, 0.060606060606060608, 0.060606060606060608, 0.078787878787878782, 0.1393939393939394, 0.34545454545454546, 0.060606060606060608, 0.060606060606060608, 0.060606060606060608], 

    #      'PTEN': [0.069182389937106917, 0.062893081761006289, 0.069182389937106917, 0.46540880503144655, 0.075471698113207544, 0.062893081761006289, 0.069182389937106917, 0.062893081761006289, 0.062893081761006289], 

    #      'KIT': [0.066225165562913912, 0.25165562913907286, 0.072847682119205295, 0.072847682119205295, 0.066225165562913912, 0.066225165562913912, 0.27152317880794702, 0.066225165562913912, 0.066225165562913912], 

    #      'BRAF': [0.066666666666666666, 0.17999999999999999, 0.073333333333333334, 0.073333333333333334, 0.093333333333333338, 0.080000000000000002, 0.29999999999999999, 0.066666666666666666, 0.066666666666666666],

    #      ...

    #     }

    gv_dict = get_gv_fea_dict(alpha, feature, df)

    # value_count is similar in get_gv_fea_dict

    value_count = train_df[feature].value_counts()

    

    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data

    gv_fea = []

    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea

    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea

    for index, row in df.iterrows():

        if row[feature] in dict(value_count).keys():

            gv_fea.append(gv_dict[row[feature]])

        else:

            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])

#             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return gv_fea
#response-coding of the Gene feature

# alpha is used for laplace smoothing

alpha = 1

# train gene feature

train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", train_df))

# test gene feature

test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df))

# cross validation gene feature

cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", cv_df))
train_gene_feature_responseCoding.shape
# We need a hyperparameter for SGD classifier

alpha = [10 ** x for x in range(-5, 1)]
# We will be using SGD classifier

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# We will also be using Calibrated Classifier to get the result into probablity format t be used for log loss

cv_log_error_array=[]



for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feature_onehotCoding, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_gene_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
# Let's plot the same to check the best Alpha value

fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()