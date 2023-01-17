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
import pandas as pd



#df = pd.read_csv('test33.csv')

#df = pd.read_csv('sampleurdunews_33_100_FINAL.csv',  encoding = "utf-8", error_bad_lines=False, header=None)

df = pd.read_csv('/kaggle/input/UrduNewsHeadlines.csv')

df.head()
df = pd.read_csv("/kaggle/input/UrduNewsHeadlines.csv", sep='\t',names = ["class", "details"])
df.head()
# Associate Category names with numerical index and save it in new column category_id

df['category_id'] = df['class'].factorize()[0]



#View first 10 entries of category_id, as a sanity check

df['category_id'][0:10]

category_id_df = df[['class', 'category_id']].drop_duplicates().sort_values('category_id')
category_id_df
category_to_id = dict(category_id_df.values)



id_to_category = dict(category_id_df[['category_id','class']].values)

id_to_category
df.sample(5, random_state=0)
df.groupby('class').category_id.count()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

df.groupby('class').details.count().plot.bar(ylim=0)

plt.show()
sw= ["آ", "آئی", "آئیں", "آئے", "آتا", "آتی", "آتے", "آخری", "آس", "آنا", "آنی", "آنے", "آپ", "آگے", "آیا", "ابھی", "اجنبی", "از", "اس", "استعمال", "اسی", "اسے", "البتہ", "الف", "ان", "اندر", "انہوں", "انہی", "انہیں", "اور", "اوپر", "اپ", "اپنا", "اپنی", "اپنے", "اکثر", "اگر", "اگرچہ", "اگلے", "ایسا", "ایسی", "ایسے", "ایک", "اے", "بار", "بارے", "باوجود", "باہر", "بظاہر", "بعد", "بعض", "بغیر", "بلکہ", "بن", "بنا", "بناؤ", "بند", "بننا", "بھر", "بھریں", "بھی", "بہت", "بیس", "بے", "تا", "تاکہ", "تب", "تجھ", "تجھے", "تحت", "تر", "تم", "تمام", "تمہارا", "تمہاری", "تمہارے", "تمہیں", "تو", "تک", "تھا", "تھی", "تھیں", "تھے", "تیری", "جا", "جاؤ", "جائیں", "جائے", "جاتا", "جاتی", "جاتے", "جانی", "جانے", "جب", "جبکہ", "جس", "جن", "جنہوں", "جنہیں", "جو", "جگہ", "جہاں", "جیسا", "جیسوں", "جیسی", "جیسے", "حاصل", "حالانکہ", "حالاں", "حصہ", "خالی", "ختم", "خلاف", "خود", "درمیان", "دسترس", "دلچسپی", "دو", "دوبارہ", "دوران", "دوسرا", "دوسروں", "دوسری", "دوسرے", "دونوں", "دوں", "دکھائیں", "دی", "دیئے", "دیا", "دیتا", "دیتی", "دیتے", "دیر", "دینا", "دینی", "دینے", "دیکھو", "دیں", "دیے", "دے", "ذریعے", "رکھا", "رکھتا", "رکھتی", "رکھتے", "رکھنا", "رکھنی", "رکھنے", "رکھو", "رکھی", "رکھے", "رہ", "رہا", "رہتا", "رہتی", "رہتے", "رہنا", "رہنی", "رہنے", "رہو", "رہی", "رہیں", "رہے", "زیادہ", "سا", "ساتھ", "سامنے", "سب", "سو", "سکا", "سکتا", "سکتے", "سی", "سے", "شان", "شاید", "صرف", "صورت", "ضرورت", "ضروری", "طرح", "طرف", "طور", "علاوہ", "عین", "غیر", "لئے", "لا", "لائی", "لائے", "لاتا", "لاتی", "لاتے", "لانا", "لانی", "لانے", "لایا", "لو", "لگ", "لگا", "لگتا", "لگی", "لگیں", "لگے", "لہذا", "لی", "لیا", "لیتا", "لیتی", "لیتے", "لیکن", "لیں", "لیے", "لے", "مجھ", "مجھے", "مزید", "مقابلے", "مل", "مکمل", "مگر", "میرا", "میری", "میرے", "میں", "نا", "نہ", "نہیں", "نیچے", "نے", "واقعی", "والا", "والوں", "والی", "والے", "وجہ", "وغیرہ", "وہ", "وہاں", "وہی", "وہیں", "وی", "ویسے", "پاس", "پایا", "پر", "پوری", "پھر", "پہلا", "پہلے", "پیچھے", "چاہئے", "چاہتے", "چاہیئے", "چاہے", "چونکہ", "چکی", "ڈالا", "ڈالنا", "ڈالنی", "ڈالنے", "ڈالی", "ڈالیں", "ڈالے", "کئے", "کا", "کافی", "کب", "کبھی", "کر", "کرتا", "کرتی", "کرتے", "کرنا", "کرنے", "کرو", "کریں", "کرے", "کس", "کسی", "کسے", "کم", "کو", "کوئی", "کون", "کونسا", "کچھ", "کہ", "کہا", "کہاں", "کہہ", "کہیں", "کہے", "کی", "کیا", "کیسے", "کیونکہ", "کیوں", "کیے", "کے", "گئی", "گئے", "گا", "گویا", "گی", "گیا", "گے", "ہاں", "ہر", "ہم", "ہمارا", "ہماری", "ہمارے", "ہمیشہ", "ہو", "ہوئی", "ہوئیں", "ہوئے", "ہوا", "ہوتا", "ہوتی", "ہوتیں", "ہوتے", "ہونا", "ہونگے", "ہونی", "ہونے", "ہوں", "ہی", "ہیں", "ہے", "یا", "یات", "یعنی", "یقینا", "یہ", "یہاں", "یہی", "یہیں"]
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(sublinear_tf=True, use_idf=True ,smooth_idf=False, min_df=3, norm='l2', encoding='utf-8', ngram_range=(1, 3), stop_words=sw)







##from sklearn.feature_extraction.text import HashingVectorizer

#hasher = HashingVectorizer( stop_words=sw, alternate_sign=False, norm=None, binary=False,encoding='utf-8' )

    #(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False

    

features = tfidf.fit_transform(df.details).toarray() # Remaps the words in the 1490 articles in the text column of 

                                                  # data frame into features (superset of words) with an importance assigned 

                                                  # based on each words frequency in the document and across documents



#features = hasher.fit_transform(df.details)        

        

labels = df.category_id                           # represents the category of each of the 1490 articles



#tfidf = HashingVectorizer(stop_words=sw, alternate_sign=False,n_features= category.n_features)

features.shape
features.shape # How many features are there ? 
category_to_id.items()
sorted(category_to_id.items())
# Use chi-square analysis to find corelation between features (importantce of words) and labels(news category) 

from sklearn.feature_selection import chi2

import numpy as np

N = 3  # We are going to look for top 3 categories



#For each category, find words that are highly corelated to it

for Category, category_id in sorted(category_to_id.items()):

  features_chi2 = chi2(features, labels == category_id)                   # Do chi2 analyses of all items in this category

  indices = np.argsort(features_chi2[0])                                  # Sorts the indices of features_chi2[0] - the chi-squared stats of each feature

  feature_names = np.array(tfidf.get_feature_names())[indices]            # Converts indices to feature names ( in increasing order of chi-squared stat values)

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]         # List of single word features ( in increasing order of chi-squared stat values)

  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]          # List for two-word features ( in increasing order of chi-squared stat values)

  print("# '{}':".format(Category))

  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:]))) # Print 3 unigrams with highest Chi squared stat

  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:]))) # Print 3 bigrams with highest Chi squared stat
features_chi2
from sklearn.manifold import TSNE



# Sampling a subset of our dataset because t-SNE is computationally expensive

SAMPLE_SIZE = int(len(features) * 0.3)

np.random.seed(0)

indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)          # Randomly select 30 % of samples

projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices]) # Array of all projected features of 30% of Randomly chosen samples 
type(projected_features)
my_id = 0 # Select a category_id

projected_features[(labels[indices] == my_id).values]
colors = ['pink', 'green', 'midnightblue', 'orange',  'black','darkgrey', 'blue','red','brown']



# Find points belonging to each category and plot them

for category, category_id in sorted(category_to_id.items()):

    points = projected_features[(labels[indices] == category_id).values]

    plt.scatter(points[:, 0], points[:, 1], s=60, c=colors[category_id], label=category)

plt.title("tf-idf feature vector for each sentece, projected on 2 dimensions.",

          fontdict=dict(fontsize=15))

plt.legend()
features.shape
#$!pip install git+https://github.com/darenr/scikit-optimize
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import RidgeClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier





from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestCentroid

from sklearn.linear_model import SGDClassifier





from sklearn.model_selection import cross_val_score
models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=1),

    

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=10, p=2,weights='uniform'),

    NearestCentroid(metric='euclidean', shrink_threshold=None),   

    LogisticRegression(random_state=0),

    MultinomialNB(),    

    Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,fit_intercept=True, max_iter=50, n_iter_no_change=5,n_jobs=None, penalty=None, random_state=0, shuffle=True, tol=0.001,validation_fraction=0.1, verbose=0, warm_start=False),    

    #SGDClassifier(alpha=0.0001, average=False, class_weight=None,  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',power_t=0.5, random_state=None, shuffle=True, tol=None,validation_fraction=0.1, verbose=0, warm_start=False), 

    

    

##    PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,

  #            early_stopping=False, fit_intercept=True, loss='hinge',

   #           max_iter=50, n_iter=None, n_iter_no_change=5, n_jobs=None,

    #          random_state=None, shuffle=True, tol=0.001,

     #         validation_fraction=0.1, verbose=0, warm_start=False),

   

    LinearSVC(penalty="l2", dual=False, tol=1e-2),

    LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,   multi_class='ovr', penalty='l2', random_state=None, tol=0.001,     verbose=0),

    

   

    RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,

        max_iter=None, normalize=False, random_state=None, solver='sag',

        tol=0.01),

  

    

    

    

    

]
CV = 5  # Cross Validate with 5 different folds of 20% data ( 80-20 split with 5 folds )



#Create a data frame that will store the results for all 5 trials of the 3 different models

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = [] # Initially all entries are empty
#For each Algorithm 

for model in models:

  model_name = model.__class__.__name__

  # create 5 models with different 20% test sets, and store their accuracies

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  # Append all 5 accuracies into the entries list ( after all 3 models are run, there will be 3x5 = 15 entries)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))

# Store the entries into the results dataframe and name its columns    

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns



ax=sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df,size=8, jitter=True, edgecolor="gray", linewidth=2 )

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
cv_df.groupby('model_name').accuracy.mean()*100
cv_df
from sklearn.model_selection import train_test_split



model = LogisticRegression(random_state=0)



#Split Data 

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)



#Train Algorithm

model.fit(X_train, y_train)



# Make Predictions

y_pred_proba = model.predict_proba(X_test)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

import seaborn as sns



conf_mat = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_mat, annot=True, fmt='d')

plt.ylabel('Actual')

plt.xlabel('Predicted')
from IPython.display import display



print(id_to_category)



for predicted in category_id_df.category_id:

  for actual in category_id_df.category_id:

    if predicted != actual and conf_mat[actual, predicted] >= 5:

      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))

      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['class', 'details']])

      print('')
model.fit(features, labels)

model.coef_
from sklearn.feature_selection import chi2



N = 5

for Category, category_id in sorted(category_to_id.items()):

  indices = np.argsort(model.coef_[category_id])   # This time using the model co-eficients / weights

  feature_names = np.array(tfidf.get_feature_names())[indices]

  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]

  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]

  print("# '{}':".format(Category))

  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))

  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
texts = [" تحریک انصاف کا وفد چین روانہ، کس کس کو ساتھ لے گئے ؟ میرٹ کی دھجیاں اڑ گئیں",

         "عید میلاد النبی ﷺ پر امیتابھ بچن کا مسلمانوں کیلئے پیغام",

         "فلمی و ادبی شخصیات کے سکینڈلز۔ ۔ ۔قسط نمبر558",

         "ریاست مدینہ کی طرز پر کام شروع ، کوئی سردار ہویا نواب درگزر نہیں کیا جائے گا",

         "جس وقت چینی قونصل خانے پر حملہ ہوا اس وقت کتنے چینی اندر موجود تھے اور وہ اب کہاں ", 

        " ""مدارس اور تعلیمی اداروں کی اصلاحات کی بات کی تھی، آصف غفور", "جنگل میں کوڑا پھیلانا ماحولیاتی مسئلے کا ممکنہ حل نہیں ہو سکتا لیکن جنوبی امریکہ کے ملک کوسٹا ریکا میں بالکل ایسا ہی ہوا", "کیا ہمیں واقعی فکرمند ہونا چاہیے کہ ’بھائی صاحب‘ سب کچھ دیکھ سن رہے ہیں", "کورونا وائرس سے بچاؤ کے لیے کراچی بندر گاہ پہنچنے والا چینی سامان ریلیز کرنے سے انکار"

        ]

text_features = tfidf.transform(texts)

predictions = model.predict(text_features)

for text, predicted in zip(texts, predictions):

  print('"{}"'.format(text))

  print("  - Predicted as: '{}'".format(id_to_category[predicted]))

  print("")