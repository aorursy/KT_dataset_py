# Dataset located here: https://www.kaggle.com/rmisra/news-category-dataset/version/2

# Based on tutorial from https://www.kaggle.com/divsinha



import pandas as pd

import json

import numpy as np

import pandas as pd

import seaborn as sns

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import pprint

pp = pprint.PrettyPrinter(indent=4)

% matplotlib inline
PATH = "../input/News_Category_Dataset_v2.json"
container = []



with open(PATH) as fr:

    for line in fr.readlines():

        row_obj = json.loads(line)

        container.append(row_obj)



df = pd.DataFrame(data=container, columns=["category", "headline", "authors", "link", "short_description", "date"])
df.head(50)
def show_5_category(category):

    return df[(df["category"] == category)][:5]
categories, categorical_data = df['category'].unique().tolist(), dict()



for category in categories:

    categorical_data["df_{}".format(category.lower().replace(" ", ""))] = df[(df['category'] == category)]

    

categorical_data.keys()
# TODO Get this working. Should produce a sample of 5 of each category of article

df_samp5 = pd.DataFrame(columns=["category", "headline", "authors", "link", "short_description", "date"])



for category in categorical_data.keys():

    df_samp5.append(categorical_data[category][:5])



df_samp5
df.info()
df.describe()
df.isnull().sum()
df['category'].value_counts().sort_values(ascending=False).plot
df['category'].value_counts().sort_values(ascending=False).plot.bar
plt.figure(figsize=(25,15))

cmapper = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

df['category'].value_counts().sort_values(ascending=False).plot.bar(color=cmapper)

plt.xticks(rotation=50)

plt.xlabel("Category of News")

plt.ylabel("Number of Articles")
# TODO make option for image



def create_wordcloud(category):

    text = " ".join(desc for desc in categorical_data[category]['short_description'])

    wordcloud = WordCloud(width=1500, height=800, max_font_size=200, background_color = 'white', stopwords = STOPWORDS).generate(text)

    plt.figure(figsize=(20,15))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Count Vectorizer for entire model 

cvector = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2), stop_words = STOPWORDS)

cvector.fit(df['headline'])
len(cvector.get_feature_names())
def create_tf_matrix(category):

    return cvector.transform(df[df.category == category].headline)
crime_matrix = create_tf_matrix('CRIME')

entertainment_matrix = create_tf_matrix('ENTERTAINMENT')

world_news_matrix = create_tf_matrix('WORLD NEWS')

impact_matrix = create_tf_matrix('IMPACT')

politics_matrix = create_tf_matrix('POLITICS')

weird_news_matrix = create_tf_matrix('WEIRD NEWS')

black_voices_matrix = create_tf_matrix('BLACK VOICES')

women_matrix = create_tf_matrix('WOMEN')

comedy_matrix = create_tf_matrix('COMEDY')

queer_voices_matrix = create_tf_matrix('QUEER VOICES')

sports_matrix = create_tf_matrix('SPORTS')

business_matrix = create_tf_matrix('BUSINESS')

travel_matrix = create_tf_matrix('TRAVEL')

media_matrix = create_tf_matrix('MEDIA')

tech_matrix = create_tf_matrix('TECH')

religion_matrix = create_tf_matrix('RELIGION')

science_matrix = create_tf_matrix('SCIENCE')

latino_voices_matrix = create_tf_matrix('LATINO VOICES')

education_matrix = create_tf_matrix('EDUCATION')

college_matrix = create_tf_matrix('COLLEGE')

parents_matrix = create_tf_matrix('PARENTS')

arts_and_culture_matrix = create_tf_matrix('ARTS & CULTURE')

style_matrix = create_tf_matrix('STYLE')

green_matrix = create_tf_matrix('GREEN')

taste_matrix = create_tf_matrix('TASTE')

healthy_living_matrix = create_tf_matrix('HEALTHY LIVING')

the_worldpost_matrix = create_tf_matrix('THE WORLDPOST')

good_news_matrix = create_tf_matrix('GOOD NEWS')

worldpost_matrix = create_tf_matrix('WORLDPOST')

fifty_matrix = create_tf_matrix('FIFTY')

arts_matrix = create_tf_matrix('ARTS')

wellness_matrix = create_tf_matrix('WELLNESS')

parenting_matrix = create_tf_matrix('PARENTING')

home_and_living_matrix = create_tf_matrix('HOME & LIVING')

style_and_beauty_matrix = create_tf_matrix('STYLE & BEAUTY')

divorce_matrix = create_tf_matrix('DIVORCE')

weddings_matrix = create_tf_matrix('WEDDINGS')

food_and_drink_matrix = create_tf_matrix('FOOD & DRINK')

money_matrix = create_tf_matrix('MONEY')

environment_matrix = create_tf_matrix('ENVIRONMENT')

culture_and_arts_matrix = create_tf_matrix('CULTURE & ARTS')
def create_term_freq(matrix):

    category_words = matrix.sum(axis=0)

    category_words_freq = [(word, category_words[0, idx]) for word, idx in cvector.vocabulary_.items()]

    return pd.DataFrame(list(sorted(category_words_freq, key = lambda x: x[1], reverse=True)),columns=['Terms', 'Frequency'])
create_term_freq(crime_matrix), create_wordcloud('df_crime')
create_term_freq(entertainment_matrix).head(10), create_wordcloud('df_entertainment')
create_term_freq(world_news_matrix).head(10), create_wordcloud('df_worldnews')
create_term_freq(impact_matrix).head(10), create_wordcloud('df_impact')
create_term_freq(politics_matrix).head(10), create_wordcloud('df_politics')
create_term_freq(weird_news_matrix).head(10), create_wordcloud('df_weirdnews')
create_term_freq(black_voices_matrix).head(10), create_wordcloud('df_blackvoices')
create_term_freq(women_matrix).head(10), create_wordcloud('df_women')
create_term_freq(comedy_matrix).head(10), create_wordcloud('df_comedy')
create_term_freq(queer_voices_matrix).head(10), create_wordcloud('df_queervoices')
create_term_freq(sports_matrix).head(10), create_wordcloud('df_sports')
create_term_freq(business_matrix).head(10), create_wordcloud('df_business')
create_term_freq(travel_matrix).head(10), create_wordcloud('df_travel')
create_term_freq(media_matrix).head(10), create_wordcloud('df_media')
create_term_freq(tech_matrix).head(10), create_wordcloud('df_tech')
create_term_freq(religion_matrix).head(10), create_wordcloud('df_religion')
create_term_freq(science_matrix).head(10), create_wordcloud('df_science')
create_term_freq(latino_voices_matrix).head(10), create_wordcloud('df_latinovoices')
create_term_freq(education_matrix).head(10), create_wordcloud('df_education')
create_term_freq(college_matrix).head(10), create_wordcloud('df_college')
create_term_freq(parents_matrix).head(10), create_wordcloud('df_parents')
create_term_freq(arts_and_culture_matrix).head(10), create_wordcloud('df_arts&culture')
create_term_freq(style_matrix).head(10), create_wordcloud('df_style')
create_term_freq(green_matrix).head(10), create_wordcloud('df_green')
create_term_freq(taste_matrix).head(10), create_wordcloud('df_taste')
create_term_freq(healthy_living_matrix).head(10), create_wordcloud('df_healthyliving')
create_term_freq(the_worldpost_matrix).head(10), create_wordcloud('df_theworldpost')
create_term_freq(good_news_matrix).head(10), create_wordcloud('df_goodnews')
create_term_freq(worldpost_matrix).head(10), create_wordcloud('df_worldpost')
create_term_freq(fifty_matrix).head(10), create_wordcloud('df_fifty')
create_term_freq(arts_matrix).head(10), create_wordcloud('df_arts')
create_term_freq(wellness_matrix).head(10), create_wordcloud('df_wellness')
create_term_freq(parenting_matrix).head(10), create_wordcloud('df_parenting')
create_term_freq(home_and_living_matrix).head(10), create_wordcloud('df_home&living')
create_term_freq(style_and_beauty_matrix).head(10), create_wordcloud('df_style&beauty')
create_term_freq(divorce_matrix).head(10), create_wordcloud('df_divorce')
create_term_freq(weddings_matrix).head(10), create_wordcloud('df_weddings')
create_term_freq(food_and_drink_matrix).head(10), create_wordcloud('df_food&drink')
create_term_freq(money_matrix).head(10), create_wordcloud('df_money')
create_term_freq(environment_matrix).head(10), create_wordcloud('df_environment')
create_term_freq(culture_and_arts_matrix).head(10), create_wordcloud('df_culture&arts')
headline = np.array(df['headline'])

category = np.array(df['category'])

# build train and test datasets



# Train/test splitting for 41 categories of news

from sklearn.model_selection import train_test_split    

headline_train, headline_test, category_train, category_test = train_test_split(headline, category, test_size=0.2, random_state=41)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



## Build Bag-Of-Words on train phrases

cv = CountVectorizer(stop_words='english',max_features=10000)

cv_train_features = cv.fit_transform(headline_train)
# build TFIDF features on train reviews

tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),

                     sublinear_tf=True)

tv_train_features = tv.fit_transform(headline_train)
cv_test_features = cv.transform(headline_test)

tv_test_features = tv.transform(headline_test)
print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)

print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)
from sklearn import metrics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.base import clone

from sklearn.preprocessing import label_binarize

from scipy import interp

from sklearn.metrics import roc_curve, auc 





def get_metrics(true_labels, predicted_labels):

    

    print('Accuracy:', np.round(

                        metrics.accuracy_score(true_labels, 

                                               predicted_labels),

                        4))

    print('Precision:', np.round(

                        metrics.precision_score(true_labels, 

                                               predicted_labels,

                                               average='weighted'),

                        4))

    print('Recall:', np.round(

                        metrics.recall_score(true_labels, 

                                               predicted_labels,

                                               average='weighted'),

                        4))

    print('F1 Score:', np.round(

                        metrics.f1_score(true_labels, 

                                               predicted_labels,

                                               average='weighted'),

                        4))

                        



def train_predict_model(classifier, 

                        train_features, train_labels, 

                        test_features, test_labels):

    # build model    

    classifier.fit(train_features, train_labels)

    # predict using model

    predictions = classifier.predict(test_features) 

    return predictions    





def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):

    

    total_classes = len(classes)

    level_labels = [total_classes*[0], list(range(total_classes))]



    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 

                                  labels=classes)

    cm_frame = pd.DataFrame(data=cm, 

                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 

                                                  labels=level_labels), 

                            index=pd.MultiIndex(levels=[['Actual:'], classes], 

                                                labels=level_labels)) 

    print(cm_frame) 

    

def display_classification_report(true_labels, predicted_labels, classes=[1,0]):



    report = metrics.classification_report(y_true=true_labels, 

                                           y_pred=predicted_labels, 

                                           labels=classes) 

    print(report)

    

    

    

def display_model_performance_metrics(true_labels, predicted_labels, classes=[1,0]):

    print('Model Performance metrics:')

    print('-'*30)

    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

    print('\nModel Classification report:')

    print('-'*30)

    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, 

                                  classes=classes)

    print('\nPrediction Confusion Matrix:')

    print('-'*30)

    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, 

                             classes=classes)





def plot_model_decision_surface(clf, train_features, train_labels,

                                plot_step=0.02, cmap=plt.cm.RdYlBu,

                                markers=None, alphas=None, colors=None):

    

    if train_features.shape[1] != 2:

        raise ValueError("X_train should have exactly 2 columnns!")

    

    x_min, x_max = train_features[:, 0].min() - plot_step, train_features[:, 0].max() + plot_step

    y_min, y_max = train_features[:, 1].min() - plot_step, train_features[:, 1].max() + plot_step

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),

                         np.arange(y_min, y_max, plot_step))



    clf_est = clone(clf)

    clf_est.fit(train_features,train_labels)

    if hasattr(clf_est, 'predict_proba'):

        Z = clf_est.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

    else:

        Z = clf_est.predict(np.c_[xx.ravel(), yy.ravel()])    

    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=cmap)

    

    le = LabelEncoder()

    y_enc = le.fit_transform(train_labels)

    n_classes = len(le.classes_)

    plot_colors = ''.join(colors) if colors else [None] * n_classes

    label_names = le.classes_

    markers = markers if markers else [None] * n_classes

    alphas = alphas if alphas else [None] * n_classes

    for i, color in zip(range(n_classes), plot_colors):

        idx = np.where(y_enc == i)

        plt.scatter(train_features[idx, 0], train_features[idx, 1], c=color,

                    label=label_names[i], cmap=cmap, edgecolors='black', 

                    marker=markers[i], alpha=alphas[i])

    plt.legend()

    plt.show()





def plot_model_roc_curve(clf, features, true_labels, label_encoder=None, class_names=None):

    

    ## Compute ROC curve and ROC area for each class

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    if hasattr(clf, 'classes_'):

        class_labels = clf.classes_

    elif label_encoder:

        class_labels = label_encoder.classes_

    elif class_names:

        class_labels = class_names

    else:

        raise ValueError('Unable to derive prediction classes, please specify class_names!')

    n_classes = len(class_labels)

    y_test = label_binarize(true_labels, classes=class_labels)

    if n_classes == 2:

        if hasattr(clf, 'predict_proba'):

            prob = clf.predict_proba(features)

            y_score = prob[:, prob.shape[1]-1] 

        elif hasattr(clf, 'decision_function'):

            prob = clf.decision_function(features)

            y_score = prob[:, prob.shape[1]-1]

        else:

            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")

        

        fpr, tpr, _ = roc_curve(y_test, y_score)      

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'

                                 ''.format(roc_auc),

                 linewidth=2.5)

        

    elif n_classes > 2:

        if hasattr(clf, 'predict_proba'):

            y_score = clf.predict_proba(features)

        elif hasattr(clf, 'decision_function'):

            y_score = clf.decision_function(features)

        else:

            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")



        for i in range(n_classes):

            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

            roc_auc[i] = auc(fpr[i], tpr[i])



        ## Compute micro-average ROC curve and ROC area

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



        ## Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points

        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):

            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr

        tpr["macro"] = mean_tpr

        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



        ## Plot ROC curves

        plt.figure(figsize=(6, 4))

        plt.plot(fpr["micro"], tpr["micro"],

                 label='micro-average ROC curve (area = {0:0.2f})'

                       ''.format(roc_auc["micro"]), linewidth=3)



        plt.plot(fpr["macro"], tpr["macro"],

                 label='macro-average ROC curve (area = {0:0.2f})'

                       ''.format(roc_auc["macro"]), linewidth=3)



        for i, label in enumerate(class_labels):

            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'

                                           ''.format(label, roc_auc[i]), 

                     linewidth=2, linestyle=':')

    else:

        raise ValueError('Number of classes should be atleast 2 or more')

        

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend(loc="lower right")

    plt.show()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2', max_iter=100, C=1)
lr_bow_predictions = train_predict_model(classifier=lr, 

                                             train_features=cv_train_features, train_labels=category_train,

                                             test_features=cv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=lr_bow_predictions,

                                      classes=df['category'].unique())
lr_tfidf_predictions = train_predict_model(classifier=lr, 

                                               train_features=tv_train_features, train_labels=category_train,

                                               test_features=tv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=lr_tfidf_predictions,

                                      classes=df['category'].unique())
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier(loss='hinge', n_iter=100)
sgd_bow_predictions = train_predict_model(classifier=sgd, 

                                             train_features=cv_train_features, train_labels=category_train,

                                             test_features=cv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=sgd_bow_predictions,

                                      classes=df['category'].unique())
sgd_tfidf_predictions = train_predict_model(classifier=sgd, 

                                                train_features=tv_train_features, train_labels=category_train,

                                                test_features=tv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=sgd_tfidf_predictions,

                                      classes=df['category'].unique())
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1)
rfc_bow_predictions = train_predict_model(classifier=rfc, 

                                             train_features=cv_train_features, train_labels=category_train,

                                             test_features=cv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=rfc_bow_predictions,

                                      classes=df['category'].unique())
rfc_tfidf_predictions = train_predict_model(classifier=rfc, 

                                                train_features=tv_train_features, train_labels=category_train,

                                                test_features=tv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=rfc_tfidf_predictions,

                                      classes=df['category'].unique())
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb_bow_predictions = train_predict_model(classifier=nb, 

                                             train_features=cv_train_features, train_labels=category_train,

                                             test_features=cv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=nb_bow_predictions,

                                      classes=df['category'].unique())
nb_tfidf_predictions = train_predict_model(classifier=nb, 

                                                train_features=tv_train_features, train_labels=category_train,

                                                test_features=tv_test_features, test_labels=category_test)

display_model_performance_metrics(true_labels=category_test, predicted_labels=nb_tfidf_predictions,

                                      classes=df['category'].unique())