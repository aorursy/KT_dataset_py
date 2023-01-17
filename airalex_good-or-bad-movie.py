import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer





data = pd.read_csv("../input/movie_metadata.csv")

clean_data = data.dropna(axis = 0)



x_list = ['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','duration',

          'num_critic_for_reviews','num_voted_users','num_user_for_reviews','budget','gross']







def token(text):

    return(text.split("|"))



cv_kw=CountVectorizer(max_features=100,tokenizer=token )

keywords = cv_kw.fit_transform(clean_data["plot_keywords"])

keywords_list = ["kw_" + i for i in cv_kw.get_feature_names()]



cv_ge=CountVectorizer(tokenizer=token )

genres = cv_ge.fit_transform(clean_data["genres"])

genres_list = ["gn_"+ i for i in cv_ge.get_feature_names()]



x_all = np.hstack([clean_data.ix[:,x_list],keywords.todense(),genres.todense()])

y_all = clean_data['imdb_score']

x_coeff = x_list+keywords_list+genres_list



x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25, random_state=0)
mid = np.percentile(y_train,50)

y_train = [int(i) for i in list(y_train>=mid)]

y_test = [int(i) for i in list(y_test>=mid)]
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import itertools



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    

def plot_test(x,y,model):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, [i[1] for i in model.predict_proba(x)])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, "b", label='AUC %0.2f' % (roc_auc))

    plt.title("AUC Curve")

    plt.show()



    auc_score = roc_auc_score(y, [i[1] for i in mod.predict_proba(x)])

    cm = confusion_matrix(y,mod.predict(x))

    plot_confusion_matrix(cm,classes = ["bad movie","good movie"],normalize=False)

    print("AUC Score: %f" % auc_score)

    print("Accuracy: %f" % (sum(mod.predict(x) == y)/float(len(y))))

mod = DecisionTreeClassifier(max_depth = 7)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
mod = RandomForestClassifier(n_estimators=100,max_features = int(np.sqrt(len(x_coeff))))

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
mod.estimators_[0]
mod = GradientBoostingClassifier(n_estimators=80, learning_rate=1.0, max_depth=1, random_state=0)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)