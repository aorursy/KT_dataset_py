# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.cross_validation import train_test_split



from sklearn import preprocessing

Encode = preprocessing.LabelEncoder()

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn import metrics





from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import MultinomialNB



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

chatbot_df = pd.read_csv("../input/Sheet_1.csv",usecols=['response_id','class','response_text'],encoding='utf-8')

chatbot_df['class_label'] = Encode.fit_transform(chatbot_df['class'])

x_train,x_test,y_train,y_test = train_test_split(chatbot_df["response_text"],

                                                 chatbot_df['class_label'],

                                                 random_state=1)



result_cols = ["Classifier", "Accuracy"]

result_frame = pd.DataFrame(columns=result_cols)



# Any results you write to the current directory are saved as output.



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    MultinomialNB()]





for clf in classifiers:

    name = clf.__class__.__name__

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),

                  ('clf', clf),])

    text_clf.fit(x_train, y_train)

    

    predicted = text_clf.predict(x_test)

    acc = metrics.accuracy_score(y_test,predicted)

    print (name+' accuracy = '+str(acc*100)+'%')

    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)

    result_frame = result_frame.append(acc_field)

    

sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



# Any results you write to the current directory are saved as output.