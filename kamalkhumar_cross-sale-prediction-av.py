# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

from pylab import plot, show, subplot, specgram, imshow, savefig

from sklearn import preprocessing

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Normalizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

%matplotlib inline
train = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")

test.head()
train.info()
train.isna().sum()
train.dtypes
train.shape
def plot_feature_importance(importance,names,model_type):



    #Create arrays from feature importance and feature names

    feature_importance = np.array(importance)

    feature_names = np.array(names)



    #Create a DataFrame using a Dictionary

    data={'feature_names':feature_names,'feature_importance':feature_importance}

    fi_df = pd.DataFrame(data)



    #Sort the DataFrame in order decreasing feature importance

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)



    #Define size of bar plot

    plt.figure(figsize=(10,8))

    #Plot Searborn bar chart

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    #Add chart labels

    plt.title(model_type + ' FEATURE IMPORTANCE')

    plt.xlabel('FEATURE IMPORTANCE')

    plt.ylabel('FEATURE NAMES')
le = LabelEncoder()

train["Gender"] = le.fit_transform(train["Gender"])

train["Vehicle_Age"] = le.fit_transform(train["Vehicle_Age"])

train["Vehicle_Damage"] = le.fit_transform(train["Vehicle_Damage"])



test["Gender"] = le.fit_transform(test["Gender"])

test["Vehicle_Age"] = le.fit_transform(test["Vehicle_Age"])

test["Vehicle_Damage"] = le.fit_transform(test["Vehicle_Damage"])
rf_model = RandomForestClassifier().fit(train.drop(["id", "Response"],axis=1),train["Response"])

plot_feature_importance(rf_model.feature_importances_,train.drop(["id", "Response"],axis=1).columns,'RANDOM FOREST')
train.dtypes
train.head()
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.heatmap(data=train.drop("id", axis=1).corr().round(2), annot = True)

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

sns.lineplot(x='Age', y='Vintage', data=train)

plt.show()
sns.set(rc={'figure.figsize':(19,8)})

sns.distplot(train['Age'], kde=True)

plt.show()
def show_donut_plot(col): #donut plot function

    

    rating_data =train.groupby(col)[['id']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['id']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot: Response Proportion for Cross-Sale ', loc='center')

    plt.show()
show_donut_plot('Response')
print(train[train.Response == 1].shape)

print(train[train.Response == 0].shape)
shuffled_train = train.sample(frac=1,random_state=4)



train_1 = shuffled_train.loc[shuffled_train['Response'] == 1]



train_0 = shuffled_train.loc[shuffled_train['Response'] == 0].sample(n = 46710,random_state=42)



new_train = pd.concat([train_1, train_0])



new_train = new_train.sample(frac=1,random_state=4)



plt.figure(figsize=(8, 8))

sns.countplot('Response', data=new_train)

plt.title('Balanced Data')

plt.show()

new_train.drop('id', axis=1, inplace=True)

Id = test['id'].tolist()

test.drop('id', axis=1, inplace=True)
print(new_train.columns)

print(test.columns)
train_ = new_train.drop(['Gender', 'Driving_License'], axis=1)

test_ = test.drop(['Gender', 'Driving_License'], axis=1)
#List of classifiers



clfs = {

    'mnb': MultinomialNB(),

    'gnb': GaussianNB(),

    'dtc': DecisionTreeClassifier(),

    'rfc': RandomForestClassifier(),

    'lr': LogisticRegression(),

    'gbc': GradientBoostingClassifier()

}
#accuracy for the list of classifiers



accuracy_scores = dict()

train_x, test_x, train_y, test_y = train_test_split(train_.drop("Response", axis=1), train_["Response"], test_size= 0.3)

for clf_name in clfs:

    

    clf = clfs[clf_name]

    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)

    accuracy_scores[clf_name] = accuracy_score(y_pred, test_y)

    print(clf, '-' , accuracy_scores[clf_name])
accuracy_scores = dict(sorted(accuracy_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse= True))

villi = list(accuracy_scores.keys())[0]

print("Classifier with high accuracy --> ",clfs[villi])

print("With the accuracy of",accuracy_scores[villi])
clfs[villi].fit(train_.drop("Response", axis=1), train_["Response"])
print(train_.columns)

print(test_.columns)
pred = clfs[villi].predict(test_)

pred = pred.tolist()
sub = pd.DataFrame({"id" : Id, "Response" : pred})
s = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/sample_submission.csv")

s.head()
s.shape
test.shape
s.to_csv("Submission_CSP.csv", index=False)
s.Response.value_counts()