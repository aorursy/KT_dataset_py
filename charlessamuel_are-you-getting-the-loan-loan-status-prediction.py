# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



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
FILEPATH = '/kaggle/input/loan-data-set/loan_data_set.csv'
df = pd.read_csv(FILEPATH)

df.head()
df.shape
df.describe(include='all')
df.info()
df.isnull().any()
df.isna().sum()
df.Credit_History.fillna(df.Credit_History.mean(), inplace=True)

df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(), inplace=True)
df.dropna(how="any",inplace=True)
df.isnull().any()
df.drop("Loan_ID", axis=1, inplace=True)
le = LabelEncoder()

cols = df.columns.tolist()

for column in cols:

    if df[column].dtype == 'object':

        df[column] = le.fit_transform(df[column])
df.dtypes
fig, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(data=df.corr().round(2), annot=True, linewidths=0.7, cmap='YlGnBu')

plt.show()
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
X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]



rand_f = RandomForestClassifier().fit(X, y)



plot_feature_importance(rand_f.feature_importances_, X.columns, 'RANDOM FOREST')
gb_m = GradientBoostingClassifier().fit(X, y)



plot_feature_importance(gb_m.feature_importances_, X.columns, 'GRADIENT BOOSTING')
ada = AdaBoostClassifier().fit(X, y)



plot_feature_importance(ada.feature_importances_, X.columns, 'ADA BOOST')
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}



svc = SVC()



grid = GridSearchCV(svc, parameters)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=30) 
grid.fit(Xtrain, ytrain)
grid.best_params_
pred = grid.best_estimator_.predict(Xtest)
confusion_matrix(ytest,pred)
print("Accuracy score: {0}%".format((accuracy_score(ytest,pred)*100).round(2)))
fig,ax=plt.subplots(figsize=(15,8))

sns.regplot(x=ytest,y=pred,marker="*")

plt.show()