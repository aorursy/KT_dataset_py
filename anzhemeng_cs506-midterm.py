# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import missingno as msno 

import scipy

from scipy.sparse import hstack

from PIL import Image

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



warnings.filterwarnings('ignore')



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/bu-cs506-spring-2020-midterm/train.csv')
data.head()
plt.title('10 Most Active Users')

data['UserId'].value_counts(sort=True).nlargest(10).plot.bar()
print('There are', str(data.shape[0]), 'records in total.')
plt.title('10 Most Rated Products')

data['ProductId'].value_counts(sort=True).nlargest(10).plot.bar()
HelpfulnessNumerator0 = data[data['HelpfulnessNumerator'] == 0]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumerator1 = data[data['HelpfulnessNumerator'] == 1]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumerator2 = data[data['HelpfulnessNumerator'] == 2]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumerator3 = data[data['HelpfulnessNumerator'] == 3]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumeratorMoreThan3 = data[data['HelpfulnessNumerator'] > 3]['HelpfulnessNumerator'].value_counts()



labels = '0', '1', '2', '3', 'more than 3'

sizes = [HelpfulnessNumerator0.values.item(), HelpfulnessNumerator1.values.item(), HelpfulnessNumerator2.values.item(), 

         HelpfulnessNumerator3.values.item(), HelpfulnessNumeratorMoreThan3.values.sum()]

explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig, ax = plt.subplots()

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title(' Portions of Amount of Helpfulness Labels')

plt.show()
HelpfulnessDenominator0 = data[data['HelpfulnessDenominator'] == 0]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominator1 = data[data['HelpfulnessDenominator'] == 1]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominator2 = data[data['HelpfulnessDenominator'] == 2]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominator3 = data[data['HelpfulnessDenominator'] == 3]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominatorMoreThan3 = data[data['HelpfulnessDenominator'] > 3]['HelpfulnessDenominator'].value_counts()



labels = '0', '1', '2', '3', 'more than 3'

sizes = [HelpfulnessDenominator0.values.item(), HelpfulnessDenominator1.values.item(), HelpfulnessDenominator2.values.item(), 

         HelpfulnessDenominator3.values.item(), HelpfulnessDenominatorMoreThan3.values.sum()]

explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig, ax = plt.subplots()

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title(' Portions of Amount of Comments Watched')

plt.show()
plt.title('Scores')

data['Score'].value_counts().plot.bar()
fig = plt.figure(figsize=(14, 10))

ax = fig.add_subplot(121)

text = data.Summary.values

wordcloud = WordCloud(

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

plt.title('Summary Keywords')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)



ax = fig.add_subplot(122)

text = data.Text.values

wordcloud = WordCloud(

    background_color = 'white',

    stopwords = STOPWORDS).generate(str(text))

plt.title('Text Keywords')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
msno.matrix(data) 
msno.heatmap(data) 
# fives = data.loc[data['Score'] == 5]

# fives = fives.sample(frac=0.5)

# data = pd.concat([data.loc[data['Score'] != 5], fives])
OHE = OneHotEncoder(sparse=True)

ID_fitter = OHE.fit(data[['ProductId', 'UserId']])

IDs = ID_fitter.transform(data[['ProductId', 'UserId']])
# # collect the model to reuse later

# from joblib import dump, load

# dump(ID_fitter, 'OHE.joblib') 
data['Text'].loc[data['Text'].isna()] = ''

data['Summary'].loc[data['Summary'].isna()] = ''
data['Helpful'] = data['HelpfulnessNumerator']

data['Unhelpful'] = data['HelpfulnessDenominator'] - data['HelpfulnessNumerator']

scaler = StandardScaler()

scalerFitter = scaler.fit(data[['Helpful', 'Unhelpful', 'Time']])

data[['Helpful', 'Unhelpful', 'Time']] = scalerFitter.transform(data[['Helpful', 'Unhelpful', 'Time']])

data = data.drop(['HelpfulnessDenominator','HelpfulnessNumerator'], axis=1)
# dump(scalerFitter, 'scaler.joblib')
text_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english')

summary_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english')

text_fitter = text_vectorizer.fit(data['Text'])

text_matrix = text_fitter.transform(data['Text'])

summary_fitter = summary_vectorizer.fit(data['Summary'])

summary_matrix = summary_fitter.transform(data['Summary'])
# dump(text_fitter, 'text.joblib')

# dump(summary_fitter, 'summary.joblib')
text_matrix, summary_matrix
numerical = scipy.sparse.csr_matrix(data[['Helpful', 'Unhelpful', 'Time']].values)
X = hstack([text_matrix, summary_matrix, numerical, IDs])
mask = data["Score"].isnull()



ind_test = mask.to_numpy().nonzero()[0]

ind_train = (~ mask).to_numpy().nonzero()[0]



train_X = scipy.sparse.csr_matrix(X)[ind_train]

test_X = scipy.sparse.csr_matrix(X)[ind_test]
# plt.spy(train_X)
train_Y = data['Score'].loc[data['Score'].isna() == False]

test_Y = data['Score'].loc[data['Score'].isna()]



train_Y = train_Y.reset_index()['Score']

test_Y = test_Y.reset_index()['Score']
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=42)

train_X, train_Y = ros.fit_resample(train_X, train_Y)
plt.title('Scores')

train_Y.value_counts().plot.bar()
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
def CVKFold(k, X, y, model):

    np.random.seed(1)

    #reproducibility

    

    highest_accuracy = float('inf')

    best_model = None



    kf = KFold(n_splits = k,shuffle =True)

    #CV loop

    

    for train_index,test_index in kf.split(X):#generation of the sets

    #generate the sets    

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        #model fitting

        model.fit(X_train,y_train)

        y_test_pred = model.predict(X_test)

    

        test_accuracy = mean_squared_error(y_test_pred, y_test)

        print("The accuracy is " + str(test_accuracy))

        if test_accuracy < highest_accuracy:

          best_model = model

          highest_accuracy = test_accuracy



    print("The highest accuracy is " + str(highest_accuracy))

    return best_model, highest_accuracy
# model = LogisticRegression(random_state = 0)

# model = model.fit(train_X, train_Y)

# dump(model, 'model.joblib')
# Logistics Regression

model = LogisticRegression(random_state = 0)

model = model.fit(train_X, train_Y)

# clf_Log, accuracy_Log = CVKFold(5, train_X, train_Y, model)

# Decision Tree

# model = DecisionTreeClassifier(random_state = 0, max_depth=20)

# clf_DTree, accuracy_DTree = CVKFold(5, train_X, train_Y, model)

# # Random Forest

# model = RandomForestClassifier(random_state = 0, max_depth=20)

# clf_RF, accuracy_RF = CVKFold(5, train_X, train_Y, model)
# accuracies = {accuracy_Log: clf_Log, accuracy_DTree: clf_DTree, accuracy_RF: clf_RF}

# clf = accuracies[min([accuracy_Log, accuracy_DTree, accuracy_RF])]
sample = pd.read_csv('/kaggle/input/bu-cs506-spring-2020-midterm/sample.csv')

predict_df = pd.DataFrame(sample)



predict_df['Score'] = model.predict(test_X)

predict_df.to_csv(r'sample.csv',index=False)