#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQV31IEomxDoLSTmbZVk8EJxSleDXZBEzn41To-p-LKAH3AYyjQ&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/BCG_COVID-19_clinical_trials-2020_06_06.csv', encoding='ISO-8859-2')

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSOtT7xti97F30Ps5veITwct1jaG7IBL8D-NF8kxkfw8NItX5Yf&usqp=CAU',width=400,height=400)
na_percent = (df.isnull().sum()/len(df))[(df.isnull().sum()/len(df))>0].sort_values(ascending=False)



missing_data = pd.DataFrame({'Missing Percentage':na_percent*100})

missing_data
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]

print(numerical_cols)
print(categorical_cols)
for col in ('ďťżNo', 'Phase', 'Target Sample Size', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38'):

    df[col] = df[col].fillna(0)
for col in ['Resion' ,'Status', 'participants segment', 'Unnamed: 14', 'Unnamed: 15']:

    df[col] = df[col].fillna('None')
#test train split



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df["Status"], df['Target Sample Size'], test_size=0.33)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)

X1 = vectorizer.fit_transform(X_train)

X_test1= vectorizer.transform(X_test)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

clf2 = LogisticRegression(C=0.1, solver='sag')

scores = cross_val_score(clf2, X1,y_train, cv=5,scoring='f1_weighted')
y_p1 = clf2.fit(X1, y_train).predict(X_test1)
from sklearn.metrics import accuracy_score



# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_test, y_p1)

print('Accuracy: %f' % accuracy)
from lime import lime_text

from sklearn.pipeline import make_pipeline

c = make_pipeline(vectorizer, clf2)
df["Status"][0]
print(c.predict_proba([df["Status"][0]]))
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer('Status')
X_test = X_test.tolist()
X_test[0]
type(y_test)
y_test = y_test.tolist()
type(y_test)
#idx = 0

#exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)

print('Document id: %d' % idx)

print('Status', c.predict_proba([X_test[idx]])[0,1])

print('True class: %s' % Status[y_test[idx]])
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
#Codes from Mario Filho https://www.kaggle.com/mariofilho/live26-https-youtu-be-zseefujo0zq

from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['Status']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Target Sample Size']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df['Status'].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('BCG Clinical Trials', size=18)

ax.set_ylabel('Status', size=10)

ax.set_xlabel('Country', size=10)
Status=df.sort_values('Status', ascending=False)

top10=Status.head(10)

f=['Country','Link to Source of information']

displ=(top10[f])

displ.set_index('Country', inplace=True)
#Code from Niharika Pandit https://www.kaggle.com/niharika41298/netflix-vs-books-recommender-analysis-eda

from IPython.display import Image, HTML



def path_to_image_html(path):

    '''

     This function essentially convert the image url to 

     '<img src="'+ path + '"/>' format. And one can put any

     formatting adjustments to control the height, aspect ratio, size etc.

     within as in the below example. 

    '''



    return '<img src="'+ path + '""/>'



HTML(displ.to_html(escape=False ,formatters=dict(small_image_url=path_to_image_html),justify='center'))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTWWI6TnpSc6AzOZQFTkuB7Rgxcry7IysGiQOnFQqDkZIL5EiiT&usqp=CAU',width=400,height=400)