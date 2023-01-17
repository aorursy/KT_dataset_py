# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import learning_curve

from sklearn.metrics import r2_score, make_scorer



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/ai4all-project/figures/classifier/auc_table.csv', encoding='ISO-8859-2')

df.head()
print(f"data shape: {df.shape}")
df.describe()
df.isna().sum()
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(df)
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    items = []

    vals = []

    for col in data.columns:

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    tt['Most frequent item'] = items

    tt['Frequence'] = vals

    tt['Percent from total'] = np.round(vals / total * 100, 3)

    return(np.transpose(tt))
most_frequent_values(df)
targets = list(df.columns[0:])

targets
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
from sklearn.preprocessing import LabelEncoder

categorical_col = ('Model', 'All', 'COVID vs No virus', 'Covid vs Other virus')

        

        

for col in categorical_col:

    label = LabelEncoder() 

    label.fit(list(df[col].values)) 

    df[col] = label.transform(list(df[col].values))



print('Shape all_data: {}'.format(df.shape))
plt.figure(figsize=(14,8))

sns.barplot(data=df,x='Model',y='COVID vs No virus',color=sns.color_palette('Set3')[0])

plt.title('COVID vs No Virus Model')

plt.xlabel('Model')

plt.ylabel('COVID vs No virus')

plt.xticks(rotation=45)

for i in range(df.shape[0]):

    count = df.iloc[i]['COVID vs No virus']

    plt.text(i,count+1,df.iloc[i]['COVID vs No virus'],ha='center')

    

from IPython.display import display, Markdown

display(Markdown("Most Number of COVID vs No Virus **20-50**"))
plt.figure(figsize = (10,8))

sns.set(style = "darkgrid")

plt.title("Distribution of COVID vs No virus", fontdict = {'fontsize':20})

ax = sns.countplot(x = "COVID vs No virus", hue = 'Model', data = df)
(sns.FacetGrid(df, hue = 'COVID vs No virus',

             height = 6,

             xlim = (0,500))

    .map(sns.kdeplot, 'Model', shade = True)

    .add_legend());
plt.figure(figsize = (10,8))

sns.barplot(x = 'COVID vs No virus', y = 'Model', data = df);
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



from warnings import filterwarnings

filterwarnings('ignore')



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import learning_curve

from sklearn.metrics import r2_score, make_scorer

from sklearn.metrics import roc_auc_score
y = df["COVID vs No virus"]

X = df.drop(["COVID vs No virus"], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
cart = DecisionTreeClassifier(max_depth = 12)
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print('Decision Tree Model')



print('Accuracy Score: {}\n\nConfusion Matrix:\n {}\n\nAUC Score: {}'

      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred), roc_auc_score(y_test,y_pred)))
pd.DataFrame(data = cart_model.feature_importances_*100,

                   columns = ["Importances"],

                   index = X_train.columns).sort_values("Importances", ascending = False)[:20].plot(kind = "barh", color = "r")



plt.xlabel("Feature Importances (%)")
# We can use the functions to apply the models and roc curves to save space.

def model(algorithm, X_train, X_test, y_train, y_test):

    alg = algorithm

    alg_model = alg.fit(X_train, y_train)

    global y_prob, y_pred

    y_prob = alg.predict_proba(X_test)[:,1]

    y_pred = alg_model.predict(X_test)



    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'

      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))

    



def ROC(y_test, y_prob):

    

    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    

    plt.figure(figsize = (10,10))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1], linestyle = '--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')
print('Model: Logistic Regression\n')

model(LogisticRegression(solver = "liblinear"), X_train, X_test, y_train, y_test)
LogR = LogisticRegression(solver = "liblinear")

cv_scores = cross_val_score(LogR, X, y, cv = 3, scoring = 'accuracy')

print('Mean Score of CV: ', cv_scores.mean())
ROC(y_test, y_prob)
print('Model: Gaussian Naive Bayes\n')

model(GaussianNB(), X_train, X_test, y_train, y_test)