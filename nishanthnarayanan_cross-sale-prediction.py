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
from catboost import CatBoostClassifier
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
%matplotlib inline
train = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
test.head()
sub = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/sample_submission.csv')
sub.head()
train.info()
train.isna().sum()
train.describe()
train.dtypes
print(train.shape)
print(test.shape)
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
from sklearn.model_selection import StratifiedKFold,KFold
# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 7, shuffle = True)
skf = StratifiedKFold(n_splits = K, random_state = 7, shuffle = True)
MAX_ROUNDS = 1000
OPTIMIZE_ROUNDS = False
#LEARNING_RATE = 0.1
train.columns
X = train.drop(columns=['id','Response'],axis=1)
y = train['Response']
X_test = test.drop(columns='id',axis=1)
y_valid_pred = 0*y
y_test_pred = 0
accuracy = 0
result={}
#specifying categorical variables indexes
cat_columns = ['Gender','Vehicle_Age','Vehicle_Damage', 'Driving_License', 'Previously_Insured']
#fitting catboost classifier model
j=1
model = CatBoostClassifier(n_estimators=MAX_ROUNDS,verbose=False)
for i, (train_index, test_index) in enumerate(kf.split(train)):

#for train_index, test_index in skf.split(X, y):  
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", j)
    #print( "\nFold ", i)
    
    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        fit_model = model.fit( X_train, y_train, 
                               eval_set=[X_valid, y_valid],cat_features=cat_columns,
                               use_best_model=True
                             )
        print( "  N trees = ", model.tree_count_ )
    else:
        fit_model = model.fit( X_train, y_train,cat_features=cat_columns )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict(X_valid)
    y_valid_pred.iloc[test_index] = pred.reshape(-1)
    print(accuracy_score(y_valid,pred))
    accuracy+=accuracy_score(y_valid,pred)
    # Accumulate test set predictions
    y_test_pred += fit_model.predict(X_test)
    result[j]=fit_model.predict(X_test)
    j+=1
results = y_test_pred / K  # Average test set predictions
print(accuracy/5)
d = pd.DataFrame()
for i in range(1, 6):
    d = pd.concat([d,pd.DataFrame(result[i])],axis=1)
d.columns=['1','2','3','4','5']
re = d.mode(axis=1)[0]
sub.Response = re
sub.to_csv('submission.csv',index = False)
sub