# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv("/kaggle/input/tweet_data.csv",encoding='ISO-8859-1')
df
col=["profileimage","tweet_location","user_timezone","sidebar_color","tweet_coord","link_color","fav_number","tweet_id","_last_judgment_at","created","tweet_created"]
df.drop(col,axis=1,inplace=True)
df
## Sanity Check ##
df.drop_duplicates(inplace=True)
df.isnull().sum()
c=["gender","gender:confidence"]
df.dropna(subset=c,how="any",inplace=True)
df
df.isnull().sum()
df["text"].fillna("", inplace=True)
df["description"].fillna("",inplace=True)
o=list(df["text"])
import re

l=[]
k=[]
for s in df["text"] :
    a=re.sub(r"http://t.co/[a-zA-Z0-9]*"," ",str(s))
    b=re.sub(r"https://t.co/[a-zA-Z0-9]*"," ",str(s))
    
    l.append(a)
    k.append(b)
    
df.replace(inplace=True, to_replace=o, value=l)
o=list(df["text"])
df.replace(inplace=True, to_replace=o, value=k)
    
df["text"].replace(regex=True, inplace=True, to_replace=r'[,!.; -@!%^&*)(]', value=' ')
    
o=list(df["description"])
import re

l=[]
k=[]
for s in df["description"] :
    s=re.sub(r"http://t.co/[a-zA-Z0-9]*"," ",str(s))
    s=re.sub(r"https://t.co/[a-zA-Z0-9]*"," ",str(s))
    
    l.append(s)
    k.append(s)
    
df.replace(inplace=True, to_replace=o, value=l)
df.replace(inplace=True, to_replace=o, value=k)

df["description"].replace(regex=True, inplace=True, to_replace=r'[,!.; -@!%^&*)(]', value=' ')
df.head(10)
df.shape
df.describe()
df.info()
df.columns
df.gender.nunique()
df.gender.unique()
df._golden.unique()
df._unit_state.unique()
num_col=df.select_dtypes(include=np.number).columns
print("Numerical Columns :\n",num_col)
cat_col=df.select_dtypes(exclude=np.number).columns
print("Categorical Columns :\n",cat_col)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])#Converts brand=0,female=1,male=2,unknown=3
df['_golden']=le.fit_transform(df['_golden'])#Converts true as 1 and false as 0
df['_unit_state']=le.fit_transform(df['_unit_state'])#converts finalized as 0 and golden as 1
df.gender.unique()
df._golden.unique()
df._unit_state.unique()
df.head(10)
df.tail()
df.gender.plot(kind='hist')

fig=plt.figure(figsize=(10,5))
plt.bar(df.gender,df.tweet_count,color='maroon',width=0.4)
plt.xlabel("Gender")
plt.ylabel("Tweet count")
plt.title("Tweet count based on gender")
plt.show()
sns.heatmap(df.corr(),annot=True,fmt='.1g',cbar=False)
matrix=np.triu(df.corr())
sns.heatmap(df.corr(),annot=True,mask=matrix)
from sklearn.model_selection import train_test_split
df.info()
def normalize_text(s):
    # just in case
    s = str(s)
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s
df['text_norm'] = [normalize_text(s) for s in df['text']]
df['description_norm'] = [normalize_text(s) for s in df['description']]
df['all_features'] = df['text_norm'].str.cat(df['description_norm'], sep=' ')
df_confident = df[df['gender:confidence']==1] ## Choosing only the one's with confidence
df_confident.shape #Now we have approx 14000 entries.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df_confident['text_norm'])
encoder = LabelEncoder()
y = encoder.fit_transform(df_confident['gender'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier
#nb = CatBoostClassifier(silent = True)

#### Multi-Nomial Naive Bayes was found to give considerably good peroformance with this data ####

nb = MultinomialNB(alpha = 0.6,fit_prior = True)
nb.fit(x_train, y_train)

print(nb.score(x_test, y_test))
### Just to illustrate how the text data looks like ###
df_just_text = pd.DataFrame(x)
df_just_text
X=df[['_unit_id','_golden','_unit_state','_trusted_judgments','gender:confidence','profile_yn:confidence','retweet_count','tweet_count']]
X.info()
df.corr()
Y=df[['gender']]
df_conf = df[df['gender:confidence']==1]
df_conf.shape
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.2)
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process,model_selection
import xgboost
from xgboost import XGBClassifier
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    #gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    #svm.SVC(probability=True),
    #svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost
    XGBClassifier()    
    ]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = Y['gender']
row_index = 0
X1 = X.copy()
for alg in MLA:
    #print(row_index)
    X = X1
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    print('Examining ',MLA_name)
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    cv_results = model_selection.cross_validate(alg, X, Y, cv  = cv_split)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    alg.fit(X, Y)
    MLA_predict[MLA_name] = alg.predict(X)
    row_index+=1
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#### Taking 4 Ensembles ###
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process,model_selection
import xgboost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    XGBClassifier(),
    CatBoostClassifier(verbose = False)    ## Just to see how it does! ##
    ]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = Y['gender']
row_index = 0
X1 = X.copy()
for alg in MLA:
    X = X1
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    print(MLA_name)
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    cv_results = model_selection.cross_validate(alg, X, Y, cv  = cv_split)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    alg.fit(X, Y)
    MLA_predict[MLA_name] = alg.predict(X)
    row_index+=1
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
#a=XGBClassifier(num_rounds = 150,min_split_leaf = 10,max_depth = 3,random_state=100)
#a=GradientBoostingClassifier(num_rounds = 150,min_split_leaf = 10,max_depth = 3,random_state=100)
#a=CatBoostClassifier(num_rounds = 150,min_split_leaf = 10,max_depth = 3,random_state=100)
#a=AdaBoostClassifier(num_rounds = 150,min_split_leaf = 10,max_depth = 3,random_state=100)

a = MultinomialNB()
a.fit(X_train,Y_train)
y_pred=a.predict(X_test)
score=accuracy_score(Y_test,y_pred)
score*100
import tensorflow as tf
model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=8,input_dim=X_train.shape[1],activation='relu'),
     tf.keras.layers.LeakyReLU(0.3),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])
model.compile(loss = 'mean_squared_error',optimizer = 'adam',metrics = ['accuracy'])
model.fit(X_train,Y_train,epochs=5)
y_pred=model.predict(X_test)
score=accuracy_score(Y_test,y_pred)
score*100