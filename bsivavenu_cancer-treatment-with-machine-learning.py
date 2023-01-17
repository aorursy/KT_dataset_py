# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train_text_df.shape,test_text_df.shape
train_variants_df.shape,test_variants_df.shape
train_variants_df.head(3)
train_text_df.head(3)
gene_group = train_variants_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
print("Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])
print("\nGenes with minimal occurences\n", minimal_occ_genes)
test_variants_df.head(3)
test_text_df.head(3)
train_text_df.Text[0]
train_variants_df.Class.unique()
plt.figure(figsize=(15,5))
sns.countplot(train_variants_df.Class,data = train_variants_df)
print(len(train_variants_df.Gene.unique()))
train_df = pd.merge(train_text_df,train_variants_df,on = 'ID')
print(train_df.shape)
train_df.head(3)
test_df = pd.merge(test_text_df,test_variants_df,on = 'ID')
print(test_df.shape)
test_df.head(3)

submission_file = pd.read_csv("../input/submissionFile")
submission_file.head()
train_df.isnull().sum()
train_df.dropna(inplace=True)
from sklearn.model_selection import train_test_split

train ,test = train_test_split(train_df,test_size=0.2) 
np.random.seed(0)
train.head()
X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values
train.isnull().sum()
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
svc = svm.LinearSVC()
rfc = RandomForestClassifier()
etrc = ExtraTreesClassifier()
xgbc = xgb.XGBClassifier()
lgbc = lgb.LGBMClassifier()
clf = [svc,rfc]
# ,etrc,xgbc,lgbc

for i in clf:
        text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True,stop_words='english',encoding='utf-8')),('tfidf', TfidfTransformer()),('clf', i)])
        text_clf = text_clf.fit(X_train,y_train)
        y_test_predicted = text_clf.predict(X_test)
        acc = np.mean(y_test_predicted == y_test)
        print('accuracy of :',str(i),'is: ',acc )