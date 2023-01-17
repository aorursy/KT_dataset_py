import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from collections import Counter
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, log_loss
import itertools
data_train_gen = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
data_train_gen.shape
data_train_gen.head()
NUMBER_OF_1_LABELS = data_train_gen.target.value_counts()[1]
NUMBER_OF_1_LABELS
count_Class=pd.value_counts(data_train_gen.target, sort= True)
count_Class.plot(kind= 'bar', color= ["blue"])
plt.title('Bar chart')
plt.show()
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
df1 = data_train_gen[data_train_gen.target == 1]
df2 = data_train_gen[data_train_gen.target == 0][:NUMBER_OF_1_LABELS]
data_train = pd.concat([df1, df2])
count_Class=pd.value_counts(data_train.target, sort= True)
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
from sklearn.utils import shuffle
data_train = shuffle(data_train)
data_train = data_train[:10000]
words = []
for sen in [s for s in data_train.question_text]:
    for word in sen.split():
        if word not in words:
            words.append(word)

len(words), len(pd.Series(words).unique())
count1 = Counter(" ".join(data_train[data_train['target']==0].question_text).split()).most_common(50)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words_in_excellent ", 1 : "count_1"})
df1.T
quiestion_words = ['what','when','why','which','who','how', 'whose', 'whome']
stop_signs = ['.',',',':','...','\'', '\"']
stop_words = set(stopwords.words('english'))
stop_words = [w for w in stop_words]
stop_words = [w for w in stop_words if w not in quiestion_words]   
cleaned_questions_train = []
 
for sentence in data_train['question_text']:
    new_sentence = [w.lower() for w in word_tokenize(sentence) if not w in stop_words]
    new_sentence = [w for w in new_sentence if w not in stop_signs]
         
    clean = ' '.join(new_sentence)    
   
    cleaned_questions_train.append(clean)

cleaned_questions_test = []
for sentence in data_test['question_text']:
    new_sentence = [w.lower() for w in word_tokenize(sentence) if not w in stop_words]
    new_sentence = [w for w in new_sentence if w not in stop_signs]
         
    clean = ' '.join(new_sentence)    
   
    cleaned_questions_test.append(clean)
data_train.insert(loc=0, column="debugged_questions", value=cleaned_questions_train)
data_test.insert(loc=0, column="debugged_questions", value=cleaned_questions_test)
data_test.head()
words = []
pd.concat([data_train.debugged_questions, data_test.debugged_questions])
for sen in [s for s in data_train.question_text]:
    for word in sen.split():
        if word not in words:
            words.append(word)
y = data_train.target
vectorizer = TfidfVectorizer("english", vocabulary = words)
X = vectorizer.fit_transform(data_train['debugged_questions'])
X_val = vectorizer.fit_transform(data_test['debugged_questions'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, max_depth=8, max_features='sqrt', subsample=0.8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('F-Score: {}'.format(f1_score(y_test, predictions, average='macro')))


grid = OrderedDict((
        ('max_depth',np.arange(2,20,4)),
        ('min_samples_leaf',np.exp(np.linspace(3,8,5)).astype(int))
        

       ))

result = {'params':[],'roc_auc_train':[],'roc_auc_valid':[], 'f1_train':[], 'f1_valid':[]}

for param_values in itertools.product(*grid.values()):
    param = dict(zip(grid.keys(),param_values))
    clf = DecisionTreeClassifier(**param)

    clf.fit(X_train,y_train)

    train_pred = clf.predict(X_train)
    valid_pred = clf.predict(X_test)

    roc_auc_train = roc_auc_score(y_train,train_pred)
    roc_auc_valid = roc_auc_score(y_test,valid_pred)
    f1_train = f1_score(y_train,train_pred)
    f1_test = f1_score(y_test,valid_pred)
    result['params'].append(param)
    result['roc_auc_train'].append(roc_auc_train)
    result['roc_auc_valid'].append(roc_auc_valid)
    result['f1_train'].append(f1_train)
    result['f1_valid'].append(f1_test)

# Выводим результаты    
(pd.DataFrame(result)   
   .style
   .background_gradient('Wistia')
)

{'max_depth': 10, 'max_features': 28, 'min_samples_leaf': 20}
f_tree = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 20)
f_tree.fit(X_train,y_train)
prediction = f_tree.predict(X_test)
f1 = f1_score(y_test, prediction)
f1
from sklearn.model_selection import GridSearchCV
first_tree = DecisionTreeClassifier()
tree_params_grid = {'max_depth' : np.arange(2,18,4),'min_samples_leaf': [20,70,244,854]}
tree_grid=GridSearchCV(first_tree,tree_params_grid,scoring='f1'
                        ,cv=5,n_jobs = -1)

tree_grid.fit(X_train,y_train)
tree_grid.best_score_
tree_grid.best_params_
{'max_depth': 10, 'max_features': 28, 'min_samples_leaf': 20}
f_tree = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 20)
f_tree.fit(X_train,y_train)
prediction = f_tree.predict(X_test)
f1 = f1_score(y_test, prediction)
f1
sub_df = pd.DataFrame({'qid':data_test.qid.values})
sub_df['prediction'] = f_tree.predict(X_val)
sub_df.to_csv('submission.csv', index=False)
