import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.head()
print(df.isna().sum()*100/df.shape[0])

to_drop = ['job_id','title','department','salary_range','company_profile','requirements','benefits']

df = df.drop(to_drop, axis = 1).sort_index() #drop columns that aren't needed

df = df.dropna(subset = ['description', 'location'])
df.shape
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

stemmer = PorterStemmer() 
stop_words = stopwords.words('english')

def preprocess(text):
    text = re.sub('[^a-zA-Z\s]', '', text) #tokenizatoin
    text = text.lower() #to lower case
    split = text.split() #getting rid of stop words and Porter2 stemming
    for word in split :
      if word in stop_words :
        word = ''
      else :
        stemmer.stem(word)
    return ' '.join([word for word in split])

df['description'] = df['description'].apply(preprocess)

df['description'].head()

#define the variables
x = df['description']
y = df['fraudulent']

#split it into training and test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

#encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

#vectorization
MAX = 2000
vectorizer = TfidfVectorizer(max_features = MAX)
vectorizer.fit(x_train)

x_trainvec = vectorizer.transform(x_train)
x_testvec = vectorizer.transform(x_test)


from sklearn.linear_model import LogisticRegression

# fit 
logreg = LogisticRegression()
logreg.fit(x_trainvec, y_train)

# predict
y_pred_lr = logreg.predict(x_testvec)

# accuracy
print("Accuracy Score of LogReg :", accuracy_score(y_pred_lr, y_test), "\n") #96.3%

# confusion matrix 
print("Confusion Matrix of LogReg:\n", confusion_matrix(y_test, y_pred_lr), "\n") # [[3324, 0], [128, 55]]

#classifcation report
print("Classification Report of LogReg:\n", classification_report(y_test, y_pred_lr), "\n")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
""" I COMMENTED THIS OUT SINCE IT TAKES FOREVER, BUT YOU CAN TRY IT ON YOUR OWN IF YOU WANT. K=1 gave me the best score


knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid={'n_neighbors':range(1,40)}, scoring='accuracy')
grid.fit(x_trainvec, y_train)

grid.best_params_

for i in range(0, len(grid.cv_results_['mean_test_score'])):
    print('N_Neighbors {}: {} '.format(i+1, grid.cv_results_['mean_test_score'][i]*100))
"""
knn1 = knn = KNeighborsClassifier(n_neighbors = 1)
knn1.fit(x_trainvec, y_train)

y_pred_knn = knn1.predict(x_testvec)

# accuracy
print("Accuracy Score of KNN :", accuracy_score(y_pred_knn, y_test), "\n") #97.9%

# confusion matrix 
print("Confusion Matrix of KNN:\n", confusion_matrix(y_test, y_pred_knn), "\n") # [[3314, 10], [61, 122]]

# classifcation report
print("Classification Report of KNN:\n", classification_report(y_test, y_pred_knn), "\n")

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')
svc.fit(x_trainvec, y_train)

y_pred_svc = svc.predict(x_testvec)

# accuracy
print("Accuracy Score of SVC :", accuracy_score(y_pred_svc, y_test), "\n") #97.3%

# confusion matrix 
print("Confusion Matrix of SVC:\n", confusion_matrix(y_test, y_pred_svc), "\n") # [[3324, 0], [96, 87]]

# classifcation report
print("Classification Report of SVC:\n", classification_report(y_test, y_pred_svc), "\n")
print(1 - y_test.mean()) #null accuracy
from sklearn.metrics import roc_curve, roc_auc_score

results_table = pd.DataFrame(columns = ['models', 'fpr','tpr','auc'])

predictions = {'LR': y_pred_lr, 'KNN': y_pred_knn, 'SVC': y_pred_svc}

for key in predictions:
    fpr, tpr, _ = roc_curve(y_test, predictions[key])
    auc = roc_auc_score(y_test, predictions[key])
    
    results_table = results_table.append({'models': key,
                                         'fpr' : fpr,
                                         'tpr' : tpr,
                                         'auc' : auc}, ignore_index=True)
    
results_table.set_index('models', inplace=True)

print(results_table)

fig = plt.figure(figsize = (8,6))

for i in results_table.index:
    plt.plot(results_table.loc[i]['fpr'], 
             results_table.loc[i]['tpr'], 
             label = "{}, AUC={:.3f}".format(i, results_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color = 'black', linestyle = '--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop = {'size':13}, loc = 'lower right')

plt.show()