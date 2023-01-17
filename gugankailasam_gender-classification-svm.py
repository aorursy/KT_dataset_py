# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)



plt.style.use("seaborn-dark")

np.random.seed(42)
data = pd.read_csv('../input/voicegender/voice.csv')

data.head()
data.shape
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')



labels = data['label'].unique()

target = data['label'].value_counts()

ax.pie(target, labels = labels,autopct='%1.2f%%')

plt.show()
data.columns
def draw_corr(features):

    #Init

    corr_matrix = data[features].corr()

    tol = 0.7



    # correlation diagram creation

    def corr_tol(x):

        return x.apply(lambda x : True if (x >= tol or x <= -tol) else False)

    bool_corr_matrix = corr_matrix.apply(lambda x : corr_tol(x))



    for i in range(0,len(bool_corr_matrix)):

        bool_corr_matrix.iloc[i,i] = False



    bool_corr_matrix = pd.DataFrame(np.tril(bool_corr_matrix, k=0), 

                                    columns=bool_corr_matrix.columns, index=bool_corr_matrix.index)



    plt.figure(figsize=(10,5))

    plt.grid(True)

    sns_plot = sns.heatmap(bool_corr_matrix, square=True, cmap=sns.cubehelix_palette(8), linewidths=0.1)

    sns_plot.set_ylim(len(bool_corr_matrix)-1, -1)

    plt.show()
draw_corr(data.columns)
# Automatic field selection



corr_matrix = data.corr()



columns = np.full((corr_matrix.shape[0],), True, dtype=bool)

for i in range(corr_matrix.shape[0]):

    for j in range(i+1, corr_matrix.shape[0]):

        if corr_matrix.iloc[i,j] >= 0.8:

            if columns[j]:

                columns[j] = False



selected_columns = data.drop('label', axis=1).columns[columns]

selected_columns
features = ['meanfreq', 'skew', 'sp.ent', 'mode', 'meanfun', 'minfun',

       'maxfun', 'meandom', 'mindom', 'modindx']
draw_corr(features)
X = data.loc[:, data.columns!='label']

y = data['label']
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(y)

y = le.transform(y)

print(le.classes_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y,

                                                   stratify = y,

                                                   test_size = 0.20)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)

lda_trans_df = lda.fit_transform(X_train, y_train)

X_train = pd.DataFrame(lda_trans_df)
pd.DataFrame(X_train).plot.kde()

plt.legend("")
from sklearn.svm import SVC

clf = SVC()



#Init

model_for_cv = clf



from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='accuracy')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()



model_for_cv = clf



from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='accuracy')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



model_for_cv = model



from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='accuracy')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.svm import SVC

clf = SVC(probability=True)

clf.fit(X_train, y_train)
X_test = lda.transform(scaler.transform(X_test))

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



cm = confusion_matrix(y_test, y_pred)

print(cm)

print("----Classification Report----")

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])



plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

# plt.savefig('Log_ROC')

plt.show()