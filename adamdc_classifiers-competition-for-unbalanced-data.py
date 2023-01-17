# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import pandas as pa

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df



import matplotlib.pyplot as plt

import itertools

def plot_confusion_matrix(cm, classes,normalize=True,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    



train = import_data("../input/train.csv")

test = import_data("../input/test.csv")

train.head()



A = train.loc[train['event'] == 'A']

d_A = len(A)

B = train.loc[train['event'] == 'B']

d_B = len(B)

C = train.loc[train['event'] == 'C']

d_C = len(C)

D = train.loc[train['event'] == 'D']

d_D = len(D)

qunt = str(d_A + d_B + d_C + d_D)

labels = ['A', 'B', 'C', 'D']

sizes = [d_A, d_B, d_C, d_D]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0, 0, 0)  

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Pie (' + qunt + ')')



plt.axis('equal')

plt.show()
train.describe()


test_id = test['id']

test.drop(['id'], axis=1, inplace=True)



import lightgbm as lgb

dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

dic1 = {'CA':0,'DA':1,'SS':3,'LOFT':4}

train["event"] = train["event"].apply(lambda x: dic[x])

train["event"] = train["event"].astype('int8')

train['experiment'] = train['experiment'].apply(lambda x: dic1[x])

test['experiment'] = test['experiment'].apply(lambda x: dic1[x])



train['experiment'] = train['experiment'].astype('int8')

test['experiment'] = test['experiment'].astype('int8')



#train.info()

y = train['event']

train.drop(['event'], axis=1, inplace=True)

from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier()]
scoring = ['recall_macro']

log_cols = ["Classifier", "macro average Recall", "Log Loss"]

log = pa.DataFrame(columns=log_cols)



for clf in classifiers:

    recall = cross_validate(clf, train, y, scoring=scoring,

                              cv=5, return_train_score=False)

    name = clf.__class__.__name__



    print("=" * 30)

    print(name)

    print('****Results****')

    Recall = np.mean(recall['test_recall_macro'])

    print("Macro Recall: {:.4%}".format(Recall))



    ll = 1 - Recall



    print("Log Loss: {}".format(ll))



    log_entry = pa.DataFrame([[name, Recall * 100, ll]], columns=log_cols)

    log = log.append(log_entry)





print("=" * 30)



sns.set_color_codes("muted")

sns.barplot(x='macro average Recall', y='Classifier', data=log, color="b")



plt.xlabel('Macro Average Recall %')

plt.xticks(np.arange(0, 100, step=10))

plt.title('recall scores -sensitivity')

plt.show()





sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

# plt.xticks(np.arange(0, 10, step=0.5))

plt.title('Classifiers Log Loss using max-min rule')

plt.show()


