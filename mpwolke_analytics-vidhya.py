#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import plot_roc_curve

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import precision_recall_curve
df = pd.read_csv('../input/analytics-vidya-customer-segmentation/Train.csv')

df.head()
sns.heatmap(df.isnull(),cmap = 'magma',cbar = False)
f = df[df['Family_Size'] == 1]

nf = df[df['Family_Size'] == 0]
f.describe()
nf.describe()
family = len(df[df['Family_Size'] == 1])/len(df)*100

no_family = len(df[df['Family_Size'] == 0])/len(df)*100

family_percentage = [family,no_family]
fig,ax = plt.subplots(nrows = 1,ncols = 3,figsize = (20,5))

plt.subplot(1,3,1)

plt.pie(family_percentage,labels = ['Family','No Family'],autopct='%1.1f%%',startangle = 90,)

plt.title('Family PERCENTAGE')



plt.subplot(1,3,2)

sns.countplot('Family_Size',data = df,)

plt.title('DISTRIBUTION OF Family Size')



plt.subplot(1,3,3)

sns.scatterplot('ID','Age',data = df,hue = 'Family_Size')

plt.title('ID vs Age w.r.t (with respect to) Family')

plt.show()
sns.heatmap(df.corr(),cmap = 'RdBu',cbar = True)
corr = df.corrwith(df['Family_Size']).sort_values(ascending = False).to_frame()

corr.columns = ['Correlations']

plt.subplots(figsize = (5,25))

sns.heatmap(corr,annot = True,cmap = 'RdBu',linewidths = 0.4,linecolor = 'black')

plt.title('CORRELATION w.r.t Family Size')
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
df[categorical_nan].isna().sum()
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
df = df[['Family_Size','ID','Work_Experience','Age']]

df.head()
import imblearn

from collections import Counter

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline



imblearn.__version__
def model(classifier):

    

    classifier.fit(x_train,y_train)

    prediction = classifier.predict(x_test)

    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)

    print("CROSS VALIDATION SCORE : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))

    print("ROC_AUC SCORE : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))

    plot_roc_curve(classifier, x_test,y_test)

    plt.title('ROC_AUC_PLOT')

    plt.show()
def model_evaluation(classifier):

    

    # CONFUSION MATRIX

    cm = confusion_matrix(y_test,classifier.predict(x_test))

    names = ['True Neg','False Pos','False Neg','True Pos']

    counts = [value for value in cm.flatten()]

    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')

    

    # CLASSIFICATION REPORT

    print(classification_report(y_test,classifier.predict(x_test)))
over = SMOTE(sampling_strategy= 0.5)

under = RandomUnderSampler(sampling_strategy = 0.1)

features = df.iloc[:,:3].values

target = df.iloc[:,3].values
steps = [('under', under),('over', over)]

pipeline = Pipeline(steps=steps)

features, target = pipeline.fit_resample(features, target)

Counter(target)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2') 
model(classifier_lr)
#model_evaluation(classifier_lr)
from sklearn.svm import SVC
classifier_svc = SVC(kernel = 'linear',C = 0.1)
#model(classifier_svc)
#model_evaluation(classifier_svc)
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
model(classifier_dt)
model_evaluation(classifier_dt)
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
model(classifier_rf)
model_evaluation(classifier_rf)
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)
model(classifier_knn)
model_evaluation(classifier_knn)