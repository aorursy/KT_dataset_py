import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline 



dataset = pd.read_csv("../input/irisdataset.csv", na_values=['NaN'])
import seaborn as sns
sns.set(style = 'ticks', color_codes= True)

sns.pairplot(dataset.dropna(), hue='class', diag_kind='hist')
dataset.loc[dataset['class'] == 'versicolor', 'class'] = 'Iris-versicolor'

dataset.loc[dataset['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'
sns.pairplot(dataset.dropna(), hue='class', diag_kind = 'hist')
dataset[dataset.isnull().any(axis=1)]
#Mean Imputation

average_petal_width = dataset.loc[dataset['class'] == 'Iris-setosa', 'petal_width_cm'].mean()

dataset.loc[(dataset['class'] == 'Iris-setosa') & (dataset['petal_width_cm'].isnull()), 'petal_width_cm'] = average_petal_width

dataset.loc[(dataset['sepal_length_cm'].isnull()) | 

        (dataset['sepal_width_cm'].isnull()) | 

        (dataset['petal_length_cm'].isnull()) | 

        (dataset['petal_width_cm'].isnull())]
sns.pairplot(dataset, hue = 'class', diag_kind = 'hist')
dataset.to_csv('icd.csv', index=False)

icd = pd.read_csv('icd.csv')

icd.head()
plt.figure(figsize = (10,10))

for index, column in enumerate(icd.columns) :

    if column == 'class' :

        continue;

    plt.subplot(2,2, index + 1)

    sns.violinplot(x = 'class', y = column, data = icd)

    
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_score
features = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']

X = icd[features]

y = icd['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
knn = KNeighborsClassifier()
scores = cross_val_score(knn,X, y, cv = 10, scoring = 'accuracy')

scores.mean()
scores.std()
from sklearn.model_selection import GridSearchCV
parameters = dict(n_neighbors = range(1,31))

grid = GridSearchCV(knn, param_grid = parameters, cv = 10)

grid.fit(X_train, y_train)
grid.best_estimator_
grid.best_score_
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(icd, title = "Download CSV file", filename = "irisdata.csv"):  

    csv = icd.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

#df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

icd = pd.read_csv('icd.csv')



# create a link to download the dataframe

create_download_link(icd)



#dataset.to_csv('icd.csv', index=False)
