import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
# Reading the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Combine boths dataframes
train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
both['subject'] = '#' + both['subject'].astype(str)
train.shape, test.shape
both.head()
both.dtypes.value_counts()
def basic_details(df):
    b = pd.DataFrame()
    b['Missing value'] = df.isnull().sum()
    b['N unique value'] = df.nunique()
    b['dtype'] = df.dtypes
    return b
basic_details(both)
activity = both['Activity']
label_counts = activity.value_counts()

plt.figure(figsize= (12, 8))
plt.bar(label_counts.index, label_counts)
Data = both['Data']
Subject = both['subject']
train = both.copy()
train = train.drop(['Data','subject','Activity'], axis =1)
# Standard Scaler
from sklearn.preprocessing import StandardScaler
slc = StandardScaler()
train = slc.fit_transform(train)

# dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9, random_state=0)
train = pca.fit_transform(train)
# Spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, activity, test_size = 0.2, random_state = 0)
# Test options and evaluation metric
num_folds = 10
seed = 0
scoring = 'accuracy'
results = {}
accuracy = {}
results
'''
{'LR': (0.9472035767792472, 0.007379785047104939),
 'LDA': (0.9191654673288584, 0.008354012458070223),
 'NB': (0.8238872701105358, 0.008640104786184547),
 'KNN': (0.9474461477662824, 0.006181737727775004),
 'CART': (0.8306822069388573, 0.007987686105087334),
 'SVM': (0.919043518267291, 0.007262868527188091),
 'AB': (0.41145392183463303, 0.05090488352549587),
 'GBM': (0.937614281164105, 0.007441670620774797),
 'ET': (0.8782626903703005, 0.007390850626754222),
 'RF': (0.8874886456133728, 0.014447203065049064),
 'XGB': (0.9265694121671839, 0.007260201028131366)}
'''
accuracy
'''
{'LR': 0.9529126213592233,
 'LDA': 0.920873786407767,
 'NB': 0.8233009708737864,
 'KNN': 0.9529126213592233,
 'CART': 0.8257281553398058,
 'SVM': 0.9247572815533981,
 'AB': 0.46067961165048543,
 'GBM': 0.9296116504854369,
 'ET': 0.8660194174757282,
 'RF': 0.8932038834951457,
 'XGB': 0.9199029126213593,
 'GScv': 0.9529126213592233}
'''
# Finalizing the model and comparing the test, predict results
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
model = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 8, p= 1, weights= 'distance')

_ = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
results["GScv"] = (_.mean(), _.std())

model.fit(X_train, y_train) 
y_predict = model.predict(X_test)

accuracy["GScv"] = accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))

cm= confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)