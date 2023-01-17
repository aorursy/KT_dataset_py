from tensorflow.python.client import device_lib

device_lib.list_local_devices()
import matplotlib.pyplot as plt

import pandas as pd

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori, association_rules



import warnings

warnings.filterwarnings("ignore")

import seaborn as sns
titanic = pd.read_csv('../input/train.csv')

nominal_cols = ['Embarked','Pclass','Age', 'Survived', 'Sex']

cat_cols = ['Embarked','Pclass','Age', 'Survived', 'Title']

titanic['Title'] = titanic.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

titanic['Title'].fillna('Title_UK', inplace=True)

titanic['Embarked'].fillna('Unknown',inplace=True)

titanic['Age'].fillna(0, inplace=True)

# Replacing Binary with String

rep = {0: "Dead", 1: "Survived"}

titanic.replace({'Survived' : rep}, inplace=True)
def binning(col, cut_points, labels=None):

  minval = col.min()

  maxval = col.max()

  break_points = [minval] + cut_points + [maxval]

  if not labels:

    labels = range(len(cut_points)+1)

  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)

  return colBin



cut_points = [1, 10, 20, 50 ]

labels = ["Unknown", "Child", "Teen", "Adult", "Old"]

titanic['Age'] = binning(titanic['Age'], cut_points, labels)

in_titanic = titanic[nominal_cols]

cat_titanic = titanic[cat_cols]
in_titanic.head()
cat_titanic.head()
for x in ['Embarked', 'Pclass','Age', 'Sex', 'Title']:

    sns.set(style="whitegrid")

    ax = sns.countplot(y=x, hue="Survived", data=titanic)

    plt.ylabel(x)

    plt.title('Survival Plot')

    plt.show()
dataset = []

for i in range(0, in_titanic.shape[0]-1):

    dataset.append([str(in_titanic.values[i,j]) for j in range(0, in_titanic.shape[1])])

# dataset = in_titanic.to_xarray()



oht = TransactionEncoder()

oht_ary = oht.fit(dataset).transform(dataset)

df = pd.DataFrame(oht_ary, columns=oht.columns_)

df.head()
oht.columns_
output = apriori(df, min_support=0.2, use_colnames=oht.columns_)

output.head()
config = [

    ('antecedent support', 0.7),

    ('support', 0.5),

    ('confidence', 0.8),

    ('conviction', 3)

]



for metric_type, th in config:

    rules = association_rules(output, metric=metric_type, min_threshold=th)

    if rules.empty:

        print ('Empty Data Frame For Metric Type : ',metric_type,' on Threshold : ',th)

        continue

    print (rules.columns.values)

    print ('-------------------------------------')

    print ('Configuration : ', metric_type, ' : ', th)

    print ('-------------------------------------')

    print (rules)



    support=rules.as_matrix(columns=['support'])

    confidence=rules.as_matrix(columns=['confidence'])



    plt.scatter(support, confidence, edgecolors='red')

    plt.xlabel('support')

    plt.ylabel('confidence')

    plt.title(metric_type+' : '+str(th))

    plt.show()
dataset = []

in_titanic=cat_titanic

for i in range(0, in_titanic.shape[0]-1):

    dataset.append([str(in_titanic.values[i,j]) for j in range(0, in_titanic.shape[1])])

# dataset = in_titanic.to_xarray()



oht = TransactionEncoder()

oht_ary = oht.fit(dataset).transform(dataset)

df = pd.DataFrame(oht_ary, columns=oht.columns_)

df.head()
output = apriori(df, min_support=0.2, use_colnames=oht.columns_)

config = [

    ('antecedent support', 0.7),

    ('confidence', 0.8),

    ('conviction', 3)

]



for metric_type, th in config:

    rules = association_rules(output, metric=metric_type, min_threshold=th)

    if rules.empty:

        print ('Empty Data Frame For Metric Type : ',metric_type,' on Threshold : ',th)

        continue

    print (rules.columns.values)

    print ('-------------------------------------')

    print ('Configuration : ', metric_type, ' : ', th)

    print ('-------------------------------------')

    print (rules)



    support=rules.as_matrix(columns=['support'])

    confidence=rules.as_matrix(columns=['confidence'])



    plt.scatter(support, confidence, edgecolors='red')

    plt.xlabel('support')

    plt.ylabel('confidence')

    plt.title(metric_type+' : '+str(th))

    plt.show()
rules[rules['confidence']==rules['confidence'].min()]
rules[rules['confidence']==rules['confidence'].max()]
rules = association_rules (output, metric='support', min_threshold=0.1)

rules[rules['confidence'] == rules['confidence'].min()]
rules[rules['confidence'] == rules['confidence'].max()]