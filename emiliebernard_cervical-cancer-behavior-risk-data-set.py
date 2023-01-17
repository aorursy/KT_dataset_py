import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt # visualisation of data

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn import tree # classification Scikit-Learn Library to build classification tree

from sklearn.model_selection import train_test_split # Training tree with dataset to build the classification tree

from sklearn.metrics import accuracy_score # To compute the best proportion between training percentage set and testing percentage set

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

%matplotlib inline
df = pd.read_csv("/kaggle/input/cervical-cancer-behavior-risk-dataset/sobar-72.csv", encoding="utf8", engine="python")

df.head()
features = [df.columns]

features
#sns.set(style="ticks", color_codes=True)

#g = sns.PairGrid(df, hue='ca_cervix')

#g.map(sns.scatterplot)
ca = df.loc[df['ca_cervix'] == 1]

ca.head()
len(ca)
#sns.pairplot(ca)
nca = df.loc[df['ca_cervix'] == 0]

nca.head()
len(nca)
#sns.pairplot(nca)
from scipy.cluster.hierarchy import dendrogram, linkage



linked = linkage(df, 'single')



plt.figure(figsize=(30, 10))

dendrogram(linked,

            orientation='top',

            distance_sort='descending',

            show_leaf_counts=True)

plt.show()
from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



features = list(df.columns)[:-2]

### Get the features data

data = df[features]

clustering_kmeans = KMeans(n_clusters=31)

data['clusters'] = clustering_kmeans.fit_predict(data)

reduced_data = PCA().fit_transform(data)

results = pd.DataFrame(reduced_data,columns=['behavior_sexualRisk', 'behavior_eating', 'behavior_personalHygine', 'intention_aggregation', 'intention_commitment', 'attitude_consistency', 'attitude_spontaneity', 'norm_significantPerson', 'norm_fulfillment', 'perception_vulnerability', 'perception_severity', 'motivation_strength', 'motivation_willingness', 'socialSupport_emotionality', 'socialSupport_appreciation', 'socialSupport_instrumental', 'empowerment_knowledge', 'empowerment_abilities','pca19'])



sns.scatterplot(x="behavior_sexualRisk", y="behavior_eating", hue=data['clusters'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="behavior_sexualRisk")

ca['behavior_sexualRisk'].mean()
ca['behavior_sexualRisk'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="behavior_sexualRisk")

nca['behavior_sexualRisk'].mean()
nca['behavior_sexualRisk'].value_counts(normalize=True) * 100
df['behavior_sexualRisk'].value_counts()

size = [0] * len(df['behavior_sexualRisk'])

i = 0;

for x in df['behavior_sexualRisk']:

    if df['ca_cervix'][i] == 1:

        size[i] = len(df[df['behavior_sexualRisk'] == x])*60;

    else:

        size[i] = len(df[df['behavior_sexualRisk'] == x])*50;

    i += 1;



plt.scatter(x=df['behavior_sexualRisk'],y=df['behavior_sexualRisk'], c=df['ca_cervix'], cmap='Reds', s=size)

plt.title("Sexual risk of patient")

plt.xlabel("Sexual risk 0 to 10")
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="behavior_eating")

ca['behavior_eating'].mean()
ca['behavior_eating'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="behavior_eating")

nca['behavior_eating'].mean()
nca['behavior_eating'].value_counts(normalize=True) * 100
df['behavior_eating'].value_counts()

size = [0] * len(df['behavior_eating'])

i = 0;

for x in df['behavior_eating']:

    if df['ca_cervix'][i] == 1:

        size[i] = len(df[df['behavior_eating'] == x])*60;

    else:

        size[i] = len(df[df['behavior_eating'] == x])*50;

    i += 1;



plt.scatter(x=df['behavior_eating'],y=df['behavior_eating'], c=df['ca_cervix'], cmap='Reds', s=size)

plt.title("Eating behavior of patient")
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="behavior_personalHygine")

ca['behavior_personalHygine'].mean()
ca['behavior_personalHygine'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="behavior_personalHygine")

nca['behavior_personalHygine'].mean()
nca['behavior_personalHygine'].value_counts(normalize=True) * 100
df['behavior_personalHygine'].value_counts()

size = [0] * len(df['behavior_personalHygine'])

i = 0;

for x in df['behavior_eating']:

    if df['ca_cervix'][i] == 1:

        size[i] = len(df[df['behavior_personalHygine'] == x])*55;

    else:

        size[i] = len(df[df['behavior_personalHygine'] == x])*50;

    i += 1;



plt.scatter(x=df['behavior_personalHygine'],y=df['behavior_personalHygine'], c=df['ca_cervix'], cmap='Reds', s=size)

plt.title("Personal Hygine behavior of patient")
plt.xlabel("Behavior personal hygine")

plt.ylabel("Behavior eating")

plt.scatter(x=df['behavior_personalHygine'],y=df['behavior_eating'], c=df['ca_cervix'], cmap='Accent')

plt.title("Behavior of patient")
plt.xlabel("Behavior sexual risk")

plt.ylabel("Behavior eating")

plt.scatter(x=df['behavior_sexualRisk'],y=df['behavior_eating'], c=df['ca_cervix'], cmap='Accent')

plt.title("Behavior of patient")
plt.xlabel("Behavior sexual risk")

plt.ylabel("Behavior personal hygine")

plt.scatter(x=df['behavior_sexualRisk'],y=df['behavior_personalHygine'], c=df['ca_cervix'], cmap='Accent')

plt.title("Behavior of patient")
from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



features = list(df.columns)[:-2]

### Get the features data

data = df[features]

clustering_kmeans = KMeans(n_clusters=2)

data['clusters'] = clustering_kmeans.fit_predict(data)

reduced_data = PCA().fit_transform(data)

results = pd.DataFrame(reduced_data,columns=['behavior_sexualRisk', 'behavior_eating', 'behavior_personalHygine', 'intention_aggregation', 'intention_commitment', 'attitude_consistency', 'attitude_spontaneity', 'norm_significantPerson', 'norm_fulfillment', 'perception_vulnerability', 'perception_severity', 'motivation_strength', 'motivation_willingness', 'socialSupport_emotionality', 'socialSupport_appreciation', 'socialSupport_instrumental', 'empowerment_knowledge', 'empowerment_abilities','pca19'])



sns.scatterplot(x="behavior_sexualRisk", y="behavior_eating", hue=df['ca_cervix'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
sns.scatterplot(x="behavior_sexualRisk", y="behavior_personalHygine", hue=df['ca_cervix'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
sns.scatterplot(x="behavior_personalHygine", y="behavior_eating", hue=df['ca_cervix'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="intention_aggregation")

ca['intention_aggregation'].mean()
ca['intention_aggregation'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="intention_aggregation")

nca['intention_aggregation'].mean()
nca['intention_aggregation'].value_counts(normalize=True) * 100
df['intention_aggregation'].value_counts()

size = [0] * len(df['intention_aggregation'])

i = 0;

for x in df['intention_aggregation']:

    if df['ca_cervix'][i] == 1:

        size[i] = len(df[df['intention_aggregation'] == x])*55;

    else:

        size[i] = len(df[df['intention_aggregation'] == x])*50;

    i += 1;



plt.scatter(x=df['intention_aggregation'],y=df['intention_aggregation'], c=df['ca_cervix'], cmap='Reds', s=size)

plt.title("Aggregation of patient")
sns.scatterplot(x="behavior_personalHygine", y="behavior_eating", hue=df['ca_cervix'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="intention_commitment")

ca['intention_commitment'].mean()
ca['intention_commitment'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="intention_commitment")

nca['intention_commitment'].mean()
nca['intention_commitment'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="attitude_consistency")

ca['attitude_consistency'].mean()
ca['attitude_consistency'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="attitude_consistency")

nca['attitude_consistency'].mean()
nca['attitude_consistency'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="attitude_spontaneity")

ca['attitude_spontaneity'].mean()
ca['attitude_spontaneity'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="attitude_spontaneity")

nca['attitude_spontaneity'].mean()
nca['attitude_spontaneity'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="norm_significantPerson")

ca['norm_significantPerson'].mean()
ca['norm_significantPerson'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="norm_significantPerson")

nca['norm_significantPerson'].mean()
nca['norm_significantPerson'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="norm_fulfillment")

ca['norm_fulfillment'].mean()
ca['norm_fulfillment'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="norm_fulfillment")

nca['norm_fulfillment'].mean()
nca['norm_fulfillment'].value_counts(normalize=True) * 100


plt.xlabel("Norm fulfillment")

plt.ylabel("Norm significant person")

plt.scatter(x=df['norm_fulfillment'],y=df['norm_significantPerson'], c=df['ca_cervix'], cmap='Accent')

plt.title("Norm of patient")
sns.scatterplot(x="norm_fulfillment", y="norm_significantPerson", hue=df['ca_cervix'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="perception_vulnerability")

ca['perception_vulnerability'].mean()
ca['perception_vulnerability'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="perception_vulnerability")

nca['perception_vulnerability'].mean()
nca['perception_vulnerability'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="perception_severity")

ca['perception_severity'].mean()
ca['perception_severity'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="perception_severity")

nca['perception_severity'].mean()
nca['perception_severity'].value_counts(normalize=True) * 100


plt.xlabel("Perception severity")

plt.ylabel("Perception vulnerability")

plt.scatter(x=df['perception_severity'],y=df['perception_vulnerability'], c=df['ca_cervix'], cmap='Accent')

plt.title("Perception of patient")
sns.scatterplot(x="perception_severity", y="perception_vulnerability", hue=df['ca_cervix'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="motivation_strength")

ca['motivation_strength'].mean()
ca['motivation_strength'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="motivation_strength")

nca['motivation_strength'].mean()
nca['motivation_strength'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="motivation_willingness")

ca['motivation_willingness'].mean()
ca['motivation_willingness'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="motivation_willingness")

nca['motivation_willingness'].mean()
nca['motivation_willingness'].value_counts(normalize=True) * 100


plt.xlabel("Motivation strength")

plt.ylabel("Motivation willingness")

plt.scatter(x=df['motivation_strength'],y=df['motivation_willingness'], c=df['ca_cervix'], cmap='Accent')

plt.title("Motivation of patient")
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="socialSupport_emotionality")

ca['socialSupport_emotionality'].mean()
ca['socialSupport_emotionality'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="socialSupport_emotionality")

nca['socialSupport_emotionality'].mean()
nca['socialSupport_emotionality'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="socialSupport_appreciation")

ca['socialSupport_emotionality'].mean()
ca['socialSupport_appreciation'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="socialSupport_appreciation")

nca['socialSupport_emotionality'].mean()
nca['socialSupport_appreciation'].value_counts(normalize=True) * 100


plt.xlabel("Social Support emotionality")

plt.ylabel("Social Support appreciation")

plt.scatter(x=df['socialSupport_emotionality'],y=df['socialSupport_appreciation'], c=df['ca_cervix'], cmap='Accent')

plt.title("Social Support of patient")
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="socialSupport_instrumental")

ca['socialSupport_instrumental'].mean()
ca['socialSupport_instrumental'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="socialSupport_instrumental")

nca['socialSupport_instrumental'].mean()
nca['socialSupport_instrumental'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="empowerment_knowledge")

ca['empowerment_knowledge'].mean()
ca['empowerment_knowledge'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="empowerment_knowledge")

nca['empowerment_knowledge'].mean()
nca['empowerment_knowledge'].value_counts(normalize=True) * 100
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="empowerment_abilities")

ca['empowerment_abilities'].mean()
ca['empowerment_abilities'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="empowerment_abilities")

nca['empowerment_abilities'].mean()
nca['empowerment_abilities'].value_counts(normalize=True) * 100
plt.xlabel("Empowerment knowledge")

plt.ylabel("Empowerment abilities")

plt.scatter(x=df['empowerment_knowledge'],y=df['empowerment_abilities'], c=df['ca_cervix'], cmap='Accent')

plt.title("Empowerment of patient")
plt.title("Patient with cervical cancer")

sns.countplot(data=ca, x="empowerment_desires")

ca['empowerment_desires'].mean()
ca['empowerment_desires'].value_counts(normalize=True) * 100
plt.title("Patient without cervical cancer")

sns.countplot(data=nca, x="empowerment_desires")

nca['empowerment_desires'].mean()
nca['empowerment_desires'].value_counts(normalize=True) * 100
plt.xlabel("Empowerment desires")

plt.ylabel("Empowerment abilities")

plt.scatter(x=df['empowerment_desires'],y=df['empowerment_abilities'], c=df['ca_cervix'], cmap='Accent')

plt.title("Empowerment of patient")
plt.xlabel("Empowerment knowledge")

plt.ylabel("Empowerment desires")

plt.scatter(x=df['empowerment_knowledge'],y=df['empowerment_desires'], c=df['ca_cervix'], cmap='Accent')

plt.title("Empowerment of patient")
record=[]

record_list=[]



def behavior_sexualRisk(value):

    return 1 if value == 10 else 0



def behavior_eating(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def behavior_personalHygine(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def intention_aggregation(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def intention_commitment(value):

    if 15 >= value >= 12:

        return 1

    return 0



def attitude_consistency(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def attitude_spontaneity(value):

    if 10 >= value >= 8:

        return 1

    return 0



def norm_significantPerson(value):

    if 5 >= value >= 4:

        return 1

    return 0



def norm_fulfillment(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def perception_vulnerability(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def perception_severity(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def motivation_strength(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def motivation_willingness(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def socialSupport_emotionality(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def socialSupport_appreciation(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def socialSupport_instrumental(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def empowerment_knowledge(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def empowerment_abilities(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def empowerment_desires(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def ca_cervix(value):

    return value



switcher = {

    0: behavior_sexualRisk,

    1: behavior_eating,

    2: behavior_personalHygine,

    3: intention_aggregation,

    4: intention_commitment,

    5: attitude_consistency,

    6: attitude_spontaneity,

    7: norm_significantPerson,

    8: norm_fulfillment,

    9: perception_vulnerability,

    10: perception_severity,

    11: motivation_strength,

    12: motivation_willingness,

    13: socialSupport_emotionality,

    14: socialSupport_appreciation,

    15: socialSupport_instrumental,

    16: empowerment_knowledge,

    17: empowerment_abilities,

    18: empowerment_desires,

    19: ca_cervix,

}

def reduce_number_to_state(column, value):

    func = switcher.get(column, value)

    

    return func(value)



for i in range(0, df.shape[0]):

    for j in range(0, df.shape[1]):

        record_list.append(reduce_number_to_state(j, df.values[i, j]))

    record.append([(record_list[k]) for k in range(0, len(record_list))])

    record_list = []
record=[]

record_list=[]



def behavior_sexualRisk(value):

    return 1 if value == 10 else 0



def behavior_eating(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def behavior_personalHygine(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def intention_aggregation(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def intention_commitment(value):

    if 15 >= value >= 12:

        return 1

    return 0



def attitude_consistency(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def attitude_spontaneity(value):

    if 10 >= value >= 8:

        return 1

    return 0



def norm_significantPerson(value):

    if 5 >= value >= 4:

        return 1

    return 0



def norm_fulfillment(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def perception_vulnerability(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def perception_severity(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def motivation_strength(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def motivation_willingness(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def socialSupport_emotionality(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def socialSupport_appreciation(value):

    if 10 >= value >= 8:

        return 2

    elif 7 >= value >= 5:

        return 1

    return 0



def socialSupport_instrumental(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def empowerment_knowledge(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def empowerment_abilities(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def empowerment_desires(value):

    if 15 >= value >= 12:

        return 2

    elif 11 >= value >= 7:

        return 1

    return 0



def ca_cervix(value):

    return value



switcher = {

    0: behavior_sexualRisk,

    1: behavior_eating,

    2: behavior_personalHygine,

    3: intention_aggregation,

    4: intention_commitment,

    5: attitude_consistency,

    6: attitude_spontaneity,

    7: norm_significantPerson,

    8: norm_fulfillment,

    9: perception_vulnerability,

    10: perception_severity,

    11: motivation_strength,

    12: motivation_willingness,

    13: socialSupport_emotionality,

    14: socialSupport_appreciation,

    15: socialSupport_instrumental,

    16: empowerment_knowledge,

    17: empowerment_abilities,

    18: empowerment_desires,

    19: ca_cervix,

}

def reduce_number_to_state(column, value):

    func = switcher.get(column, value)

    

    return func(value)



for i in range(0, df.shape[0]):

    for j in range(0, df.shape[1]):

        record_list.append(reduce_number_to_state(j, df.values[i, j]))

    record.append([(record_list[k]) for k in range(0, len(record_list))])

    record_list = []
X = np.array(record)[:, :19] # Prepare datasets --> X = Independent Sets

Y = np.array(record)[:, 19] # --> Y = Dependent Sets

print(X)

print(Y)



# setting the proportion of test data

test_proportion = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

accuracy = []



for i in range(10):

    X_data,Y_data,X_label,Y_label = train_test_split(X, Y, test_size = test_proportion[i], random_state = 42)

    # Using decision tree for classification

    clf = tree.DecisionTreeClassifier(criterion='entropy')

    clf = clf.fit(X_data, X_label)

    print("clf: " + str(clf))

    predictedY = clf.predict(Y_data)

    accuracy.append(accuracy_score(Y_label, predictedY))
# Visulaize the accuracy results

plt.title('result(accuracy)')

plt.xlabel('test proportion')

plt.ylabel('accuracy')

plt.plot(test_proportion, accuracy)

plt.show()
feature_names = [

        'behavior_sexualRisk',

        'behavior_eating',

        'behavior_personalHygine',

        'intention_aggregation',

        'intention_commitment',

        'attitude_consistency',

        'attitude_spontaneity',

        'norm_significantPerson',

        'norm_fulfillment',

        'perception_vulnerability',

        'perception_severity',

        'motivation_strength',

        'motivation_willingness',

        'socialSupport_emotionality',

        'socialSupport_appreciation',

        'socialSupport_instrumental',

        'empowerment_knowledge',

        'empowerment_abilities',

        'empowerment_desires'

    ]



X_data,Y_data,X_label,Y_label = train_test_split(X, Y, test_size = 0.45, random_state = 42)

# Using decision tree for classification

clf = tree.DecisionTreeClassifier(criterion='entropy')

clf = clf.fit(X_data, X_label)



with open("./ID3.dot", 'w') as f:

    f = tree.export_graphviz(clf, feature_names = feature_names, out_file = f)

record2=[]

record_list2=[]



def behavior_sexualRisk2(value):

    return 1 if value == 10 else 0



def behavior_eating2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def behavior_personalHygine2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def intention_aggregation2(value):

    if 10 >= value >= 5:

        return 1

    return 0



def intention_commitment2(value):

    if 15 >= value >= 12:

        return 1

    return 0



def attitude_consistency2(value):

    if 10 >= value >= 5:

        return 1

    return 0



def attitude_spontaneity2(value):

    if 10 >= value >= 8:

        return 1

    return 0



def norm_significantPerson2(value):

    if 5 >= value >= 4:

        return 1

    return 0



def norm_fulfillment2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def perception_vulnerability2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def perception_severity2(value):

    if 10 >= value >= 5:

        return 1

    return 0



def motivation_strength2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def motivation_willingness2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def socialSupport_emotionality2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def socialSupport_appreciation2(value):

    if 10 >= value >= 5:

        return 1

    return 0



def socialSupport_instrumental2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def empowerment_knowledge2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def empowerment_abilities2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def empowerment_desires2(value):

    if 15 >= value >= 7:

        return 1

    return 0



def ca_cervix2(value):

    return value



switcher2 = {

    0: behavior_sexualRisk2,

    1: behavior_eating2,

    2: behavior_personalHygine2,

    3: intention_aggregation2,

    4: intention_commitment2,

    5: attitude_consistency2,

    6: attitude_spontaneity2,

    7: norm_significantPerson2,

    8: norm_fulfillment2,

    9: perception_vulnerability2,

    10: perception_severity2,

    11: motivation_strength2,

    12: motivation_willingness2,

    13: socialSupport_emotionality2,

    14: socialSupport_appreciation2,

    15: socialSupport_instrumental2,

    16: empowerment_knowledge2,

    17: empowerment_abilities2,

    18: empowerment_desires2,

    19: ca_cervix2,

}

def reduce_number_to_state2(column, value):

    func2 = switcher2.get(column, value)

    

    return func2(value)



for i in range(0, df.shape[0]):

    for j in range(0, df.shape[1]):

        record_list2.append(reduce_number_to_state2(j, df.values[i, j]))

    record2.append([(record_list2[k]) for k in range(0, len(record_list2))])

    record_list2 = []
record2
record_df=pd.DataFrame(record2)

frequent_itemsets = apriori(record_df, min_support=0.7, use_colnames=False)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.head()
rules[ (rules['lift'] >= 1) &

    (rules['confidence'] >= 0.9)]
support=rules['support'].values

confidence=rules['confidence'].values

lift=rules['lift'].values

plt.scatter(support, confidence, alpha=0.5)

plt.xlabel('support')

plt.ylabel('confidence')

plt.show()
plt.scatter(support, lift, alpha=0.5)

plt.xlabel('support')

plt.ylabel('lift')

plt.show()
fit = np.polyfit(lift, confidence, 1)

fit_fn = np.poly1d(fit)

plt.plot(lift, confidence, 'yo', lift, fit_fn(lift))

plt.xlabel('lift')

plt.ylabel('confidence')

plt.show()