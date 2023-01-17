import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cluster
from sklearn import tree
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('classic')
import seaborn as sns
from IPython.display import HTML, display
import tabulate
import graphviz 
from itertools import combinations

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
students_df = pd.read_csv("../input/StudentsPerformance.csv")
def key_index(array):
    for i in range(0,len(array)):
        yield((array[i],i))

# Dictionaries for encoding the cathegorical fields
race_ethnicity_enc = {key:i for (key,i) in key_index(students_df['race/ethnicity'].unique())}
gender_enc = {key:i for (key,i) in key_index(students_df['gender'].unique())}
parental_education = ['some high school', 'high school','some college',
                      "associate's degree", "bachelor's degree", "master's degree"]
parental_education_enc = {key:i for (key,i) in key_index(parental_education)}
lunch_enc = {key:i for (key,i) in key_index(students_df['lunch'].unique())}
test_prep_course_enc = {key:i for (key,i) in key_index(students_df['test preparation course'].unique())}

# Disctionaries for decoding the cathegorical fields
race_ethnicity_val = {i:key for (key,i) in key_index(tuple(race_ethnicity_enc.keys()))}
gender_val = {i:key for (key,i) in key_index(tuple(gender_enc.keys()))}
parental_education_val = {i:key for (key,i) in key_index(tuple(parental_education_enc.keys()))}
lunch_val = {i:key for (key,i) in key_index(tuple(lunch_enc.keys()))}
test_prep_course_val = {i:key for (key,i) in key_index(tuple(test_prep_course_enc.keys()))}
table = [
    ['race/ethnicity']+['value']+[a for a in race_ethnicity_enc.keys()],
    [' ']+['code']+[a for a in race_ethnicity_enc.values()], 
    ['gender']+['value']+[a for a in gender_enc.keys()],
    [' ']+['code']+[a for a in gender_enc.values()],
    ['parental level of education']+['value']+[a for a in parental_education_enc.keys()],
    [' ']+['code']+[a for a in parental_education_enc.values()],
    ['lunch']+['value']+[a for a in lunch_enc.keys()],
    [' ']+['code']+[a for a in lunch_enc.values()],
    ['test preparation course']+['value']+[a for a in test_prep_course_enc.keys()],
    [' ']+['code']+[a for a in test_prep_course_enc.values()]
]
display(HTML(tabulate.tabulate(table, tablefmt='html')))
students_df.loc[:,'race/ethnicity'] = students_df['race/ethnicity'].apply(lambda x: race_ethnicity_enc[x])
students_df.loc[:,'gender'] = students_df['gender'].apply(lambda x: gender_enc[x])
students_df.loc[:,'parental level of education'] = students_df['parental level of education'].apply(lambda x: parental_education_enc[x])
students_df.loc[:,'lunch'] = students_df['lunch'].apply(lambda x: lunch_enc[x])
students_df.loc[:,'test preparation course'] = students_df['test preparation course'].apply(lambda x: test_prep_course_enc[x])
students_df.describe()
sns.set()
fig=plt.figure(figsize=(20,3))
gs=gridspec.GridSpec(1,3) # 2 rows, 3 columns

ax00=fig.add_subplot(gs[0,0]) # First row, first column
ax01=fig.add_subplot(gs[0,1]) # First row, second column
ax02=fig.add_subplot(gs[0,2]) # First row, first column

sns.boxplot(x="gender", y="math score", data=students_df, ax=ax00)
sns.boxplot(x="gender", y="reading score", data=students_df, ax=ax01)
sns.boxplot(x="gender", y="writing score", data=students_df, ax=ax02)

ax00.set_xticklabels(['Male', 'Female'])
ax01.set_xticklabels(['Male', 'Female'])
ax02.set_xticklabels(['Male', 'Female'])

ax00.set_title('Math scores')
ax01.set_title('Reading scores')
_ = ax02.set_title('Writing scores')
sns.set()
fig=plt.figure(figsize=(20,3))
gs=gridspec.GridSpec(1,3) # 2 rows, 3 columns

ax00=fig.add_subplot(gs[0,0]) # First row, first column
ax01=fig.add_subplot(gs[0,1]) # First row, second column
ax02=fig.add_subplot(gs[0,2]) # First row, first column

sns.scatterplot(x="writing score", y="math score", data=students_df, ax=ax00, hue="gender")
sns.scatterplot(x="writing score", y="reading score", data=students_df, ax=ax01, hue="gender")
sns.scatterplot(x="math score", y="reading score", data=students_df, ax=ax02, hue="gender")

ax00.set_title('Math scores')
ax01.set_title('Reading scores')
_ = ax02.set_title('Writing scores')
sns.set()
corr = students_df.loc[:,['math score', 'reading score', 'writing score']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr,annot=True,annot_kws={"size": 7.5},linewidths=.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right");
outliers = None
for col in ['math score', 'reading score', 'writing score']:
    iq = students_df[col].quantile(.75) - students_df[col].quantile(.25)
    mean = students_df[col].mean()
    if (outliers is None):
        outliers = students_df[col] < (mean - 1.5*iq)   
    else: 
        outliers = outliers | (students_df[col] < (mean - 1.5*iq))
        
outliers_by_gender = {'0':None, '1':None}
for gender in ['0','1']:
    gender_filter = students_df['gender'] == int(gender)
    for col in ['math score', 'reading score', 'writing score']:
        iq = students_df.loc[gender_filter, col].quantile(.75) - students_df.loc[gender_filter, col].quantile(.25)
        mean = students_df.loc[gender_filter, col].mean()
        out = students_df[col] < (mean - 1.5*iq)
        
        if (outliers_by_gender[gender] is None):
            outliers_by_gender[gender] = gender_filter & out
        else: 
            outliers_by_gender[gender] = gender_filter & (outliers_by_gender[gender] | out)
np.sum(outliers_by_gender['0'])
len(students_df[outliers_by_gender['0'] | outliers_by_gender['1']])
len(students_df[outliers])
students_df.loc[:,'outlier'] = outliers
sns.set()
fig=plt.figure(figsize=(20,3))
gs=gridspec.GridSpec(1,3) # 2 rows, 3 columns

ax00=fig.add_subplot(gs[0,0]) # First row, first column
ax01=fig.add_subplot(gs[0,1]) # First row, second column
ax02=fig.add_subplot(gs[0,2]) # First row, first column

sns.catplot(x='parental level of education', kind="count", ax=ax00, data=students_df, color='lightgrey')
sns.catplot(x='parental level of education', kind="count", ax=ax00, data=students_df[students_df['gender']==0], color='#cc8963')
sns.catplot(x='parental level of education', kind="count", ax=ax00, data=students_df[students_df['gender']==1], color='#5975a4')

a= sns.catplot(x='race/ethnicity', kind="count", data=students_df, ax=ax01, color='lightgrey')
b= sns.catplot(x='race/ethnicity', kind="count", data=students_df[students_df['gender']==0], ax=ax01, color='#cc8963')
c= sns.catplot(x='race/ethnicity', kind="count", data=students_df[students_df['gender']==1], ax=ax01, color="#5975a4")

sns.catplot(x='test preparation course', kind="count", data=students_df, ax=ax02, color='lightgrey')
sns.catplot(x='test preparation course', kind="count", data=students_df[students_df['gender']==0], ax=ax02, color='#cc8963')
sns.catplot(x='test preparation course', kind="count", data=students_df[students_df['gender']==1], ax=ax02, color='#5975a4')

ax02.legend(['Total','Female','Male'])

ax01.set_xticklabels([race_ethnicity_val[i] for i in range(0, len(race_ethnicity_val.keys()))])
ax02.set_xticklabels([test_prep_course_val[i] for i in range(0, len(test_prep_course_val.keys()))])
print('Keys for the firstbar chart ',parental_education_val)

plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)
plt.close(6)
plt.close(7)
plt.close(8)
plt.close(9)
plt.close(10)
a = np.array(students_df.loc[:,['math score', 'reading score', 'writing score']])
b = np.mean(a, axis=1)
b = b.reshape((len(a),1))

students_df.loc[:,'score_class'] = (np.ceil(b / 20)).astype(int)
students_df = students_df.drop('math score',axis="columns")
students_df = students_df.drop('reading score',axis="columns")
students_df = students_df.drop('writing score',axis="columns")
students_df.head()
sns.set()
corr = students_df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, mask=mask,annot=True,annot_kws={"size": 7.5},linewidths=.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right");
def test_classifiers(classifiers, y_values, X_train, X_test, y_train, y_test):
    sns.set(font_scale=1.1)
    fig=plt.figure(figsize=(18,9))
    gs=gridspec.GridSpec(2,2) # 2 rows, 3 columns
    gs.update(wspace=0.15, hspace=0.35)

    axis_heatmap =[
        fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[0,1])
    ]

    axis_violin =[
        fig.add_subplot(gs[1,0]),
        fig.add_subplot(gs[1,1])
    ]

    clf_index = 0
    for clf in classifiers:
        print(f'Training {clf}')
        clf = clf.fit(X_train, y_train)
        print(f'Score : {clf.score(X_test, y_test)}')

        results = pd.DataFrame()
        results.loc[:,'real'] = y_test.copy()
        results.loc[:,'predicted'] = list(map(lambda x:clf.predict([x,])[0], X_test))

        axis_heatmap[clf_index].set_title(str(clf).split('(')[0])
        axis_violin[clf_index].set_title(str(clf).split('(')[0])

        g = sns.catplot(x="real", y="predicted", data=results, kind="violin", ax=axis_violin[clf_index])
        confusion_matrix = np.zeros((len(y_values),len(y_values)))
        for i in range(0, len(results)):
            confusion_matrix[results.loc[i,'predicted']-1][results.loc[i,'real']-1] += 1

        # Confusion matrix dispolayed with a heatmap        
        fig = sns.heatmap(confusion_matrix, robust=True, annot=True, linewidths=.3, ax=axis_heatmap[clf_index])
        fig.set_xlabel('real')
        fig.set_ylabel('predicted')
        fig.yaxis.set_ticklabels(y_values)
        _ = fig.xaxis.set_ticklabels(y_values)

        # Table with the summary for each of the stats
        TP = np.array([confusion_matrix[i,i] for i in range(0,len(y_values))])
        FP = np.sum(confusion_matrix, axis=1) - TP
        FN = np.sum(confusion_matrix, axis=0) - TP
        TN = np.array([len(results) for i in range(len(y_values))]) - TP - FP - FN

        precision = TP / np.maximum(np.ones(len(y_values)), (TP + FP))
        recall = TP / np.maximum(np.ones(len(y_values)), (TP + FN))

        header = []
        table = [
            [' ']+y_values,
            ['TP',] + list(TP),
            ['FP',] + list(FP),
            ['TN',] + list(TN),
            ['FN',] + list(FN),
            ['Precision'] + list(precision),
            ['Recall'] + list(recall)
        ]

        display(HTML(tabulate.tabulate(table, tablefmt='html')))
        clf_index += 1

    plt.close(2)
    plt.close(3)
X = np.array(students_df.loc[:,['parental level of education', \
                                             'test preparation course',
                                             'lunch',
                                             'race/ethnicity']]).copy()
y = np.array(students_df.loc[:,'score_class']).copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
classifiers = []
classifiers.append(tree.DecisionTreeClassifier())
classifiers.append(naive_bayes.ComplementNB())
test_classifiers(classifiers, [1,2,3,4,5], X_train, X_test, y_train, y_test)
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
X = np.array(students_df.loc[:,['test preparation course', \
                                'gender', \
                                'race/ethnicity', \
                                 'parental level of education'\
                                ,'score_class', ]]).copy()
y = np.array(students_df.loc[:,'lunch']).copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.23, random_state=42)
classifiers = []
classifiers.append(naive_bayes.ComplementNB())
classifiers.append(RandomForestClassifier(n_estimators=30, max_depth=5,random_state=0))
test_classifiers(classifiers, [1,0], X_train, X_test, y_train, y_test)
X = np.array(students_df.loc[:,['parental level of education', \
                                             'test preparation course',
                                             'lunch',
                                             'race/ethnicity']]).copy()
Y = np.array(students_df.loc[:,'outlier']).copy()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=['parental level of education', 
                                'test preparation course', 
                                'lunch', 
                                'race/ethnicity'],  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
#graph # uncomment to show the decision tree
results = pd.DataFrame()
results.loc[:,'real'] = Y.copy()
results.loc[:,'predicted'] = list(map(lambda x:clf.predict([x,])[0], X))

confusion_matrix = np.zeros((2,2))
for i in range(0, len(results)):
    confusion_matrix[results.loc[i,'predicted']-1][results.loc[i,'real']-1] += 1
    
TP = np.array([confusion_matrix[i,i] for i in range(0,2)])
FP = np.sum(confusion_matrix, axis=1) - TP
FN = np.sum(confusion_matrix, axis=0) - TP
TN = np.array([len(results) for i in range(2)]) - TP - FP - FN

precision = TP / np.maximum(np.ones(2), (TP + FP))
recall = TP / np.maximum(np.ones(2), (TP + FN))

table = [
    [' ', '1', '0'],
    ['TP',] + list(TP),
    ['FP',] + list(FP),
    ['TN',] + list(TN),
    ['FN',] + list(FN),
    ['Precision'] + list(precision),
    ['Recall'] + list(recall)
]

display(HTML(tabulate.tabulate(table, tablefmt='html')))


fig = sns.heatmap(confusion_matrix, robust=True, annot=True, linewidths=.3)
fig.set_title('Confusion matrix')
fig.set_xlabel('real')
fig.set_ylabel('predicted')
fig.yaxis.set_ticklabels(['True', 'False'])
_ = fig.xaxis.set_ticklabels(['True', 'False'])
X = np.array(students_df.loc[:,['parental level of education', \
                                             'test preparation course',
                                             'lunch',
                                             'race/ethnicity']]).copy()
Y = np.array(students_df.loc[:,'outlier']).copy()

clf = tree.DecisionTreeClassifier(max_depth=7, 
                                  class_weight="balanced", 
                                  presort=True, 
                                  max_leaf_nodes=50, 
                                  min_samples_leaf=4, 
                                  min_samples_split=6)
clf = clf.fit(X,Y)
results = pd.DataFrame()
results.loc[:,'real'] = Y.copy()
results.loc[:,'predicted'] = list(map(lambda x:clf.predict([x,])[0], X))

confusion_matrix = np.zeros((2,2))
for i in range(0, len(results)):
    confusion_matrix[results.loc[i,'predicted']-1][results.loc[i,'real']-1] += 1
    
TP = np.array([confusion_matrix[i,i] for i in range(0,2)])
FP = np.sum(confusion_matrix, axis=1) - TP
FN = np.sum(confusion_matrix, axis=0) - TP
TN = np.array([len(results) for i in range(2)]) - TP - FP - FN

precision = TP / np.maximum(np.ones(2), (TP + FP))
recall = TP / np.maximum(np.ones(2), (TP + FN))

table = [
    [' ', '1', '0'],
    ['TP',] + list(TP),
    ['FP',] + list(FP),
    ['TN',] + list(TN),
    ['FN',] + list(FN),
    ['Precision'] + list(precision),
    ['Recall'] + list(recall)
]

display(HTML(tabulate.tabulate(table, tablefmt='html')))
fig = sns.heatmap(confusion_matrix, robust=True, annot=True, linewidths=.3)
fig.set_title('Confusion matrix')
fig.set_xlabel('real')
fig.set_ylabel('predicted')
fig.yaxis.set_ticklabels(['True', 'False'])
_ = fig.xaxis.set_ticklabels(['True', 'False'])
social_factors = np.array(students_df.loc[:,['parental level of education', \
                                             'test preparation course',
                                             'lunch',
                                             'race/ethnicity']]).copy()
kmeans = cluster.MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=200)
for i in range(5):
    kmeans = kmeans.partial_fit(social_factors[i:(200 + i*200),:])
    
    
    clusters = np.zeros((social_factors.shape[0],1))
    clusters[:,0] = kmeans.predict(social_factors)
    #social_factors = np.append(social_factors,clusters,axis=1)
    sc_df = pd.DataFrame(np.append(social_factors,clusters,axis=1), columns=['parental level of education', 'test preparation course', 'lunch', 'race/ethnicity','cluster'])
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    sns.catplot(x="parental level of education", 
                y='race/ethnicity', 
                hue='cluster', 
                data=sc_df)