import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline



from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV, SelectKBest



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/HR_comma_sep.csv')

# We clean up this dataset quickly

# ----------------------- sales <- department

cols = list(df.columns)

cols[-2] = 'department'

df.columns = cols

# ----------------------- salarybracket to int

df.salary = df.salary.map({'low': 0, 'medium': 1, 'high': 2})

# ----------------------- department similarities

department_groups = {'sales': 1, 'marketing': 1, 'product_mng': 1, #selling stuff

                     'technical': 2, 'IT': 2, 'RandD': 2, # making stuff

                     'accounting': 3, 'hr': 3, 'support': 3, 'management': 3 # maintenance

                    }

df['deptgrps'] = df.department.map(department_groups)

# ----------------------- department one hot

for dept in df.department.unique():

    df['dept_'+dept] = (df.department == dept).astype(int)

df = df.drop('department', axis=1)

# ----------------------- 

df.info()


x, y = df.drop('left', axis=1), df['left']
# We visualize the first two principal components.

data = PCA(n_components=2).fit_transform(x)

temp = pd.DataFrame(data, columns=['a', 'b'])

temp['target'] = y

sns.lmplot('a', 'b', data=temp, hue='target', fit_reg=False)
# No use I see. Is there balance in the label classes? 0 seems to outnumber 1 by a lot

df.left.value_counts()
# There's imbalance in the data

# We will handle that in some other notebook. For now let's compare classifier performance.

classifiers = [('rfg', RandomForestClassifier(n_jobs=-1, criterion='gini')),

               ('rfe', RandomForestClassifier(n_jobs=-1, criterion='entropy')),

               ('ada', AdaBoostClassifier()),

               ('extf', ExtraTreesClassifier(n_jobs=-1)),

               ('knn', KNeighborsClassifier(n_jobs=-1)),

               ('dt', DecisionTreeClassifier()),

               ('Et', ExtraTreeClassifier()),

               ('Logit', LogisticRegression()),

               ('gnb', GaussianNB()),

               ('bnb', BernoulliNB()),

              ]

allscores = []

for name, classifier in classifiers:

    scores = []

    for i in range(3): # three runs

        roc = cross_val_score(classifier, x, y, scoring='roc_auc', cv=20)

        scores.extend(list(roc))

    scores = np.array(scores)

    print(name, scores.mean())

    new_data = [(name, score) for score in scores]

    allscores.extend(new_data)

        
temp = pd.DataFrame(allscores, columns=['classifier', 'score'])

sns.violinplot('classifier', 'score', data=temp, inner=None, linewidth=0.3)
classifier = RandomForestClassifier(n_jobs=-1, n_estimators=100)

rfecv = RFECV(estimator=classifier, cv=15, scoring='roc_auc')

rfecv.fit(x, y)

print("Optimal number of features : {}".format(rfecv.n_features_))
plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, '.-')

plt.plot([rfecv.n_features_, rfecv.n_features_], [0.9, 1], '-.')

plt.show()
ranks = list(zip(rfecv.ranking_, x.columns))

ranks.sort()

ranks
selector = SelectKBest(k=7)

best7 = selector.fit_transform(x, y)

ranks = list(zip(selector.scores_, x.columns))

ranks.sort(reverse=True)

best = [i[1] for i in ranks[:7]]

best
classifier = RandomForestClassifier(n_jobs=-1, n_estimators=100)

best_x = x[best]

scores = []

for i in range(5):

    run_scores = cross_val_score(classifier, best_x, y, scoring='roc_auc', cv=15)

    print(i, run_scores.mean())

    scores.extend(list(run_scores))

scores = np.array(scores)
#sns.distplot(scores)