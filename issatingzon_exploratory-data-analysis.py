import matplotlib.pyplot as plt

from math import pi

import seaborn as sns

import pandas as pd

import numpy as np
from sklearn import preprocessing

from sklearn import decomposition

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from matplotlib import rcParams

rcParams['axes.titlepad'] = 40
#df = pd.read_csv("../input/Family Income and Expenditure.csv")

df = pd.read_csv("Family-Income-and-Expenditure.csv")
df.head()
df.info()
df.describe()
df.describe(include=['O'])
target = 'Total Household Income'

df[target].describe()
df[target] = df[target]/12

target_new = 'Total Household Income (Monthly)'

df = df.rename(columns={target:target_new})

target = target_new

df[target] = df[target].astype(float)

df[target].describe()
f, ax = plt.subplots(figsize=(10, 5))

s = sns.distplot(df[target])

s.set(ylabel='Density')

plt.show()
print("Skewness:", df[target].skew())
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
k = 10

corrmat = df.corr()

cols = corrmat.nlargest(k, target)[target].index

f, ax = plt.subplots(figsize=(10, 7))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

s = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

s.set_title("Top 10 Variables Most Correlated with Total Household Income", size=15)

plt.show()
sns.set()

sns.pairplot(df[cols[:5]], size = 3)

plt.show();
var = 'Type of Household'

f, ax = plt.subplots(figsize=(3, 5))

s = sns.countplot(x=var, data=df)

s.set_xticklabels(s.get_xticklabels(), rotation=90)

s.set(ylabel='Count')

plt.show()
var = 'Total Number of Family members'

fig, ax = plt.subplots(figsize=(10,5))

s = sns.countplot(x=var, data=df)

s.set(ylabel='Count')

plt.show()
s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set(ylabel=target)

s.set_ylim(0,115000)

plt.show()
var = 'Total number of family members employed'

fig, ax = plt.subplots(figsize=(7,5))

s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set(ylabel=target)

s.set_ylim(0,115000)

plt.show()
var = 'Members with age less than 5 year old'

fig, ax = plt.subplots(figsize=(5,5))

sns.countplot(x=var, data=df)

s.set(ylabel='Count')

plt.show()
var = 'Members with age less than 5 year old'

fig, ax = plt.subplots(figsize=(5,5))

s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set_ylim(0,60000)

plt.show()
var = 'Members with age 5 - 17 years old'

fig, ax = plt.subplots(figsize=(7,5))

sns.countplot(x=var, data=df)

s.set(ylabel='Count')

plt.show()
fig, ax = plt.subplots(figsize=(5,5))

s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set_ylim(0,60000)

plt.show()
var = 'Household Head Age'

ages = pd.cut(df[var], 10)
fig, ax = plt.subplots(figsize=(7,5))

s = sns.countplot(y=ages)

s.set(xlabel='Count')

plt.show()
fig, ax = plt.subplots(figsize=(7,5))

s = sns.boxplot(y=ages, x=df[target], fliersize=0)

s.set_xlim(0,75000)

plt.show()
var = 'Total number of family members employed'

fig, ax = plt.subplots(figsize=(7,5))

s = sns.countplot(x=var, data=df)

s.set(ylabel='Count')

plt.show()
fig, ax = plt.subplots(figsize=(7,5))

s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set_ylim(0,125000)

plt.show()
sex = 'Household Head Sex' 

fig, ax = plt.subplots(figsize=(2,5))

sns.countplot(x=sex, data=df)

plt.show()
sex = 'Household Head Sex' 

fig, ax = plt.subplots(figsize=(2,5))

s = sns.boxplot(x=sex, y=target, data=df, fliersize=0)

s.set(ylabel=target)

s.set_ylim(0,60000)

plt.show()
var = 'Main Source of Income'

fig, ax = plt.subplots(figsize=(8,5))

s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set_ylim(0,60000)

plt.show()
var = 'Household Head Marital Status'

s = sns.countplot(x=var, hue=sex, data=df)

s.set_xticklabels(s.get_xticklabels(), rotation=30)

plt.show()
var = 'Household Head Class of Worker' 

a = df[var].astype('category').cat.categories

b = {i:df[var].value_counts()[i] for i in a}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)

a = [i[0] for i in b if i[0] != 'Other']

c = [i[1] for i in b if i[0] != 'Other']
f, ax = plt.subplots(figsize=(10, 5))

s = sns.barplot(x=c, y=a)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel='Number of Filipino Workers')

for i, v in enumerate(c):

    s.text(v + 3, i + .25, str(v), color='gray')

plt.show()
f, ax = plt.subplots(figsize=(10, 5))

s = sns.boxplot(y=df[var], x=df[target], fliersize=0)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel='Average Monthly Income')

s.set_xlim(0, 100000)

plt.show()
var = 'Household Head Occupation' 

a = df[var].astype('category').cat.categories

b = {i:df[var].value_counts()[i] for i in a}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)[:20]

a = [i[0] for i in b if i[0] != 'Other']

c = [i[1] for i in b if i[0] != 'Other']
f, ax = plt.subplots(figsize=(18, 10))

s = sns.barplot(x=c, y=a)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel='Count')

s.set_title("Top 20 Most Common Jobs in the Philippines", size=20)

for i, v in enumerate(c):

    s.text(v + 3, i + .25, str(v), color='gray')

plt.show()
var = 'Household Head Occupation' 

a = df[var].astype('category').cat.categories

b = {i:df[df[var]==i][target].mean() for i in a}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)

d = [i for i in b[:20]]

a = [i[0] for i in d]

c = [i[1] for i in d]

f, ax = plt.subplots(figsize=(18, 10))

s = sns.barplot(x=c, y=a)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel=target)

s.set_title("Top 20 Highest-Paying Occupations in the Philippines", size=20)

for i, v in enumerate(c):

    s.text(v + 3, i + .25, str(round(v,2)), color='gray')

s.set_xlim(0,180000)

plt.show()
var = 'Household Head Occupation' 

a = df[var].astype('category').cat.categories

b = {i:df[df[var]==i][target].mean() for i in a}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)

d = [i for i in b[-20:]]

a = [i[0] for i in d]

c = [i[1] for i in d]

f, ax = plt.subplots(figsize=(18, 10))

s = sns.barplot(x=c, y=a)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel=target)

s.set_title("20 Lowest Paying Occupations in the Philippines", size=20)

for i, v in enumerate(c):

    s.text(v + 3, i + .25, str(round(v,2)), color='gray')

s.set_xlim(0,180000)

plt.show()
var = 'Household Head Occupation' 

a = df[var].astype('category').cat.categories

fems = df[df[sex] == 'Female'][var].value_counts()

b = {i:fems[i] for i in a if i in fems}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)

d = [i for i in b[:20]]

a = [i[0] for i in d if i[0] != 'Other']

c = [i[1] for i in d if i[0] != 'Other']

f, ax = plt.subplots(figsize=(18, 10))

s = sns.barplot(x=c, y=a)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel='Count')

s.set_title("Top 20 Occupations with the Largest Share of Women", size=20)

for i, v in enumerate(c):

    s.text(v + 3, i + .25, str(v), color='gray')

plt.show()
var = 'Household Head Occupation' 

a = df[var].astype('category').cat.categories

f, ax = plt.subplots(figsize=(18, 10))

b = {i:df[df[var]==i][target].mean() for i in a}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)

d = [i for i in b[:20]]

a = [i[0] for i in d]

occs = {i:dict(df[df[var]==i][sex].value_counts()) for i in a}

c = [i[1] for i in d]



menMeans = []

womenMeans = []

for i in occs:

    male, female = 0, 0

    if 'Male' in occs[i]: male = occs[i]['Male']

    if 'Female' in occs[i]: female = occs[i]['Female']

    menMeans.append(male)

    womenMeans.append(female)



N = len(occs)

ind = np.arange(N)    # the x locations for the groups

width = 0.35       # the width of the bars: can also be len(x) sequence



p1 = plt.barh(ind, menMeans, width, color='#d62728')

p2 = plt.barh(ind, womenMeans, width,left=menMeans)



plt.ylabel(var)

plt.xlabel('Count')

plt.title('Share of Men and Women Household Heads in the Highest-paying Occupations', size = 15)

plt.yticks(ind, [i for i in occs])

#plt.yticks(np.arange(0, 81, 10))

plt.legend((p1[0], p2[0]), ('Male', 'Female'))



plt.show()
var = 'Household Head Highest Grade Completed'

df[var] = df[var].replace('Other Programs of Education at the Third Level, First Stage, of the Type that Leads to a Baccalaureate or First University/Professional Degree (HIgher Education Level, First Stage, or Collegiate Education Level)', 'Programs of Education at the Third Level');

df[var] = df[var].replace('Other Programs in Education at the Third Level, First Stage, of the Type that Leads to an Award not Equivalent to a First University or Baccalaureate Degree', 'Third Level that Leads to Non-Baccalureate Award')
var = 'Household Head Highest Grade Completed' 

f, ax = plt.subplots(figsize=(20, 15))

s = sns.countplot(y=var, data=df, order=df[var].value_counts().index)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel='Number of Filipinos')

s.set_title("Highest Educational Attainment by the Numbers", size=20)

plt.show()
var = 'Household Head Highest Grade Completed' 

f, ax = plt.subplots(figsize=(20, 15))

s = sns.boxplot(y=var, x=target, data=df, fliersize=0)

s.set_yticklabels(s.get_yticklabels())

s.set(ylabel=var, xlabel=target)

s.set_xlim(0,250000)

s.set_title("Total Household Income based on the Highest Educational Attainment", size=20)

plt.show()
var = 'Household Head Highest Grade Completed' 

f, ax = plt.subplots(figsize=(20, 15))

a = df[var].astype('category').cat.categories

b = {i:dict(df[df[var]==i][sex].value_counts()) for i in a}

b = sorted(b.items(), key=lambda kv: kv[1], reverse=False)



menMeans = []

womenMeans = []

for i in b:

    male, female = 0, 0

    if 'Male' in i[1]: male = i[1]['Male']

    if 'Female' in i[1]: female = i[1]['Female']

    menMeans.append(male)

    womenMeans.append(female)



N = len(b)

ind = np.arange(N)    # the x locations for the groups

width = 0.35      # the width of the bars: can also be len(x) sequence



p1 = plt.barh(ind, np.array(menMeans), width, color='#d62728')

p2 = plt.barh(ind, np.array(womenMeans), width, left=menMeans)



plt.ylabel(var)

plt.title('Share of Men and Women Household Heads in Highest Educational Attainment', size = 15)

plt.yticks(ind, [i[0] for i in b])

plt.legend((p1[0], p2[0]), ('Male', 'Female'))

plt.ylabel(var)

plt.xlabel('Count')

plt.show()
var = 'Region' 

s = sns.boxplot(x=var, y=target, data=df, fliersize=0)

s.set(ylabel=target)

s.set_ylim(0,80000)

s.set_xticklabels(s.get_xticklabels(), rotation=90)

plt.show()
# Source: https://stackoverflow.com/questions/42227409/tutorial-for-python-radar-chart-plot

def radar_plot():

    

    regions = df['Region'].astype('category').cat.categories.tolist()

    region_dict = {i:region for i,region in enumerate(regions)}

    fig, ax = plt.subplots(int(len(regions)) , 1)

    fig.subplots_adjust(hspace=0.5)

    fig.set_figheight(100)

    fig.set_figwidth(100)



    for i in range(0, len(regions)):

        title = region_dict[i]

        var = 'Region'

        regions = df['Region'].astype('category').cat.categories.tolist()

        cat = [c for c in df.columns if ('Expenditure' in c)]

        values = [df[df[var]==regions[i]][c].mean() for c in cat]



        N = len(cat)

        x_as = [n / float(N) * 2 * pi for n in range(N)]

        values += values[:1]

        x_as += x_as[:1]



        # Set color of axes

        plt.rc('axes', linewidth=0.5, edgecolor="#888888")



        # Create polar plot

        ax = plt.subplot(int(len(regions)), 1, i+1, polar=True)



        # Set clockwise rotation. That is:

        ax.set_theta_offset(pi / 2)

        ax.set_theta_direction(-1)



        # Set position of y-labels

        ax.set_rlabel_position(0)



        # Set color and linestyle of grid

        ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

        ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)



        # Set number of radial axes and remove labels

        plt.xticks(x_as[:-1], [])



        # Set yticks

        max_ = 11

        plt.yticks([x*10000 for x in range(1,max_+1) if x %2 == 0], [str(x)+ "e4"for x in range(1,max_+1) if x %2 == 0], size=10)



        # Plot data

        ax.plot(x_as, values, linewidth=0, linestyle='solid', zorder=3)



        # Fill area

        ax.fill(x_as, values, 'b', alpha=0.5)



        # Set axes limits

        max_val = max_*10000

        plt.ylim(0, max_val)



        # Draw ytick labels to make sure they fit properly

        for i in range(N):

            angle_rad = i / float(N) * 2 * pi



            if angle_rad == 0:

                ha, distance_ax = "center", 15000

            elif 0 < angle_rad < pi:

                ha, distance_ax = "left", 100

            elif angle_rad == pi:

                ha, distance_ax = "center", 15000

            else:

                ha, distance_ax = "right", 100



            ax.text(angle_rad, max_val + distance_ax + 15000, cat[i], size=10, horizontalalignment=ha, verticalalignment="center")



        plt.title('Expenditures of Region ' + title)
radar_plot()

plt.show()
var = 'Type of Walls'

s = sns.countplot(y=var, data=df)

plt.show()
var = 'Type of Roof'

s = sns.countplot(y=var, data=df)

plt.show()
var = 'Type of Building/House'

target_ = 'Housing and water Expenditure'

s = sns.boxplot(y=var, x=target_, data=df, fliersize=0)

s.set_xlim(0,175000)

s.set_yticklabels(s.get_yticklabels())

plt.show()
var = 'Type of Building/House'

target_ = 'Imputed House Rental Value'

s = sns.boxplot(y=var, x=target_, data=df, fliersize=0)

s.set_xlim(0,100000)

s.set_yticklabels(s.get_yticklabels())

plt.show()
var = 'Number of bedrooms'

target_ = 'House Floor Area'

s = sns.boxplot(x=var, y=target_, data=df, fliersize=0)

s.set_ylim(0,400)

plt.show()
df.isnull().sum()
df['Household Head Occupation'] = df['Household Head Occupation'].replace(np.nan, 'Other');

df['Household Head Class of Worker'] = df['Household Head Class of Worker'].replace(np.nan, 'Other');
y = df[target]

X = df[df.columns.difference([target])]
bins = [0, 15000, 100000000]

y = pd.cut(y, bins, labels=["low income", "high income"])

#y = y.astype('object')
y.value_counts()
cols = list(X.columns[X.dtypes != object])

std_scale = preprocessing.StandardScaler().fit(X[cols])

X[cols] = pd.DataFrame(std_scale.transform(X[cols]), columns=cols)
cols = list(X.columns[X.dtypes == object])

X = pd.DataFrame(pd.get_dummies(X, prefix=cols, columns=cols))
X.info()
X.describe(include='all')
#pca = decomposition.PCA(n_components=100)

#X = pca.fit_transform(X)
def train(X, y):

    test_size = 0.2

    seed = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    classifiers = dict() 

    classifiers['GaussianNB'] = GaussianNB()

    classifiers['SVM'] = SVC()

    classifiers['MLPClassifier'] = MLPClassifier()

    classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=300)



    # Iterate over dictionary

    for clf_name, clf in classifiers.items(): #clf_name is the key, clf is the value

        scores = cross_val_score(clf, X, y, cv=5)

        print(clf_name + ' cross_val_score: ' + str(np.mean(scores)))

        

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        score = metrics.accuracy_score(y_test, pred)

        print(clf_name + ': ' + str(score))

        print(metrics.classification_report(y_test, pred))

        
#train(X, y)