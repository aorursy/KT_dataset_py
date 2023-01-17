import dython.nominal as dm

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



plt.style.use('seaborn')

pd.set_option("display.max_rows", 100)
data = pd.read_csv("../input/mushrooms.csv")
data.head()
data.describe()
data = data.drop(['veil-type'], axis=1)
def donut_chart(data):

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(aspect="equal"))



    recipe = list(data.value_counts().index)



    info = data.value_counts()



    def pcts(val_list):

        pct = []

        for val in val_list:

            pct.append(" ({:.1f}k obs - {:.1f}%)".format(val/1000, 100*val/np.sum(val_list)))

        return pct



    recipe2 = pcts(info)



    wedges, texts = ax.pie(info, wedgeprops=dict(width=0.5), startangle=-40)



    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),

              bbox=bbox_props, zorder=0, va="center")



    for i, p in enumerate(wedges):

        ang = (p.theta2 - p.theta1)/2. + p.theta1

        y = np.sin(np.deg2rad(ang))

        x = np.cos(np.deg2rad(ang))

        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

        connectionstyle = f"angle,angleA=0,angleB={ang}"

        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        kw["color"] = 'k'

        ax.annotate(recipe[i]+recipe2[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                     horizontalalignment=horizontalalignment, **kw)



    ax.set_title("Proportion of classes")

    plt.show()

    

donut_chart(data['class'])
data.isnull().sum()
def generate_freq_heatmaps(feature):

    wild_ct = pd.crosstab(data[feature], 

                          data['class'])



    wild_ct_pct = wild_ct.apply(lambda r: r/r.sum(), axis=0)

    wild_ct_pct2 = wild_ct.apply(lambda r: r/r.sum(), axis=1)

    

    cram_v = dm.cramers_v(data[feature], data['class'])

    t_u_given_class = dm.theils_u(data[feature], data['class'])

    t_u_given_feature = dm.theils_u(data['class'], data[feature])

    

    plt.figure(figsize=(12,10))



    ax1 = plt.subplot2grid((3,2), (0, 0), colspan=2, 

                title=f"Overall Count of {feature} \n Cramer's V: {cram_v:.3f}")

    ax2 = plt.subplot2grid((3,2), (1, 0), 

                title=f"P({feature}|class) \n Theil's U: {t_u_given_class:.3f}")

    ax3 = plt.subplot2grid((3,2), (1, 1), 

                title=f"P(class|{feature}) \n Theil's U: {t_u_given_feature:.3f}")



    sns.heatmap(wild_ct, cmap=sns.color_palette("BuGn"), annot=True, fmt='g', ax=ax1)

    sns.heatmap(wild_ct_pct, cmap=sns.color_palette("GnBu"), annot=True, fmt='.0%', ax=ax2)

    sns.heatmap(wild_ct_pct2, cmap=sns.color_palette("GnBu"), annot=True, fmt='.0%', ax=ax3)



    plt.tight_layout()

    plt.show()
features = [col for col in data.columns if col != 'class']



for feature in features:

    generate_freq_heatmaps(feature)
def get_nominal_scores(func):

    scores = pd.DataFrame(features, columns=['feature'])

    scores['score'] = scores['feature'].apply(lambda x: func(data['class'], data[x]))

    scores = scores.sort_values(by='score', ascending=False)

    return scores
cramers_vs = get_nominal_scores(dm.cramers_v)

theils_us = get_nominal_scores(dm.theils_u)
ax = sns.barplot(x="score", y="feature", data=cramers_vs)

ax.set_title("Cramer's V with respect to class")

plt.show()
ax = sns.barplot(x="score", y="feature", data=theils_us)

ax.set_title("Theil's U with respect to class given the feature")

plt.show()
data = data.reindex(np.random.permutation(data.index))

data = pd.get_dummies(data, drop_first=True)

data.head()
y = data.iloc[:,0:1]

X = data.iloc[:,1:]
from sklearn.model_selection import cross_validate



def eval_model(model):

    cv_results = cross_validate(model, X, y.values.ravel(), cv=5, scoring=('roc_auc', 

                                                                           'precision',

                                                                           'recall',

                                                                           'f1'))

    

    print("=== Mean Test Results for {} ===".format(type(model).__name__))

    print("RECALL: {:.3f}".format(cv_results.get('test_recall').mean()))

    print("ROC AUC: {:.3f}".format(cv_results.get('test_roc_auc').mean()))

    print("PRECISION: {:.3f}".format(cv_results.get('test_precision').mean()))

    print("F1 SCORE: {:.3f} \n".format(cv_results.get('test_f1').mean()))

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.svm import SVC



X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.1, random_state=0)



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='recall')

clf.fit(X_train, y_train)



means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']



for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
from sklearn.metrics import classification_report



svc_tuning = SVC(kernel="linear", C=1)

svc_tuning.fit(X_train, y_train)

svc_tuning_preds = svc_tuning.predict(X_test)



print(classification_report(y_test, svc_tuning_preds))
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

 

gnb = GaussianNB()

logit = LogisticRegression(solver='liblinear')

svc = SVC(kernel="linear", C=1)

knn = KNeighborsClassifier(3)

xgb = XGBClassifier()



classifiers = [gnb, logit, svc, xgb]



for classifier in classifiers:

    eval_model(classifier)
logit.fit(X_train, y_train)

logit_pred = logit.predict(X_test)

print(classification_report(y_test, logit_pred))
df_log = pd.DataFrame({'feature': np.array(X.columns), 'beta':logit.coef_[0]})

df_log['OR'] = np.exp(df_log['beta'])



bases = ['class_e', 'cap-shape_b', 'cap-surface_f', 'cap-color_b', 'bruises_f', 'odor_a',

 'gill-attachment_a', 'gill-spacing_c', 'gill-size_b', 'gill-color_b', 'stalk-root_?',

 'stalk-color-above-ring_b', 'stalk-color-below-ring_b', 'veil-color_n',

 'ring-number_n', 'ring-type_e', 'spore-print-color_b', 'population_a', 'habitat_d',

'stalk-shape_e', 'stalk-surface-above-ring_f', 'stalk-surface-below-ring_f'] 



pd_bases = pd.DataFrame.from_dict({

    'feature': bases,

    'beta': np.zeros(len(bases)),

    'OR': np.ones(len(bases))

})



df_log = pd.concat([df_log, pd_bases])

df_log
def graph_ors(feature):

    ors = df_log[df_log['feature'].str.contains(feature)]

    ors = ors.sort_values(by='OR', ascending=False)



    ax = sns.barplot(x="beta", y="feature", data=ors)

    ax.set_title(f"Log of Odds Ratio for {feature}")

    plt.show()
for feature in features:

    graph_ors(feature)