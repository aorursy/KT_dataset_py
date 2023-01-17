import dython.nominal as dm

import itertools

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



plt.style.use('seaborn')

pd.set_option("display.max_rows", 100)
data = pd.read_csv("../input/pulsar_stars.csv")

data.columns = ['mean_ip', 'sd_ip', 'ek_ip', 'skw_ip', 'mean_dm', 'sd_dm', 'ek_dm', 'skw_dm', 'pulsar']

data.head()
data.describe()
def donut_chart(data):

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(aspect="equal"))



    recipe = [str(i) for i in list(data.value_counts().index)]



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

    

donut_chart(data['pulsar'])
from matplotlib import colors



preds = [col for col in data.columns if col != "pulsar"]

num_preds = len(preds)

fig, axes = plt.subplots(num_preds, 2, figsize=(10,20))



for i, j in itertools.zip_longest(preds, range(num_preds)):

    N, bins, patches = axes[j, 0].hist(data[i], bins="auto", density=True)

    axes[j, 0].set_title(f"PDF of {i}")

    

    axes[j, 0].axvline(data[i].mean(), color = "c", linestyle="dashed", label="mean", linewidth=3)

    axes[j, 0].axvline(data[i].std(), color = "m", linestyle="dotted", label="std", linewidth=3)

    axes[j, 0].legend(("mean", "std"), loc="best")

    

    fracs = N / N.max()

    norm = colors.Normalize(fracs.min(), fracs.max())



    for thisfrac, thispatch in zip(fracs, patches):

        color = plt.cm.plasma(norm(thisfrac))

        thispatch.set_facecolor(color)

    

    axes[j, 1].hist(data[i], bins="auto", cumulative=True, density=True)

    axes[j, 1].set_title(f"CDF of {i}")



plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()
pulsars = data[data['pulsar'] == 1].drop('pulsar', axis=1)

non_pul = data[data['pulsar'] == 0].drop('pulsar', axis=1)



fig, axes = plt.subplots(num_preds, 2, figsize=(10,20))



for i, j in itertools.zip_longest(preds, range(num_preds)):

    axes[j, 0].hist(pulsars[i], bins="auto", label="pulsars", color = "g", alpha=0.5, density=True)

    axes[j, 0].hist(non_pul[i], bins="auto", label="non-pulsars", color = "r", alpha=0.5, density=True)

    axes[j, 0].set_title(f'PDF comparison for {i}')

    axes[j, 0].legend(loc="best")

    

    axes[j, 1].hist(pulsars[i], bins="auto", label="pulsars", color = "g", alpha=0.5)

    axes[j, 1].hist(non_pul[i], bins="auto", label="non-pulsars", color = "r", alpha=0.5)

    axes[j, 1].set_title(f'Frequency comparison for {i}')

    axes[j, 1].legend(loc="best")



plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()
plt.figure(figsize=(15,25))



for i in range(num_preds):

    plt.subplot(10,1,i+1)

    sns.violinplot(x=data['pulsar'],y=data.iloc[:, i],

                   palette=sns.color_palette("hls", 7),alpha=.5)

    plt.title(data.columns[i])

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(13,25))



for i in range(num_preds):

    plt.subplot(10,1,i+1)

    sns.boxplot(data['pulsar'],y=data.iloc[:, i], 

                palette=sns.color_palette("husl", 7), color="w")

    plt.title(data.columns[i])

    

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()
g = sns.pairplot(data,hue="pulsar")

plt.show()
correlation = data.corr()

ax = sns.heatmap(correlation, annot=True, cmap=sns.color_palette("pastel"),

                 linewidth=2,edgecolor="k")

plt.title("Correlation Between Features")

plt.show()
corr_ratios = [dm.correlation_ratio(data['pulsar'], data[pred]) for pred in preds]

df_cr = pd.DataFrame({'feature': preds, 'correlation_ratio':corr_ratios})



ax = sns.barplot(x="correlation_ratio", y="feature", data=df_cr)

ax.set_title("Correlation Ratio for features")

plt.show()
def get_outliers(feature):

    data_out = data[feature]

    q1 = data_out.quantile(0.25)

    q3 = data_out.quantile(0.75)

    iqr = q3 - q1



    out_l = data_out[data_out < (q1 - 1.5*iqr)].index

    out_r = data_out[data_out > (q3 + 1.5*iqr)].index

    

    return(iqr, out_l, out_r)
def print_outlier_group():

    for i in preds:

        print(f'Predictor: {i}\n')



        iqr, out_l, out_r = get_outliers(i)



        full_count = pd.DataFrame()



        lower_count = data.iloc[out_l].groupby('pulsar').size().to_frame('lower_count')

        upper_count = data.iloc[out_r].groupby('pulsar').size().to_frame('upper_count')



        if not lower_count.empty:

            lower_count['lower_ratio'] = lower_count/lower_count.sum()

        else:

            lower_count['lower_ratio'] = np.nan

        if not upper_count.empty:

            upper_count['upper_ratio'] = upper_count/upper_count.sum()

        else:

            upper_count['upper_ratio'] = np.nan



        full_count = pd.concat([lower_count, upper_count], axis=1)

        print(full_count)

        print('\n======\n')



print_outlier_group()
def update_outlier_dict(ou, out_list, label, i):

    for elem in list(out_list):

        if ou.get(elem) is None:

            ou.update({elem: dict.fromkeys(preds, None)})

        ou.get(elem).update({i: label})

        

def print_outliers_df():

    ou = {}

    for i in preds:

        iqr, out_l, out_r = get_outliers(i)

        

        update_outlier_dict(ou, out_l, "L", i)

        update_outlier_dict(ou, out_r, "R", i)



    df = pd.DataFrame.from_dict(ou, orient='index')

    df['count'] = df.count(axis=1)

    df['pulsar'] = data['pulsar'].iloc[df.index]

    #print(df.groupby('pulsar').size().to_frame('count_r'))

    print(df.sort_values('count', ascending=False))

    #print(df.groupby('count').size())

    plt.hist(df[df['pulsar']==0]['count'], bins='auto', color = "m", alpha=0.5)

    plt.hist(df[df['pulsar']==1]['count'], bins='auto', color = "y", alpha=0.5)

    plt.legend(labels=['non-pulsars', 'pulsars'])

    plt.title("Frequency of observations that are outliers for x features")

    plt.xlabel("Number of features")

    plt.ylabel("Frequency of observations")

    plt.show()



print_outliers_df()
from sklearn.preprocessing import RobustScaler



float_data = data[preds].astype(np.float64)

robust_trans = RobustScaler().fit(float_data)

robust_data = pd.DataFrame(robust_trans.transform(data[preds]), 

                     columns= ['mean_ip', 'sd_ip', 'ec_ip', 

                               'sw_ip', 'mean_dm', 'sd_dm', 

                               'ec_dm', 'sw_dm'])



robust_data.describe()
from sklearn.decomposition import PCA



pca_all = PCA()

pca_all.fit(robust_data)



cum_var = (np.cumsum(pca_all.explained_variance_ratio_))

n_comp = [i for i in range(1, pca_all.n_components_ + 1)]



ax = sns.pointplot(x=n_comp, y=cum_var)

ax.set(xlabel='# components', ylabel='cumulative explained variance')

plt.show()
pca_2 = PCA(2)

pca_2.fit(robust_data)

data_2pc = pca_2.transform(robust_data)



ax = sns.scatterplot(x=data_2pc[:,0], 

                     y=data_2pc[:,1], 

                     hue=data['pulsar'],

                     palette=sns.color_palette("muted", n_colors=2))



ax.set(xlabel='1st PC', ylabel='2nd PC', title='Scatterplot for first two Principal Components')

plt.show()
from mpl_toolkits.mplot3d import Axes3D



pca_3 = PCA(3)

pca_3.fit(robust_data)

data_3pc = pca_3.transform(robust_data)



pulsar_index = list(pulsars.index)

non_pulsar_index = list(non_pul.index)



x = data_3pc[:,0]

y = data_3pc[:,1]

z = data_3pc[:,2]



fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(np.take(x, pulsar_index), 

           np.take(y, pulsar_index), 

           np.take(z, pulsar_index),

           edgecolors='none', color='g',

           alpha=0.8, label='pulsar')





ax.scatter(np.take(x, non_pulsar_index), 

           np.take(y, non_pulsar_index), 

           np.take(z, non_pulsar_index),

           edgecolors='none', color='r',

           alpha=0.8, label='non-pulsar')



ax.set_xlabel("1st PC", fontsize=20)

ax.set_ylabel("2nd PC", fontsize=20)

ax.set_zlabel("3rd PC", fontsize=20)

ax.legend(fontsize="xx-large")

plt.show()
X = data[preds]

y = data[['pulsar']]
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_validate



def eval_model(model):

    pipe = make_pipeline(RobustScaler(), model)

    cv_results = cross_validate(pipe, X, y.values.ravel(), cv=3, scoring=('balanced_accuracy',

                                                                           'recall'))

    

    print("=== Mean Test Results for {} ===".format(type(model).__name__))

    print("BALANCED ACCURACY: {:.3f}".format(cv_results.get('test_balanced_accuracy').mean()))

    print("RECALL: {:.3f}".format(cv_results.get('test_recall').mean()))

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 

                                                    test_size=0.1, random_state=0,

                                                    stratify=y)



tuned_parameters = {

    'n_neighbors': [3, 5, 11, 19],

    'weights': ['uniform', 'distance'],

    'metric': ['euclidean', 'manhattan']

}



rob_scal = RobustScaler()

rob_scal.fit(X_train)

X_train = rob_scal.transform(X_train)



clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=3, scoring='balanced_accuracy')

clf.fit(X_train, y_train)



means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']



for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
from sklearn.metrics import balanced_accuracy_score



knn_tuning = KNeighborsClassifier(3, weights='distance', metric='euclidean')

knn_tuning.fit(X_train, y_train)

X_test = rob_scal.transform(X_test)

knn_tuning_preds = knn_tuning.predict(X_test)



print(balanced_accuracy_score(y_test, knn_tuning_preds))
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

 

gnb = GaussianNB()

logit = LogisticRegression(solver='liblinear', class_weight='balanced')

knn = KNeighborsClassifier(3, weights='distance', metric='euclidean')

xgb = XGBClassifier()



classifiers = [gnb, logit, knn, xgb]



for classifier in classifiers:

    eval_model(classifier)
def eval_model_pca(model):

    pipe = make_pipeline(RobustScaler(), PCA(3), model)

    cv_results = cross_validate(pipe, X, y.values.ravel(), cv=3, scoring=('balanced_accuracy',

                                                                           'recall'))

    

    print("=== Mean Test Results for {} ===".format(type(model).__name__))

    print("BALANCED ACCURACY: {:.3f}".format(cv_results.get('test_balanced_accuracy').mean()))

    print("RECALL: {:.3f}".format(cv_results.get('test_recall').mean()))

classifiers = [gnb, logit, knn, xgb]



for classifier in classifiers:

    eval_model_pca(classifier)