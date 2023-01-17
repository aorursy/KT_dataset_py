import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

# For plot marker colours

import colorlover as cl

from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import cross_val_score

from scipy import stats

from sklearn import metrics
sc = pd.read_csv("/kaggle/input/skillcraft/SkillCraft.csv")

sc.head()
print(f"Dimension of the data set is{sc.shape}\n")

print(f"Data Types are:")

print(sc.dtypes)
sc.drop(columns=["Age", "HoursPerWeek", "TotalHours"], inplace=True)
print("Number of missing value for each feature:")

print(sc.isnull().sum())
sc.describe()
%matplotlib inline



# Name the leagues

league_lbls = ["Bronze","Silver","Gold","Platinum","Diamond","Master","Grandmaster"]

league_indexs = sc["LeagueIndex"].unique()

league_indexs.sort()

league_lbls_dict = dict()

for i, ind in enumerate(league_indexs):

    league_lbls_dict[ind] = league_lbls[i]

league_labeled = sc["LeagueIndex"].replace(league_lbls_dict)





def clrgb_to_hex(rgb):

    rgb = re.search("\(([^\)]+)\)", rgb).group(1).split(",")

    hex_clr = "#"

    for n in rgb:

        val = hex(int(n))[2:]

        if len(val)<2:

            val = "0"+val

        hex_clr+=val

    return hex_clr





# Define league colours for consistency

league_colours_raw = cl.scales['8']['qual']['Paired']

league_colours = []

for i, clr in enumerate(league_colours_raw):

    league_colours.append(clrgb_to_hex(league_colours_raw[i]))



league_colours_dict = dict()

for i, lbl in enumerate(league_lbls):

    league_colours_dict[lbl] = league_colours[i]

    

    

def box_hist_plot(x, title, w, h):

    fig, (ax_box, ax_hist)= plt.subplots(2, sharex=True,gridspec_kw={"height_ratios": (.15,.85)})

    fig.set_size_inches(w, h)

    

    ax_box.set_xlim(0,x.max())

    ax_hist.set_xlim(0,x.max())

    

    sns.boxplot(x, ax=ax_box)

    sns.distplot(x, ax=ax_hist)

    ax_box.set(yticks=[])

    

    sns.despine(ax=ax_hist)

    sns.despine(ax=ax_box, left=True)

    ax_box.set_title(title)

    ax_hist.set_title(None)

    plt.show()

    

def violin_plot(y, title, w, h):

    plt.figure(figsize=(w, h))

    ax1 = sns.violinplot(x=league_labeled, y=y, palette=league_colours_dict, order=league_lbls)

    ax1.set_ylim(0,)

    ax1.set(xlabel='League')

    plt.title(title)

    plt.show()

    

def auto_plot(feature, fig_num):

    box_hist_plot(sc[feature], f"Figure {fig_num}: {feature} Distribution", 11, 8)

    violin_plot(sc[feature], f"Figure {fig_num+1}: {feature} by League", 11, 8)
def league_dist():

    global fig_count

    

    #labels

    lab = league_lbls

    #values: counts for each category

    val = sc["LeagueIndex"].value_counts().sort_index().values.tolist()

    pct = [x/sum(val)for x in val]

    

    fig1, ax1 = plt.subplots()

    #ax1.pie(val, labels=lab, autopct='%1.2f%%', pctdistance=0.8,shadow=True, startangle=90)

    

    wedges, texts = ax1.pie(pct, wedgeprops=dict(width=0.5), startangle=90, colors=league_colours)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")

    

    for i, p in enumerate(wedges):

        ang = (p.theta2 - p.theta1)/2. + p.theta1

        y = np.sin(np.deg2rad(ang))

        x = np.cos(np.deg2rad(ang))

        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

        connectionstyle = "angle,angleA=0,angleB={}".format(ang)

        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        ax1.annotate(f"{val[i]} {league_lbls[i]} ({pct[i]*100:.2f}%)", xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)

    

    ax1.set_title("Figure 1: Distribution of Leagues", y=1.2)

    

league_dist()
auto_plot("APM",2)
auto_plot("SelectByHotkeys",4)
auto_plot("AssignToHotkeys",6)
auto_plot("UniqueHotkeys",8)
auto_plot("MinimapAttacks",10)

auto_plot("MinimapRightClicks",12)
auto_plot("NumberOfPACs",14)

auto_plot("GapBetweenPACs",16)

auto_plot("ActionLatency",18)

auto_plot("ActionsInPAC",20)
auto_plot("TotalMapExplored",22)
auto_plot("WorkersMade",24)
auto_plot("UniqueUnitsMade",26)

auto_plot("ComplexUnitsMade",28)

auto_plot("ComplexAbilitiesUsed",30)
leagues = sc.LeagueIndex

sc_data = sc.drop(columns='LeagueIndex')

leagues.value_counts()
# Reencode

leagues.replace({2:1, 3:2, 4:3, 5:4, 6:5, 7:5, 8:5}, inplace=True)

# New Value Counts

leagues.value_counts()
def normalize_data(data):

    scaler = preprocessing.MinMaxScaler()

    data_norm = scaler.fit_transform(data)

    

    # When the data has been transformed, a np.array is returned,

    # So we have to convert it back to a dataframe, and insert column names

    return pd.DataFrame(data_norm, columns=data.columns)

    

sc_norm = normalize_data(sc_data)

sc_norm.describe()
def plot_scores():

    def get_k_best(data, target, method, k):

        skb = SelectKBest(method, k = k)

        skb.fit(data.values, target.values)

        fs_indices = np.argsort(skb.scores_)[::-1]



        return pd.DataFrame({"features": data.columns[fs_indices].values, 

                      "scores": skb.scores_[fs_indices]})

    

    fig, axs = plt.subplots(ncols=3, figsize=(20,6))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

    

    titles = []

    data = []

    titles.append("ANOVA F-value")

    data.append(get_k_best(sc_norm, leagues, f_classif, len(sc_norm.columns)))

    titles.append("Mutual Information")

    data.append(get_k_best(sc_norm, leagues, mutual_info_classif, len(sc_norm.columns)))

    titles.append("Chi-squared")

    data.append(get_k_best(sc_norm, leagues, chi2, len(sc_norm.columns)))

    

    for i in range(3):

        p = sns.barplot(x='features', y='scores', data=data[i], ax=axs[i])

        p.set_xticklabels(p.get_xticklabels(), rotation=90)

        p.set_title(titles[i])

        



plot_scores()
sc_train, sc_test, leagues_train, leagues_test = train_test_split(sc_norm, leagues, 

                                                                  test_size = 0.3, random_state=1,

                                                                  stratify = leagues)



print(f"Training dataset shape: {sc_train.shape}")

print(f"Test dataset shape: {sc_test.shape}")

print(f"Training target shape: {leagues_train.shape}")

print(f"Test target shape: {leagues_test.shape}")
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV



cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
def run_KNN_pipe(n_neighbours, p):

    pipe_KNN = Pipeline([('selector', SelectKBest()), 

                         ('knn', KNeighborsClassifier())])



    params_pipe_KNN = {'selector__score_func': [f_classif, mutual_info_classif, chi2],

                       'selector__k': [3, 4, 5, 6, 7, 10, sc_norm.shape[1]],

                       'knn__n_neighbors': n_neighbours,

                       'knn__p': p}



    gs_pipe_KNN = GridSearchCV(estimator=pipe_KNN, 

                               param_grid=params_pipe_KNN, 

                               cv=cv,

                               n_jobs = -1,

                               scoring='accuracy',

                               verbose=0)



    gs_pipe_KNN.fit(sc_train, leagues_train);

    

    return gs_pipe_KNN





gs_pipe_KNN = run_KNN_pipe([150, 160 ,170 ,180 ,190, 200], [1,2,5])
gs_pipe_KNN.best_params_
gs_pipe_KNN.best_score_
# custom function to format the search results as a Pandas data frame

def get_search_results(gs):



    def model_result(scores, params):

        scores = {'mean_score': np.mean(scores),

             'std_score': np.std(scores),

             'min_score': np.min(scores),

             'max_score': np.max(scores)}

        return pd.Series({**params,**scores})



    models = []

    scores = []



    for i in range(gs.n_splits_):

        key = f"split{i}_test_score"

        r = gs.cv_results_[key]        

        scores.append(r.reshape(-1,1))



    all_scores = np.hstack(scores)

    for p, s in zip(gs.cv_results_['params'], all_scores):

        models.append((model_result(s, p)))



    pipe_results = pd.concat(models, axis=1).T.sort_values(['mean_score'], ascending=False)



    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']

    columns = columns_first + [c for c in pipe_results.columns if c not in columns_first]



    return pipe_results[columns]



results_KNN = get_search_results(gs_pipe_KNN)

results_KNN.head(5)
def plot_KNN_results(res):

    def get_selector_data(d, p, sel):

        return d[(d.knn__p==p) & (d.selector__score_func==sel)].iloc[:,[0,4,6]]

    

    rows = len(res["knn__p"].unique())

    

    fig, axs = plt.subplots(ncols=3, nrows=rows, figsize=(20,rows*6), sharey='all')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    

    titles = []

    data = []

    temp = res.copy().infer_objects()

    for i, p in enumerate(res["knn__p"].unique()):

        titles.append(f"ANOVA F-value, p={p}")

        data.append(get_selector_data(temp, p, f_classif))

        titles.append(f"Mutual Information, p={p}")

        data.append(get_selector_data(temp, p, mutual_info_classif))

        titles.append(f"Chi-squared, p={p}")

        data.append(get_selector_data(temp, p, chi2))

    

    row = 0

    col = 0

    ax = None

    for i in range(rows*3):

        if col%3==0 and col!=0:

            col=0

            row+=1

        if rows == 1:

            ax = axs[col]

        else:

            ax=axs[row,col]

        p = sns.lineplot(x='knn__n_neighbors', 

                         y='mean_score', 

                         hue='selector__k', 

                         data=data[i], 

                         ax=ax, 

                         palette=sns.color_palette("Set1", 7))

        p.set_title(titles[i])

        p.legend(loc='lower right')

        col+=1

        

plot_KNN_results(results_KNN)
gs_pipe_KNN2 = run_KNN_pipe([210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350], [1])
gs_pipe_KNN2.best_params_
gs_pipe_KNN2.best_score_
results_KNN2 = get_search_results(gs_pipe_KNN2)

results_KNN2.head(5)
plot_KNN_results(results_KNN2)
def run_dt_pipe(max_depth, min_split):

    pipe_DT = Pipeline([('selector', SelectKBest()), 

                         ('dt', DecisionTreeClassifier(criterion='gini'))])



    params_pipe_DT = {'selector__score_func': [f_classif, mutual_info_classif, chi2],

                       'selector__k': [3, 4, 5, 6, 7, 10, sc_norm.shape[1]],

                       'dt__max_depth': max_depth,

                       'dt__min_samples_split': min_split}

 

    gs_pipe_DT = GridSearchCV(estimator=pipe_DT, 

                               param_grid=params_pipe_DT, 

                               cv=cv,

                               n_jobs = -1,

                               scoring='accuracy',

                               verbose=0)



    gs_pipe_DT.fit(sc_train, leagues_train);

    

    return gs_pipe_DT





gs_pipe_DT = run_dt_pipe([5, 7, 9], [2, 3, 5, 7, 9, 11])
gs_pipe_DT.best_params_
gs_pipe_DT.best_score_
results_DT = get_search_results(gs_pipe_DT)

results_DT.head(5)
def plot_DT_results(res):

    def get_selector_data(d, p, sel):

        return d[(d.dt__max_depth==p) & (d.selector__score_func==sel)].iloc[:,[0,5,6]]

    

    rows = len(res["dt__max_depth"].unique())

    

    fig, axs = plt.subplots(ncols=3, nrows=rows, figsize=(20,rows*6), sharey='all')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    

    titles = []

    data = []

    temp = res.copy().infer_objects()

    for i, p in enumerate(res["dt__max_depth"].unique()):

        titles.append(f"ANOVA F-value, max_depth={p}")

        data.append(get_selector_data(temp, p, f_classif))

        titles.append(f"Mutual Information, max_depth={p}")

        data.append(get_selector_data(temp, p, mutual_info_classif))

        titles.append(f"Chi-squared, max_depth={p}")

        data.append(get_selector_data(temp, p, chi2))

    

    row = 0

    col = 0

    ax = None

    for i in range(rows*3):

        if col%3==0 and col!=0:

            col=0

            row+=1

        if rows == 1:

            ax = axs[col]

        else:

            ax=axs[row,col]

        p = sns.lineplot(x='dt__min_samples_split', 

                         y='mean_score', 

                         hue='selector__k', 

                         data=data[i], 

                         ax=ax, 

                         palette=sns.color_palette("Set1", 7))

        p.set_title(titles[i])

        p.legend(loc='lower right')

        col+=1



        

plot_DT_results(results_DT)
def compare_depths():

    def get_selector_data(d, p, sel):

        return d[(d.dt__min_samples_split==p) & (d.selector__score_func==sel)].iloc[:,[0,4,6]]

    

    pipe_DT = Pipeline([('selector', SelectKBest()), 

                         ('dt', DecisionTreeClassifier(criterion='gini'))])



    params_pipe_DT = {'selector__score_func': [f_classif],

                       'selector__k': [3, 4, 5],

                       'dt__max_depth': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],

                       'dt__min_samples_split': [2]}

 

    gs_pipe_DT = GridSearchCV(estimator=pipe_DT, 

                               param_grid=params_pipe_DT, 

                               cv=cv,

                               n_jobs = -1,

                               scoring='accuracy',

                               verbose=0)



    res = get_search_results(gs_pipe_DT.fit(sc_train, leagues_train));

    res = res.infer_objects()

    p = sns.lineplot(x='dt__max_depth', 

                     y='mean_score', 

                     hue='selector__k', 

                     data=get_selector_data(res,2,f_classif), 

                     palette=sns.color_palette("Set1", 3))

    #p.set_title(titles[i])

    p.legend(loc='lower right')

    

compare_depths()
gs_pipe_DT2 = run_dt_pipe([5], [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201])
gs_pipe_DT2.best_params_
gs_pipe_DT2.best_score_
results_DT2 = get_search_results(gs_pipe_DT2)

results_DT2.head(5)
plot_DT_results(results_DT2)
np.random.seed(1)



def run_NB_pipe(var_smoothing):

    pipe_NB = Pipeline([('selector', SelectKBest()), 

                         ('nb', GaussianNB())])



    params_pipe_NB = {'selector__score_func': [f_classif, mutual_info_classif],

                       'selector__k': [3, 4, 5, 6, 7, 10, sc_norm.shape[1]],

                       'nb__var_smoothing': var_smoothing}



    gs_pipe_NB = GridSearchCV(estimator=pipe_NB, 

                               param_grid=params_pipe_NB, 

                               cv=cv,

                               n_jobs = -1,

                               scoring='accuracy',

                               verbose=0)

    

    sc_train_transformed = PowerTransformer().fit_transform(sc_train)

    gs_pipe_NB.fit(sc_train_transformed, leagues_train);

    

    return gs_pipe_NB





gs_pipe_NB = run_NB_pipe(np.logspace(2,-2, num=100))
gs_pipe_NB.best_params_
gs_pipe_NB.best_score_
results_NB = get_search_results(gs_pipe_NB)

results_NB.head(5)
def plot_NB_results(res):

    def get_selector_data(d, sel):

        return d[(d.selector__score_func==sel)].iloc[:,[0,4,5]]

    

    fig, axs = plt.subplots(ncols=2, figsize=(20,6), sharey='all')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    

    

    titles = []

    data = []

    temp = res.copy().infer_objects()

    

    titles.append(f"ANOVA F-value")

    data.append(get_selector_data(temp, f_classif))

    titles.append(f"Mutual Information")

    data.append(get_selector_data(temp, mutual_info_classif))

    

    for i in range(2):

        p = sns.lineplot(x='nb__var_smoothing', 

                         y='mean_score', 

                         hue='selector__k', 

                         data=data[i], 

                         ax=axs[i], 

                         palette=sns.color_palette("Set1", 7))

        p.set_xscale("log")

        p.set_title(titles[i])

        p.legend(loc='lower right')





plot_NB_results(results_NB)
cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2)



cv_results_KNN = cross_val_score(estimator=gs_pipe_KNN2.best_estimator_,

                                 X=sc_test,

                                 y=leagues_test, 

                                 cv=cv2, 

                                 n_jobs=-1,

                                 scoring='accuracy')

cv_results_KNN.mean()
cv_results_DT = cross_val_score(estimator=gs_pipe_DT2.best_estimator_,

                                X=sc_test,

                                y=leagues_test, 

                                cv=cv2, 

                                n_jobs=-1,

                                scoring='accuracy')

cv_results_DT.mean()
sc_test_transformed = PowerTransformer().fit_transform(sc_test)



cv_results_NB = cross_val_score(estimator=gs_pipe_NB.best_estimator_,

                                X=sc_test_transformed,

                                y=leagues_test, 

                                cv=cv2, 

                                n_jobs=-1,

                                scoring='accuracy')

cv_results_NB.mean()
print(stats.ttest_rel(cv_results_KNN, cv_results_DT))

print(stats.ttest_rel(cv_results_KNN, cv_results_NB))

print(stats.ttest_rel(cv_results_DT, cv_results_NB))
pred_KNN = gs_pipe_KNN.predict(sc_test)



pred_DT = gs_pipe_DT2.predict(sc_test)



sc_test_transformed = PowerTransformer().fit_transform(sc_test)

pred_NB = gs_pipe_NB.predict(sc_test_transformed)



print("\nK-Nearest Neighbour Report") 

print(metrics.classification_report(leagues_test, pred_KNN))

print("\nDecision Tree Report") 

print(metrics.classification_report(leagues_test, pred_DT))

print("\nNaive Bayes Report") 

print(metrics.classification_report(leagues_test, pred_NB))
print("\nConfusion matrix for K-Nearest Neighbour") 

print(metrics.confusion_matrix(leagues_test, pred_KNN))

print("\nConfusion matrix for Decision Tree") 

print(metrics.confusion_matrix(leagues_test, pred_DT))

print("\nConfusion matrix for Naive Bayes") 

print(metrics.confusion_matrix(leagues_test, pred_NB))