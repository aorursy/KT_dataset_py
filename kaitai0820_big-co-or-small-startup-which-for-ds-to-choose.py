# dependency installation



!pip install pywaffle

!pip install hdbscan
import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from pywaffle import Waffle

from hdbscan import HDBSCAN

from sklearn.manifold import TSNE 

from sklearn.decomposition import PCA

from scipy.stats import chi2_contingency



pd.options.display.max_columns = 300



%matplotlib inline
path = "/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv"

kaggle_df = pd.read_csv(path, low_memory=False)
# load subset for data scientists

ds_df = kaggle_df.query("Q5 == 'Data Scientist'").iloc[1:, :]
fig, ax = plt.subplots(figsize=(12, 6))



counts = ds_df["Q6"].value_counts()



cmap = plt.get_cmap("tab20c")

ax.barh(counts[:2].index, counts[:2].values, color=cmap([0]))

ax.barh(counts[2:].index, counts[2:].values, color=cmap([19]))



ax.axvline(ds_df.shape[0] / ds_df["Q6"].nunique(), color="r")

ax.annotate("Average number per group", xy=(820, 3.5), 

            xytext=(900, 4), arrowprops=dict(facecolor='black', shrink=0.01), fontsize=13)

ax.set_xlabel("Number of participants per group", fontsize=12)

plt.title("Average Number of Data Scientist Respondents per Group", fontsize=14);
TARGET_COMPANIES = ["0-49 employees", "> 10,000 employees"]

df = ds_df[ds_df["Q6"].isin(TARGET_COMPANIES)]



[STARTUP_NUM, BIG_CORP_NUM] = df.groupby("Q6").size()
# Make a generic function used for transforming survey dataset.

def _normalized_value_counts(df: pd.DataFrame, col: str, reindexed=False) -> pd.DataFrame:

    """Reusable function to get normalized value counts.

    

    Args:

        df: dataset

        col: column name for value count

        reindexed: False or List, default is False. If a list is assigned, will use it in

            reindexing.

            

    Returns:

        DataFrame

    """

    sub_df = df.groupby("Q6")[col].value_counts(normalize=True).unstack("Q6")

    if reindexed:

        return sub_df.reindex(reindexed)

    return sub_df
# Make a customized multi bar plot function to reduce repeated code.

def customized_multi_bar_plot(df: pd.DataFrame, col: str, xticks: list, title: str):

    """Customized multi bar plot with several options for customization."""

    df_bar = _normalized_value_counts(df, col=col)

    (

        df_bar

        .apply(lambda x: x.round(3) * 100)

        .reindex(xticks)

        .plot(kind='bar', colormap="tab20c", figsize=(12, 8))

    )



    plt.xticks(rotation=60)

    plt.grid(axis='y', color="black", alpha=0.5)

    plt.ylabel("Percentage across the group (%)")

    plt.title(title, fontsize=14);
# Make a customized barh plot

def customized_barh(df: pd.DataFrame, ordered_cols: list, colors: list, legend_loc: tuple, title: str):

    """Customized barh plot with several options."""

    fig, ax = plt.subplots(figsize=(20, 2))

    

    for idx, col in enumerate(TARGET_COMPANIES):

        bar_start = 0

        for c, order in zip(colors, ordered_cols):

            value = df.loc[order, col]

            ax.barh(y=col, width=value, height=0.6, left=bar_start, color=c)

            plt.text(bar_start + value/2 - 0.01, idx - 0.1, "{:.0%}".format(value), fontsize=12)

            bar_start += value

    

    for spine in plt.gca().spines.values():

        spine.set_visible(False)



    ax.get_xaxis().set_visible(False)

    ax.tick_params(axis='y', labelsize=14)

    ax.set_facecolor("white")

    leg = ax.legend(ordered_cols, loc=legend_loc, fontsize=14)

    plt.title(title, fontsize=18);
def _transform_df_q2(df: pd.DataFrame, col: str, col_name: str) -> pd.DataFrame:

    """Internal dataframe transformation method for data preparation.

    

    Args:

        df: dataset

        col: size of company, either "0-49 employees" or "> 10,000 employees"

    """

    group_size = BIG_CORP_NUM

    if col == "0-49 employees":

        group_size = STARTUP_NUM



    sub_df = (

        df[df["Q6"] == col]

            .groupby("Q1")["Q2"]

            .value_counts()

            .unstack("Q2")

            .fillna(0)

            .apply(lambda x: (x / group_size).round(3) * 100)

    )

    sub_df.columns.name = col_name

    

    return sub_df
def concat_df_bar_plot(df: pd.DataFrame, s: pd.Series):

    """Concatenate data and visualize bar plot after certain data cleaning."""

    concat_df = _normalized_value_counts(pd.concat([df['Q6'], s], axis=1), col="sumed")

    (

        concat_df

        .round(3)

        .fillna(0) * 100 

    ).plot(kind='bar', figsize=(20, 5), cmap="tab20c")
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
q4 = _normalized_value_counts(df, col="Q4")

cats = q4.index



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12), dpi=120)



q4.plot(kind="pie", colormap="plasma", labels=None, subplots=True, legend=False,

        ax=ax, autopct=lambda x: "{:.1%}".format(x/100),pctdistance=1.15,

        wedgeprops=dict(width=0.3, edgecolor='white'))



plt.suptitle("Q4: What is the highest level of formal education you have attained", y=0.7)

plt.text(-3.4, 0, f"{STARTUP_NUM} Respondents")

plt.text(-0.35, -0, f"{BIG_CORP_NUM} Respondents")

plt.legend(cats, loc='lower right', bbox_to_anchor=(0, -0.4, 0.5, 1));
# Question 15

year_idx = ["I have never written code", "< 1 years", "1-2 years", "3-5 years", 

            "5-10 years", "10-20 years", "20+ years"]



customized_multi_bar_plot(df, col="Q15", xticks=year_idx, 

                          title="Q15: How long have you been writing code to analyze data")



# Question 23

year_idx = ["< 1 years", "1-2 years", "2-3 years", "3-4 years", "4-5 years", 

            "5-10 years", "10-15 years", "20+ years"]



customized_multi_bar_plot(df, col="Q23", xticks=year_idx, 

                          title="Q23: For long many years have you used ML methods")
# Question 14

q14 = _normalized_value_counts(df, "Q14")

cats = q14.index

explode=[0.2, 0, 0, 0,0, 0]

cmap = plt.get_cmap("tab20c")

colors = cmap(np.array([0, 3, 7, 8, 13, 18]))



fig, ax = plt.subplots(figsize=(12, 12), dpi=120)



q14.plot(kind='pie', subplots=True, ax=ax, legend=False,

         figsize=(12, 12),

         colors=colors, 

         explode=[0.2, 0, 0, 0, 0, 0],

         autopct=lambda x: "{:.1%}".format(x / 100), labels=None)



plt.legend(cats, loc='lower right', bbox_to_anchor=(0, -0.4, 0.5, 1))

plt.suptitle("Q14: What is primary tool that you use at work to analyze data", y=0.7);
# Question 18

# Data prepration

q18 = df.filter(regex="Q18_Part").dropna(axis=1, how="all").dropna(axis=0, how="all")

q18.columns = [q18[col].dropna().unique()[0] for col in q18]

sum_q18 = pd.Series(q18.notnull().astype(int).sum(axis=1), name="sumed")



# First bar plot

concat_df_bar_plot(df, sum_q18)



for spine in plt.gca().spines.values():

    spine.set_alpha(0.3)

plt.title("Q18: How many programming languages do you use at work", fontsize=16)

plt.grid(axis="y", color="black", alpha=0.5)

plt.xlabel("Number of programming language used at work", fontsize=12)

plt.ylabel("Percentage - %", fontsize=12);



# Heatmap plot

concat_df = pd.concat([df['Q6'], q18.notnull().astype(int)], axis=1)

transformed_df = (concat_df.groupby("Q6").sum().div([STARTUP_NUM, BIG_CORP_NUM], axis=0))

    

plt.figure(figsize=(20, 5))

sns.heatmap(transformed_df, cmap="RdPu")

plt.title("What programming language you use at work", fontsize=14);
questions = [24, 25, 28, 29]

names = ["Q24: ML algorithms", "Q25: ML tools", "Q28: ML frameworks", "Q29: cloud platforms"]



for n, name in zip(questions, names):

    sub_df = df.filter(regex=f"Q{n}_Part").dropna(axis=1, how="all")

    cats = [sub_df[col].dropna().unique()[0] for col in sub_df]

    sub_df.columns = cats



    concat_df = pd.concat([df["Q6"], sub_df.notnull().astype(int)],

                          axis=1).groupby("Q6")[cats].sum().T



    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 10))

    

    i =1

    for col in concat_df:

        txt = ",".join(concat_df[col].sort_values(ascending=False).index)

        plt.subplot(2, 2, i)

        wc = WordCloud(colormap="tab20c", width=1600, height=900).generate(txt)

        plt.imshow(wc, interpolation='bilinear')

        plt.title(col + "\n" + f"{name}")

        plt.axis("off")

        i += 1
startup_df = _transform_df_q2(df, col="0-49 employees", col_name="Q2 (Start-up)")

big_corp_df = _transform_df_q2(df, col="> 10,000 employees", col_name="Q2 (Big Corp)")



kwargs = {

    "kind": "bar",

    "figsize": (12, 8),

    "legend": True,

    "ylim": (0, 30),

    "stacked": True,

    "cmap": "tab20c"

}



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)



big_corp_df.plot(ax=ax1, 

                 title="Age and gender distribution comparison between big corporations and startups", 

                 **kwargs)

startup_df.plot(ax=ax2, **kwargs)



ax1.grid(axis='y')

ax2.grid(axis='y')

ax2.invert_yaxis()



plt.tight_layout()

plt.subplots_adjust(hspace=0)

plt.xticks(rotation=60)

plt.show()
# Question 7

year_idx = ["0", "1-2", "3-4", "5-9", "10-14", "15-19", "20+"]



customized_multi_bar_plot(df, col="Q7", xticks=year_idx, 

                          title="Q7: Approximately how many individuals are responsible for data science workloads")
# Question 10

ordered_cols = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999','4,000-4,999',

                '5,000-7,499', '7,500-9,999', '10,000-14,999','15,000-19,999',

                '20,000-24,999', '25,000-29,999', '30,000-39,999', '40,000-49,999', 

                '50,000-59,999', '60,000-69,999', '70,000-79,999', '80,000-89,999', 

                '90,000-99,999', '100,000-124,999', '125,000-149,999', '150,000-199,999', 

                '200,000-249,999', '250,000-299,999', '300,000-500,000', '> $500,000']



q10 = _normalized_value_counts(df, "Q10", reindexed=ordered_cols)

q10.index.name = "Annual Salary($)"

q10.style.format("{:.0%}").background_gradient(cmap="RdPu", axis=0)
# Question 11

q11 = _normalized_value_counts(df, "Q11")

cmap = plt.get_cmap("tab20c")

colors = cmap(np.array([0, 3, 7, 8, 13, 18]))

ordered_cols = ['$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999', '> $100,000 ($USD)']



customized_barh(q11, 

                colors=colors, 

                ordered_cols=ordered_cols, 

                legend_loc=(0.8, -1.35), 

                title="Q11: Approximately how much money have you spent on learning ML/cloud products at work")
# Question 8

q8 = _normalized_value_counts(df, col="Q8")

cmap = plt.get_cmap("tab20c")

colors = cmap(np.array([0, 3, 7, 8, 13, 18]))

ordered_cols = ['I do not know', 'No (we do not use ML methods)',

 'We are exploring ML methods (and may one day put a model into production)',

 'We use ML methods for generating insights (but do not put working models into production)',

 'We recently started using ML methods (i.e., models in production for less than 2 years)',

 'We have well established ML methods (i.e., models in production for more than 2 years)']



customized_barh(q8, 

                ordered_cols=ordered_cols,

                colors=colors,

                legend_loc=(0.325, -1.3), 

                title="Q8: Does your current employer incorporate machine learning methods into buisiness")

# Question 9

q9 = df.filter(regex="Q9_Part").dropna(axis=1, how="all").dropna(axis=0, how="all")

q9.columns = [q9[col].dropna().unique()[0] for col in q9]

sum_q9 = pd.Series(q9.notnull().astype(int).sum(axis=1), name="sumed")



# bar plot

concat_df_bar_plot(df, sum_q9)



for spine in plt.gca().spines.values():

    spine.set_alpha(0.3)

plt.title("Q9: How many activities that make up at your work", fontsize=16)

plt.grid(axis='y', color="black", alpha=0.5)

plt.xlabel("Number of activities involved at work", fontsize=14)

plt.ylabel("Percentage - %", fontsize=14);





q9 = df.filter(regex="Q9_Part_")

q9_cols = [q9[col].dropna().unique()[0] for col in q9]

q9 = q9.applymap(lambda x: 0 if x is np.nan else 1)

q9.columns = q9_cols



q9_agg = (

    pd.concat([df["Q6"], q9], axis=1)

    .groupby("Q6")[q9_cols]

    .sum()

    .div([STARTUP_NUM, BIG_CORP_NUM], axis=0)

)

# pandas dataframe styler

cm = sns.light_palette("blue", as_cmap=True)

q9_agg.style.format("{:.1%}").background_gradient(cmap="RdPu", axis=1)
# Question 21

q21 = df.filter(regex="Q21_Part").dropna(axis=1, how="all").dropna(axis=0, how="all")

q21.columns = [q21[col].dropna().unique()[0] for col in q21]

sum_q21 = pd.Series(q21.notnull().astype(int).sum(axis=1), name="sumed")



# bar plot

concat_df_bar_plot(df, sum_q21)



for spine in plt.gca().spines.values():

    spine.set_alpha(0.3)

plt.title("How many types of hardware do you use on a regular basis", fontsize=16)

plt.xlabel("Number of different types of harware used at work", fontsize=12)

plt.grid(axis="y", color="black", alpha=0.5)

plt.ylabel("Percentage - %", fontsize=12);





# Heatmap plot

concat_df = pd.concat([df['Q6'], q21.notnull().astype(int)], axis=1)

transformed_df = (concat_df.groupby("Q6").sum().div([STARTUP_NUM, BIG_CORP_NUM], axis=0))

   

plt.figure(figsize=(20, 5))

sns.heatmap(transformed_df, cmap="RdPu")

plt.title("Q21: Which types of spcialized hardware do you use on a regular basis", fontsize=14);
reordered_cols = ['Never', 'Once', '2-5 times', '6-24 times', '> 25 times']

q22 = _normalized_value_counts(df, col="Q22", reindexed=reordered_cols)

q22 = q22.applymap(lambda x: round(x * 100))



n_cat = q22.shape[0]

colors = [plt.cm.tab20c(i/n_cat) for i in range(n_cat)]



fig = plt.figure(

    FigureClass=Waffle,

    rows=5,

    plots={

        "211": {

            'values': q22['0-49 employees'].to_dict(),

            'title': {"label": "0-49 employees", "loc": "right"}

        },

        "212": {

            'values': q22['> 10,000 employees'].to_dict(),

            'title': {"label": "> 10,000 employees", "loc": "right"}

        }

    },

    colors=colors,

    figsize=(12, 10),

    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'ncol': n_cat, 'framealpha': 0}

)

plt.suptitle("Q22: Have you ever used a TPU", x=0.5, y=0.9, fontsize=14);
def filter_df(df: pd.DataFrame, col: str) -> pd.DataFrame:

    """Filter the needed columns and binarize them."""

    df = df.copy()

    sub_df = df.filter(regex=f"{col}_Part")

    return sub_df.applymap(lambda x: 1 if x is not np.nan else 0)
# Binarize the columns

q9 = filter_df(df, "Q9")

q18 = filter_df(df, "Q18")

q21 = filter_df(df, "Q21")



# For those columns, only get_dummies is enough

cols = ["Q1", "Q2", "Q4", "Q6", "Q7", "Q8", "Q10", "Q11", "Q14", "Q15", "Q19", "Q22", "Q23"]

df_cats = pd.get_dummies(df[cols], prefix=cols)



# Concatenate them into dataset as input of clustering

df_cluster = pd.concat([df_cats, q9, q18, q21], axis=1)



# Dimension reduction

pca_50 = PCA(n_components=50, random_state=2019).fit_transform(df_cluster)

tsne_output = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, 

                   random_state=42).fit_transform(pca_50)
clustered = HDBSCAN(min_cluster_size=500, min_samples=2).fit(tsne_output)

fig = plt.figure(figsize=(16, 10))

plt.title("New segment of data Scientist respondents from startups and big firms", fontsize=16)

sns.scatterplot(tsne_output[:, 0], tsne_output[:, 1], hue=clustered.labels_, legend='full', 

                palette=sns.color_palette('hls', len(np.unique(clustered.labels_))), alpha=0.5)

plt.xlabel("Decomposed Dimension 1", fontsize=14)

plt.ylabel("Decomposed Dimension 2", fontsize=14);
fig = plt.figure(figsize=(16, 10))

plt.title("Data Scientist respondents cluster based on the size of their company", fontsize=16)

sns.scatterplot(tsne_output[:, 0], tsne_output[:, 1], hue=df["Q6"], legend='full', 

                palette=sns.color_palette('hls', 2), alpha=0.5)

plt.xlabel("Decomposed Dimension 1", fontsize=14)

plt.ylabel("Decomposed Dimension 2", fontsize=14);
fig, ax = plt.subplots(figsize=(12, 8))



cols = ["Q1", "Q2", "Q4", "Q7", "Q8", "Q10", "Q11", "Q14", "Q15", "Q19", "Q22", "Q23"]

ax.barh(cols, [cramers_v(df['Q6'], df[col]) for col in cols], color="#cccccc")

highs = ["Q7", "Q8", "Q10", "Q11"]

ax.barh(highs, [cramers_v(df['Q6'], df[col]) for col in highs], color="#5a9bf8")

ax.axvline(0.25, color="r")



plt.title("Correlation between company size and other features we use in analysis", fontsize=16)

plt.xlabel("Correlation index", fontsize=12)

plt.ylabel("Features selected in this analysis");