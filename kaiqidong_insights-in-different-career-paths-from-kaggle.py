import os
print(os.listdir("../input"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Any results you write to the current directory are saved as output.
mcr = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False).iloc[1:, :]
CONSIDERED_JOB_TITLE = ['Business Analyst', 'Chief Officer', 'Consultant',
       'DBA/Database Engineer', 'Data Analyst', 'Data Engineer',
       'Data Journalist', 'Data Scientist', 'Developer Advocate',
       'Manager', 'Marketing Analyst','Principal Investigator', 'Product/Project Manager',
       'Research Assistant', 'Research Scientist', 'Salesperson',
       'Software Engineer', 'Statistician']
mcr_jobs = mcr.loc[mcr["Q6"].isin(CONSIDERED_JOB_TITLE)]
plt.figure(figsize=(18, 10))
mcr_jobs["Q6"].value_counts().plot(kind='bar')
_ = plt.style.use('ggplot')
_ = plt.title("Number of respondents in each job title")
_ = plt.xlabel("Job title")
_ = plt.ylabel("Counts")
CODING_TIME = ['0% of my time', '1% to 25% of my time',
       '25% to 49% of my time','50% to 74% of my time', 
       '75% to 99% of my time', '100% of my time']

mcr_jobs[["Q6", "Q23"]]\
        .dropna().groupby(["Q6", "Q23"]).size()\
        .groupby("Q6").apply(lambda x: 100 * x / float(x.sum()))\
        .unstack(1)[CODING_TIME]\
        .plot(kind="bar", stacked=True, figsize=(18, 10), 
              fontsize=15, colormap=plt.get_cmap("tab20")
             )

plt.style.use("ggplot")
plt.title("Percentage of time on coding for different jobs", fontsize=20)
plt.ylabel("Percentage of Time on Coding", fontsize=15)
plt.xlabel("Job Titles", fontsize=15)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=14)
plt.show()
CODING_YEAR = ['I have never written code and I do not want to learn',
        'I have never written code but I want to learn', '< 1 year', '1-2 years', 
        '3-5 years', '5-10 years', '10-20 years', '20-30 years', 
        '30-40 years', '40+ years']
mcr_code_year = mcr_jobs[["Q6", "Q24"]]\
        .groupby(["Q6", "Q24"])\
        .size()\
        .to_frame("counts")\
        .unstack(1)\
        .fillna(0)
mcr_code_year = mcr_code_year["counts"][CODING_YEAR]
mcr_code_year = mcr_code_year.apply(lambda x: ((x / mcr_code_year.sum(axis=1)) * 100).round(0))
plt.figure(figsize=(20, 10))
sns.heatmap(mcr_code_year, annot=True)
_ = plt.title("Percentage of how many years of experience people have to use coding skill to analyze data for different jobs")
_ = plt.xlabel("Years of experience")
_ = plt.ylabel("Job titles")
ML_YEARS = ['I have never studied machine learning and I do not plan to',
        'I have never studied machine learning but plan to learn in the future',
        '< 1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', 
        '10-15 years', '20+ years']
mcr_ml_year = mcr_jobs[["Q6", "Q25"]]\
        .groupby(["Q6", "Q25"])\
        .size()\
        .to_frame("counts")\
        .unstack(1)\
        .fillna(0)
mcr_ml_year = mcr_ml_year["counts"][ML_YEARS]
# create features for ml and non-ml

no_ml = mcr_ml_year.iloc[:, :2].sum(axis=1).values
with_ml = mcr_ml_year.iloc[:, 2:].sum(axis=1).values
mcr_ml_year_bi = pd.DataFrame(index=CONSIDERED_JOB_TITLE)
mcr_ml_year_bi["No machine learning"] = no_ml
mcr_ml_year_bi["With machine learning"] = with_ml

mcr_ml_year_bi = mcr_ml_year_bi.apply(lambda x: ((x / mcr_ml_year_bi.sum(axis=1)) * 100).round(0))\
                            .stack()\
                            .reset_index()\
                            .rename(columns={"level_0": "Q6", "level_1": "Q25", 0: "percentage"})

sns.catplot(x="Q6", y='percentage', hue='Q25', height=12, data=mcr_ml_year_bi, kind='bar', palette="muted")
_ = plt.xticks(rotation=90)
_ = plt.title("Percentage of positions having machine learning experience")
mcr_ml_year_multi = mcr_ml_year.apply(lambda x: ((x / mcr_ml_year.sum(axis=1)) * 100).round(0))
plt.figure(figsize=(20, 10))
sns.heatmap(mcr_ml_year_multi, annot=True)
_ = plt.title("Percentage of people's ML experience in different job positions")
_ = plt.xlabel("Years of experience")
_ = plt.ylabel("Job titles")
DATA_TYPE_LIST = ['Audio Data', 'Categorical Data', 'Genetic Data',
       'Geospatial Data', 'Image Data', 'Numerical Data', 'Sensor Data',
       'Tabular Data', 'Text Data', 'Time Series Data', 'Video Data',
       'Other Data']
mcr_data_type = mcr_jobs.filter(regex="^(?!.*TEXT)(Q31)").fillna(0)
ori_list = mcr_data_type.columns.tolist()

for col in mcr_data_type:
    mcr_data_type[col] = mcr_data_type[col].apply(lambda x: 1 if x!=0 else 0)
    
mcr_data_type = mcr_data_type.rename(columns=dict(zip(ori_list, DATA_TYPE_LIST)))
mcr_data_type = mcr_data_type.merge(mcr_jobs[["Q6"]], left_index=True, right_index=True)
sum_datatype = mcr_data_type.groupby("Q6")[DATA_TYPE_LIST].sum().T
fig, ax = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(20, 30))
n = 1
for job in sum_datatype:
    plt.subplot(6, 3, n)
    sum_datatype[job].plot(kind="bar", title=job, fontsize=8)
    n +=1
plt.subplots_adjust(hspace = 1)
plt.show()
ACTIVITIES = ["Understand data", "Build service", 
              "Build infrastructure", "Build prototype", 
              "Advanced research", "Not related", "Other"]

mcr_activity = mcr_jobs.filter(regex="^(?!.*TEXT)(Q11)").fillna(0)
activity_list = mcr_activity.columns.tolist()
mcr_activity = mcr_activity.rename(columns=dict(zip(activity_list, ACTIVITIES)))
for col in mcr_activity:
    mcr_activity[col] = mcr_activity[col].apply(lambda x: 1 if x!=0 else 0)
mcr_activity = mcr_activity.merge(mcr_jobs[["Q6"]], left_index=True, right_index=True)
transposed_activities = mcr_activity.groupby("Q6")[ACTIVITIES].sum().T
full_activity = ["Analyze and understand data to influence product or business decisions",
                "Build and/or run a machine learning service that operationally improves my product or workflows",
                "Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data",
                "Build prototypes to explore applying machine learning to new areas",
                "Do research that advances the state of the art of machine learning",
                "None of these activities are an important part of my role at work",
                "Others"]
colors = ["Red", "Blue", "Purple", "Grey", "Yellow", "Green", "Pink"]
for col, act in zip(colors, full_activity):
    print( col + "  ->  " + act)
fig, ax = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(20, 20))
n = 1
for job in transposed_activities:
    plt.subplot(6, 3, n)
    transposed_activities[job].plot(kind="bar", title=job, fontsize=6)
    n +=1
plt.subplots_adjust(hspace = 1)
plt.show()

WORK_DETAILS = ['Gathering data', 'Cleaning data', 
            'Visualizing data', 'Model building/model selection', 
            'Putting the model into production', 
            'Finding insights in the data and communicating with stakeholders']
mcr_time = mcr_jobs.filter(regex=("^(?!.*TEXT)(Q6|Q34)"), axis=1)\
                .dropna()
mcr_time.columns = ["Q6"] + WORK_DETAILS
mcr_time[WORK_DETAILS] = mcr_time[WORK_DETAILS].astype(float)
mcr_time.groupby("Q6").agg(np.mean)\
    .plot.barh(stacked=True, figsize=(20, 10), fontsize=15)

plt.title("Average time percentage devoted on data projects for different titles", fontsize=20)
plt.ylabel("Job Titles", fontsize=15)
plt.xlabel("Time percentage", fontsize=15)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=14)
plt.show()
mcr_insight = mcr_jobs[["Q6", "Q46"]]\
                        .dropna()\
                        .groupby(["Q6", "Q46"])\
                        .size()\
                        .unstack(1)\
                        .fillna(0)

mcr_insight = mcr_insight.apply(lambda x: (x / mcr_insight.sum(axis=1) * 100).round(0))

plt.figure(figsize=(20, 10))
sns.heatmap(mcr_insight, cmap="YlGnBu")

_ = plt.title("Distribution of percentage of time people have on exploring model insights.")
_ = plt.xlabel("Slots")
_ = plt.ylabel("Job titles")
job_index = mcr_jobs.index
mcr_ffr = pd.read_csv("../input/freeFormResponses.csv", low_memory=False).filter(regex="Q12").iloc[job_index, :]
drop_cols = mcr_ffr.columns.tolist()
TOOLS = ['jupyter', 'python', 'pycharm', 'anaconda', 'matlab', 'rstudio', 'visual studio', 'excel', 'docker', 'sql'
    , 'bash', 'spark', 'scala', 'emacs', 'tensor', 'torch', 'scikit', 'xgb', 'hadoop', 'jupyterlab', 'cloudera'
, 'c++', 'tableau', 'pandas', 'keras', 'ide', 'orange', 'colab', 'aws', 'mxnet', 'databricks', 'spyder', 'java',
'h2o', 'slack', 'zeppelin', 'sas', 'spss', 'nltk', 'powerbi']

mcr_ffr["summary"] = mcr_ffr.apply(lambda x: set([i for i in x if i is not np.nan]), axis=1)
mcr_ffr["summary"] = mcr_ffr["summary"].apply(lambda x: [i.lower() for i in x])
mcr_ffr = mcr_ffr[mcr_ffr["summary"].apply(lambda x: len(x)) == 1]
for t in TOOLS:
    mcr_ffr[t] = 0
    
for idx, r in mcr_ffr.iterrows():
    for t in TOOLS:
        if t in r['summary'][0]:
            mcr_ffr.set_value(idx, t, 1)
            
drop_cols = drop_cols + ["summary"]
mcr_tool = mcr_ffr.merge(mcr_jobs[['Q6']], right_index=True, left_index=True)\
        .drop(drop_cols, axis=1)
fig, ax = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(20, 20))
n = 1
for job in CONSIDERED_JOB_TITLE:
    sub_job = mcr_tool[mcr_tool["Q6"] == job]
    txt = ",".join(sub_job[TOOLS].sum(axis=0).sort_values(ascending=False).index)
    plt.subplot(6, 3, n)
    wordcloud = WordCloud(colormap="Reds", width=900, height=480,
                      normalize_plurals=False).generate(txt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(job)
    plt.axis("off")
    n +=1
plt.subplots_adjust(hspace = 0.2)
plt.show()
mcr_tool["num_tools"] = mcr_tool.iloc[:, :-1].apply(lambda x: sum(x), axis=1)
mcr_num_tool = mcr_tool[["num_tools", "Q6"]]
pal = sns.cubehelix_palette(10, rot=-.25, light=1)
g = sns.FacetGrid(mcr_num_tool, row="Q6", hue="Q6", aspect=15, height=1, palette=pal)

g.map(sns.kdeplot, "num_tools", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "num_tools", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "num_tools")
g.fig.subplots_adjust(hspace=.25)

g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
TRAIN_CAT = ["Self-taught", "Online courses", "Work", "University", "Kaggle competitions", "Other"]
mcr_train = mcr_jobs.filter(regex="^(?!.*TEXT)(Q35|Q6)").dropna(axis=0)
mcr_train.columns = ["Q6"] + TRAIN_CAT
mcr_train[TRAIN_CAT] = mcr_train[TRAIN_CAT].astype(float)

mcr_train.groupby("Q6").mean()\
        .plot(kind="bar", stacked=True, 
              figsize=(18, 10), fontsize=15, 
              colormap=plt.get_cmap("tab20"))

plt.title("Percentage of training on ML fell on different categories", fontsize=20)
plt.ylabel("Percentage", fontsize=15)
plt.xlabel("Job Titles", fontsize=15)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=14)
plt.show()
EDU_BACKGROUND = ["Bachelor’s degree", "Master’s degree", "Doctoral degree", 
                  "Professional degree", "Some college/university study without earning a bachelor’s degree",
                  "No formal education past high school"]

mcr_job_edu = mcr_jobs[["Q6", "Q4"]]\
        .groupby(["Q6", "Q4"])\
        .size()\
        .to_frame("counts")\
        .unstack(1)\
        .fillna(0)
mcr_job_edu = mcr_job_edu['counts'][EDU_BACKGROUND]
mcr_job_edu = mcr_job_edu.apply(lambda x: ((x / mcr_job_edu.sum(axis=1)) * 100).round(0))
plt.figure(figsize=(20, 10))
sns.heatmap(mcr_job_edu, cmap="Oranges", annot=True)

_ = plt.title("Distribution of percentage of education background people have in different occupations.")
_ = plt.xlabel("Educations")
_ = plt.ylabel("Job titles")
mcr_age = mcr_jobs[["Q6", "Q2"]]
random_sampled_age = mcr_age["Q2"].apply(lambda x: np.random.randint(int(x.split("-")[0]), int(x.split("-")[1])) 
                                                  if (len(x.split("-")) == 2) else 80)
mcr_age["age"] = random_sampled_age
plt.figure(figsize=(30, 15))
sns.swarmplot(x="Q6", y="age", hue="Q2",
              palette="RdBu", data=mcr_age)
_ = plt.xticks(rotation=90)
_ = plt.title("Age distribution among different job occupations")
_ = plt.xlabel("Job titles")
_ = plt.ylabel("Age")
