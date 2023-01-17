#importing packages

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#setting options

pd.options.display.max_colwidth = 500

pd.set_option('display.max_columns', None)
#Loading the data

df = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

df = df.drop(df.index[0])
df.Q5.value_counts()
#Loading the Schema

schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
#Create a function to obtain all the questions that are excluded for a given question in a schema

def schema_check(schema, n):

    schema_n = schema.iloc[n][1:]

    excluded_n = []

    for i, v in enumerate(schema_n):

        if v == '1':

            excluded_n.append(schema_n.index[i])

    return excluded_n



#Checking the schema for row 4, which correspond to questions that a

schema_check(schema, 4)
#Slice the data to only focus on the 2 groups of interest

df = df.loc[df['Q5'].isin(['Student', 'Data Scientist'])]

df.shape
salary_order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999', 

                '5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000-19,999', '20,000-24,999', 

                '25,000-29,999', '30,000-39,999', '40,000-49,999', '50,000-59,999', '60,000-69,999', 

                '70,000-79,999', '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999', 

                '150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000', '> $500,000']

fig_q10, ax_q10 = plt.subplots(figsize=(15,7))

ax_q10 = sns.countplot(x="Q10", data=df, order=salary_order).set_xticklabels(labels = salary_order, rotation=45)
Europe = ['France', 'Germany', 'Russia', 'Portugal', 'Italy', 'Spain',

         'Sweden', 'Hungary', 'Ireland', 'Belarus', 'Norway', 'Netherlands',

         'United Kingdom of Great Britain and Northern Ireland', 'Switzerland',

         'Denmark', 'Ukraine', 'Poland', 'Greece', 'Belgium', 'Austria', 

         'Romania', 'Czech Republic', ]

Asia = ['India', 'China', 'Japan', 'Pakistan', 'Turkey', 'Bangladesh',

       'South Korea', 'Taiwan', 'Thailand', 'Israel', 'Viet Nam', 'Singapore',

       'Republic of Korea', 'Iran, Islamic Republic of...', 'Malaysia', 

       'Indonesia', 'Philippines', 'Hong Kong (S.A.R.)', 'Saudi Arabia' ]

North_america =  ['United States of America', 'Canada', 'Mexico' ]

South_america = ['Argentina', 'Brazil', 'Chile', 'Columbia', 'Peru' ]

Oceania = ['Australia', 'New Zealand']

Africa = ['Algeria', 'South Africa', 'Egypt', 'Nigeria', 'Morocco', 'Tunisia',

         'Kenya', ]
def map_continent(i):

    if i in Europe:

        return 'Europe'

    if i in Asia:

        return 'Asia'

    if i in North_america:

        return 'North America'

    if i in South_america:

        return 'South America'

    if i in Oceania:

        return 'Oceania'

    if i in Africa:

        return 'Africa'

    else:

        return 'Other'

    

df['Continent'] = df['Q3'].apply(map_continent)
fig_cont, ax = plt.subplots(6, 1, figsize=(15,15), sharex=True, sharey=False)



ax[0] = sns.countplot(x="Q10", data=df[df.Continent == 'Oceania'], order=salary_order, ax=ax[0])

ax[1] = sns.countplot(x="Q10", data=df[df.Continent == 'South America'], order=salary_order, ax=ax[1])

ax[2] = sns.countplot(x="Q10", data=df[df.Continent == 'Africa'], order=salary_order, ax=ax[2])

ax[3] = sns.countplot(x="Q10", data=df[df.Continent == 'Europe'], order=salary_order, ax=ax[3])

ax[4] = sns.countplot(x="Q10", data=df[df.Continent == 'North America'], order=salary_order, ax=ax[4])

ax[5] = sns.countplot(x="Q10", data=df[df.Continent == 'Asia'], order=salary_order, ax=ax[5])



ax[0].set_title('Oceania')

ax[1].set_title('South America')

ax[2].set_title('Africa')

ax[3].set_title('Europe')

ax[4].set_title('North America')

ax[5].set_title('Asia')



ax[0].set_xlabel('')

ax[1].set_xlabel('')

ax[2].set_xlabel('')

ax[3].set_xlabel('')

ax[4].set_xlabel('')

ax[5].set_xlabel('')



ax_rotate = ax[5].set_xticklabels(salary_order, rotation=45)
def get_columns(question, df):

    col_names = []

    for i in range(len(df.columns)):

        if question in df.columns[i]:

            col_names.append(df.columns[i])

    return col_names
def clean_column_name(df1, df2):

    new_column_name = [] 

    for i in range(len(df1.columns)):

        new_name = df1.columns[i].replace(df2.columns[i] + '_', '')

        new_name = new_name.strip()

        new_column_name.append(new_name)

    return new_column_name
def create_dummy_multiple_question(num, df):

    question = 'Q' + str(num)

    df_q5 = df[['Q5']] 

    df_q = df[get_columns(question, df)[:-1]].fillna(0)

    df_dummy = pd.get_dummies(df_q, columns=df_q.columns)

    df_dummy = df_dummy[[c for c in df_dummy.columns if c.lower()[-1:] != '0']]

    df_dummy.columns = clean_column_name(df_dummy, df_q)

    return df_dummy.join(df_q5)
def plot_dummy(num, df):

    df_q = create_dummy_multiple_question(num, df).groupby('Q5').sum().transpose()

    df_q.plot.barh(figsize=(7,12))
def plot_dummy_vertical(num, df):

    df_q = create_dummy_multiple_question(num, df).groupby('Q5').sum().transpose()

    df_q.plot.bar(figsize=(15,7)).set_xticklabels(df_q.index.tolist(), rotation=25)
plot_dummy(9, df)
# Code for the heatmap adopted from Drazen Zaric.

# Visit https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec to

# read about his amazing work on creating this heatmap. 

# Visit https://www.kaggle.com/drazen/heatmap-with-sized-markers to see the code for this in action



def heatmap(x, y, **kwargs):

    if 'color' in kwargs:

        color = kwargs['color']

    else:

        color = [1]*len(x)



    if 'palette' in kwargs:

        palette = kwargs['palette']

        n_colors = len(palette)

    else:

        n_colors = 256 # Use 256 colors for the diverging color palette

        palette = sns.color_palette("Blues", n_colors) 



    if 'color_range' in kwargs:

        color_min, color_max = kwargs['color_range']

    else:

        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation



    def value_to_color(val):

        if color_min == color_max:

            return palette[-1]

        else:

            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range

            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1

            ind = int(val_position * (n_colors - 1)) # target index in the color palette

            return palette[ind]



    if 'size' in kwargs:

        size = kwargs['size']

    else:

        size = [1]*len(x)



    if 'size_range' in kwargs:

        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]

    else:

        size_min, size_max = min(size), max(size)



    size_scale = kwargs.get('size_scale', 500)



    def value_to_size(val):

        if size_min == size_max:

            return 1 * size_scale

        else:

            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range

            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1

            return val_position * size_scale

    if 'x_order' in kwargs: 

        x_names = [t for t in kwargs['x_order']]

    else:

        x_names = [t for t in sorted(set([v for v in x]))]

    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}



    if 'y_order' in kwargs: 

        y_names = [t for t in kwargs['y_order']]

    else:

        y_names = [t for t in sorted(set([v for v in y]))]

    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}



    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid

    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot



    marker = kwargs.get('marker', 's')



    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [

         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'

    ]}



    ax.scatter(

        x=[x_to_num[v] for v in x],

        y=[y_to_num[v] for v in y],

        marker=marker,

        s=[value_to_size(v) for v in size], 

        c=[value_to_color(v) for v in color],

        **kwargs_pass_on

    )

    ax.set_xticks([v for k,v in x_to_num.items()])

    ax.set_xticklabels([k for k in x_to_num], rotation=0, horizontalalignment='center')

    ax.set_yticks([v for k,v in y_to_num.items()])

    ax.set_yticklabels([k for k in y_to_num])



    ax.grid(False, 'major')

    ax.grid(True, 'minor')

    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)

    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)



    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])

    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    ax.set_facecolor('#F1F1F1')



    # Add color legend on the right side of the plot

    if color_min < color_max:

        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot



        col_x = [0]*len(palette) # Fixed x coordinate for the bars

        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars



        bar_height = bar_y[1] - bar_y[0]

        ax.barh(

            y=bar_y,

            width=[5]*len(palette), # Make bars 5 units wide

            left=col_x, # Make bars start at 0

            height=bar_height,

            color=palette,

            linewidth=0

        )

        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle

        ax.grid(False) # Hide grid

        ax.set_facecolor('white') # Make background white

        ax.set_xticks([]) # Remove horizontal ticks

        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max

        ax.yaxis.tick_right() # Show vertical ticks on the right 
y_bin_labels = ['0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+']

x_bin_labels = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']



ds_size = df.groupby(['Q6', 'Q7']).count()[['Q5']].reset_index().replace(np.nan, 0)



plt.figure(figsize=(15, 7))

heatmap(

    x=ds_size['Q6'],

    y=ds_size['Q7'],

    size=ds_size['Q5'],

    #color=ds_size['Q5'],

    marker='s',

    x_order=x_bin_labels,

    y_order=y_bin_labels

)
fig_q2, ax_q2 = plt.subplots(figsize=(15,7))

ax_q2 = sns.countplot(x="Q2", hue="Q5", data=df, palette='muted', order=df.Q2.unique())
age_order = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']

fig_q1, ax_q1 = plt.subplots(figsize=(15,7))

ax_q1 = sns.countplot(x="Q1", hue="Q5", data=df, palette='muted', order=age_order)
fig_q4, ax_q4 = plt.subplots(figsize=(7,7))

ax_q4 = sns.countplot(y="Q4", hue="Q5", data=df, palette='muted', order=df.Q4.unique())

#ax_q4.set_xticklabels(rotation=90)
df['Q15'] = df['Q15'].replace('I have never written code', 'Never')
year_exp_order = ['Never', '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']

fig_q15, ax_q15 = plt.subplots(figsize=(7,7))

ax_q15 = sns.countplot(x="Q15", hue="Q5", data=df, palette='muted', order=year_exp_order)
df['Q12_Part_1'] = df['Q12_Part_1'].replace('Twitter (data science influencers)', 'Twitter')

df['Q12_Part_2'] = df['Q12_Part_2'].replace('Hacker News (https://news.ycombinator.com/)', 'Hacker News')

df['Q12_Part_3'] = df['Q12_Part_3'].replace('Reddit (r/machinelearning, r/datascience, etc)', 'Reddit')

df['Q12_Part_4'] = df['Q12_Part_4'].replace('Kaggle (forums, blog, social media, etc)', 'Kaggle')

df['Q12_Part_5'] = df['Q12_Part_5'].replace('Course Forums (forums.fast.ai, etc)', 'Course Forums')

df['Q12_Part_6'] = df['Q12_Part_6'].replace('YouTube (Cloud AI Adventures, Siraj Raval, etc)', 'YouTube')

df['Q12_Part_7'] = df['Q12_Part_7'].replace('Podcasts (Chai Time Data Science, Linear Digressions, etc)', 'Podcasts')

df['Q12_Part_8'] = df['Q12_Part_8'].replace('Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)', 'Blogs')

df['Q12_Part_9'] = df['Q12_Part_9'].replace('Journal Publications (traditional publications, preprint journals, etc)', 'Journal Publications')

df['Q12_Part_10'] = df['Q12_Part_10'].replace('Slack Communities (ods.ai, kagglenoobs, etc)', 'Slack Communities')
plot_dummy_vertical(12, df)
df['Q13_Part_6'] = df['Q13_Part_6'].replace("Kaggle Courses (i.e. Kaggle Learn)", "Kaggle Courses")

df['Q13_Part_10'] = df['Q13_Part_10'].replace("University Courses (resulting in a university degree)", "University Courses")
plot_dummy_vertical(13, df)
df['Q14'] = df['Q14'].replace('Advanced statistical software (SPSS, SAS, etc.)', 'Advanced statistical software')

df['Q14'] = df['Q14'].replace('Local development environments (RStudio, JupyterLab, etc.)', 'Local development environments')

df['Q14'] = df['Q14'].replace('Basic statistical software (Microsoft Excel, Google Sheets, etc.)', 'Basic statistical software')

df['Q14'] = df['Q14'].replace('Cloud-based data software & APIs (AWS, GCP, Azure, etc.)', 'Cloud-based data software & APIs')

df['Q14'] = df['Q14'].replace('Business intelligence software (Salesforce, Tableau, Spotfire, etc.)', 'Business intelligence software')
fig_q14, ax_q14 = plt.subplots(figsize=(15,7))

ax_q14 = sns.countplot(x="Q14", hue="Q5", data=df, palette='muted', order=df.Q14.value_counts().index)

label_14 = ax_q14.set_xticklabels(labels=df.Q14.value_counts().index, rotation=15, horizontalalignment='center')
df['Q16_Part_1'] = df['Q16_Part_1'].replace('Jupyter (JupyterLab, Jupyter Notebooks, etc) ', 'Jupyter')

df['Q16_Part_6'] = df['Q16_Part_6'].replace(' Visual Studio / Visual Studio Code ', 'Visual Studio')
plot_dummy_vertical(16, df)
df['Q17_Part_1'] = df['Q17_Part_1'].replace(' Kaggle Notebooks (Kernels) ', 'Kaggle Kernels')

df['Q17_Part_4'] = df['Q17_Part_4'].replace(' Google Cloud Notebook Products (AI Platform, Datalab, etc) ', 'Google Cloud Notebook Products')

df['Q17_Part_10'] = df['Q17_Part_10'].replace('AWS Notebook Products (EMR Notebooks, Sagemaker Notebooks, etc) ', 'AWS Notebook Products')
plot_dummy_vertical(17, df)
plot_dummy_vertical(18, df)
plot_dummy_vertical(20, df)
plot_dummy(24, df)
plot_dummy(25, df)
plot_dummy(26, df)
plot_dummy(27, df)
plot_dummy_vertical(28, df)
plot_dummy_vertical(29, df)
plot_dummy_vertical(30, df)
plot_dummy_vertical(31, df)
plot_dummy_vertical(32, df)
plot_dummy_vertical(34, df)