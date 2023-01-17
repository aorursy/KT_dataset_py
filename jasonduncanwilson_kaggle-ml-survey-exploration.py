import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from textwrap import wrap
sns.set(style="whitegrid")

#turn off warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# Load data
survey_data = pd.read_csv('../input/multipleChoiceResponses.csv')
survey_schema = pd.read_csv('../input/SurveySchema.csv')

# Split out titles from the first row of multiple choice response survey data
survey_data_titles = survey_data.iloc[0]
survey_data = survey_data.iloc[1:]
# Describe data
survey_data.describe(include = 'all')
# Get columns that will do well in correlation
corr_columns = [cname for cname in survey_data.columns 
                if survey_data[cname].count()/len(survey_data[cname]) > 0.5
                and survey_data[cname].nunique() > 2
                and survey_data[cname].mode().iloc[0] != -1
               ]

# Factorize data so it does well in correlation
corr_data = survey_data[corr_columns].apply(lambda x: pd.factorize(x)[0])

# Run correlation
corrmat = corr_data.corr()

# Show correlations on a heatmap
plt.subplots(figsize=(12,9))
plt.title("Correlation between most data and all participants")
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corrmat, mask=mask, vmax=0.9, cmap="YlGnBu",
            square=True, cbar_kws={"shrink": .5})
# Get columns that will do well in correlation
corr_columns = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
corr_data = survey_data[corr_columns]

# Whittle data down to just non-students
corr_data = corr_data[corr_data["Q6"].isin(['Not employed','Student']) == False]

# Factorize data so it does well in correlation
corr_data = corr_data.apply(lambda x: pd.factorize(x)[0])

# Layer better titles in for each column
for i in corr_columns:
    new_title = '\n'.join(wrap(survey_data_titles[i],30))
    corr_data.rename(columns={i: new_title}, inplace=True)

# Run correlation
corrmat = corr_data.corr()

# Show correlations on a heatmap
plt.subplots(figsize=(12,9))
plt.title("Correlation of a few key columns for the workforce")
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corrmat, mask=mask, vmax=0.9, cmap="YlGnBu",
            square=True, cbar_kws={"shrink": .5})
# Get columns that will do well in correlation
corr_columns = ["Q1","Q2","Q3","Q4","Q5","Q6"]
corr_data = survey_data[corr_columns]

# Whittle data down to just students
corr_data = corr_data[corr_data["Q6"].isin(['Not employed','Student'])]

# Factorize data so it does well in correlation
corr_data = corr_data.apply(lambda x: pd.factorize(x)[0])

# Layer better titles in for each column
for i in corr_columns:
    new_title = '\n'.join(wrap(survey_data_titles[i],30))
    corr_data.rename(columns={i: new_title}, inplace=True)

# Run correlation
corrmat = corr_data.corr()

# Show correlations on a heatmap
plt.subplots(figsize=(12,9))
plt.title("Correlation of a few key columns for students")
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corrmat, mask=mask, vmax=0.9, cmap="YlGnBu",
            square=True, cbar_kws={"shrink": .5})
# Create a countplot of the Q1 through Q10 columns
for i in range(1,11):
    col_name = 'Q'+str(i)
    col_title = '\n'.join(wrap(survey_data_titles[col_name],30))
    col_name_list = [col_name]
    survey_data[col_name_list].head(5)
    fig = plt.figure(figsize=(12, 5))
    ax = sns.countplot(data = survey_data[col_name_list], 
                       x=col_name,
                       order = survey_data[col_name].value_counts().index
                      )
    ax.set_title(col_title)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
# set up plot data and columns to create histograms for all of the columns
plot_columns = survey_data.columns
plot_column_titles = survey_data_titles[plot_columns]
plot_data = survey_data[plot_columns]
num_data_columns = len(plot_columns)

# metadata for plot
num_plot_columns = 5
height_subplots = 3
vertical_above_subplots = 2
max_bins = 10
num_plot_rows = math.ceil(num_data_columns/num_plot_columns)

# initialize plot figure
fig = plt.figure(figsize=(12, num_plot_rows*(height_subplots+vertical_above_subplots)))

# loop through columns to generate subplots
for i in plot_columns:
    col_num = plot_data.columns.get_loc(i)
    
    ax = plt.subplot(num_plot_rows, num_plot_columns, col_num+1)
    title = '\n'.join(wrap(plot_column_titles[i],20))
    ax.set_title(title)
    ax.get_yaxis().set_visible(False)
    # f.axes.set_ylim([0, train.shape[0]])
    
    # factorize object data
    if plot_data[i].dtype == "object":
        plot_data[i] = pd.factorize(plot_data[i])[0]
    
    unique_vals = np.size(plot_data[i].unique())
    
    # add plots of data
    if plot_data[i].dtype == "object" and unique_vals < max_bins:
        sns.countplot(data = plot_data[i], x=i, ax=ax)
    else:
        if unique_vals < max_bins:
            bins = unique_vals
        else:
            bins = max_bins
        sns.distplot(plot_data[i], bins=bins, ax=ax)

plt.subplots_adjust(top=vertical_above_subplots)
plt.tight_layout()