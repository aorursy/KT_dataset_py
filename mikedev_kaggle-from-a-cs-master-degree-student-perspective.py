# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.preprocessing import minmax_scale, scale
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
survey_schema = pd.read_csv('../input/SurveySchema.csv')
free_form = pd.read_csv('../input/freeFormResponses.csv')
multiple_choice = pd.read_csv('../input/multipleChoiceResponses.csv')
all_na_values = (multiple_choice.loc[1:, :].isna().sum() == (len(multiple_choice) - 1))
print('Dropped columns ', multiple_choice.columns[all_na_values].tolist(),
      ' from multipleChoiceResponses.csv because are all na')
multiple_choice = multiple_choice.loc[:, ~all_na_values]
# Many support functions
def cols_part_questionary(q):
    return multiple_choice.columns[multiple_choice.columns.str.startswith(q) & multiple_choice.columns.str.contains('_Part_')]

def get_cols_values(values):
    return values.iloc[1:, :].apply(lambda col: col[col.first_valid_index()])

def plot_correlation_question(*qs):
    q_multiple_answers = []
    for q in qs:
        q_multiple_answers += cols_part_questionary(q).tolist()
    q_choices = multiple_choice.loc[1:, q_multiple_answers]
    map_index = q_choices.apply(lambda col: col[col.first_valid_index()], axis=0).reset_index(drop=True)
    if not map_index.str.match('[0-9]+').any():
        q_choices = ~q_choices.isna()
#       plt.subplots(figsize=(15, 11))
#         plt.title(survey_schema.loc[0, q])
        sns.heatmap(cramerv(q_choices, q_choices.columns.tolist()),
                    xticklabels=map_index.values,
                    yticklabels=map_index.values)
        plt.show()
    return map_index, q_choices

def plot_countbar(responses, values):
    counts = values.sum()
    counts.index = responses.values
    plt.subplots(figsize=(10, 7))
    ax = sns.barplot(counts.index, counts.values)
    ax.set_xticklabels(counts.index.values, rotation=90)
    return ax

def plot_cross_correlation(q1, q2):
    q1 = cols_part_questionary(q1).tolist()
    q2 = cols_part_questionary(q2).tolist()
    q_choices = multiple_choice.loc[1:, q1 + q2]
    map_index = q_choices.apply(lambda col: col[col.first_valid_index()], axis=0).reset_index(drop=True)
    q_choices = ~q_choices.isna()
    corr_matrix = cramerv(q_choices, q1 + q2)
    ax = sns.heatmap(corr_matrix[:len(q1), len(q1):], yticklabels=map_index[:len(q1)], xticklabels=map_index[len(q1):])
    return ax

from scipy.stats import chi2_contingency
def _cramerv(col1: np.array, col2: np.array):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2, p_value = chi2_contingency(confusion_matrix)[0:2]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if not phi2corr == 0 else 0


def cramerv(dataset: pd.DataFrame, cols) -> np.matrix:
    """
    Apply the cramerv test on each pair of columns. The cramerv test is used to
    measure in [0,1] the correlation between categorical variables
    :param dataset: Dataframe where the correlation is measured
    :param cols: the columns of dataset where measure the correlation
    :return: the correlation matrix of cramerv on the specified columns of dataset
    """
    matr = np.matrix([[0] * len(cols)] * len(cols), dtype='float')
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            if i == j:
                matr[i, j] = 1
            else:
                corr = _cramerv(dataset[cols[i]], dataset[cols[j]])
                matr[i, j] = corr
                matr[j, i] = corr
    return matr

def normalized_crosstab(q1, q2):
    """Get the crosstab normalized by q1"""
    crosstab = pd.crosstab(multiple_choice.loc[1:, q1], multiple_choice.loc[1:, q2])
    norm_crosstab = crosstab.divide(crosstab.sum(axis=1), axis='rows')
    return norm_crosstab.T
multiple_choice.Q2[1:].value_counts().plot.barh(title='Age of kagglers survey respondants')
plt.show()
multiple_choice.Q8[1:].value_counts().plot.barh(title=multiple_choice.Q8[0])
plt.show()
multiple_choice.Q10[1:].value_counts().plot.barh(title=multiple_choice.Q10[0])
plt.show()
q49_cols = multiple_choice.loc[:, cols_part_questionary('Q49')]
q49_values = get_cols_values(q49_cols)
counts_q49 = (~q49_cols.isna()).sum()
counts_q49.index = q49_values.values
counts_q49.plot.barh(title=survey_schema.Q49[0])
plt.show()
multiple_choice.loc[1:, 'Q5'].value_counts().plot.barh(title=multiple_choice.Q5[0])
plt.show()
q16_cols = multiple_choice.loc[1:, cols_part_questionary('Q16')]
q16_values = get_cols_values(q16_cols)
counts_q16 = (~q16_cols.isna()).sum()
counts_q16.index = q16_values.values
counts_q16.plot.barh(title=survey_schema.Q16[0])
plt.show()
multiple_choice.Q18[1:].value_counts().plot.barh(title=multiple_choice.Q18[0])
plt.show()
ax = plot_cross_correlation('Q11', 'Q16')
plt.show()
crosstab = pd.crosstab(multiple_choice.Q17[1:], multiple_choice.Q5[1:])
norm_crosstab = crosstab.divide(crosstab.sum(axis=1), axis='rows')
ax = sns.heatmap(norm_crosstab.T)
ax.set_xlabel('')
ax.set_ylabel('')

plt.show()
ax = sns.heatmap(normalized_crosstab('Q17', 'Q10'))
# sns.heatmap(pd.crosstab(multiple_choice.Q17[1:], multiple_choice.Q10[1:]).T)
plt.show()
counts = multiple_choice.Q17[1:].value_counts()
q9_true_order = [0, 1, 5,8, 10, 12, 14, 15, 16, 17, 2, 3, 4 ,6 , 7, 9, 11, 13, 18]
crosstab = (pd.crosstab(multiple_choice.Q17[1:], multiple_choice.Q9[1:]))
crosstab = crosstab.iloc[:, q9_true_order]
crosstab = crosstab.rename({crosstab.columns[-1]:"I don't want to disclose"}, axis='columns')
norm_crosstab = crosstab.divide(crosstab.sum(axis=1), axis='rows')
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(norm_crosstab, ax=ax)
plt.tight_layout()
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()
ax = plot_cross_correlation('Q13', 'Q16')
plt.show()
mlf_col = 'Q19'
mlf_cols = cols_part_questionary(mlf_col)
mlf_counts = (~multiple_choice.loc[1:, mlf_cols].isna()).sum()
cols_values = get_cols_values(multiple_choice.loc[:, mlf_cols])
mlf_counts.index = cols_values.values
mlf_counts.plot.barh()
plt.show()
ax = plot_cross_correlation('Q11', mlf_col)
ax.set_title(survey_schema.Q11[0])
plt.show()
ax = sns.heatmap(normalized_crosstab('Q20', 'Q10'))
plt.show()
_ = plot_correlation_question('Q27')
del _
plt.subplots(figsize=(8, 5))
_ = plot_correlation_question('Q29')
del _
plt.subplots(figsize=(10, 7))
_ = plot_correlation_question('Q28')
del _
legend_cols = multiple_choice.loc[0, cols_part_questionary('Q35')].str.split(' - ').tolist()[:-1]
legend_cols = list(map(lambda el: el[1], legend_cols))
time_use = multiple_choice.loc[1:, cols_part_questionary('Q35')[:-1]].fillna(0).astype('float')
time_use.columns = legend_cols
time_use.plot.hist(bins=15, title=survey_schema.Q35[0])
plt.xlim((1, 100))
plt.show()
multiple_choice.Q40[1:].value_counts().plot.barh(title=multiple_choice.Q40[0])
plt.show()