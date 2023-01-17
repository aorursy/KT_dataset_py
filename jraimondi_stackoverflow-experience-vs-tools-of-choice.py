import pandas as pd

survey_results = pd.read_csv('../input/survey_results_public.csv');
survey_results.head()
print('Amount of records: {:,}'.format(survey_results.size))
survey_results['Professional'].value_counts()
str_professional_dev = 'Professional developer'
str_professional_non_dev = 'Professional non-developer who sometimes writes code'
are_professionals_dev = survey_results['Professional'] == str_professional_dev
are_professionals_non_dev = survey_results['Professional'] == str_professional_non_dev
professionals = survey_results[are_professionals_dev | are_professionals_non_dev | True]
print('Amount of professionals (dev or non-dev): {:,}'.format(professionals.size))
professionals['YearsProgram'].value_counts().plot.barh()
# from https://stackoverflow.com/a/48120674/147507
def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]

def split_df(dataframe, col_name, sep):
    orig_col_index = dataframe.columns.tolist().index(col_name)
    orig_index_name = dataframe.index.name
    orig_columns = dataframe.columns
    dataframe = dataframe.reset_index()  # we need a natural 0-based index for proper merge
    index_col_name = (set(dataframe.columns) - set(orig_columns)).pop()
    df_split = pd.DataFrame(
        pd.DataFrame(dataframe[col_name].str.split(sep).tolist())
        .stack().reset_index(level=1, drop=1), columns=[col_name])
    df = dataframe.drop(col_name, axis=1)
    df = pd.merge(df, df_split, left_index=True, right_index=True, how='inner')
    df = df.set_index(index_col_name)
    df.index.name = orig_index_name
    # merge adds the column to the last place, so we need to move it back
    return change_column_order(df, col_name, orig_col_index)
professionals_years_languages = professionals[['HaveWorkedLanguage', 'YearsProgram']]
professionals_years_languages.dropna(inplace=True)
prof_years_languages_str = professionals_years_languages.astype(str)

years_languages = split_df(prof_years_languages_str, 'HaveWorkedLanguage', '; ')
from statsmodels.graphics.mosaicplot import mosaic
# unreadable, but le
mosaic_plot, mosaic_dict = mosaic(years_languages, ['YearsProgram', 'HaveWorkedLanguage'], axes_label=True)
mosaic_plot.set_size_inches(30, 30)
# from https://stackoverflow.com/a/42563850/147507
from matplotlib import colors
import matplotlib.pyplot as plt

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]
crosstab_data = pd.crosstab(years_languages.HaveWorkedLanguage, years_languages.YearsProgram)
crosstab_data = change_column_order(crosstab_data, 'Less than a year', 0)
crosstab_data = change_column_order(crosstab_data, '1 to 2 years', 1)
crosstab_data = change_column_order(crosstab_data, '2 to 3 years', 2)
crosstab_data = change_column_order(crosstab_data, '3 to 4 years', 3)
crosstab_data = change_column_order(crosstab_data, '4 to 5 years', 4)
crosstab_data = change_column_order(crosstab_data, '5 to 6 years', 5)
crosstab_data = change_column_order(crosstab_data, '6 to 7 years', 6)
crosstab_data = change_column_order(crosstab_data, '7 to 8 years', 7)
crosstab_data = change_column_order(crosstab_data, '8 to 9 years', 8)
crosstab_data = change_column_order(crosstab_data, '9 to 10 years', 9)
crosstab_data = change_column_order(crosstab_data, '10 to 11 years', 10)
crosstab_data = change_column_order(crosstab_data, '11 to 12 years', 11)
crosstab_data = change_column_order(crosstab_data, '12 to 13 years', 12)
crosstab_data = change_column_order(crosstab_data, '13 to 14 years', 13)
crosstab_data = change_column_order(crosstab_data, '14 to 15 years', 14)
crosstab_data = change_column_order(crosstab_data, '15 to 16 years', 15)
crosstab_data = change_column_order(crosstab_data, '16 to 17 years', 16)
crosstab_data = change_column_order(crosstab_data, '17 to 18 years', 17)
crosstab_data = change_column_order(crosstab_data, '18 to 19 years', 18)
crosstab_data = change_column_order(crosstab_data, '19 to 20 years', 19)
crosstab_data = change_column_order(crosstab_data, '20 or more years', 20)
crosstab_data_styler = crosstab_data.style.apply(background_gradient,
                                                cmap='Greens',
                                                m=crosstab_data.min().min(),
                                                M=crosstab_data.max().max(),
                                                low=0,
                                                high=1)
crosstab_data_styler
crosstab_data.to_csv('crosstab_languages.csv')
