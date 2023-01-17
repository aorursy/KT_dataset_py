import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.metrics import confusion_matrix

from IPython.display import Markdown, HTML

from collections import Counter

from itertools import chain

from functools import reduce

# from src.jupyter import grid_df_display, combination_matrix



pd.set_option('display.max_columns',   500)

pd.set_option('display.max_colwidth',   -1)



%load_ext autoreload

%autoreload 2
# Source: https://github.com/JamesMcGuigan/kaggle-digit-recognizer/blob/master/src/utils/confusion_matrix.py

from typing import Union



import pandas as pd

from pandas.io.formats.style import Styler





def combination_matrix(dataset: pd.DataFrame, x: str, y: str, z: str,

                       format=None, unique=True) -> Union[pd.DataFrame, Styler]:

    """

    Returns a combination matrix, showing all valid combinations between three DataFrame columns.

    Sort of like a heatmap, but returning lists of (optionally) unique values



    :param dataset: The dataframe to create a combination_matrx from

    :param x: column name to use for the X axis

    :param y: column name to use for the Y axis

    :param z: column name to use for the Z axis (values that appear in the cells)

    :param format: '', ', '-', ', '\n'    = format value lists as "".join() string

                    str, bool, int, float = cast value lists

    :param unique:  whether to return only unique values or not - eg: combination_matrix(unique=False).applymap(sum)

    :return: returns nothing

    """

    unique_y = sorted(dataset[y].unique())

    combinations = pd.DataFrame({

        n: dataset.where(lambda df: df[y] == n)

            .groupby(x)[z]

            .pipe(lambda df: df.unique() if unique else df )

            .apply(list)

            .apply(sorted)

        for n in unique_y

    }).T



    if isinstance(format, str):

        combinations = combinations.applymap(

            lambda cell: f"{format}".join([str(value) for value in list(cell) ])

            if isinstance(cell, list) else cell

        )

    if format == str:   combinations = combinations.applymap(lambda cell: str(cell)      if isinstance(cell, list) and len(cell) > 0 else ''     )

    if format == bool:  combinations = combinations.applymap(lambda cell: True           if isinstance(cell, list) and len(cell) > 0 else False  )

    if format == int:   combinations = combinations.applymap(lambda cell: int(cell[0])   if isinstance(cell, list) and len(cell)     else ''     )

    if format == float: combinations = combinations.applymap(lambda cell: float(cell[0]) if isinstance(cell, list) and len(cell)     else ''     )



    combinations.index.rename(y, inplace=True)

    combinations.fillna('', inplace=True)

    if format == '\n':

        return combinations.style.set_properties(**{'white-space': 'pre-wrap'})  # needed for display

    else:

        return combinations  # Allows for subsequent .applymap()
dataset = pd.read_csv('../input/bengaliai-cv19/train.csv'); 

dataset['base_graphemes'] = dataset['grapheme'].apply(list)

dataset.head()
base_diacritics_unique = sorted(set(chain(*dataset['base_graphemes'].values)))

base_diacritics_stats  = {

    "mean":   round( dataset['base_graphemes'].apply(len).mean(), 2),

    "median": np.median( dataset['base_graphemes'].apply(len) ),

    "min":    dataset['base_graphemes'].apply(len).min(),

    "max":    dataset['base_graphemes'].apply(len).max(),

    "std":    dataset['base_graphemes'].apply(len).std(),    

    "unique": len( set(chain(*dataset['base_graphemes'].values))),

    "count":  len(list(chain(*dataset['base_graphemes'].values))),

    "mean_duplicated_bases":  dataset['base_graphemes'].apply(lambda value: (len(value) - len(set(value)))).mean(),

    "max_duplicated_bases":   dataset['base_graphemes'].apply(lambda value: (len(value) - len(set(value)))).max(),    

    "count_duplicated_bases": dataset['base_graphemes'].apply(lambda value: (len(value) - len(set(value))) != 0).sum(),        

}

base_diacritics_counter = dict( 

    sum(dataset['base_graphemes'].apply(Counter), Counter()).most_common()

)



display( pd.DataFrame([base_diacritics_counter]) / base_diacritics_stats['count'] )

display( " ".join(base_diacritics_unique) )

display( base_diacritics_stats )
base_diacritic_sets = {

    key: dataset.groupby(key)['base_graphemes']

                .apply(lambda group: reduce(lambda a,b: set(a) & set(b), group)) 

                .apply(sorted)     

    for key in [ 'vowel_diacritic', 'consonant_diacritic', 'grapheme_root' ]

}

display(

    pd.DataFrame(base_diacritic_sets)

        .applymap(lambda x: x if x is not np.nan else set())        

        .applymap(lambda group: "\n".join(group))

        .T

        .style.set_properties(**{'white-space': 'pre-wrap'})

)
for key in [ 'vowel_diacritic', 'consonant_diacritic', 'grapheme_root' ]:

    base_key = key.split('_')[0] + '_base'

    zfill = 3 if key == 'grapheme_root' else 2

    dataset[base_key] = (

        dataset[key]

            .apply(lambda value: [ str(value).zfill(zfill)] + sorted(base_diacritic_sets[key][value]))

            .apply(lambda value: " ".join(value))            

            .fillna('')

    )

# Make numeric strings sortable

dataset.head()
combination_matrix(dataset, x='consonant_base', y='vowel_base', z='grapheme', format=' ')
combination_matrix(dataset, x='grapheme_base', y='vowel_base', z='grapheme', format='\n')
combination_matrix(dataset, x='grapheme_base', y='consonant_base', z='grapheme', format='\n')
combination_matrix(dataset, x=['vowel_base','consonant_base'], y='grapheme_base', z='grapheme', format=' ').T
combination_matrix(dataset, x=['vowel_base','consonant_base'], y='grapheme_base', z='grapheme', format=' ')