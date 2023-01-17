import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.metrics import confusion_matrix

from IPython.display import Markdown, HTML

# from src.jupyter import grid_df_display, combination_matrix



pd.set_option('display.max_columns',   500)

pd.set_option('display.max_colwidth',   -1)



%load_ext autoreload

%autoreload 2
# Source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side/50899244#50899244

import pandas as pd

from IPython.display import display,HTML



def grid_df_display(list_dfs, rows = 2, cols=3, fill = 'cols'):

    if fill not in ['rows', 'cols']: print("grid_df_display() - fill must be one of: 'rows', 'cols'")



    html_table = "<table style='width:100%; border:0px'>{content}</table>"

    html_row   = "<tr style='border:0px'>{content}</tr>"

    html_cell  = "<td style='width:{width}%;vertical-align:top;border:0px'>{{content}}</td>"

    html_cell  = html_cell.format(width=100/cols)



    cells = [ html_cell.format(content=df.to_html()) for df in list_dfs[:rows*cols] ]

    cells += cols * [html_cell.format(content="")] # pad



    if fill == 'rows':   # fill in rows first (first row: 0,1,2,... col-1)

        grid = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,rows*cols,cols)]

    elif fill == 'cols': # fill columns first (first column: 0,1,2,..., rows-1)

        grid = [ html_row.format(content="".join(cells[i:rows*cols:rows])) for i in range(0,rows)]

    else:

        grid = []



    # noinspection PyTypeChecker

    display(HTML(html_table.format(content="".join(grid))))



    # add extra dfs to bottom

    [display(list_dfs[i]) for i in range(rows*cols,len(list_dfs))]





if __name__ == "main":

    list_dfs = []

    list_dfs.extend((pd.DataFrame(2*[{"x":"hello"}]),

                     pd.DataFrame(2*[{"x":"world"}]),

                     pd.DataFrame(2*[{"x":"gdbye"}])))



    grid_df_display(3*list_dfs)
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
!ls ../input/bengaliai-cv19/
dataset = pd.read_csv('../input/bengaliai-cv19/train.csv'); 

# for key in ['grapheme_root','vowel_diacritic','consonant_diacritic','grapheme']:

#     dataset[key] = dataset[key].astype('category')  # ensures groupby().count() shows zeros

dataset['graphemes'] = dataset['grapheme'].apply(list)

dataset.head()
unique = dataset.apply(lambda col: col.nunique()); unique
combination_matrix(dataset, x='consonant_diacritic', y='vowel_diacritic', z='consonant_diacritic', unique=False).applymap(len)
root_vowels            = dataset.groupby('grapheme_root')['vowel_diacritic'].unique().apply(sorted).to_frame().T

root_consonants        = dataset.groupby('grapheme_root')['consonant_diacritic'].unique().apply(sorted).to_frame().T

root_vowels_values     = root_vowels.applymap(len).values.flatten()

root_consonants_values = root_consonants.applymap(len).values.flatten()



display(root_vowels)

display({

    "mean":   root_vowels_values.mean(),

    "median": np.median( root_vowels_values ),

    "min":    root_vowels_values.min(),

    "max":    root_vowels_values.max(),

    "unique_vowels":    unique['vowel_diacritic'],

    "root_combine_0":   sum([ 0 in lst for lst in root_vowels.values.flatten() ]),

    "unique_roots":     unique['grapheme_root'],

    "root_combine_not_0": str([ index for index, lst in enumerate(root_vowels.values.flatten()) if 0 not in lst ]),    

    "root_combine_all":       [ index for index, lst in enumerate(root_vowels.values.flatten()) if len(lst) == unique['vowel_diacritic'] ],

})

# print('--------------------')

display(root_consonants)

display({

    "mean":   root_consonants_values.mean(),

    "median": np.median( root_consonants_values ),

    "min":    root_consonants_values.min(),

    "max":    root_consonants_values.max(),

    "unique_consonants":  unique['consonant_diacritic'],

    "root_combine_0": sum([ 0 in lst for lst in root_consonants.values.flatten() ]),

    "unique_roots":   unique['grapheme_root'],

    "root_combine_not_0": str([ index for index, lst in enumerate(root_consonants.values.flatten()) if 0 not in lst ]),        

    "root_combine_all":       [ index for index, lst in enumerate(root_consonants.values.flatten()) if len(lst) == unique['consonant_diacritic'] ],

})
combination_matrix(dataset, x='consonant_diacritic', y='vowel_diacritic', z='grapheme_root', format=', ')
from collections import Counter



def filter_pairs_diacritics(pairs, diacritics, key=None):

    previous_diacritics = set(chain(*[ diacritics[k] for k,v in diacritics.items() if k != key ]))

    return [ pair for pair in pairs if pair[0] not in previous_diacritics ]



def print_conflicts(pairs, diacritics, key):

    valid = filter_pairs_diacritics(pairs, diacritics, key)

    if len(valid) == 0:

        conflict_key   = [ k for k,v in diacritics.items() if pairs[0][0] in diacritics[k].values ][0]

        conflict_dict  = { v:k for k,v in diacritics[conflict_key].items() }

        display({

            "source":   ( key, pairs[:4] ),

            "conflict": ( conflict_key, conflict_dict[pairs[0][0]], pairs[0][0] ),

        })

    return pairs
diacritics_raw = {

    "vowel_diacritic":     None,

    "consonant_diacritic": None,

    "grapheme_root":       None

}

for key in [ 'vowel_diacritic', 'consonant_diacritic', 'grapheme_root' ]:

    diacritics_raw[key] = (

        dataset.groupby(key)

            .apply(lambda group:   sum(group['graphemes'].apply(set).apply(Counter), Counter()) )   # -> Counter()

            .apply(lambda counter: counter.most_common() )                                          # -> [ tuple(symbol, count), ]

            .apply(lambda pairs:   pairs[0][0] if len(pairs) else '?' )

    )



    

### Hardcode conflict resolution and deduplicate - TODO: Verify correctness        

diacritics_resolutions = {

    "vowel_diacritic":     pd.Series({ 0: '্', 1: 'া', 2: 'ি' }),

    "consonant_diacritic": pd.Series({ 0: '্' }),

    "grapheme_root":       pd.Series({ 4: 'য' })

}

diacritics = { k:v.copy() for k,v in diacritics_resolutions.items() }

for key in [ 'vowel_diacritic', 'consonant_diacritic', 'grapheme_root' ]:

    diacritics[key] = (

        dataset.groupby(key)

            ### NOTE: group['graphemes'].apply(set) removes duplicate unicode diacritics       

            .apply(lambda group:   sum(group['graphemes'].apply(set).apply(Counter), Counter()) )   # -> Counter()

            .apply(lambda counter: counter.most_common() )                                          # -> [ tuple(symbol, count), ]

            .apply(lambda pairs:   print_conflicts(pairs, diacritics, key) ) 

            .apply(lambda pairs:   filter_pairs_diacritics(pairs, diacritics, key)) 

            .apply(lambda pairs:   pairs[0][0] if len(pairs) else '?' )

    )

    for index, symbol in diacritics_resolutions[key].items():

        diacritics[key][index] = symbol

    

display("Before Conflict Resolution")

display( pd.DataFrame(diacritics_raw).fillna('').T )



display("Deduplicated")

display( pd.DataFrame(diacritics).fillna('').T )
combination_matrix(dataset, x='consonant_diacritic', y='vowel_diacritic', z='grapheme', format=' ')
combination_matrix(dataset, x='grapheme_root', y='vowel_diacritic', z='grapheme', format=' ')
combination_matrix(dataset, x='grapheme_root', y='consonant_diacritic', z='grapheme', format=' ')
from itertools import chain

{

    "combinations": len(list(chain( 

        *combination_matrix(dataset, x='consonant_diacritic', y='vowel_diacritic', z='grapheme_root')

        .values.flatten() 

    ))),

    "unique_graphemes": unique['grapheme']

}
dataset.apply(lambda row: row.isnull()).sum()
( 

    dataset

    .groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

    .nunique(dropna=False) > 1 

).sum()
( 

    dataset

    .groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

    .nunique(dropna=False) > 1

).query("grapheme != False")
multilabled_graphemes = {

    "64-3-2": dataset.query("grapheme_root == 64 & vowel_diacritic == 3 & consonant_diacritic == 2")['grapheme'].unique().tolist(),

    "64-7-2": dataset.query("grapheme_root == 64 & vowel_diacritic == 7 & consonant_diacritic == 2")['grapheme'].unique().tolist(),

    "72-0-2": dataset.query("grapheme_root == 72 & vowel_diacritic == 0 & consonant_diacritic == 2")['grapheme'].unique().tolist(),

}

multilabled_graphemes
multilabled_grapheme_list   = list(chain(*multilabled_graphemes.values())); multilabled_grapheme_list

multilabled_grapheme_dict   = { grapheme: list(grapheme) for grapheme in multilabled_grapheme_list }

display(multilabled_grapheme_list)

display(multilabled_grapheme_dict)
dataset[ dataset['grapheme'].isin(multilabled_grapheme_list) ].groupby(['grapheme']).count()['image_id'].to_dict()