import pandas as pd

from IPython.display import HTML, display
%%time

df_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv", delimiter='\t')
def display_row(row):

    txt = row.Text

    txt_html = ''

    last_offset = 0

    for key_offset, offset in row.T[['Pronoun-offset', 'A-offset', 'B-offset']].sort_values().iteritems():

        key = key_offset.split('-')[0]

        if key in key in {'A', 'B'}:

            is_coref = row[f'{key}-coref']

        else:

            is_coref = ''

        txt_html = txt_html + txt[last_offset:offset] + f'<span class="{key} {is_coref}">' + row[key] + '</span>'

        last_offset = offset + len(row[key])

        

    txt_html = txt_html + txt[last_offset:]



    display(HTML('<li>' + txt_html + '</li>'))
# Add styles

display(HTML('''

<style>

.A, .B, .Pronoun { font-weight: bold }

.True { color: green }

.False { color: red }

</style>

'''))



display(HTML('<ul>'))



for _, row in df_train.iloc[:20].iterrows():

    display_row(row)



HTML('</ul>')