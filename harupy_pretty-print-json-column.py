import pandas as pd



pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', 100)
%ls ../input/data-science-bowl-2019 -lh
sample = pd.read_csv('../input/data-science-bowl-2019/train.csv', nrows=10)
import json

from IPython.display import display, HTML



def pretty_json(data):

    return json.dumps(json.loads(data), indent=2, sort_keys=False)





def pretty_display(df):

    style = """

    <style>

      tr {

        text-align: left !important;

      }

      td {

        font-family: monospace;

        white-space: pre !important;

        text-align: left !important;

      }

    </style>

    """

    return display(HTML(df.to_html().replace('\\n', '<br>') + style))
sample['event_data_pretty'] = sample['event_data'].map(pretty_json)

pretty_display(sample[['event_id', 'event_code', 'event_data_pretty']])