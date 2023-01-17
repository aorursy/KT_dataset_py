import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt

import json

from IPython.display import HTML

import warnings

warnings.filterwarnings("ignore")
def read_data(kernel=True):

    print('Reading data...')

    if not kernel:

        suffix = 'pkl'

        read_function = pd.read_pickle

    else:

        suffix = 'csv'

        read_function = pd.read_csv 

    input_dir = '../input/data-science-bowl-2019/'

    train_df = read_function('{0}train.{1}'.format(input_dir, suffix))

    test_df = read_function('{0}test.{1}'.format(input_dir, suffix))

    train_labels_df = read_function('{0}train_labels.{1}'.format(input_dir, suffix))

    specs_df = read_function('{0}specs.{1}'.format(input_dir, suffix))

    sample_submission_df = read_function('{0}sample_submission.{1}'.format(input_dir, suffix))

    print('Data has been read...')

    return train_df, test_df, train_labels_df, specs_df, sample_submission_df

%%time

kernel=True

train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data(kernel=kernel)
#taken from https://www.kaggle.com/jakevdp/altair-kaggle-renderer

# Define and register a kaggle renderer for Altair





KAGGLE_HTML_TEMPLATE = """

<style>

.vega-actions a {{

    margin-right: 12px;

    color: #757575;

    font-weight: normal;

    font-size: 13px;

}}

.error {{

    color: red;

}}

</style>

<div id="{output_div}"></div>

<script>

requirejs.config({{

    "paths": {{

        "vega": "{base_url}/vega@{vega_version}?noext",

        "vega-lib": "{base_url}/vega-lib?noext",

        "vega-lite": "{base_url}/vega-lite@{vegalite_version}?noext",

        "vega-embed": "{base_url}/vega-embed@{vegaembed_version}?noext",

    }}

}});

function showError(el, error){{

    el.innerHTML = ('<div class="error">'

                    + '<p>JavaScript Error: ' + error.message + '</p>'

                    + "<p>This usually means there's a typo in your chart specification. "

                    + "See the javascript console for the full traceback.</p>"

                    + '</div>');

    throw error;

}}

require(["vega-embed"], function(vegaEmbed) {{

    const spec = {spec};

    const embed_opt = {embed_opt};

    const el = document.getElementById('{output_div}');

    vegaEmbed("#{output_div}", spec, embed_opt)

      .catch(error => showError(el, error));

}});

</script>

"""



class KaggleHtml(object):

    def __init__(self, base_url='https://cdn.jsdelivr.net/npm'):

        self.chart_count = 0

        self.base_url = base_url

        

    @property

    def output_div(self):

        return "vega-chart-{}".format(self.chart_count)

        

    def __call__(self, spec, embed_options=None, json_kwds=None):

        # we need to increment the div, because all charts live in the same document

        self.chart_count += 1

        embed_options = embed_options or {}

        json_kwds = json_kwds or {}

        html = KAGGLE_HTML_TEMPLATE.format(

            spec=json.dumps(spec, **json_kwds),

            embed_opt=json.dumps(embed_options),

            output_div=self.output_div,

            base_url=self.base_url,

            vega_version=alt.VEGA_VERSION,

            vegalite_version=alt.VEGALITE_VERSION,

            vegaembed_version=alt.VEGAEMBED_VERSION

        )

        return {"text/html": html}

    

alt.renderers.register('kaggle', KaggleHtml())

print("Define and register the kaggle renderer. Enable with\n\n"

      "    alt.renderers.enable('kaggle')")
%%time

i = -1

#each activity within a game session lasts from its timestamp until the timestamp of the next activity

train_df['timestamp'] = pd.to_datetime(train_df['timestamp']).dt.tz_localize(None)

train_df['timestamp{0}'.format(i)] =train_df.groupby(['installation_id', 'game_session'])['timestamp'].shift(i)
alt.renderers.enable('kaggle')

for installation_id in train_labels_df.installation_id.unique()[:5]:

    print('installation_id - {0}'.format(installation_id))

    display(train_labels_df[train_labels_df['installation_id']==installation_id])

    data = train_df[train_df['installation_id']==installation_id]

    data = data.dropna()

    for day in data.timestamp.dt.day.unique():

        d= data[data['timestamp'].dt.day==day]

        print(d.timestamp.iloc[0],'-',d.timestamp.iloc[-1])

        d['from'] = d['timestamp']

        d['to'] = d['timestamp-1']

        d['activity'] = d['title']

        #data

        c = alt.Chart(d).mark_bar().encode(

            x='from',

            x2='to',

            y='activity',

            color=alt.Color('activity', scale=alt.Scale(scheme='dark2')))

        c.display()