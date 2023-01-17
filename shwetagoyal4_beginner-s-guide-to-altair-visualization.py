# Importing libraries

import pandas as pd
import numpy as np
import altair as alt
#alt.data_transformers.enable('default', max_rows=None) 
music = pd.read_csv("../input/data-analytics-to-study-music-streaming-patterns/spotify.csv", index_col=0)
music.head()
# Source: https://www.kaggle.com/omegaji/altair-render-script/notebook

import json
from IPython.display import HTML

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
alt.renderers.enable('kaggle')  
chart = alt.Chart(music)
alt.Chart(music).mark_bar().encode(
    x='Danceability',
    y='Energy'
)
alt.Chart(music).mark_boxplot().encode(
    x='Genre',
    y='Speechiness'
)
alt.Chart(music).mark_boxplot(extent=2.0).encode(
    x='Genre',
    y='Speechiness'
)
alt.Chart(music).mark_circle(
    color='red',
    opacity=0.3
).encode(
    x='Genre',
    y='Speechiness'
)
alt.Chart(music).mark_point().encode(
    y='Positivity',
    x='Artist', 
    color='Genre'
)
alt.Chart(music).mark_point().encode(
    y='Length(seconds):O',
    x='Artist:N', 
    color='Speechiness:N'
)
alt.Chart(music).mark_point().encode(
    y='Length(seconds):O',
    x='Artist:N', 
    color='Speechiness:Q'
) 
alt.Chart(music).mark_bar().encode(
    alt.X('Positivity', bin=True),
    y='count()'
 )   
alt.Chart(music).mark_square().encode(
    alt.X('Beats per minute', bin=alt.Bin(maxbins=20)),
    alt.Y('Positivity', bin=True),
    size='count()',
    color='Artist:N'
  )
alt.Chart(music).mark_square().encode(
    alt.X('Beats per minute', bin=alt.Bin(maxbins=20)),
    alt.Y('Positivity', bin=True),
    size='count()',
    color='mean(Popularity):Q'
  )
alt.Chart(music).mark_point().encode(
    x='Positivity:Q',
    y='Popularity:Q',
    color='Artist:N',
    tooltip='Genre'
).interactive()
interval = alt.selection_interval()
alt.Chart(music).mark_point().encode(
    x='Danceability:Q',
    y='Positivity:Q',
    color='Artist:N'
).add_selection(
    interval
)
alt.Chart(music).mark_point().encode(
    x='Danceability:Q',
    y='Positivity:Q',
    color=alt.condition(
        interval, 'Artist:N', alt.value('lightgray'))
).add_selection(
     interval 
) 
single = alt.selection_single()
alt.Chart(music).mark_point().encode(
    x='Danceability:Q',
    y='Positivity:Q',
    color=alt.condition(
        single, 'Artist:N', alt.value('lightgray'))
).add_selection(
     single 
) 
alt.Chart(music).mark_bar().encode(
    y='Genre:N',
    color='Artist:N',
    x='count(Genre):Q'
)
base = alt.Chart(music).mark_point().encode(
    y='Popularity',
    color=alt.condition(interval, 'Artist', alt.value('lightgray')),
    tooltip='Genre'
).properties(
    selection=interval
)
base.encode(x='Positivity') | base.encode(x='Speechiness')
points = alt.Chart(music).mark_point().encode(
    x='Danceability:Q',
    y='Positivity:Q',
    color=alt.condition(
        interval, 'Artist:N', alt.value('grey'))
).add_selection(
     interval 
) 

bars = alt.Chart(music).mark_bar().encode(
    y='Genre:N',
    color='Artist:N',
    x='count(Genre):Q'
).transform_filter(
    interval
)

points & bars