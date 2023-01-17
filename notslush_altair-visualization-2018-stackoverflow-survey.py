# Setup
import csv
import json
import re
import numpy as np
import pandas as pd
import altair as alt

from collections import Counter, OrderedDict
from IPython.display import HTML
from  altair.vega import v3

# The below is great for working but if you publish it, no charts show up.
# The workaround in the next cell deals with this.
#alt.renderers.enable('notebook')

HTML("This code block contains import statements and setup.")
##-----------------------------------------------------------
# This whole section 
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
    "This code block sets up embedded rendering in HTML output and<br/>",
    "provides the function `render(chart, id='vega-chart')` for use below."
)))
#----------------------------------------------------------------------
# Read the schema: it contains the column name and the survey question.

rdr = csv.reader(open("../input/survey_results_schema.csv"))
next(rdr)  # Discard the first line -- it's the header.

# An OrderedDict preserves column order
questions = OrderedDict()
for column, full_question in rdr:
    questions[column] = full_question

#-----------------------
# Group related columns.
# --> Some columns have the same name with just a number at the end.
grouped_q_matcher = r'(?P<col>\D+)(?P<num>\d+)'  # Alphabetical then numeric
grouped_questions = {}
for q, description in questions.items():
    match = re.match(grouped_q_matcher, q)
    if match:
        col, num = match['col'], match['num']
        if col not in grouped_questions:
            grouped_questions[col] = {}
        # Next prune the descriptions to only use the last sentence
        desc = re.split(r'(important. |preferred. |interested. |statements[?:] )', description)[-1]
        grouped_questions[col][num] = desc


#----------------
# Ingest the data

colnames = list(questions.keys())
# The currency columns are combined into `ConvertedSalary` so we don't need the original values.
unused_cols = ['Currency', 'Salary', 'SalaryType', 'CurrencySymbol']
# Also remove opinions about the survey.
unused_cols.extend(['SurveyTooLong', 'SurveyEasy'])
usecols = [c for c in colnames if c not in unused_cols]
types = dict((c, np.dtype('float') if 'Salary' in c else np.dtype('str')) for c in colnames)

df = pd.read_csv('../input/survey_results_public.csv',
                 names=colnames,
                 skiprows=1,
                 index_col='Respondent',
                 usecols=usecols,
                 dtype=types)
        
        
HTML(
    "This block does the following:<br/><ul>"
    "<li>Loads the column names and questions into <code>questions</code>.</li>"
    "<li>Separates out groups of questions (e.g. 'AssessJob1-5') in <code>grouped_questions</code>.</li>"
    "<li>Ingests the data into a data frame <code>df</code>.</li>"
    "<li>Sets every column except <code>ConvertedSalary</code> to be a string.</li>"
    "</ul>"
)
# The below is great for working but if you publish it, no charts show up.
# Use the `render(chart, id='vega-chart')` workaround instead
#v3.renderers.enable('notebook')  

def word_cloud(df, pixwidth=720, pixheight=450, column="index", counts="count"):
    data= [dict(name="dataset", values=df.to_dict(orient="records"))]
    wordcloud = {
        "$schema": "https://vega.github.io/schema/vega/v3.json",
        "width": pixwidth,
        "height": pixheight,
        "padding": 0,
        "title": "Hover to see number of responese from each country",
        "data": data
    }
    scale = dict(
        name="color",
        type="ordinal",
        range=["cadetblue", "royalblue", "steelblue", "navy", "teal"]
    )
    mark = {
        "type":"text",
        "from":dict(data="dataset"),
        "encode":dict(
            enter=dict(
                text=dict(field=column),
                align=dict(value="center"),
                baseline=dict(value="alphabetic"),
                fill=dict(scale="color", field=column),
                tooltip=dict(signal="datum.count + ' respondents'")
            )
        ),
        "transform": [{
            "type": "wordcloud",
            "text": dict(field=column),
            "size": [pixwidth, pixheight],
            "font": "Helvetica Neue, Arial",
            "fontSize": dict(field="datum.{}".format(counts)),
            "fontSizeRange": [10, 60],
            "padding": 2
        }]
    }
    wordcloud["scales"] = [scale]
    wordcloud["marks"] = [mark]
    # return v3.vega(wordcloud)  ## return the dictionary instead when using `render` instead of altair.vega.v3
    return wordcloud

HTML("This block defines the function <code>word_cloud</code> that's used below.<br/>"
          "With tooltips -- thanks Altair!")
render(word_cloud(df.Country.value_counts().to_frame(name="count").reset_index(), pixheight=600, pixwidth=900))
def plot_col(colname, width=550, height=350, meaning=None, cutoff=.6, pad=.25):
    # The purpose of the padding is s the highlight will turn off
    # in the bigger matrices
    pad = alt.Scale(paddingOuter=pad)
    meaning = meaning or colname
    title = "{}: {}".format(meaning, questions[colname])
    tmp = df[colname].str.get_dummies(';')
    pcts_bycol = tmp.mean().sort_values(ascending=False)
    tots_bycol = tmp.sum().sort_values(ascending=False)
    tmp = tmp.reindex(pcts_bycol.index, axis=1)
    tc = tmp.corr()
    tc['Count'] = tots_bycol
    tc['Percent of total'] = pcts_bycol
    tc=tc.reset_index().melt(id_vars=['index', 'Count', 'Percent of total'], var_name=colname, value_name="Percent overlap")

    index_selector = alt.selection(name="index_selector", on="mouseover", type="single", encodings=['x'])
    col_selector = alt.selection(name="col_selector", on="mouseover", type="single", encodings=['y'])

    heatmap = alt.Chart(
    ).mark_rect().encode(
        x=alt.X('index', axis=dict(title=meaning), sort=None, type="nominal", scale=pad),
        y=alt.Y(colname, axis=None, sort=None, type="nominal", scale=pad),
        color=alt.Color('Percent overlap',
                        scale=alt.Scale(scheme='blues', domain=[0, cutoff]),
                        type="quantitative",
                        legend=alt.Legend(format=".0p", title="Percent overlap (truncated)")),
        opacity=dict(value=.35,
                     condition=dict(
                     value=1,
                     selection={"or": ["index_selector", "col_selector"]}))
    ).properties(
        width=height,
        height=height,
        selection=index_selector + col_selector
    )

    bars = alt.Chart(
    ).mark_bar().encode(
        x=alt.X('Count', type="quantitative", aggregate='max', axis=alt.Axis(title="Number of respondents")),
        y=alt.Y('index', type="nominal", sort=None, axis=alt.Axis(title=None), scale=pad),
        color=alt.ColorValue('skyblue')
    ).properties(
        width=width - height,
        height=height
    )
    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text=alt.Text('Percent of total', aggregate='max', format='.0p'),
        color=alt.ColorValue('#000000')
    )

    return alt.hconcat(bars + text, heatmap, data=tc, title=title)


HTML("This block defines the function <code>plot_col</code> that's used below.")
render(plot_col('DevType', meaning='Developer type'))
devType = df['DevType'].str.get_dummies(';')

def augment_dev(colnames):
    dev_type_colnames = devType.columns
    tmp = devType.copy()
    for c in colnames:
        tmp[c] = df[c]
    tmp = tmp.melt(id_vars=colnames, value_vars=dev_type_colnames, var_name='DevType')
    tmp = tmp[tmp.value == 1][colnames + ['DevType']]
    return tmp.dropna()


def plot_preference_change(category, height=450, width=650):
    #------------------------------------------ Data prep
    id_col = 'index'
    used = 'WorkedWith'
    wanted = 'DesireNextYear'
    #-------------------------- used --
    colname = category + used
    aug = augment_dev([colname])
    dev_plus = aug.DevType.reset_index()
    tmp = (
        aug[colname]
          .str.get_dummies(';')
          .reset_index()
          .melt(id_vars=[id_col], var_name='thing', value_name='used')
    )
    dev_plus = pd.merge(dev_plus, tmp, on='index')
    #---------------------- wanted --
    colname = category + wanted
    tmp2 = (
        augment_dev([colname])[colname]
          .str.get_dummies(';')
          .reset_index()
          .melt(id_vars=[id_col], var_name='thing', value_name='wants')
    )
    dp = (
        pd.merge(dev_plus, tmp2, on=[id_col, 'thing'])
          .groupby(['DevType', 'thing', 'used', 'wants'])
          .size()
          .reset_index(name='Count')
    )
    dp = dp.assign(usedwants=(
        dp[['used', 'wants']].apply(lambda x: ('used' if x.used else '') + ('want' if x.wants else ''), axis=1)
    ))
    dp = dp[dp.usedwants != '']
    dp = dp.groupby(['DevType', 'thing','usedwants']).Count.sum().unstack().reset_index()
    dp['used'] = - dp.used
    dp['want_stacked'] = dp.usedwant + dp.want
    dp.sort_values(by=['want_stacked'], ascending=False, inplace=True)
    #------------------------------------------ Chart
    dev_selector = alt.selection(name="dev_selector", type="single", encodings=['y'])
    agg_sort = dict(op='sum', field='want_stacked', order='descending')
    x_ax = alt.Axis(title='Respondents')
    y_ax = alt.Axis(title='', offset=3, ticks=False, domain=False)
    dev_type_bar = (
        alt.Chart(title='Click on a developer type').mark_bar()
          .encode(y=alt.Y('DevType',  sort=agg_sort),
                  x=alt.X('sum(want_stacked):Q'),
                  color=alt.condition(dev_selector,
                                      alt.ColorValue('mediumslateblue'),
                                      alt.ColorValue('rebeccapurple')))
          .properties(height=height, width=width/5, selection=dev_selector)
    )
    bar = (
        alt.Chart().mark_bar()
          .encode(y=alt.Y('thing:N', sort=None, axis=y_ax))
    )
    used = bar.encode(x=alt.X('sum(used):Q', axis=x_ax), color=alt.ColorValue('goldenrod'))
    usedwant = bar.encode(x=alt.X('sum(usedwant):Q'), color=alt.ColorValue('gray'))
    want = bar.encode(x=alt.X('sum(usedwant):Q'), x2=alt.X2('sum(want_stacked):Q'), color=alt.ColorValue('teal'))
    #f = {"and": [{'selection': 'dev_selector'}, {'field':'category', 'equal':category}]}
    layered_chart = (
      alt.LayerChart(layer=(used, usedwant, want), title=category + ": yellow(plan to stop using)  gray(used + will use)  teal(learning next year)")
        .properties(height=height, width=4*width/5)
    )
    return alt.HConcatChart(data=dp, hconcat=[dev_type_bar, layered_chart.transform_filter(dev_selector)])

    
HTML(
    "This block defines the function <code>plot_preference_change</code> that's used below.<br/>"
    "The colors on the right-hand chart are:<ul>"
    "<li><span style='color:goldenrod'>Goldenrod</span> — used a tool this year and won't use it next year.</li>"
    "<li><span style='color:gray'>Gray</span> — used a tool this year and will continue using it.</li>"
    "<li><span style='color:teal'>Teal</span> — have not used the tool but want to use it next year.</li></ul>"
)
render(plot_col('LanguageWorkedWith', meaning='Coding languages', width=720, height=600, pad=.5))
render(plot_preference_change('Language'))
render(plot_col('DatabaseWorkedWith', meaning='Databases', width=500, height=400, pad=.42))
render(plot_preference_change('Database', width=650))
render(plot_col('PlatformWorkedWith', meaning='Platforms', pad=.42))
render(plot_preference_change('Platform'))
render(plot_col('FrameworkWorkedWith', meaning='Frameworks', pad=.42))
render(plot_preference_change('Framework', height=300, width=500))
devType = df['DevType'].str.get_dummies(';')
print("Dimensions:", devType.shape,
      "total roles: {:,}".format(devType.sum().sum()),
      "Avg roles per person: {:0.1f}".format(devType.sum().sum() / devType.shape[0]))

def plot_cat_vs_dev(colname, title="", block_size=12, cutoff=0.5, padding=.8):
    # First construct the dataset. Rows: developer, Cols: the `colname` argument
    dev_plus = augment_dev([colname])
    expanded_category = dev_plus[colname].str.get_dummies(';')
    dev_plus = pd.concat([expanded_category, dev_plus.drop([colname], axis=1)], axis=1)
    dev_plus = dev_plus.reindex(['DevType'] + list(expanded_category.columns), axis=1)
    value_name = "Percent of developers"
    group_means = dev_plus.groupby('DevType').mean().reset_index().melt(
        id_vars=['DevType'],
        var_name=colname,
        value_name=value_name)
    #
    # Next set up the chart.
    x_selector = alt.selection(name="x_selector", on="mouseover", type="single", encodings=['x'])
    y_selector = alt.selection(name="y_selector", on="mouseover", type="single", encodings=['y'])
    pad = alt.Scale(paddingOuter=padding)
    sort_order = alt.SortField(op="mean", field=value_name, order="descending")
    #
    # Last make the chart.
    heatmap = alt.Chart(data=group_means, title=title
    ).mark_rect().encode(
        x=alt.X(colname, sort=sort_order, type="nominal", scale=pad),
        y=alt.Y('DevType', type="nominal", scale=pad),
        color=alt.Color(value_name,
                        scale=alt.Scale(scheme='blues', domain=[0, cutoff]),
                        type="quantitative",
                        legend=alt.Legend(format=".0p",
                                          title="{} (cut off at {})".format(value_name, cutoff))),
        opacity=dict(value=.5,
                     condition=dict(
                     value=1,
                     selection={"or": ["x_selector", "y_selector"]}))
    ).properties(
        width=len(group_means[colname].unique()) * block_size,
        height= len(group_means["DevType"].unique())* block_size,
        selection=x_selector + y_selector
    )
    return heatmap


HTML("This block defines the function <code>augment_dev</code> that's used below.")
render(
    plot_cat_vs_dev(
        "LanguageWorkedWith",
        title = "Fraction of group using language (people are double counted in every group they marked)")
)
new_col = 'Years'
dev_plus = augment_dev(['YearsCodingProf', 'YearsCoding'])
dev_plus = dev_plus.melt(id_vars=['DevType'], var_name='Situation', value_name=new_col)
dev_tots = dev_plus.groupby(['DevType', 'Situation']).size().reset_index(name="Total")
__ = dev_plus.groupby(['DevType', 'Situation', new_col]).size().reset_index(name="Count")
__ = __.merge(dev_tots)
__['Percent of group'] = __.Count / __.Total


##------------------------------------
# For the sorted data:
#   Because of a quirk in Vega with multi-layered plots
#   (https://github.com/vega/vega-lite/issues/2038)
#   you can't sort categories by giving a literal sort order.
#   --> But you *can* sort the dataset and then force `sort=None`
#       which is what we will do.
year_order = sorted(__[new_col].unique(), key=lambda s: float(re.match('\d+', s)[0]), reverse=True)
dev_order = (
    dev_plus[dev_plus.Situation=='YearsCodingProf']
      .groupby('DevType')[new_col]
      .agg(lambda x: x.str.extract('(?P<yr>^\d+)').astype('float').mean())
      .rename('AvgYears').sort_values(ascending=False).reset_index()
)

# All of this work is still to get sorted data
foo = pd.DataFrame({new_col:year_order, "i":range(len(year_order))}).merge(__).sort_values("i")
bar = dev_order.merge(foo).sort_values(["Situation", "i", "AvgYears"], ascending=False)
baz = bar.drop("i", axis=1)

# For mouseover
pad = alt.Scale(paddingOuter=.8)
x_selector = alt.selection(name="x_selector", on="mouseover", type="single", encodings=['x'])
y_selector = alt.selection(name="y_selector", on="mouseover", type="single", encodings=['y'])

base = alt.Chart().encode(
        alt.X(new_col, type="nominal", sort=None, scale=pad, title=""),
        alt.Y('DevType',
              type="nominal",
              sort=None,
              scale=pad,
              title="DevType (sorted by average experience)")
)

overall = base.mark_rect().encode(
    alt.Color('Percent of group',
              type='quantitative',
              scale=alt.Scale(scheme='blues', domain=[0, baz['Percent of group'].max()]),
              legend=alt.Legend(format=".0p", title="Overall: percent"))
).transform_filter(
    'datum.Situation == "YearsCoding"'
)

professional = base.mark_point().encode(
    alt.Size('Percent of group',
             type='quantitative',
             legend=alt.Legend(format=".0p", title="As professional: percent")),
    alt.Color(value='slategray',
              condition=dict(value='navy', selection={"or": ["x_selector", "y_selector"]}))
).transform_filter(
    'datum.Situation == "YearsCodingProf"'
).properties(
  selection=x_selector + y_selector
)

avg_years = alt.Chart().mark_bar().encode(
        alt.Y('DevType', type="nominal", sort=None, scale=alt.Scale(paddingOuter=.8, paddingInner=.2), axis=None),
        alt.X('AvgYears', type="quantitative", title="Average years coding"),
        alt.ColorValue('#bdd7e7')
)
year_text = (
    avg_years
      .mark_text(align='right', baseline='middle', dx=-3, fontWeight=alt.FontWeightNumber(100))
      .encode(
          alt.Text('AvgYears', type="quantitative", format=".1f"),
          alt.ColorValue("navy")
      )
)

HTML("This block defines the chart layers <code>overall</code> and <code>professional</code> used below.")
render(
    alt.HConcatChart(
        data=baz,
        hconcat=[
            alt.LayerChart(layer=[overall, professional], title="Distribution of experience levels", width=300),
            alt.LayerChart(layer=[avg_years, year_text], title="Average years experience", width=200)
        ]
    )
)
def get_lower(field):
    n = field.split(' ', 1)[0]
    try:
        return float(n.replace(',', ''))
    except ValueError:
        return 0
    
def get_upper(field):
    n = field.split(' ')[2]
    try:
        return float(n.replace(',', ''))
    except ValueError:
        return 15000
    
co_data = augment_dev(['CompanySize']).groupby(['DevType', 'CompanySize']).size().reset_index(name='Count')
co_data['LowerBound'] = co_data['CompanySize'].map(get_lower)
co_data['UpperBound'] = co_data['CompanySize'].map(get_upper)
co_data.sort_values(by='Count', ascending=False, inplace=True)
    
dev_selector = alt.selection(name="dev_selector", type="single", encodings=['y'], resolve="global", empty="all")
co_selector = alt.selection(name="co_selector", type="single", encodings=['x'], resolve="global", empty="all")

y_ax = alt.Axis(title='', offset=3, ticks=False, domain=False)
dev_type_bar = (
    alt.Chart(title='Click on a developer type').mark_bar()
        .encode(y=alt.Y('DevType:N',  sort=None),  #dict(op='sum', field='Count', order='descending')),
                x=alt.X('sum(Count):Q', axis=alt.Axis(title='Respondents')),
                color=dict(value='rebeccapurple',
                           condition=dict(value='mediumslateblue', selection='dev_selector')))    
        .properties(selection=dev_selector, width=100, height=300)
        .transform_filter(co_selector)
)
company_size_rect = (
    alt.Chart(title='...or click on a company size (click outside to reset)').mark_rect()
        .encode(y=alt.Y('sum(ht):Q', scale=dict(type='log'), axis=alt.Axis(title='Respondents')),
                x=alt.X('LowerBound:Q',
                        axis=alt.Axis(title='Company size (top bin is 10,000+ employees)',
                                      maxExtent=max(co_data.UpperBound))),
                x2=alt.X2('UpperBound:Q'),
                color=dict(value='steelblue',
                           condition=dict(value='lightblue', selection='co_selector')))
        .properties(selection=co_selector, width=480, height=300)
        .transform_calculate(ht='datum.Count / (datum.UpperBound - datum.LowerBound)')
        .transform_filter(dev_selector)
)
 
HTML(
    "This block provides <code>co_data, dev_type_bar</code>, and <code>company_size_rect</code> used below."
)
render(alt.HConcatChart(data=co_data, hconcat=[dev_type_bar, company_size_rect]))
order = [
    'Less than a month', 'One to three months',  'Three to six months',
    'Six to nine months', 'Nine months to a year', 'More than a year'
]
    
up_data = augment_dev(['TimeFullyProductive']).groupby(['DevType', 'TimeFullyProductive']).size().reset_index(name='Count')
up_data.sort_values(by='Count', ascending=False, inplace=True)

dev_selector = alt.selection(name="dev_selector", type="single", encodings=['y'], resolve="global", empty="all")
up_selector = alt.selection(name="up_selector", type="single", encodings=['x'], resolve="global", empty="all")

dev_type_bar = (
    alt.Chart(title='Click on a developer type').mark_bar()
        .encode(y=alt.Y('DevType:N',  sort=None),  #dict(op='sum', field='Count', order='descending')),
                x=alt.X('sum(Count):Q', axis=alt.Axis(title='Respondents')),
                color=dict(value='rebeccapurple',
                           condition=dict(value='mediumslateblue', selection='dev_selector')))    
        .properties(selection=dev_selector, width=100, height=300)
        .transform_filter(up_selector)
)

y_ax = alt.Axis(title='', offset=3, ticks=False, domain=False)
time_bar = (
    alt.Chart(title='...or click on expected new hire time to productivity').mark_rect()
        .encode(y=alt.Y('sum(Count):Q', axis=alt.Axis(title='Respondents')),
                x=alt.X('TimeFullyProductive:N',
                        sort=order,
                        axis=alt.Axis(title='')),
                color=dict(value='steelblue',
                           condition=dict(value='lightblue', selection='up_selector')))
        .properties(selection=up_selector, width=450, height=300)
        .transform_filter(dev_selector)
)
 
HTML(
    "This block provides <code>up_data</code> and <code>time_bar</code> used below."
)
render(alt.HConcatChart(data=up_data, hconcat=[dev_type_bar, time_bar]))
import statsmodels.api as sm;  # Semicolon to suppress deprecation warning
from statsmodels.formula.api import ols;


def number_of(colname):
    return re.search('\d+$', colname).group(0)
    

def anova_for(category, convert_map=None):
    print('Testing for difference across developers in {}:'.format(category))
    colnames = [category + k for k in sorted(grouped_questions[category])]
    dev_plus = augment_dev(colnames)
    all_counts = dev_plus.melt(id_vars='DevType', var_name=category)
    if convert_map is not None:
        all_counts.value = [convert_map[v] if v in convert_map else None for v in all_counts.value]
    all_counts.value = all_counts.value.astype(float)
    #fit = ols('value ~ {}'.format(category), data=all_counts).fit()
    fit_with_dev = ols('value ~ {} + DevType'.format(category), data=all_counts).fit()
    table = sm.stats.anova_lm(fit_with_dev, typ=2) # Type 2 Anova DataFrame
    return table

HTML(
    "This block defines <code>anova_for</code> which compares variation "
    "in response values across developer type.")
def plot_grouped(category, title=None, convert_map=None, width=620, height=None):
    title = title if title else category
    if convert_map is None:
        order = sorted(grouped_questions[category])
    else:
        order = [x[0] for x in sorted(convert_map.items(), key=lambda x:x[-1])]
    for q in sorted(grouped_questions[category].values()):
        print(q)
    colnames = [category + k for k in sorted(grouped_questions[category])]
    # Set the individual chart dimensions
    height = height or 20 * len(colnames)
    individual_chart_width = min(height, width / (1 +len(colnames)))
    tmp = df[colnames]
    tmp.columns = [grouped_questions[category][k] for k in sorted(grouped_questions[category])]
    tmp = tmp.melt(var_name=title)
    tmp[title]
    tots = tmp.groupby(title).size().reset_index(name="Total")
    __ = tmp.groupby([title, "value"]).size().reset_index(name="Count")
    __ = __.merge(tots)
    __["Percent"] = __.Count / __.Total
    c = alt.Chart(__).mark_bar().encode(
        y=alt.Y("value", title="Choice rank", sort=order),
        x=alt.X('Percent', title="Percent of group", axis=alt.Axis(format='.0p')),
        color=alt.ColorValue('skyblue'),
        column=title,
        tooltip=alt.Tooltip("my_tooltip:N")
    ).transform_calculate(
        my_tooltip='datum["{}"] + " (" + round(datum.Percent*100) + "%)"'.format(title)
    ).properties(width=individual_chart_width, height=height)
    return c


def plot_different_cats(category, width=600, height=300, convert_map=None, xtitle=None):
    #------------------------------------------------------------------------ Data
    for q in sorted(grouped_questions[category].values()):
        print(q)
    colnames = [category + k for k in sorted(grouped_questions[category])]
    dev_plus = augment_dev(colnames)
    choice_min, choice_max = 1, len(colnames)
    dev_plus.columns = [
        c if c == 'DevType' else grouped_questions[category][number_of(c)]
        for c in dev_plus.columns
    ]
    tmp = dev_plus.melt(id_vars='DevType', var_name=category)
    if convert_map is not None:
        tmp.value = [convert_map[v] if v in convert_map else None for v in tmp.value]
        choice_min, choice_max = min(tmp.value), max(tmp.value)
    xtitle = xtitle or "Group average rank ({} to {}) — smaller is better".format(choice_min, choice_max)
    tmp.value = tmp.value.astype(float)
    __ = tmp.groupby(['DevType', category]).value.mean().reset_index(name="Value")
    #----------------------------------------------------------------------------- Chart
    domain = [max(choice_min, __.Value.min() - .2), min(choice_max, __.Value.max() + .2)]
    scale = alt.Scale(domain=domain)
    dev_selector = alt.selection(name="dev_selector", on="mouseover", type="single", fields=['DevType'])
    c = (
        alt.Chart(__, title="Hover over the points in the line to see developer type")
        .mark_line(strokeWidth=3, point=alt.MarkConfig(shape="diamond"))
        .encode(
            x=alt.X("Value:Q", scale=scale, title=xtitle),
            y=alt.Y(category, type="nominal"),
            color=alt.Color('DevType:N', scale=alt.Scale(scheme='viridis'), legend=None),
            tooltip='DevType:N',
            opacity=alt.condition(dev_selector, alt.OpacityValue(1), alt.OpacityValue(.1)))
        .properties(selection=dev_selector)
    )
    points = (
        alt.Chart().mark_point(shape="diamond", size=15)
        .encode(
            x=alt.X("Value:Q", scale=scale),
            y=alt.Y(category, type="nominal"),
            tooltip='DevType:N',
            color=alt.Color('DevType:N', scale=alt.Scale(scheme='viridis'), legend=None))
        .transform_filter(dev_selector)
    )
    return c.properties(width=width, height=height)  #alt.layer(c, points, data=__, width=width, height=height)

HTML("This block defines <code>plot_grouped</code> and <code>plot_different_cats</code> "
     "to show the aggregate responses below.")
lookups = {'Strongly agree':2, 'Agree':1, 'Neither Agree nor Disagree':0 , 'Disagree':-1, 'Strongly disagree':-2}
render(
    plot_different_cats('AgreeDisagree',
                        convert_map=lookups,
                        xtitle='Strongly disagree = -2; neutral = 0; Strongly agree = 2')
)
lookups = {'Strongly agree':2, 'Agree':1, 'Neither Agree nor Disagree':0, 'Disagree':-1, 'Strongly disagree':-2}
print(anova_for('AgreeDisagree', convert_map=lookups))
category = 'HypotheticalTools'

lookups = {'Not at all interested':5,
           'Somewhat interested':4,
           'A little bit interested':3,
           'Very interested':2,
           'Extremely interested':1}
render(plot_grouped(category, title='SO Tools', convert_map=lookups))
#category = 'HypotheticalTools'
print(anova_for(category, convert_map=lookups))
render(plot_different_cats(category, convert_map=lookups))
import difflib  # For sorting the response keys later

#-------------------------------------------------
# Set up the questions that come in related groups
related_groups = ['AdBlocker', 'AI', 'Ethic', 'Hours', 'StackOverflow', 'Survey']
related_questions = {g:{} for g in related_groups}
for q, desc in questions.items():
    if any([q.startswith(g) for g in related_groups]):
        g = next(g for g in related_groups if q.startswith(g))
        subtopic = q[len(g):]
        related_questions[g][subtopic] = desc

        
def get_question_order(colname, desired):
    responses = df[colname].dropna().unique()
    def match(d):
        return difflib.get_close_matches(d, responses, cutoff=0)[0]
    return [match(d) for d in desired]

        
def plot_related(groupname, width=100, height=80, ncol=None, with_dev=False, subset=None, question_orders={}):
    #-------------------------------------------------------------------- Data setup
    if subset is None:
        group_cols = [groupname + q for q in related_questions[groupname]]
    else:
        group_cols = [groupname + s for s in subset]
    if with_dev:
        dev_plus = augment_dev(group_cols)
        responses = dev_plus.groupby(['DevType'] + group_cols).size().reset_index(name='Count')
        if responses.shape[0] > 5000:
            print("Switching to plot without `DevType` option because the dataset is too big.")
            with_dev = False
    if not with_dev:
        responses = df.groupby(group_cols).size().reset_index(name='Count')
    #-------------------------------------------------------------------- Chart setup
    if with_dev:
        choice = alt.binding_select(options=sorted(dev_plus.DevType.dropna().unique()))
        select_dev = alt.selection_single(name="Developer", fields=['DevType'], bind=choice)
    # One chart for each entry in group_cols
    nchart = len(group_cols)
    ncol = ncol or min(4, nchart)
    nrow = nchart // ncol + 0 if nchart % ncol == 0 else 1
    charts = [[None for j in range(ncol+1)] for i in range(nrow+1)]
    for ij, col in enumerate(group_cols):
        print(questions[col])
        i, j = ij // ncol, ij % ncol
        properties = dict(width=width, height=height)
        transform_window = dict(
            window=[alt.WindowFieldDef(op='sum', field='Count', **{'as': 'TotalCount'})],
            frame=[None, None]
        )
        # question orders -- to specifically sort the Y axis
        try:
            q = next(q for q in question_orders if q in col)
            order = get_question_order(col, question_orders[q])
            y = alt.Y(col + ':N', sort=order, axis=alt.Axis(title=''))
        except StopIteration:  # Didn't find it
            y = alt.Y(col + ':N', axis=alt.Axis(title=''))
        # Add the 'DevType' if we're using it
        if with_dev:
            transform_window['groupby'] = ['DevType']
            if i == j == 0:
                # Add selection to the first chart only
                properties['selection'] = select_dev       
        chart = (
            alt.Chart(title=questions[col]).mark_bar()
            .encode(
                x=alt.X('sum(PercentOfTotal):Q', axis=alt.Axis(title='Respondents', format='.0%')),
                y=y)
            .transform_window(**transform_window)
            .transform_calculate(
                PercentOfTotal="datum.Count / datum.TotalCount")
            .properties(**properties)
        )
        charts[i][j] = chart if not with_dev else chart.transform_filter(select_dev)
    hcharts = []
    for i in range(nrow + 1):
        hchart = None
        for c in charts[i]:
            if c is not None:
                hchart = c if hchart is None else hchart | c
        if hchart is not None:
            hcharts.append(hchart)
    return alt.VConcatChart(data=responses, vconcat=hcharts)


HTML("This block contains code for the <code>plot_related</code> function used below.")
# This question has huge titles, so you have to scroll to see all the answers.
render(plot_related('Ethic', ncol=2))
render(
    plot_related('AI', with_dev=True, ncol=2, width=200, question_orders={
                 'Responsible': ['Nobody', 'developers', 'industry', 'government']
             })
)
render(plot_col('HackathonReasons', meaning="Why attend a hackathon", width=400, height=250))
render(
    plot_related(
        'StackOverflow',
        subset=['DevStory', 'Jobs', 'JobsRecommend'],
        height=120,
        question_orders={
            'Visit': ['Less', 'per month', 'per week', 'Daily', 'Multiple'],
            'JobsRecommend': [str(i) for i in range(10)] + ['10 (']
        }
    )
)
render(
  (
    plot_cat_vs_dev("AdBlockerReasons", title = "Fraction of group disabling adblocker for a given reason")
    |  # hconcat is 'or'
    plot_col('AdBlockerReasons', meaning="Reasons for disabling adblocker", width=200, height=150)
  ).resolve_legend(color='independent')
)
print('\n'.join(df.AdsActions.str.get_dummies(';').columns))
render(
  (
    plot_cat_vs_dev("AdsActions",
                title = "Fraction of group taking given action")
    |  # hconcat is 'or'
    plot_col('AdsActions', meaning="AdsActions", width=300, height=200)
  ).resolve_legend(color='independent')
)
render(plot_different_cats('AdsPriorities'))
print(anova_for('AdsPriorities'))
render(plot_grouped('AdsPriorities', title="Important components of an Ad"))
category = 'AdsAgreeDisagree'
lookups = {
    'Strongly disagree': -2,
    'Somewhat disagree': -1,
    'Neither agree nor disagree': 0,
    'Somewhat agree': 1,
    'Strongly agree': 2}
render(plot_grouped(category, title='Ads', convert_map=lookups, width=500, height=150))
category = 'AdsAgreeDisagree'
print(anova_for(category, convert_map=lookups))
render(plot_different_cats(category, convert_map=lookups, xtitle='Strongly disagree = -2; neutral = 0; Strongly agree = 2'))
render(plot_grouped('JobContactPriorities', title='Contact method'))
print(anova_for('JobContactPriorities'))
render(plot_different_cats('JobContactPriorities'))
category = "JobEmailPriorities"
render(plot_grouped(category, title='Email'))
print(anova_for('JobContactPriorities'))
render(plot_different_cats('JobEmailPriorities'))
category = "AssessJob"
lookup = dict((str(k), '{:>2}'.format(k)) for k in range(1, 11))
render(plot_grouped(category, title='Job', convert_map=lookup, width=750))
category = "AssessJob"
print(anova_for(category))
render(plot_different_cats(category))
category = "AssessBenefits"
lookup = dict((str(k), '{:>2}'.format(k)) for k in range(1, 12))
render(plot_grouped(category, title='Benefits', convert_map=lookup, width=800))
category = "AssessBenefits"
print(anova_for(category))
render(plot_different_cats(category))
#--------------------------------- Data
dev_plus = augment_dev(['Country','ConvertedSalary'])

quantiles = (
    dev_plus.groupby('DevType')
    .ConvertedSalary.quantile(q=[.25,.5,.75])
    .unstack().reset_index()
)
quantiles.columns = ['DevType', 'Q1', 'Q2', 'Q3']
quantiles.sort_values(by='Q2', ascending=False, inplace=True)
quantiles['MedianString'] = ['Median: ${:,.0f}'.format(v) for v in quantiles.Q2]
quantiles['IQR'] = ['Q1: ${:,.0f} — Q2: ${:,.0f}'.format(*v) for v in zip(quantiles.Q1, quantiles.Q3)]

by_country = (
    dev_plus.groupby(['DevType', 'Country'])
    .ConvertedSalary.median()
    .reset_index(name="Median Salary")
    .merge(quantiles[['DevType', 'Q2']])
    .sort_values(by=['Q2'], ascending=False)
    .drop('Q2', axis=1)
)


#------------------ Chart
width = 700
height = 400

choice = alt.binding_select(options=sorted(by_country.Country.dropna().unique()))
selector = alt.selection_single(name="Country", fields=['Country'], bind=choice)


country_chart = alt.Chart(by_country).mark_tick(size=10).encode(
    x=alt.X('Median Salary:Q', axis=alt.Axis(title="Median annual salary, USD")),
    y=alt.Y('DevType:N', sort=None),
    color=alt.condition(selector, alt.ColorValue('rebeccapurple'), alt.ColorValue('thistle')),
    opacity=alt.condition(selector, alt.OpacityValue(1), alt.OpacityValue(0.25)),   
    tooltip='Country:N'
).properties(width=width, height=height, selection=selector)


quantile_chart = alt.Chart(quantiles).mark_bar(color='lightblue', opacity=0.75).encode(
    x='Q1:Q',
    x2='Q3:Q',
    y=alt.Y('DevType:N', sort=None),
    tooltip='IQR:N'
).properties(width=width, height=height)

median_chart = alt.Chart(quantiles).mark_tick(color='black', size=21, strokeOpacity=.5).encode(
    x='Q2:Q',
    y=alt.Y('DevType:N', sort=None),
    tooltip='MedianString:N'
).properties(width=width, height=height)


HTML(
    "This section defines <code>quantile_chart</code>, <code>country_chart</code>, "
    "and <code>median_chart</code>, which are used below to make the Salary plot."
)
render(
    alt.LayerChart(
        title='Median salary by country and developer type. Zoom + pan for detail + pick a country at the bottom.',
        layer=[quantile_chart, country_chart, median_chart]
    ).configure_tick(thickness=2).interactive()
)