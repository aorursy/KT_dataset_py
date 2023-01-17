# Kaggle specific code

!pip install factor_analyzer

# https://www.kaggle.com/general/63534#672910
!pip install altair vega_datasets notebook vega # needs internet in settings (right panel)

# https://www.kaggle.com/jakevdp/altair-kaggle-renderer
# Define and register a kaggle renderer for Altair

import json
import altair as alt
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
# Import necessary libraries 

import pandas as pd
import numpy as np 
import altair as alt
import statsmodels.formula.api as smf
import statsmodels.api as sm  
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy import stats
import itertools
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
from scipy.stats import levene, normaltest
# https://github.com/nytimes/covid-19-data
# Cumulative counts of coronavirus cases in the US at the county level
county_infection = pd.read_csv('../input/county-covid-related/us-counties.csv')
county_infection.head()
# let's sort the counties first by date

county_infection['date'] = pd.to_datetime(county_infection['date'])
county_infection = county_infection.sort_values(by='date')
county_infection.tail()
county_infection[(county_infection['state'] == 'Illinois') & (county_infection['county'] == 'Cook')]
county_infection[(county_infection['state'] == 'California') & (county_infection['county'] == 'Santa Clara')]
# https://en.wikipedia.org/wiki/County_(United_States)
# County population and density
county_population = pd.read_csv('../input/county-covid-related/county-population.csv')
county_population.head()
# https://en.wikipedia.org/wiki/Political_party_strength_in_U.S._states
# https://en.wikipedia.org/wiki/List_of_United_States_governors
# State party affiliation based on house representation
state_party_line = pd.read_csv('../input/county-covid-related/state_party_line.csv')
state_party_line.head()
# Source: https://www.countyhealthrankings.org/
# Access: https://app.namara.io/#/data_sets/579ee1c6-8f66-418c-9df9-d7b5b618c774?organizationId=5ea77ea08fb3bf000c9879a1
# County health information
county_health = pd.read_csv('../input/uncover/UNCOVER/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv')
county_health.head()
county_health.columns[:75]
# Aggregate data related to county infection and basic characteristics
county = county_infection.merge(
    county_population, left_on=['county', 'state'], right_on=['county', 'state']
).merge(
    state_party_line, left_on=['state'], right_on=['state']
)
county.sample(5)
# Count the number of days each county data has
def count_days(series):
    time_series = pd.to_datetime(series)
    first_date = time_series.iloc[0]
    last_date = time_series.iloc[-1]
    
    return (last_date - first_date).days + 1
grouped_county = county.groupby(['state', 'county']).agg(days_counted=('date', count_days))
grouped_county.describe()
grouped_county.shape
# Find the value at the 50 day mark
def county_cumulative_days(series, days = 50):
    # This may not be 100% accurate because perhaps some days are missing, 
    # but that seems to happen rarely. So this should be accurate enough.
    if len(series) < days:
        return series.iloc[-1]
    else:
        return series.iloc[days - 1]
# Group our data in terms of county and aggregate some columns to show overall infection rate 
# and death rate as well as at the 50 day mark
def group_county_data(data):
    grouped_data = data.groupby(['state', 'county']).agg(
        population=('population', lambda x: x.iloc[-1]),
        density_km=('density_km', lambda x: x.iloc[-1]),
        state_house_blue_perc=('state_house_blue_perc', lambda x: x.iloc[-1]),
        state_governor_party=('state_governor_party', lambda x: x.iloc[-1]),
        days_counted=('date', count_days),
        case_sum=('cases', lambda x: x.iloc[-1]),
        death_sum=('deaths', lambda x: x.iloc[-1]),
        case_count_50_days=('cases', county_cumulative_days),
        death_count_50_days=('deaths', county_cumulative_days)
    )
    
    grouped_data = grouped_data[grouped_data['days_counted'] >= 50]
    grouped_data['infection_rate'] = grouped_data['case_sum']/grouped_data['population']*100
    grouped_data['death_rate'] = grouped_data['death_sum']/grouped_data['case_sum']*100
    grouped_data = grouped_data[grouped_data['infection_rate'] != float("inf")]
    grouped_data['infection_rate_50_days'] = grouped_data['case_count_50_days']/grouped_data['population']*100
    grouped_data['death_rate_50_days'] = grouped_data['death_count_50_days']/grouped_data['case_count_50_days']*100
    
    return grouped_data.reset_index()
grouped_county = group_county_data(county)
grouped_county
grouped_county.sample(5)
# Remove state total rows first
county_health = county_health.dropna(subset=['county'])
county_health.sample(5)
county_health.columns
county_health.columns[:100]
excluded_column_words = [
    'quartile',
    'ci_high',
    'ci_low',
    'fips',
    'num',
    'denominator',
    'ratio',
    'population',
]
filtered_columns = county_health.columns[~county_health.columns.str.contains('|'.join(excluded_column_words))]
print(str(len(filtered_columns)) + ' columns remain!')
filtered_county_health = county_health[filtered_columns]
county = grouped_county.merge(
    filtered_county_health, left_on=['county', 'state'], right_on=['county', 'state']
)
county
# Let's see the columns at near 90% cutoff points
county.dropna(thresh=1300, axis=1).info(max_cols=200)
county.dropna(thresh=1370, axis=1).dropna()
county = county.dropna(thresh=1370, axis=1).dropna()
# Exclude columns that won't be used as explanatory variables and can't used in factor analysis
excluded_columns = [
    'state',
    'county', 
    'population',
    'state_house_blue_perc',
    'state_governor_party',
    'days_counted', 
    'case_sum', 
    'death_sum', 
    'case_count_50_days',
    'death_count_50_days', 
    'infection_rate', 
    'death_rate',
    'infection_rate_50_days', 
    'death_rate_50_days',
    'presence_of_water_violation'
]
county_non_factor = county[excluded_columns]
county_factor = county.drop(excluded_columns, axis=1)
len(county_factor.columns)
county_factor.columns
fig = county_factor.hist(
    column=county_factor.columns, 
    xlabelsize=0.1, 
    ylabelsize=0.1, 
    layout=(11, 7), 
    figsize=(10, 10),
    bins=50
)  
[x.title.set_size(0) for x in fig.ravel()]
plt.show()
levene(*county_factor.to_numpy(), center='trimmed')
levene(*county_factor.to_numpy(), center='mean')
kmo_all, kmo_model = calculate_kmo(county_factor)
kmo_model
fa = FactorAnalyzer()

# Using the varimax rotation because it makes it easier to identify each variable with a single factor.
fa.set_params(rotation='varimax')
fa.fit(county_factor)
ev, v = fa.get_eigenvalues()
ev[:30]
plt.scatter(range(1, len(ev)+1), ev)
plt.plot(range(1, len(ev)+1), ev)
plt.title('Scree plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
fa = FactorAnalyzer()
fa.set_params(n_factors=14, rotation='varimax')
fa.fit(county_factor)
factor_loading = pd.DataFrame(fa.loadings_)
factor_loading.index = county_factor.columns
factor_loading.shape
factor_loading
def filter_decent_loadings(factor):
    return factor[(factor > 0.3) | (factor < -0.3)]
for factor in factor_loading.columns:
    print('Factor ' + str(factor + 1) + ' loadings: ')
    print()
    print(filter_decent_loadings(factor_loading[factor]))
    print()
    print()
fa.get_factor_variance()
fa_score_columns = [
    'poor_general_wellbeing_fa_score',
    'housing_burden_fa_score',
    'hispanic_relative_population_fa_score',
    'inverse_sucicde_rate_fa_score',
    'uninsured_rate_fa_score',
    'care_provider_accessibility_fa_score',
    'population_youth_fa_score',
    'crime_risk_fa_score',
    'overall_income_fa_score',
    'population_density_fa_score',
    'native_relative_population_fa_score',
    'black_relative_population_fa_score',
    'urbanization_level_fa_score',
    'poor_food_environment_fa_score',
]

transformed_county_factor = pd.DataFrame(
    fa.transform(county_factor),
    columns=fa_score_columns
)
county = county_non_factor.reset_index(drop=True).join(transformed_county_factor)
# Remove some columns we are interested in for sure
county = county.drop(columns=[
    'population', 
    'state_house_blue_perc', 
    'days_counted',
    'case_sum',
    'death_sum',
    'case_count_50_days',
    'death_count_50_days',
    'presence_of_water_violation'
])
county.sample(5)
county.info()
alt.Chart(county).mark_bar().encode(
    alt.X("infection_rate_50_days", bin=alt.Bin(extent=[0, 3], step=0.02)),
    y='count()',
).properties(
    width=800,
    height=400,
    title='Infection rate at 50 days since first case'
)
infection_rate_50_days_boxcox, lmbda = stats.boxcox(county['infection_rate_50_days'])
lmbda
county['infection_rate_50_days_boxcox'] = infection_rate_50_days_boxcox
alt.Chart(county).mark_line().encode(
    x='infection_rate_50_days_boxcox',
    y='infection_rate_50_days'
).properties(
    title='Infection rate Boxcox transformation relationship'
)
alt.Chart(county).mark_bar().encode(
    alt.X("infection_rate_50_days_boxcox", bin=alt.Bin(extent=[-5, 2], step=0.1)),
    y='count()',
).properties(
    width=800,
    height=400,
    title='Boxcox infection rate at 50 days since first case'
)
county[county['state_governor_party'] == 'blue'].corr(method='pearson')['infection_rate_50_days_boxcox']
county[county['state_governor_party'] == 'red'].corr(method='pearson')['infection_rate_50_days_boxcox']
county.corr(method='pearson')['infection_rate_50_days_boxcox']
alt_y = alt.Y(
    'infection_rate_50_days_boxcox', 
    axis=alt.Axis(values=list(np.linspace(-6, 2, 81))),
    scale=alt.Scale(domain=(-5, 2), clamp=True)
)
alt.Chart(county).mark_point(filled=True, size=22).encode(
    x='inverse_sucicde_rate_fa_score',
    y=alt_y,
    color='state_governor_party'
).properties(
    width=800,
    height=400,
    title='Inverse suicide factor score vs Boxcox infection rate'
)
alt.Chart(county).mark_bar().encode(
    alt.X("inverse_sucicde_rate_fa_score", bin=alt.Bin(extent=[-3, 3], step=0.1)),
    y='count()',
)
alt.Chart(county).mark_point(filled=True, size=22).encode(
    x='overall_income_fa_score',
    y=alt_y,
    color='state_governor_party'
).properties(
    width=800,
    height=400,
    title='Overall income factor score vs Boxcox infection rate'
)
alt.Chart(county).mark_bar().encode(
    alt.X("overall_income_fa_score", bin=alt.Bin(extent=[-3, 3], step=0.1)),
    y='count()',
)
alt.Chart(county).mark_point(filled=True, size=22).encode(
    x='urbanization_level_fa_score',
    y=alt_y,
    color='state_governor_party'
).properties(
    width=800,
    height=400,
    title='Urbanization factor score vs Boxcox infection rate'
)
alt.Chart(county).mark_bar().encode(
    alt.X("urbanization_level_fa_score", bin=alt.Bin(extent=[-3, 3], step=0.1)),
    y='count()',
)
alt.Chart(county).mark_point(filled=True, size=22).encode(
    x='black_relative_population_fa_score',
    y=alt_y,
    color='state_governor_party'
).properties(
    width=800,
    height=400,
    title='Black population factor score vs Boxcox infection rate'
)
alt.Chart(county).mark_bar().encode(
    alt.X("black_relative_population_fa_score", bin=alt.Bin(extent=[-3, 3], step=0.1)),
    y='count()',
)
interaction_term = 'state_governor_party'
response_variable = 'infection_rate_50_days_boxcox'
explanatory_variables = fa_score_columns
explanatory_variables
variable_combinations = []

for variable in explanatory_variables:
    variable_combinations.append([variable, variable + '*' + interaction_term])
formula_combinations = list(itertools.product(*variable_combinations))
print('There are ' + str(len(formula_combinations)) + ' combinations.')
models = []
rsquared_adjs = []
formulas = []
aics = []
bics = []

for combo in formula_combinations:
    explanatory_variable_part = ' + '.join(combo)
    formula = ' '.join([
        'infection_rate_50_days_boxcox ~',
        explanatory_variable_part
    ])
    
    mod = smf.ols(formula=formula, data=county)
    res = mod.fit()

    models.append(res)
    formulas.append(formula)
    rsquared_adjs.append(res.rsquared_adj)
    aics.append(res.aic)
    bics.append(res.bic)
    
    if len(models)%1600 == 0:
        print(str(len(models)) + ' models finished so far!')
result = pd.DataFrame({
    'formula': formulas,
    'rsquared_adj': rsquared_adjs,
    'aic': aics,
    'bic': bics,
    'model': models
})
result.sort_values(by='rsquared_adj', ascending=False).head()
result.sort_values(by='aic').head()
result.sort_values(by='bic').head()
aic_res = result.iloc[result.sort_values(by='aic').iloc[0].name]['model']
bic_res = result.iloc[result.sort_values(by='bic').iloc[0].name]['model']
model_df = pd.DataFrame({
    'aic_model_residual': county['infection_rate_50_days_boxcox'].values - aic_res.fittedvalues,
    'bic_model_residual': county['infection_rate_50_days_boxcox'].values - bic_res.fittedvalues,
    'real_val': county['infection_rate_50_days_boxcox'],
    'aic_model_pred': aic_res.fittedvalues,
    'bic_model_pred': bic_res.fittedvalues,
})
# QQ plot for the AIC model residuals
plt.show(qqplot(model_df['aic_model_residual'], line='s'))
# QQ plot for the BIC model residuals
plt.show(qqplot(model_df['bic_model_residual'], line='s'))
alt.Chart(model_df).mark_point().encode(
    x='aic_model_pred',
    y='aic_model_residual',
) | alt.Chart(model_df).mark_point().encode(
    x='bic_model_pred',
    y='bic_model_residual',
)
print(aic_res.summary())
print(bic_res.summary())