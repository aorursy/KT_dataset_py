import sympy as sm

from sympy import Function

from sympy.stats import Normal, cdf



import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import warnings

from IPython.display import Image, HTML

import io





# import sys

# sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
# Define and register a kaggle renderer for Altair



import altair as alt

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
from IPython.display import Math, HTML



def load_mathjax_in_cell_output():

    display(HTML("<script src='https://www.gstatic.com/external_hosted/"

               "mathjax/latest/MathJax.js?config=default'></script>"))



def init_print(sm, use_matplotlib):

    if use_matplotlib:

        sm.init_printing(use_latex='matplotlib')

    else:

        get_ipython().events.register('pre_run_cell', load_mathjax_in_cell_output)



def get_items(items_id, dict_var):

    return [dict_var.get(item_id) for item_id in items_id]
alt.renderers.enable('kaggle')
# init_print(sm, use_matplotlib=True)

sm.init_printing()
s, k, tau, t, T, r, sigma, x, y = sm.symbols('s, k, tau, t, T, r, sigma, x, y', real=True)

s, k, tau, t, T, r, sigma, x, y
class Option:

    def __init__(self, option, name):

        self.name = name

        self.option = option

        self.delta = self.option.diff(s).simplify()

        self.gamma = self.delta.diff(s).simplify()

        self.theta = -self.option.diff(tau).simplify()

        self.vega = self.option.diff(sigma).simplify()

        self.rho = self.option.diff(r).simplify()



        self.option_np = sm.lambdify([s, k, tau, r, sigma], self.option, modules=['numpy', 'sympy'])

        self.delta_np = sm.lambdify([s, k, tau, r, sigma], self.delta, modules=['numpy', 'sympy'])

        self.gamma_np = sm.lambdify([s, k, tau, r, sigma], self.gamma, modules=['numpy', 'sympy'])

        self.theta_np = sm.lambdify([s, k, tau, r, sigma], self.theta, modules=['numpy', 'sympy'])

        self.vega_np = sm.lambdify([s, k, tau, r, sigma], self.vega, modules=['numpy', 'sympy'])

        self.rho_np = sm.lambdify([s, k, tau, r, sigma], self.rho, modules=['numpy', 'sympy'])

  

    def _plot(self, name, s_vals, k_val, tau_vals, r_val, sigma_val, numeric_tau):

        fn = getattr(self, name + '_np')

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            data_dict = {('tau = ' + str(tau_i)) : [fn(x, k_val, tau_i, r_val, sigma_val) for x in s_vals] for tau_i in tau_vals}



        source = pd.DataFrame({

          's': s_vals,

            **data_dict

        }, dtype='float64')



        if 0 in tau_vals:

            source.loc[s_vals == k_val, 'tau = 0'] = 0 if np.isnan(source.loc[s_vals == k_val, 'tau = 0'].values)[0] else source.loc[s_vals == k_val, 'tau = 0']

        # source.fillna(0, inplace=True)



        data = source.melt('s', value_name=name, var_name='tau')

        if numeric_tau:

            data.tau = [x.replace('tau = ', '') for x in data.tau]

            data.tau = data.tau.astype('float64')

        return data



    def plot(self, name, s_vals, k_val, tau_vals, r_val, sigma_val, numeric_tau=True, interactive=True):

        data = self._plot(name, s_vals, k_val, tau_vals, r_val, sigma_val, numeric_tau)

        plot = alt.Chart(data, title=self.name).mark_line().encode(

            x='s',

            y=name,

            color='tau'

        )

        if interactive:

            plot = plot.interactive()

        plot = plot.properties(width=325, height=250)

        return plot



    def plot2(self, ax, palette, name, s_vals, k_val, tau_vals, r_val, sigma_val, numeric_tau=True):

        data = self._plot(name, s_vals, k_val, tau_vals, r_val, sigma_val, numeric_tau)

        plot = sns.lineplot(x="s", y=name, hue="tau", data=data, ax=ax, palette=palette).set_title(self.name)

        return plot
N =  sm.simplify(cdf(Normal('x', 0, 1)))

N
d1 = (sm.ln(s/k) + (r + sigma**2/2)*tau)/(sigma*sm.sqrt(tau))

d2 = d1 - sigma*sm.sqrt(tau)

d1, d2
call_price = s*N(d1) - k*sm.exp(-r*tau)*N(d2)

put_price = call_price + k*sm.exp(-r*tau) - s

call_put_price = call_price - put_price
call = Option(call_price, 'call')

put = Option(put_price, 'put')

call_put = Option(call_put_price, 'call-put')
def plot(name):

    chart = alt.vconcat(alt.hconcat(call.plot(name, s_vals, k_val, tau_vals, r_val, sigma_val,numeric_tau=numeric_tau_val, interactive=interactive_val),

                                  put.plot(name, s_vals, k_val, tau_vals, r_val, sigma_val,numeric_tau=numeric_tau_val, interactive=interactive_val)),

                      call_put.plot(name, s_vals, k_val, tau_vals, r_val, sigma_val,numeric_tau=numeric_tau_val, interactive=interactive_val))

#     with io.StringIO() as f:

#         chart.save(f, format='html')

#         chart_html = f.getvalue()

    return chart



def plot2(name):

    palette = sns.color_palette("Blues_d")

    fig, axs = plt.subplots(2,2, figsize=(15,15))

    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    call.plot2(axs[0, 0], palette, name, s_vals, k_val, tau_vals, r_val, sigma_val,numeric_tau=numeric_tau_val)

    put.plot2(axs[0, 1], palette, name, s_vals, k_val, tau_vals, r_val, sigma_val,numeric_tau=numeric_tau_val)

    call_put.plot2(axs[1, 0], palette, name, s_vals, k_val, tau_vals, r_val, sigma_val,numeric_tau=numeric_tau_val)

    fig.delaxes(axs[1, 1])

    plt.show()
s_vals = np.arange(50, 150, 0.5)

k_val = 100

tau_vals = [0, 0.1, 1, 3, 5, 9, 12] # month

# r_val = 0.01

sigma_val = 0.1

interactive_val = False

numeric_tau_val = True
r_val = 0.01
name = 'option'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'delta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'gamma'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'theta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'vega'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'rho'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
r_val = 0.0
name = 'option'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'delta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'gamma'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'theta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'vega'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'rho'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
r_val = 0.075
name = 'option'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'delta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'gamma'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'theta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'vega'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'rho'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
r_val = 0.1
name = 'option'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'delta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'gamma'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'theta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'vega'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'rho'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
r_val = 0.15
name = 'option'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'delta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'gamma'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'theta'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'vega'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)
name = 'rho'

print('call - put   ' + name + ' : ')

display(getattr(call_put, name))

plot(name)