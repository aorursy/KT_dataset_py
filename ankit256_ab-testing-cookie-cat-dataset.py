import matplotlib.pyplot as plt

from scipy.stats import binom

import numpy as np

import pandas as pd 

from scipy.stats import norm

import math as mt

np.set_printoptions(suppress=True)

import statsmodels.stats.api as sms

from scipy.stats import norm

import scipy.stats as stats

import plotly.graph_objs as go

import plotly.graph_objects as go

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode, plot_mpl

def sample_sd_when_population_sd_is_not_given(p,n):

    sd = np.sqrt((p*(1-p))/n)

    return round(sd,4)



#Returns: z-score for given alpha

def get_z_score(alpha):

    return norm.ppf(alpha)



def ztest_comparing_two_proportions(X1,X2,n1,n2):

    p1_hat = X1/n1

    p2_hat = X2/n2

    p_bar = (X1+X2)/(n1+n2)

    q_bar = (1-p_bar)

    z_numerator= p1_hat-p2_hat

    z_denominator = np.sqrt((1/n1+1/n2)*p_bar*q_bar)

    z_statistic = z_numerator/z_denominator

    p_value = norm().cdf(z_statistic)

    return z_numerator, p_value,z_denominator,z_statistic
data = pd.read_csv('../input/mobile-games-ab-testing/cookie_cats.csv')

print(data.info())

print(data['version'].value_counts())
gate_30 = data[data['version']=='gate_30']  #control

gate_40 = data[data['version']=='gate_40']  #Treatment

print(gate_30.head())

print(gate_40.head())
x0 = gate_30['sum_gamerounds']

x1 = gate_40['sum_gamerounds'] 

fig = go.Figure()

fig.add_trace(go.Box(x=x0, name='gate_30 (control_group)'))

fig.add_trace(go.Box(x=x1, name='gate_40 (experiment_group)'))

fig.update_layout(legend=dict(x=.6,y=0.97, traceorder='reversed', font_size=16))

fig.show()
gate_30=gate_30[gate_30['sum_gamerounds']!=gate_30['sum_gamerounds'].max()]

x0 = gate_30['sum_gamerounds']

x1 = gate_40['sum_gamerounds'] 



fig = go.Figure()

fig.add_trace(go.Box(x=x0, name='gate_30 (control_group)'))

fig.add_trace(go.Box(x=x1, name='gate_40 (experiment_group)'))

fig.update_layout(legend=dict(x=.6,y=0.97, traceorder='reversed', font_size=16))

fig.show()
gameround_gate_40 = gate_40['sum_gamerounds'].value_counts()

gameround_gate_40=gameround_gate_40.sort_values(ascending=False)

gameround_gate_30 = gate_30['sum_gamerounds'].value_counts()

gameround_gate_30=gameround_gate_30.sort_values(ascending=False)

pd.set_option("max_rows", None)

pd.set_option('display.max_columns', None)

import warnings

warnings.filterwarnings('ignore')

colors=['#151515','#f0c24f']

fig = go.Figure()

fig.add_trace(go.Scatter(y=gameround_gate_30, mode='lines+markers',marker_color=colors[0], name='gate_30 (experiment_group)'))

fig.add_trace(go.Scatter(y=gameround_gate_40, mode='lines+markers',marker_color=colors[1], name='gate_40 (control_group)'))

fig.update_layout(title="Count of userid vs count of games played", 

                 legend=dict(x=.05,y=0.95, traceorder='reversed', font_size=16), width=600,height=600,

                 yaxis=dict(title="Count of useid",titlefont=dict(

                          color="#1f77b4"),tickfont=dict(color="#1f77b4")))

fig.show()
gate_30_mean= gate_30['sum_gamerounds'].mean()

gate_30_std=gate_30['sum_gamerounds'].std()

gate_40_mean= gate_40['sum_gamerounds'].mean()

gate_40_std=gate_40['sum_gamerounds'].std()

poopled_std = np.sqrt(((gate_30_std)**2/gate_30.shape[0]) + ((gate_40_std)**2/gate_40.shape[0]))

pooled_mean = gate_30_mean-gate_40_mean

z_score =  pooled_mean/poopled_std

p= norm().cdf(z_score)

print(f"Zscore is {z_score:0.2f}, p-value is {(1-p)*2:0.3f} (two tailed), {p:0.3f} (one tailed)")
from scipy.stats import ttest_ind

order_value_control_group = gate_30['sum_gamerounds'].to_list()

order_value_experimental_group = gate_40['sum_gamerounds'].to_list()

zscore, prob= ttest_ind(order_value_control_group, order_value_experimental_group, equal_var=False)

print(f"Zscore is {zscore:0.2f}, p-value is {prob:0.3f} (two tailed), {prob/2:0.3f} (one tailed)")
colors=['#de3d83','#00b8b8','#e4bd0b']

fig = go.Figure()

click_rate = np.linspace(0, 100, 10000)

prob_a = norm(gate_40_mean, gate_30_std).pdf(click_rate)

prob_b = norm(gate_30_mean, gate_40_std).pdf(click_rate)

fig.add_trace(go.Scatter(y=prob_a , x=click_rate, mode='lines+markers',marker_color=colors[0], name='gate_40'))

fig.add_trace(go.Scatter(y=prob_b , x=click_rate, mode='lines+markers',marker_color=colors[1], name='gate_30'))

fig.update_layout(title="Probability Density Function for gate_30 and gate_40 <br> for  sum_gamerounds (Invariate Matrix) ", 

                 legend=dict(x=.05,y=0.95, traceorder='reversed', font_size=16), 

                 width=600,

                 height=600,

                 yaxis=dict(

                          title="Probability Density Function P(x)",

                 titlefont=dict(

                          color="#1f77b4"

                                ),

                 tickfont=dict(

                        color="#1f77b4"

                               )

  ))

fig.show()
import statsmodels.stats.api as sms

baseline_cvr=0.448

alpha=0.05

power=0.8

mini_diff=0.01

effect_size=sms.proportion_effectsize(baseline_cvr, baseline_cvr+mini_diff)

sample_size=sms.NormalIndPower().solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1)

print('Required sample size for Retention Day 1 metric ~ {0:.1f}'.format(sample_size) + ' per group')
baseline_cvr_day7=0.19018

alpha_day7=0.05

power_day7=0.8

mini_diff_day7=0.01

effect_size_day7=sms.proportion_effectsize(baseline_cvr_day7, baseline_cvr_day7+mini_diff_day7)

sample_size_day7=sms.NormalIndPower().solve_power(effect_size=effect_size_day7, power=power_day7, alpha=alpha_day7, ratio=1)

print('Required sample size for Retention Day 7 metric ~ {0:.1f}'.format(sample_size_day7) + ' per group')
samplesize_list=[]

baseline=0.448

deltas=np.arange(0.005, 0.025, 0.001)

for delta in deltas:

  prob2=baseline+delta

  effect_size=sms.proportion_effectsize(baseline, prob2)

  sample_size=sms.NormalIndPower().solve_power(effect_size=effect_size, power=0.8, alpha=0.05, ratio=1)

  samplesize_list.append(sample_size)

colors=['#151515','#f0c24f']

fig = go.Figure()

fig.add_trace(go.Scatter(y=samplesize_list , x=deltas, mode='lines+markers',marker_color=colors[1]))

fig.update_layout(title="Minimum required sample size for minimum detectable delta/effective size <br> for Retention Day 1 metric", 

                 legend=dict(x=.05,y=0.95, traceorder='reversed', font_size=16), 

                 width=500,

                 height=500,

                 yaxis=dict(

                          title="Required Sample Size",

                 titlefont=dict(

                          color="#1f77b4"

                                ),

                 tickfont=dict(

                        color="#1f77b4"

                               )),

                xaxis=dict(

                          title="Minimum Detectable Effective Size",

                 titlefont=dict(

                          color="#1f77b4"

                                ),

                 tickfont=dict(

                        color="#1f77b4"

                               )

  ))

fig.show()
samplesize_list=[]

baseline=0.19018

deltas=np.arange(0.005, 0.025, 0.001)

for delta in deltas:

  prob2=baseline+delta

  effect_size=sms.proportion_effectsize(baseline, prob2)

  sample_size=sms.NormalIndPower().solve_power(effect_size=effect_size, power=0.8, alpha=0.05, ratio=1)

  samplesize_list.append(sample_size)

colors=['#00b8b8','#f0c24f']

fig = go.Figure()

fig.add_trace(go.Scatter(y=samplesize_list , x=deltas, mode='lines+markers',marker_color=colors[0]))

fig.update_layout(title="Minimum required sample size for minimum detectable delta/effective size <br> for Retention Day 7 metric", 

                 legend=dict(x=.05,y=0.95, traceorder='reversed', font_size=16), 

                 width=500,

                 height=500,

                 yaxis=dict(

                          title="Required Sample Size",

                 titlefont=dict(

                          color="#1f77b4"

                                ),

                 tickfont=dict(

                        color="#1f77b4"

                               )),

                xaxis=dict(

                          title="Minimum Detectable Effective Size",

                 titlefont=dict(

                          color="#1f77b4"

                                ),

                 tickfont=dict(

                        color="#1f77b4"

                               )

  ))

fig.show()
#1-day rretention

#h0 = gate40<=gate30

#ha = gate40>30

game_30_day1_retention = gate_30['retention_1'].sum()

game_40_day1_retention = gate_40['retention_1'].sum()

game_30_count=gate_30.shape[0]

game_40_count=gate_40.shape[0]

print(f'game_30 p1 {game_30_day1_retention/game_30_count:.3f}')

print(f'game_40 p2 {game_40_day1_retention/game_40_count:.3f}')

mean, p_value,me,z_statistic1 = ztest_comparing_two_proportions(game_40_day1_retention, game_30_day1_retention,game_40_count,game_30_count)

print(f'zscore for 1-day retention {z_statistic1:.3f}')

print(f'p-value for 1-day retention {1-p_value:.3f}')

print(f'Probablity that gate_40 increasted the 1-day retention (gate_40-gate_30>0) {p_value:.3f}')
import plotly.graph_objects as go

fig = go.Figure()

colors=['#00b8b8','#de3d83']



rate_b=game_40_day1_retention/game_40_count

rate_a=game_30_day1_retention/game_30_count

std_a = sample_sd_when_population_sd_is_not_given(rate_a,game_30_count)

std_b=sample_sd_when_population_sd_is_not_given(rate_b,game_40_count)



p_hat_1day=rate_b - rate_a

p_hat_1day_std = np.sqrt(std_a**2 + std_b**2)



z_score = p_hat_1day / p_hat_1day_std

p = norm(p_hat_1day, p_hat_1day_std)

x = np.linspace(-0.02, 0.02, 1000)

y = p.pdf(x)

area_under_curve = p.sf(0)





print(f"zscore is {z_score:0.3f}, with p-value {norm().sf(z_score):0.3f}")



x_value = norm(loc=p_hat_1day,scale=p_hat_1day_std).isf(norm().sf(z_score))

px=np.arange(0.0,.02,0.00100)

fig.add_trace(go.Scatter(y=y , x=x, mode='lines+markers',marker_color=colors[0], name='pdf(gate_40-gate_30)'))

fig.add_trace(go.Scatter(

    y=norm.pdf(px,loc=p_hat_1day,scale=p_hat_1day_std), 

    x=px,  mode='lines',name="prob(gate_40-gate_30>0) ",

    fill='tozeroy',

    fillcolor=colors[1],

    line=dict(color='#f43530', width=.1))) # fill to trace0 y



fig.add_annotation(x=.002,y=10,

            text="Area under the curve {}".format(area_under_curve.round(4)))

fig.update_annotations(dict(xref="x",yref="y",showarrow=True, arrowhead=7,ax=130,ay=-60))



fig.update_layout(title="PDF of gate_40-gate_30 1-Day Retention", 

                 legend=dict(x=.40,y=0.95, traceorder='reversed', font_size=16),  width=600,height=600,yaxis=dict(title="Probability Density Function P(x)", titlefont=dict(color="#1f77b4"),tickfont=dict(color="#1f77b4")))

fig.show()
#7-day rretention

#h0 = gate40<=gate30

#ha = gate40>30

game_30_day7_retention = gate_30['retention_7'].sum()

game_40_day7_retention = gate_40['retention_7'].sum()

game_30_count=gate_30.shape[0]

game_40_count=gate_40.shape[0]

print(f'game_30_ret7 p1 {game_30_day7_retention/game_30_count:.4f}')

print(f'game_40_ret7 p2 {game_40_day7_retention/game_40_count:.4f}')

mean7, p_value7,me7,z_statistic7 = ztest_comparing_two_proportions(game_40_day7_retention, game_30_day7_retention,game_40_count,game_30_count)

print(f'zscore for 7-day retention {z_statistic7:.4f}')

print(f'p-value for 7-day retention {1-p_value7:.4f}')

print(f'Probablity that gate_40 increasted the 7-day retention (gate_40-gate_30>0) {p_value7:.3f}')
import plotly.graph_objects as go

fig = go.Figure()

colors=['#00b8b8','#de3d83']



rate_b_day7=game_40_day7_retention/game_40_count

rate_a_day7=game_30_day7_retention/game_30_count

std_a_day7 = sample_sd_when_population_sd_is_not_given(rate_a_day7,game_30_count)

std_b_day7=sample_sd_when_population_sd_is_not_given(rate_b_day7,game_40_count)





p_hat_day7=rate_b_day7 - rate_a_day7

p_std_hat_day7= np.sqrt(std_a_day7**2 + std_b_day7**2)





z_score_day7 = (p_hat_day7) / (p_std_hat_day7)

p_day7 = norm(p_hat_day7, (p_std_hat_day7))

x_day7 = np.linspace(-0.02, 0.02, 1000)

y_day7 = p_day7.pdf(x_day7)

area_under_curve_day7 = p_day7.sf(0)



print(f"zscore is {z_score_day7:0.3f}, with p-value {norm().sf(z_score_day7):0.3f}")



x_value_day7 = norm(loc=p_hat_day7,scale=p_std_hat_day7).isf(norm().sf(z_score_day7))

px_day7=np.arange(0.0,.02,0.00100)

fig.add_trace(go.Scatter(y=y_day7 , x=x_day7, mode='lines+markers',marker_color=colors[0], name='pdf(gate_40-gate_30)'))

fig.add_trace(go.Scatter(

    y=norm.pdf(px_day7,loc=p_hat_day7,scale=p_std_hat_day7), 

    x=px_day7,  mode='lines',name="prob(gate_40-gate_30>0)",

    fill='tozeroy',

    fillcolor=colors[1],

    line=dict(color='#f43530', width=.1))) # fill to trace0 y



fig.add_annotation(x=.002,y=10,

            text="Area under the curve {}".format(area_under_curve_day7.round(3)))

fig.update_annotations(dict(xref="x",yref="y",showarrow=True, arrowhead=7,ax=130,ay=-60))



fig.update_layout(title="PDF of gate_40-gate_30 7-Day Retention", 

                 legend=dict(x=.40,y=0.95, traceorder='reversed', font_size=16),  width=600,height=600,yaxis=dict(title="Probability Density Function P(x)", titlefont=dict(color="#1f77b4"),tickfont=dict(color="#1f77b4")))

fig.show()