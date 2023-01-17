import anemoi as an
import pandas as pd
import numpy as np
import scipy as sp
import sklearn as skl

import scipy.optimize as spyopt
from scipy.special import gamma

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import seaborn as sns
sns.set_context('talk')

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.tools as tls
offline.init_notebook_mode(connected=True)

# Colors for plotting
edf_green = '#509E2F'
edf_orange = '#FE5815'
edf_blue = '#001A70'

print('Anemoi version: {}'.format(an.__version__))
print('Pandas version: {}'.format(pd.__version__))
print('Numpy version: {}'.format(np.__version__))
print('Scikit Learn version: {}'.format(skl.__version__))
print('Matplotlib version: {}'.format(mpl.__version__))
# %%timeit: 24.9 ms ± 793 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
mast_data = pd.read_parquet('../input/demo_mast.parquet')
ref_data = pd.read_parquet('../input/demo_refs.parquet')

primary_ano = 'SPD_59_COMB_AVG'
primary_vane = 'DIR_95_COMB_AVG'
mast_data.head()
ref_data.head()
mast_data.count()/mast_data.shape[0]*100.0
mast = an.MetMast(data=mast_data, name='mast', lat=45, lon=-90, height=60, elev=500, primary_ano='SPD_59_COMB_AVG', primary_vane='DIR_95_COMB_AVG')
mast
mast.metadata
mast.data.head()
ano_vane_data = mast.return_primary_ano_vane_data().dropna()
ano_vane_data.columns = ano_vane_data.columns.get_level_values('sensor')
energy_wind_roses = an.analysis.wind_rose.return_directional_energy_frequencies(ano_vane_data, ws_sensor=mast.primary_ano, dir_sensor=mast.primary_vane)
energy_wind_roses
def rose_axis_settings(ax):
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['', '', 'E', '', 'S', '', 'W', ''])
    ax.set_yticklabels('')

fig = plt.figure(figsize=[12,6])
ax1 = fig.add_subplot(121, projection='polar')
ax2 = fig.add_subplot(122, projection='polar')
ax1.bar(np.radians(energy_wind_roses.index.values), energy_wind_roses.dir, width=np.radians(360.0/energy_wind_roses.shape[0]+1), color=edf_green)
ax2.bar(np.radians(energy_wind_roses.index.values), energy_wind_roses.energy, width=np.radians(360.0/energy_wind_roses.shape[0]+1), color=edf_blue)
[rose_axis_settings(ax) for ax in [ax1,ax2]]
ax1.set_title('Wind Rose')
ax2.set_title('Energy Rose')
plt.tight_layout()
plt.show()
momm_meas = an.utils.mast_data.return_momm(mast.return_primary_ano_data()).T.MoMM[0]
print('MoMM at measurement height: {:.3f} m/s'.format(momm_meas))
shear = an.analysis.shear.mast_annual(mast)
shear
shear_fig = an.plotting.shear.annual_mast_results(shear)
offline.iplot(shear_fig)
shear_mean = shear.melt().value.mean()
shear_top = shear.loc[pd.IndexSlice['COMB',47],59]
print('Mean shear: {:.3f} and top shear: {:.3f}'.format(shear_mean, shear_top))
ref_monthly_means = ref_data.groupby(ref_data.index.month).mean()

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(111)
(ref_monthly_means/ref_data.mean()).plot(legend=False, xlim=[1,12], ax=ax, title='Normalized annual profiles')
ax.set_xticks(ref_monthly_means.index.values)
ax.set_xlabel('Month')
plt.show()
ref_yearly_monthly_means = ref_data.resample('MS').mean()
ref_monthly_means_repeat = ref_monthly_means.loc[ref_yearly_monthly_means.index.month,:]
ref_monthly_means_repeat.index = ref_yearly_monthly_means.index
ref_normalized_yearly_monthly_means = ref_yearly_monthly_means/ref_monthly_means_repeat
normalized_rolling_monthly_average_figure = ref_normalized_yearly_monthly_means.rolling(12, min_periods=10).mean().dropna(how='all')

plotly_fig = an.plotting.references.normalized_rolling_monthly_average_figure(normalized_rolling_monthly_average_figure)
offline.iplot(plotly_fig)
daily_mast_data = mast.return_primary_ano_data().resample('D').mean()
daily_mast_data.columns = daily_mast_data.columns.get_level_values('sensor')
daily_data = pd.concat([daily_mast_data,ref_data], axis=1)
daily_data.dropna().head()
an.analysis.correlate.ws_correlation_binned_by_month(daily_data, site_ws_col=mast.primary_ano, ref_ws_col='1')
# %%timeit: 7.6 s ± 110 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
valid_refs = np.arange(1,9).astype(str)
daily_corr_results = [an.analysis.correlate.ws_correlation_binned_by_month(daily_data, site_ws_col=mast.primary_ano, ref_ws_col=ref) for ref in valid_refs]
daily_corr_results = pd.concat(daily_corr_results, axis=0, keys=valid_refs, names=['ref','month'])
daily_corr_results.head(20)
syn_data = [an.analysis.correlate.apply_daily_results_by_month_to_mast_data(daily_data, daily_corr_results.loc[ref,:], ref_ws_col=ref, site_ws_col=mast.primary_ano, splice=True).syn_splice for ref in valid_refs]
syn_data = pd.concat(syn_data, axis=1, keys=valid_refs)
long_term_predictions = an.utils.mast_data.return_momm(syn_data).T.MoMM
long_term_predictions
(long_term_predictions/momm_meas-1)*100.0
lt_mws = long_term_predictions.mean()
print('Long-term mean wind speed at 59 m: {:.3f} m/s'.format(lt_mws))
hub_height = 120 #m
meas_height = 59 #m
lt_hh_mws_top_alpha = lt_mws * (hub_height/meas_height)**shear_top
lt_hh_mws_mean_alpha = lt_mws * (hub_height/meas_height)**shear_mean
print('Long-term hub-height mean wind speed using top sensor combination for shear: {:.3f} m/s'.format(lt_hh_mws_top_alpha))
print('Long-term hub-height mean wind speed using mean of all sensor combinations for shear: {:.3f} m/s'.format(lt_hh_mws_mean_alpha))
alpha_time_series = an.analysis.shear.alpha_time_series(mast.data, wind_speed_sensors=['SPD_47_COMB_AVG','SPD_59_COMB_AVG'], heights=[47,59])
alpha_time_series[alpha_time_series>1] = np.nan
alpha_time_series[alpha_time_series<-1] = np.nan

ano_data = mast.return_primary_ano_data()
ano_data.columns = ano_data.columns.get_level_values('sensor')
hub_height_time_series = ano_data[mast.primary_ano] * (hub_height/meas_height)**alpha_time_series.alpha

hub_height_mws = an.utils.mast_data.return_momm(hub_height_time_series.to_frame('hh_mws')).T.MoMM[0]
hub_height_adj = lt_hh_mws_top_alpha/hub_height_mws
lt_hh_mws_time_series = (hub_height_time_series*hub_height_adj).dropna()
A,k = an.analysis.weibull.euro_atlas_fit(lt_hh_mws_time_series.values)
freq_by_ws = lt_hh_mws_time_series.groupby(lt_hh_mws_time_series.round()).count() / lt_hh_mws_time_series.shape[0]
freq_by_ws.index = freq_by_ws.index.astype(int)

x = np.linspace(0,23,100)

fig = plt.figure(figsize=[9,6])
ax = fig.add_subplot(111)
freq_by_ws.plot(kind='bar', color=edf_green, width=0.9, ax=ax)
ax.plot(x, sp.stats.exponweib(1, k, scale=A, loc=0).pdf(x), color=edf_blue)
ax.set_xlim([0,25])
ax.set_title('Long-term hub-height wind speed frequency distribution')
ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Frequency')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
hub_height_wind_speed_and_dir_data = pd.concat([lt_hh_mws_time_series, ano_vane_data[mast.primary_vane]], axis=1)
hub_height_wind_speed_and_dir_data.columns = ['ws','dir']
tab_file_table = an.analysis.wind_rose.return_tab_file(hub_height_wind_speed_and_dir_data, ws_sensor='ws', dir_sensor='dir', dir_sectors=16, ws_bin_width=1.0, half_first_bin=False, freq_as_label=False)
tab_file_table
