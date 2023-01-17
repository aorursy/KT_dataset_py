import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import os
import pandas as pd

DATA_DIR = '../input/cand-multi-stellar-dr1'
DASCH_DIR = os.path.join(DATA_DIR, 'dasch')
dim_candidates = pd.read_csv(os.path.join(DATA_DIR, 'clustered-dim-candidates.csv'), dtype={'source_id': str})
len(dim_candidates[dim_candidates['dasch_id'] != 'None'])
bright_candidates = pd.read_csv(os.path.join(DATA_DIR, 'clustered-bright-candidates.csv'), dtype={'source_id': str})
len(bright_candidates[bright_candidates['dasch_id'] != 'None'])
normal_controls = pd.read_csv(os.path.join(DATA_DIR, 'normal-controls.csv'), dtype={'source_id': str})
len(normal_controls[normal_controls['dasch_id'] != 'None'])
# See http://dasch.rc.fas.harvard.edu/database.php#AFLAGS_ext
AFLAGS_BAD_BIN = 0x800000
AFLAGS_DEFECTIVE = 0x2000000
def read_dasch_lc(source_id, dasch_id, is_gaia_dr2_id=True):
    file_name_prefix = 'short_' + ('Gaia_DR2_' if is_gaia_dr2_id else 'TYC_') + source_id + '_' + dasch_id
    files = os.listdir(DASCH_DIR)
    matching_files = [file for file in files if file.startswith(file_name_prefix)]
    lmf = len(matching_files)
    if lmf == 0:
        raise ValueError('No matching file starting with %s' % file_name_prefix)
    if lmf > 1:
        print('WARNING: More than one matching file for source_id=%s' % source_id)        
    file_name = matching_files[0]
    raw_table = pd.read_table(os.path.join(DASCH_DIR, file_name), skiprows=[0,2])
    # Remove defective data points
    aflags = raw_table['AFLAGS'];
    is_defective = (aflags & AFLAGS_BAD_BIN != 0) | (aflags & AFLAGS_DEFECTIVE != 0) 
    corrected_table = raw_table[~is_defective]
    # Simplify the table
    result = pd.DataFrame()
    result['year'] = corrected_table['year']
    result['Date'] = corrected_table['Date']
    result['magcal_magdep'] = corrected_table['magcal_magdep']
    return result
dasch_frame_map = dict()
def extract_dasch_frames(gaia_frame):
    has_tycho2_id = 'tycho2_id' in gaia_frame.columns 
    for _, row in gaia_frame.iterrows():
        source_id = row['source_id']
        dasch_id = row['dasch_id']
        if dasch_id != 'None':
            retrieval_id_column = 'tycho2_id' if has_tycho2_id else 'source_id'
            retrieval_id = row[retrieval_id_column]
            dasch_frame = read_dasch_lc(retrieval_id, dasch_id, is_gaia_dr2_id=not has_tycho2_id)
            dasch_frame_map[source_id] = dasch_frame
extract_dasch_frames(normal_controls)
extract_dasch_frames(dim_candidates)
extract_dasch_frames(bright_candidates)
from scipy import stats
import numpy as np

def get_trend_frame(gaia_frame, pre_mg=False):
    has_tycho2_id = 'tycho2_id' in gaia_frame.columns
    results = pd.DataFrame(columns=['source_id', 'tycho2_id', 'slope', 'confidence_interval', 'sigmas'])
    for _, row in gaia_frame.iterrows():
        source_id = row['source_id']
        tycho2_id = row['tycho2_id'] if has_tycho2_id else ''
        dasch_frame = dasch_frame_map.get(source_id)
        if dasch_frame is not None:
            if pre_mg:
                dasch_frame = dasch_frame[dasch_frame['year'] <= 1960].reset_index(drop=True)
            if len(dasch_frame) < 200:
                continue
            slope, _, _, _, std_err = stats.linregress(dasch_frame['year'] - 1880, dasch_frame['magcal_magdep'])
            confidence_interval = 1.96 * std_err
            slope_cent = slope * 100
            confidence_interval_cent = confidence_interval * 100
            results.loc[len(results)] = [source_id, tycho2_id, slope_cent, confidence_interval_cent, slope / std_err]
    return results
def style_function(value):
    significant = np.abs(value) > 3.0
    color = ('red' if value < 0 else 'blue') if significant else 'black'
    font_weight = 'bold' if significant else 'plain'
    return 'color: %s; font-weight: %s' % (color, font_weight,)
def show_results(gaia_frame, pre_mg=False):
    trend_frame = get_trend_frame(gaia_frame, pre_mg=pre_mg)
    return trend_frame.style.applymap(style_function, subset=['sigmas'])
show_results(dim_candidates)
show_results(bright_candidates)
show_results(dim_candidates, pre_mg=True)
show_results(bright_candidates, pre_mg=True)
from scipy.stats import ttest_ind
import statsmodels.stats.api as sms

def print_t_test(gaia_frame1, gaia_frame2, column_name='slope', pre_mg=False):
    trend_frame1 = get_trend_frame(gaia_frame1, pre_mg=pre_mg)
    trend_frame2 = get_trend_frame(gaia_frame2, pre_mg=pre_mg)
    values1 = trend_frame1[column_name].values
    values2 = trend_frame2[column_name].values
    cm = sms.CompareMeans(sms.DescrStatsW(values1), sms.DescrStatsW(values2))
    confint_low, confint_high = cm.tconfint_diff()
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    statistic, pvalue = ttest_ind(values1, values2)
    print('Mean Difference: %.3f [95%% CI %.3f to %.3f]' % (mean1 - mean2, confint_low, confint_high,))
    print('Statistic: %.4f | p-value: %.6f' % (statistic, pvalue,))
print_t_test(dim_candidates, normal_controls)
print_t_test(dim_candidates, normal_controls, pre_mg=True)
print_t_test(bright_candidates, normal_controls)
print_t_test(bright_candidates, normal_controls, pre_mg=True)
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=False)
def plot_group_diffs(exp_group, title):
    exp_trend_frame = get_trend_frame(exp_group)
    normal_trend_frame = get_trend_frame(normal_controls)
    trace0 = go.Box(
        name='Candidates',
        y=exp_trend_frame['slope']
    )
    trace1 = go.Box(
        name='Ordinary stars',  
        y=normal_trend_frame['slope']
    )
    data = [trace0, trace1]
    layout = go.Layout(
        showlegend=False,
        title=title,
        yaxis=dict(
            title='DASCH slope (magnitudes / century)',
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
plot_group_diffs(dim_candidates, 'Anomalously dim candidates vs. controls')
plot_group_diffs(bright_candidates, 'Anomalously bright candidates vs. controls')
def show_dasch_lightcurve(source_id, source_name):
    dasch_frame = dasch_frame_map[source_id]
    year_series = dasch_frame['year']
    points_trace = go.Scatter(
                    name='Observation',
                    x=year_series,
                    y=dasch_frame['magcal_magdep'],
                    mode='markers',
                    marker=go.scatter.Marker(color='rgb(255, 127, 14)')
                    )
    slope, intercept, _, _, std_err = stats.linregress(year_series, dasch_frame['magcal_magdep'])
    line_trace = go.Scatter(x=year_series, 
                    name='Fit',
                    y=year_series * slope + intercept,
                    mode='lines',
                    line=dict(color='blue', width=3)
                    )
    annotation = go.layout.Annotation(
                    x=1970,
                    y=1979 * slope + intercept - 1.5,
                    text='%.2f ± %.2f mag/century' % (slope * 100, 1.96 * std_err * 100,),
                    showarrow=False,
                    font=go.layout.annotation.Font(size=16, color='blue'),
                  )
    layout = go.Layout(
                    showlegend=False,
                    title='DASCH light curve of ' + source_name,
                    annotations=[annotation],
                    xaxis=dict(
                        title='Year'
                    ),
                    yaxis=dict(
                        title='Magnitude',
                        autorange='reversed'
                    )
                )
    data = [points_trace, line_trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
show_dasch_lightcurve('2097850907148567424', 'TYC 3105-1212-1')