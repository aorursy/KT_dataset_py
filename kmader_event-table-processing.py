%matplotlib inline
import os, pandas as pd, numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from pandas.tseries.holiday import USFederalHolidayCalendar
def business_dates(start, end):
    us_cal = USFederalHolidayCalendar()
    kw = dict(start=start, end=end)
    return pd.DatetimeIndex(freq='B', **kw).drop(us_cal.holidays(**kw))
@lru_cache()
def count_business_dates(start, end):
  us_cal = USFederalHolidayCalendar()
  kw = dict(start=start, end=end)
  return pd.DatetimeIndex(freq='B', **kw).drop(us_cal.holidays(**kw)).shape[0]
from IPython.display import FileLink
data_path = '../input/event_table-csv/event_table.csv'
with open(data_path, 'r', encoding="utf8", errors='ignore') as f:
    events_df = pd.read_csv(f, dtype={'CASE_ID_PO': str, 'VENDOR': str})
events_df['EVENT_TIME'] = pd.to_datetime(events_df['EVENT_TIME'])
print('From', events_df['EVENT_TIME'].min(), 'to', events_df['EVENT_TIME'].max())
events_df.sample(3)
print(events_df.shape[0], 'events found')
num_desc_df = events_df.describe()
display(Markdown('## Numerical Variables'))
display(num_desc_df.T)
cat_desc_df = events_df.describe(include='O')
display(Markdown('## Categorical Variables'))
display(cat_desc_df.T)
np.random.seed(2018)
test_process_df = events_df[events_df['CASE_ID_PO']==np.random.choice(events_df['CASE_ID_PO'].values)].sort_values('EVENT_TIME', ascending=True)
test_process_df
fig, m_axs = plt.subplots(4, 3, figsize = (15, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
d_col_names = [x for x in test_process_df.columns 
               if x not in ['EVENT_TIME', 'CASE_ID_PO']]
elapsed_time = (test_process_df['EVENT_TIME']-test_process_df['EVENT_TIME'].min()).dt.total_seconds()/(3600*24.0)
for c_ax, c_col in zip(m_axs.flatten(), d_col_names):
    c_col_vec = test_process_df[c_col]
    if np.issubdtype(c_col_vec.dtype, np.number):
        c_ax.plot(elapsed_time, c_col_vec,
                  '.-', label=test_process_df['CASE_ID_PO'])
    else:
        code_vec = c_col_vec.astype('category').cat
        c_ax.plot(elapsed_time, code_vec.codes, 
                  '.-', label=test_process_df['CASE_ID_PO'])
        if len(code_vec.categories)>1:
            for i,j,k in zip(elapsed_time.values, code_vec.codes, c_col_vec.values):
                c_ax.text(i,j,k)
        else:
            c_ax.set_yticks(range(code_vec.categories.shape[0]))
            c_ax.set_yticklabels([x[:20] for x in code_vec.categories])
    c_ax.axis('on')
    c_ax.set_xlabel('Elapsed Days')
    c_ax.set_title(c_col)
np.random.seed(2018)
few_processes_df = events_df[events_df['CASE_ID_PO'].isin(np.random.choice(events_df['CASE_ID_PO'].values, 8))].sort_values('EVENT_TIME', ascending=True)
few_processes_df.head(5)
fig, m_axs = plt.subplots(4, 3, figsize = (15, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
# groupby screws with the order
for c_count, c_pid in enumerate(np.unique(few_processes_df['CASE_ID_PO'])):
    cur_case_mask = (few_processes_df['CASE_ID_PO']==c_pid)
    time_vec = few_processes_df[cur_case_mask]['EVENT_TIME']
    elapsed_time = (time_vec-time_vec.min()).dt.total_seconds()/(3600*24)
    for c_ax, c_col in zip(m_axs.flatten(), d_col_names):
        c_col_vec = few_processes_df[c_col]
        if np.issubdtype(c_col_vec.dtype, np.number):
            c_ax.plot(elapsed_time, c_col_vec[cur_case_mask],
                      '.-', label=c_pid)
        else:
            code_vec = c_col_vec.astype('category').cat
            c_ax.plot(elapsed_time, 
                      code_vec.codes[cur_case_mask], 
                      '.-', label=c_pid)
            if len(code_vec.categories)>1:
                for i,j,k in zip(elapsed_time.values, 
                                 code_vec.codes[cur_case_mask], 
                                 c_col_vec.values[cur_case_mask]):
                    c_ax.text(i,j,k)
                c_ax.set_yticks([])
            else:
                if c_count==0:
                    c_ax.set_yticks(range(code_vec.categories.shape[0]))
                    c_ax.set_yticklabels([x[:20] for x in code_vec.categories])
        c_ax.axis('on')
        c_ax.set_xlabel('Elapsed Days')
        c_ax.set_title(c_col)
[c_ax.legend() for c_ax in m_axs.flatten()];    
print(np.unique(events_df['CASE_ID_PO']).shape[0], 'total processes')
proc_length_df = events_df.groupby(['CASE_ID_PO']).\
    apply(lambda c_rows: 
          pd.Series({'event_count': c_rows.shape[0],
                    'total_days': (c_rows['EVENT_TIME'].max()-c_rows['EVENT_TIME'].min()).total_seconds()/(24*3600),
                    'unique_users': len(np.unique(c_rows['USER_ID'])),
                    'unique_groups': len(np.unique(c_rows['EVENT_GROUP']))
                    })
         ).reset_index()
proc_length_df.sample(5)
sns.pairplot(proc_length_df[['event_count', 'total_days', 'unique_groups', 'unique_users']])
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
log_bins = np.logspace(0, np.log10(proc_length_df['event_count'].max()), 50)
event_bins = np.arange(np.percentile(proc_length_df['event_count'], 1)-1, 
                       np.percentile(proc_length_df['event_count'], 99)+1)
ax1.hist(proc_length_df['event_count'], event_bins)
ax1.set_title('Event Count')

day_bins = np.linspace(np.percentile(proc_length_df['total_days'], 1)-1, 
                       np.percentile(proc_length_df['total_days'], 99)+1, 31)
ax2.hist(proc_length_df['total_days'], day_bins)
ax2.set_title('Number of Days')
fig, ax1 = plt.subplots(1,1, figsize = (20, 5))
h, _, _ = np.histogram2d(proc_length_df['event_count'], 
           proc_length_df['total_days'],
          bins = (event_bins, day_bins));
sns.heatmap(h.astype(int), 
            annot=True, 
            fmt = 'd', ax=ax1, 
            vmax=np.percentile(h.ravel(), 90))
ax1.set_ylabel('Number of Events')
ax1.set_yticklabels(event_bins.astype(int))
ax1.set_xlabel('Number of Days')
ax1.set_xticklabels(day_bins.astype(int));
def sort_and_idx(in_rows):
    out_rows = in_rows.sort_values(['EVENT_TIME'], ascending=True)
    out_rows['time_idx'] = range(in_rows.shape[0])
    out_rows['elapsed_days'] = (out_rows['EVENT_TIME']-out_rows['EVENT_TIME'].min()).dt.total_seconds()/(24*3600)
    return out_rows
process_chains_df = pd.merge(events_df, proc_length_df).\
    groupby(['CASE_ID_PO']).\
    apply(sort_and_idx).\
    reset_index(drop=True)
process_chains_df.sample(5)
# show the most frequent first steps
start_ele = process_chains_df.query('time_idx==0').groupby(['EVENT_GROUP']).\
  size().reset_index(name='counts').sort_values('counts', ascending=False).\
  set_index('EVENT_GROUP')
start_ele/start_ele.sum()
# show the most frequent last step
last_ele = process_chains_df.query('time_idx==(event_count-1)').\
  groupby(['EVENT_GROUP']).size().reset_index(name='counts').\
  sort_values('counts', ascending=False).set_index('EVENT_GROUP')
last_ele/last_ele.sum()
process_chains_df.sample(3)
process_chains_df.to_csv('process_chains.csv', index=False)
process_chains_df.sample(5)
join_cols = proc_length_df.columns.tolist() + ['time_idx']
next_step_chain_df = process_chains_df.copy()
next_step_chain_df['time_idx'] = next_step_chain_df['time_idx'] - 1
transition_chain_df = pd.merge(process_chains_df, 
                               next_step_chain_df, 
                               on=join_cols, 
                               suffixes=['_current', '_next'])
transition_chain_df.sample(5)
transition_chain_df['transition_time'] = transition_chain_df['elapsed_days_next']-transition_chain_df['elapsed_days_current']
transition_chain_df.to_csv('transition_chains.csv', index=False)
min_trans_df = transition_chain_df[['EVENT_GROUP_current', 
                                    'EVENT_GROUP_next', 
                                    'transition_time']]
def _flip_order_events(in_df):
  """
  flip the order of events to see if the backwards event is more likely
  """
  out_df = in_df[['EVENT_GROUP_next',
                  'EVENT_GROUP_current', 
                  'transition_time']].copy()
  out_df.columns = in_df.columns
  out_df['transition_time']*=-1
  return out_df

ordered_trans_df = pd.concat([min_trans_df, 
           _flip_order_events(min_trans_df)]).\
  query('EVENT_GROUP_current<=EVENT_GROUP_next')

ordered_trans_df['transition_direction'] = ordered_trans_df['transition_time'].\
  map(lambda x: 'forward' if x>0 else ('backward' if x<0 else 'same'))

event_bias_df = ordered_trans_df.\
  groupby(['EVENT_GROUP_current', 'EVENT_GROUP_next', 'transition_direction']).\
  size().\
  reset_index(name='count').\
  pivot_table(index=['EVENT_GROUP_current', 'EVENT_GROUP_next'], 
             columns='transition_direction', 
              values='count',
             aggfunc='sum',
             fill_value=0).\
  reset_index()

forward_events = event_bias_df[event_bias_df['forward']>event_bias_df['backward']]
backward_events = event_bias_df[event_bias_df['forward']<event_bias_df['backward']].copy()
backward_events.columns=['EVENT_GROUP_next', 'EVENT_GROUP_current', 
                         'forward', 'backward', 'same']
comb_events_df = pd.concat([forward_events, backward_events])
comb_events_df['forward_bias'] = comb_events_df.\
  apply(lambda c_row: c_row['forward']/(c_row['backward']+
                                        c_row['forward']+c_row['same']), 1)
comb_events_df.sort_values(['forward_bias'])
sorted_event_list = [[k['EVENT_GROUP_current'], k['EVENT_GROUP_next']] 
 for _, k in  comb_events_df.iterrows()]
sorted_event_set = {tuple(k) for k in sorted_event_list}
eg_keys = list(np.unique(events_df['EVENT_GROUP']))
print(eg_keys)
last_eg_keys = []
for _ in range(10000):
  last_eg_keys = eg_keys.copy()
  for i in range(len(eg_keys)-1):
    for j in range(i, len(eg_keys)):
      if (eg_keys[i], eg_keys[j]) in sorted_event_set:
        eg_keys[i], eg_keys[j] = eg_keys[j], eg_keys[i]
  if last_eg_keys==eg_keys:
    print('finished')
    break
eg_keys
ge_order_dict = {v: k for k, v in enumerate(eg_keys)}
def sort_and_idx_fancy(in_rows, verbose=False):
    out_rows = in_rows.sort_values(['EVENT_TIME'], ascending=True)
    guess_idxs = list(range(in_rows.shape[0]))
    out_rows['time_idx'] = guess_idxs
    out_rows['elapsed_days'] = (out_rows['EVENT_TIME']-out_rows['EVENT_TIME'].min()).dt.total_seconds()/(24*3600)
    if (np.diff(out_rows['elapsed_days'])==0).sum()>0:
      for (_, c_row), (_, n_row) in zip(out_rows.iloc[0:].iterrows(), 
                                        out_rows.iloc[1:].iterrows()):
        if c_row['EVENT_TIME']==n_row['EVENT_TIME']:
          if verbose:
            print('Found Match', c_row['EVENT_GROUP'],'->', n_row['EVENT_GROUP'])
          if (c_row['EVENT_GROUP'], n_row['EVENT_GROUP']) not in sorted_event_set:
            # swap
            if verbose:
              print('Swap')
            guess_idxs[c_row['time_idx']] = n_row['time_idx']
            guess_idxs[n_row['time_idx']] = c_row['time_idx']
          else:
            if verbose:
              print('Preserve')
    out_rows['time_idx'] = guess_idxs
    return out_rows
def sort_and_idx(in_rows, verbose=False):
    out_rows = in_rows.copy()
    out_rows['_order_value'] = -1*out_rows['EVENT_GROUP'].map(ge_order_dict.get)
    out_rows = out_rows.sort_values(['EVENT_TIME', '_order_value'], ascending=True)
    out_rows.drop(['_order_value'], axis=1, inplace=True)
    guess_idxs = list(range(in_rows.shape[0]))
    out_rows['time_idx'] = guess_idxs
    out_rows['elapsed_days'] = (out_rows['EVENT_TIME']-out_rows['EVENT_TIME'].min()).dt.total_seconds()/(24*3600)
    return out_rows
test_case_id = transition_chain_df[transition_chain_df['transition_time']==0]['CASE_ID_PO'].values[0]
test_case_df = events_df[events_df['CASE_ID_PO']==test_case_id]
pd.merge(test_case_df, proc_length_df).\
    groupby(['CASE_ID_PO']).\
    apply(lambda x: sort_and_idx(x, True)).\
    reset_index(drop=True)
%%time
process_chains_df = pd.merge(events_df, proc_length_df).\
    groupby(['CASE_ID_PO']).\
    apply(sort_and_idx).\
    reset_index(drop=True)
process_chains_df.sample(5)
process_chains_df.to_csv('process_chains.csv', index=False)
process_chains_df.sample(5)
join_cols = proc_length_df.columns.tolist() + ['time_idx']
next_step_chain_df = process_chains_df.copy()
next_step_chain_df['time_idx'] = next_step_chain_df['time_idx'] - 1
transition_chain_df = pd.merge(process_chains_df, 
                               next_step_chain_df, 
                               on=join_cols, 
                               suffixes=['_current', '_next'])
transition_chain_df['transition_time'] = transition_chain_df['elapsed_days_next']-transition_chain_df['elapsed_days_current']
transition_chain_df.to_csv('transition_chains.csv', index=False)
transition_chain_df.sample(5)
FileLink('process_chains.csv')
FileLink('transition_chains.csv')
trans_count_df = transition_chain_df.pivot_table(index='EVENT_GROUP_current', 
                                                    values='CASE_ID_PO', 
                                                    columns = 'EVENT_GROUP_next',
                                                   aggfunc='count',
                                                   fill_value = 0)
# determine the normal order of events
normal_order_df = process_chains_df.groupby(['EVENT_GROUP']).apply(lambda c_row: 
                                                 pd.Series({'relative_position': 
                                                            np.mean(c_row['time_idx']/c_row['event_count'])})).\
  sort_values('relative_position').reset_index()
normal_order_df
trans_count_df = trans_count_df[normal_order_df['EVENT_GROUP']].loc[reversed(normal_order_df['EVENT_GROUP'])]
transition_matrix = trans_count_df.values.astype('float')
norm_mat = np.tile(np.expand_dims(np.sum(transition_matrix, 1), -1), 
                   [1, transition_matrix.shape[1]])
fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
sns.heatmap(100*transition_matrix/norm_mat, annot=True, fmt = '2.1f')
ax1.xaxis.tick_top()
ax1.set_xticklabels(trans_count_df.columns, rotation=45, 
                    horizontalalignment = 'left')
ax1.set_yticklabels(trans_count_df.index, rotation=0);
prob_mask = transition_matrix/norm_mat<0.05
fig.suptitle('State Transition Probability (%)', x=0, y=1.02);
prob_time_df = transition_chain_df.\
  groupby(['EVENT_GROUP_current', 'EVENT_GROUP_next']).\
  apply(lambda c_rows: pd.Series({'avg_time': np.mean(c_rows['transition_time']),
                                 'times_occured': c_rows.shape[0]})).reset_index()
prob_time_df.sample(5)
import networkx as nx  # For the graphs
G = nx.MultiDiGraph()
labels={}
edge_labels={}

states = trans_count_df.columns
for i, c_row  in prob_time_df.iterrows():
  if c_row['times_occured']>11:
    origin_state = c_row['EVENT_GROUP_current']
    destination_state = c_row['EVENT_GROUP_next']
    c_label = "{times_occured:2.0f} ({avg_time:2.1f} days)".format(**c_row)
    G.add_edge(origin_state,
               destination_state,
               weight=np.sqrt(c_row['times_occured']),
               label=c_label)
    edge_labels[(origin_state, destination_state)] = c_label
fig, ax1 = plt.subplots(1, 1, figsize=(20,20))
pos=nx.spring_layout(G, k=10, iterations=1000)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size = 300)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.20, ax=ax1)
nx.draw_networkx_labels(G, pos, font_weight=2, font_color='blue', ax=ax1)
nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1)
ax1.axis('off');
from IPython.display import Image
dt_rep = nx.nx_pydot.to_pydot(G)
#dt_rep.set_splines('curved')
Image(dt_rep.create_png())
rare_transitions_df = pd.melt(trans_count_df.reset_index(), id_vars=['EVENT_GROUP_current']).query('value>0 and value<=2')
rare_transitions_df.sort_values('value')
trans_time_df = transition_chain_df.pivot_table(index='EVENT_GROUP_current', 
                                                    values='transition_time', 
                                                    columns = 'EVENT_GROUP_next',
                                                   aggfunc='mean',
                                                   fill_value = np.NAN)
trans_time_df = trans_time_df[normal_order_df['EVENT_GROUP']].loc[reversed(normal_order_df['EVENT_GROUP'])]

transition_matrix = trans_time_df.values.astype('float')
norm_mat = np.tile(np.expand_dims(np.sum(transition_matrix, 1), -1), 
                   [1, transition_matrix.shape[1]])
fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
sns.heatmap(transition_matrix, annot=True, fmt = '2.0f', mask = prob_mask)
ax1.xaxis.tick_top()
ax1.set_xticklabels(trans_time_df.columns, rotation=45, 
                    horizontalalignment = 'left')
ax1.set_yticklabels(trans_time_df.index, rotation=0);
fig.suptitle('Average Days Taken for Common Transitions (>5%)', y=1.05, 
            horizontalalignment='right');
transition_chain_df['transaction_date'] = transition_chain_df['EVENT_TIME_next'].dt.date
flow_df = transition_chain_df.groupby(['transaction_date', 
                     'EVENT_GROUP_current', 
                     'EVENT_GROUP_next']).size().reset_index(name='count')
fig, ax1 = plt.subplots(1, 1, figsize = (10, 4))
ax1.plot(flow_df['transaction_date'], flow_df['count'])
flow_df['day_count'] = (flow_df['transaction_date']-flow_df['transaction_date'].min()).dt.total_seconds()/(24*3600)
fig, ax1 = plt.subplots(1, 1, figsize = (10, 4))
ax1.plot(flow_df['day_count'], flow_df['count'])
flow_df['busday_count'] = flow_df['transaction_date'].map(lambda x: count_business_dates(flow_df['transaction_date'].min(), x))
fig, ax1 = plt.subplots(1, 1, figsize = (10, 4))
ax1.plot(flow_df['busday_count'], flow_df['count'])
from sklearn.preprocessing import LabelEncoder
state_enc = LabelEncoder()
state_enc.fit(normal_order_df['EVENT_GROUP'])
flow_df['id_current'] = state_enc.transform(flow_df['EVENT_GROUP_current'])
flow_df['id_next'] = state_enc.transform(flow_df['EVENT_GROUP_next'])
flow_df.sample(3)
fig, ax1 = plt.subplots(1, 1, figsize = (20, 4))
ax1.quiver(flow_df['busday_count'], flow_df['id_current'], 
            [0]*flow_df.shape[0], flow_df['id_next']-flow_df['id_current'],
           units='y', angles='xy')
ax1.set_xlabel('Business Day')
ax1.set_ylabel('EVENT_GROUP')
ax1.set_yticks(range(len(state_enc.classes_)+1))
ax1.set_yticklabels(state_enc.classes_);
fig, m_axs = plt.subplots(4, 4, figsize = (20, 20))
for c_ax, (c_id, c_df) in zip(m_axs.flatten(), 
                              flow_df.groupby('EVENT_GROUP_current')):
  for c_grp, c_rows in c_df.groupby('EVENT_GROUP_next'):
    c_ax.plot(c_df['busday_count'], c_df['count'], label=c_grp)
  c_ax.legend()
  c_ax.set_title(c_id)
fig, ax1 = plt.subplots(1, 1, figsize=(20,20))

def draw_graph(pos = None, filter_cmd = lambda x: x):
  ax1.cla()
  G = nx.MultiDiGraph()
  labels={}
  edge_labels={}
  
  states = trans_count_df.columns
  new_df = filter_cmd(flow_df).groupby(['EVENT_GROUP_current', 
                                        'EVENT_GROUP_next']).agg({'count': 'sum'}).reset_index()
  for i, c_row  in new_df.iterrows():
    origin_state = c_row['EVENT_GROUP_current']
    destination_state = c_row['EVENT_GROUP_next']
    c_label = "{count:2.0f}".format(**c_row)
    G.add_edge(origin_state,
               destination_state,
               weight=np.sqrt(c_row['count']),
               label=c_label)
    edge_labels[(origin_state, destination_state)] = c_label
  if pos is None:
    pos=nx.circular_layout(G)
  weight_dict = nx.get_edge_attributes(G, 'weight')
  weights = [weight_dict[(u,v, 0)] for u,v in G.edges()]
  
  weights = np.log2(weights / np.max(weights))
  weights+=weights.min()
  weights/=weights.max()/3
  nx.draw_networkx_nodes(G, pos, ax=ax1, node_size = 300)
  nx.draw_networkx_edges(G, pos, width=weights, alpha=0.20, ax=ax1, color='green')
  nx.draw_networkx_labels(G, pos, font_weight=4, font_color='black', ax=ax1)
  nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1)
  ax1.axis('off');
  return pos
cur_pos = draw_graph();

#@title Show Interactive Results
day_to_show = 31 #@param {type:"slider", min:0, max:100, step:1}

filt_func = lambda x: x.query(f'busday_count=={day_to_show}')
draw_graph(pos=cur_pos, filter_cmd=filt_func)
fig
FileLink('process_chains.csv')
events_std_df = pd.merge(events_df, proc_length_df.query('events==7'))
events_std_df.sample(5)
def sort_and_idx(in_rows):
    out_rows = in_rows.sort_values(['EVENT_TIME'], ascending=True)
    out_rows['time_idx'] = range(in_rows.shape[0])
    out_rows['elapsed_days'] = (out_rows['EVENT_TIME']-out_rows['EVENT_TIME'].min()).dt.days
    return out_rows
events_std_df = events_std_df.\
    groupby(['CASE_ID_PO']).\
    apply(sort_and_idx).\
    reset_index(drop=True)
events_std_df.sample(5)
events_std_df['EVENT_GROUP_ID'] = events_std_df['EVENT_GROUP'].astype('category').cat.codes
events_time_df = events_std_df.pivot_table(values='EVENT_GROUP', 
                          index=['CASE_ID_PO', 'days', 'events'], 
                          columns=['time_idx'],
                         aggfunc='first')
events_time_df.sample(20)
events_time_df['pattern'] = events_time_df.apply(lambda c_row: ';'.join(c_row), 1)
pats = events_time_df['pattern'].value_counts()
print(len(pats), 'number of different order patterns')
pats[:5]
events_time_df['pattern_id'] = events_time_df['pattern'].astype('category').cat.codes
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
sns.boxplot(x='pattern_id', 
               y='days',  
               data=events_time_df.reset_index(),
            ax=ax1)
