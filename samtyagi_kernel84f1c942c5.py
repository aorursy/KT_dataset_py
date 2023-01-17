import pandas as pd
import numpy as np
from tqdm import *
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline


df = pd.read_csv('../input/homepage_actions.csv')
df.info() # getting the basic info about our dataset
# total number of actions
df.action.count()

# number of unique users
df.id.nunique()
# size of the control and the experiment group
df.groupby('group').nunique()['id']
# converting the timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])
# duration of experiment
df.timestamp.max() - df.timestamp.min()
# experiment group users
experiment_gr = df.query("group == 'experiment'")
# click through rate for experiment group users
experiment_ctr = experiment_gr.query("action == 'click'").nunique()['id']/experiment_gr.query("action == 'view'").nunique()['id']
experiment_ctr
# control group users
control_gr = df.query("group == 'control'")
# click through rate for experiment group users
control_ctr = control_gr.query("action == 'click'").nunique()['id']/control_gr.query("action == 'view'").nunique()['id']
control_ctr
# bootstrapping the sampling distribution
diffs = []
for i in tqdm(range(10000)):
    sample = df.sample(4000 ,replace = True)
    experiment_gr = sample.query("group == 'experiment'")
    control_gr = sample.query("group == 'control'")
    experiment_ctr = experiment_gr.query("action == 'click'").nunique()['id']/experiment_gr.query("action == 'view'").nunique()['id']
    control_ctr = control_gr.query("action == 'click'").nunique()['id']/control_gr.query("action == 'view'").nunique()['id']
    diffs.append( experiment_ctr - control_ctr)
# ploting the sampling distribution
plt.hist(diffs)
# null values
null_vals = np.random.normal(0 ,np.array(diffs).std() ,10000)
# ploting null values
# with the observed stats
plt.hist(null_vals)
plt.axvline(x = np.array(diffs).mean() ,color = 'r') # we can see the observed stats are way out of the range of mean null vals
# getting p-values
p_vals = (null_vals > np.array(diffs).mean()).mean()
p_vals # with a p-value of less then 1 % , we can safely reject the null
