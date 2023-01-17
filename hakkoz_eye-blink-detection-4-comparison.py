import pandas as pd 

from matplotlib import gridspec

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go



# import utility functions

from utils import *

from utils3 import *
#define path and dataset_name

path1 = "../input/eye-blink-detection-1-simple-model"

dataset_name = "talking"



# load datasets

c_pred1, b_pred1, df1, c_test1, b_test1, s_str1 = load_datasets(path1, dataset_name)
#define path and dataset_name

path2 = "../input/eye-blink-detection-2-adaptive-model-v2"

dataset_name = "talking"



# load datasets

c_pred2, b_pred2, df2, c_test2, b_test2, s_str2 = load_datasets(path2, dataset_name)
#define path

path2 = "../input/../input/eye-blink-detection-3-ml-model-part2/"



# load datasets

c_pred3 = pd.read_pickle(path2+'red_df_ml.pkl')

c_pred4 = pd.read_pickle(path2+'red_df_ml2.pkl')
df_test = pd.read_pickle("../input/eye-blink-detection-3-ml-model-part1/test/test_set.pkl")

ground_truth = df_test['blink_annot'].tolist() 
#EAR values

data1 = df1['avg_ear']
# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(len(data1)),y=data1,

                    mode='lines+markers',name='EAR'))

fig.add_trace(go.Scatter(x=np.arange(len(data1)),y=[float('nan') if x==0 else x-0.98 for x in ground_truth],

                    mode='markers',name='ground_truth'))

fig.add_trace(go.Scatter(x=np.arange(len(data1)),y=[float('nan') if x==0 else x-0.99 for x in c_pred1],

                    mode='markers',name='simple model (Precision:46% Recall:97% F1:62%)'))

fig.add_trace(go.Scatter(x=np.arange(len(data1)),y=[float('nan')]*222+[float('nan') if x==0 else x-1.00 for x in c_pred2],

                    mode='markers',name='adaptive model (Precision:61% Recall:68% F1:64%)'))

fig.add_trace(go.Scatter(x=np.arange(len(data1)),y=[float('nan') if x==0 else x-1.01 for x in c_pred3],

                    mode='markers',name='ML model (Precision:86% Recall:87% F1:86%)'))

fig.add_trace(go.Scatter(x=np.arange(len(data1)),y=[float('nan') if x==0 else x-1.02 for x in c_pred4],

                    mode='markers',name='ML model-RLDD (Precision:50% Recall:97% F1:66%)'))

fig.update_layout(

    autosize=False,

    width=1000,

    height=500,

    legend=dict(x=0, y=-0.4))

fig.show()