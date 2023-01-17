import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import plotly.figure_factory as ff

cols = px.colors.qualitative.Plotly
import cv2, pandas as pd, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
print('Examples WITH Melanoma')
imgs = train.loc[train.target==1].sample(10).image_name.values
plt.figure(figsize=(20,8))
for i,k in enumerate(imgs):
    img = cv2.imread('../input/jpeg-melanoma-128x128/train/%s.jpg'%k)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2,5,i+1); plt.axis('off')
    plt.imshow(img)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
print('Examples WITHOUT Melanoma')
imgs = train.loc[train.target==0].sample(10).image_name.values
plt.figure(figsize=(20,8))
for i,k in enumerate(imgs):
    img = cv2.imread('../input/jpeg-melanoma-128x128/train/%s.jpg'%k)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2,5,i+1); plt.axis('off')
    plt.imshow(img)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
cols
B = []
for i in range(0,8):
    B.append(pd.read_csv('../input/siimisc/20epochs_base/20epochs_base/history-B' + str(i) + '-20.csv'))
    
S = []
for i in range(0,8):
    S.append(pd.read_csv('../input/siimisc/20epochs_base_sprinkles/20epochs_base_sprinkles/history-B' + str(i) + '-20-sprinkles.csv'))

size = [224, 240, 260, 300, 380, 456, 528, 600]

lr = dict()

lr['10'] = []
for i in range(0, 8):
    if i == 1:
        lr['10'].append(pd.read_csv('../input/siimisc/lr_0.0001/B'+str(i)+'/history-B'+str(i)+'-'+str(size[i])+'-2.csv'))
    else:
        lr['10'].append(pd.read_csv('../input/siimisc/lr_0.0001/B'+str(i)+'/history-B'+str(i)+'-'+str(size[i])+'.csv'))

lr['16'] = []
lr['16'].append(pd.read_csv('../input/siimisc/lr_0.00016/B0/history-B0-224.csv'))
lr['16'].append(pd.read_csv('../input/siimisc/lr_16/history-B1-240.csv'))
for i in range(2, 8):
    lr['16'].append(pd.read_csv('../input/siimisc/lr_0.00016/B'+str(i)+'/history-B'+str(i)+'-'+str(size[i])+'.csv'))

lr['32'] = []
for i in range(0, 8):
    lr['32'].append(pd.read_csv('../input/siimisc/lr_0.00032/lr_0.00032/B'+str(i)+'/history-B'+str(i)+'-'+str(size[i])+'.csv'))

lr['48'] = []
for i in range(0,8):
    lr['48'].append(pd.read_csv('../input/siimisc/lr_0.00048/lr_0.00048/history-B'+str(i)+'-'+str(size[i])+'.csv'))

name = ['10', '16', '32', '48']
S[0]
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=B[i]['val_auc'],
                        mode='lines',
                        name='B'+str(i),
                        line=dict(color=cols[i])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Validation Accuracy"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=S[i]['val_auc'],
                        mode='lines',
                        name='B'+str(i),
                        line=dict(color=cols[i])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Validation Accuracy with Sprinkles"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=B[i]['val_loss'],
                        mode='lines+markers',
                        name='B'+str(i),
                        line=dict(color=cols[i])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Validation Loss"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=S[i]['val_loss'],
                        mode='lines+markers',
                        name='B'+str(i),
                        line=dict(color=cols[i])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Validation Loss with Sprinkles"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=B[i]['auc'],
                        mode='lines',
                        name='Auc B'+str(i),
                        line=dict(color=cols[i])))

    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Training Accuracy"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=S[i]['auc'],
                        mode='lines',
                        name='Auc B'+str(i),
                        line=dict(color=cols[i])))

    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Training Accuracy with Sprinkles"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=B[i]['loss'],
                        mode='lines+markers',
                        name='Loss B'+str(i),
                        line=dict(color=cols[i])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Training Loss"
)

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

for i in range(0,8):
    fig.add_trace(go.Scatter(x=x, y=S[i]['loss'],
                        mode='lines+markers',
                        name='Loss B'+str(i),
                        line=dict(color=cols[i])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Training Loss with Sprinkles"
)

fig.show()
fig = go.Figure(data=[go.Table(
    header=dict(values=['Model', 'Max Val Auc', 'Min Val Auc', 'Max Auc', 'Min Auc', 'Max Val Auc Sprinkles', 
                        'Min Val Auc Sprinkles', 'Max Auc Sprinkles', 'Min Auc Sprinkles'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[[i for i in range(0,8)], # 1st column
                       [max(B[i]['val_auc']) for i in range(0,8)],
                       [min(B[i]['val_auc']) for i in range(0,8)],
                       [max(B[i]['auc']) for i in range(0,8)],
                       [min(B[i]['auc']) for i in range(0,8)],
                       [max(S[i]['val_auc']) for i in range(0,8)],
                       [min(S[i]['val_auc']) for i in range(0,8)],
                       [max(S[i]['auc']) for i in range(0,8)],
                       [min(S[i]['auc']) for i in range(0,8)]], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=900, height=500)
fig.show()
x = [i for i in range(1,21)]
title = ['auc', 'val_auc', 'loss', 'val_loss']
t, r = 1, 1

fig = make_subplots(rows=2, cols=4, subplot_titles=['B'+str(i) for i in range(0,8)])

for i in range(0,8):
    temp = 0
    if t==5:
        t, r= 1, 2 
    for j in title:
        if i==0:
            fig.add_trace(go.Scatter(x=x, y=B[i][j],
                                    mode='lines+markers',
                                    name=j,
                                    line=dict(color=cols[temp])), row=r, col=t)
        fig.add_trace(go.Scatter(x=x, y=B[i][j],
                                    mode='lines+markers',
                                    name=j,
                                    showlegend=False,
                                    line=dict(color=cols[temp])), row=r, col=t)
        temp += 1
    t += 1      
        
fig.update_layout( title_text="Data for All Efficient")

fig.show()
x = [i for i in range(1,21)]
title = ['auc', 'val_auc', 'loss', 'val_loss']
t, r = 1, 1

fig = make_subplots(rows=2, cols=4, subplot_titles=['B'+str(i) for i in range(0,8)])

for i in range(0,8):
    temp = 0
    if t==5:
        t, r= 1, 2 
    for j in title:
        if i==0:
            fig.add_trace(go.Scatter(x=x, y=S[i][j],
                                    mode='lines+markers',
                                    name=j,
                                    line=dict(color=cols[temp])), row=r, col=t)
        fig.add_trace(go.Scatter(x=x, y=S[i][j],
                                    mode='lines+markers',
                                    name=j,
                                    showlegend=False,
                                    line=dict(color=cols[temp])), row=r, col=t)
        temp += 1
    t += 1      
        
fig.update_layout( title_text="Data for All Efficient with Sprinkles")

fig.show()
x = [i for i in range(1,21)]

t, r = 1, 1

fig = make_subplots(rows=2, cols=4, subplot_titles=['B'+str(i) for i in range(0,8)])

for i in range(0,8):
    temp = 0
    if t==5:
        t, r= 1, 2 
    if i==0:
        fig.add_trace(go.Scatter(x=x, y=S[i]['val_auc'],
                                mode='lines+markers',
                                name='S val',
                                line=dict(color=cols[0])), row=r, col=t)
        fig.add_trace(go.Scatter(x=x, y=B[i]['val_auc'],
                                mode='lines+markers',
                                name='B val',
                                line=dict(color=cols[1])), row=r, col=t)
    fig.add_trace(go.Scatter(x=x, y=S[i]['val_auc'],
                                mode='lines+markers',
                                name='val',
                                showlegend=False,
                                line=dict(color=cols[temp])), row=r, col=t)
    temp += 1
    fig.add_trace(go.Scatter(x=x, y=B[i]['val_auc'],
                                mode='lines+markers',
                                name='val',
                                showlegend=False,
                                line=dict(color=cols[temp])), row=r, col=t)
    
    t += 1      
        
fig.update_layout( title_text="val_auc Comparision")

fig.show()
x = [i for i in range(1,21)]

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=B[0]['lr'],
                    mode='lines+markers',
                    name='LR B',
                    line=dict(color=cols[0])))
    
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    title_text="Learning Rate"
)


fig.show()
x = [i for i in range(1,21)]

t, r = 1, 1

fig = make_subplots(rows=3, cols=3, subplot_titles=['B'+str(i) for i in range(0,8)])

count = 0

for i in range(0,8):
    temp = 0
    if t==4:
        if count == 0:
            t, r = 1, 2 
            count = 1
        else:
            t, r = 1, 3
    if i==0:
        for j in name:
            temp += 1
            fig.add_trace(go.Scatter(x=x, y=lr[j][i]['val_auc'],
                                    mode='lines+markers',
                                    name= j+' val',
                                    line=dict(color=cols[temp])), row=r, col=t)
    else:
        for j in name:
            temp += 1
            fig.add_trace(go.Scatter(x=x, y=lr[j][i]['val_auc'],
                                    mode='lines+markers',
                                    name= j+' val',
                                    showlegend=False,
                                    line=dict(color=cols[temp])), row=r, col=t)
    
    t += 1      
        

fig.update_layout(height=900, width=1200, title_text="val_auc Comparision")

fig.show()
x = [i for i in range(1,21)]

t, r = 1, 1

fig = make_subplots(rows=3, cols=3, subplot_titles=['B'+str(i) for i in range(0,8)], shared_yaxes=True)

count = 0

for i in range(0,8):
    temp = 0
    if t==4:
        if count == 0:
            t, r = 1, 2 
            count = 1
        else:
            t, r = 1, 3
    if i==0:
        for j in name:
            temp += 1
            fig.add_trace(go.Scatter(x=x, y=lr[
                j][i]['lr'],
                                    mode='lines+markers',
                                    name= j+' val',
                                    line=dict(color=cols[temp])), row=r, col=t)
    else:
        for j in name:
            temp += 1
            fig.add_trace(go.Scatter(x=x, y=lr[j][i][
                'lr'],
                                    mode='lines+markers',
                                    name= j+' val',
                                    showlegend=False,
                                    line=dict(color=cols[temp])), row=r, col=t)
    fig.update_yaxes(showexponent = 'all', exponentformat = 'e')
    fig.update_xaxes(title_font=dict(size=18, family='Courier', color='crimson'))

    
    t += 1      
        

fig.update_layout(height=900, width=1200, title_text="Learning Rate Comparision")

fig.show()
fig = go.Figure(data=[go.Table(
    header=dict(values=['Model', 'Max 10', 'Max 16', 'Max 32', 'Max 48'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[[i for i in range(0,8)], # 1st column
                       [max(lr['10'][i]['val_auc']) for i in range(0,8)],
                       [max(lr['16'][i]['val_auc']) for i in range(0,8)],
                       [max(lr['32'][i]['val_auc']) for i in range(0,8)],
                       [max(lr['48'][i]['val_auc']) for i in range(0,8)],
                       ], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=900, height=400)
fig.show()
t = dict()
t['10'] = [0.8973, 0.8889, 0.8876, 0.9018, 0.9080, 0.9017, 0.9057, 0.9114]
t['32'] = [0.8867, 0.8968, 0.8961, 0.8861, 0.9008, 0.8903, 0.8829, 0.8948]
t['16'] = [0.892, 0.9021, 0.9066, 0.8993, 0.8956, 0.904, 0.9061, 0.8868]
t['48'] = [0.8912, 0.8909, 0.8997, 0.8947, 0.8855, 0.8907, 0.8876, 0.8949]
x = [i for i in range(0, 8)]

fig = go.Figure()

for i in range(0,8):
    temp = 0
    if i == 0:
        for j in name:
            temp += 1
            fig.add_trace(go.Scatter(x=x, y=t[j],
                                mode='lines+markers',
                                name=str(j),
                                line=dict(color=cols[temp])))
    else:
        for j in name:
            temp += 1
            fig.add_trace(go.Scatter(x=x, y=t[j],
                                mode='lines+markers',
                                name=str(j),
                                showlegend=False,
                                line=dict(color=cols[temp])))

fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
)

fig.show()
fig = go.Figure(data=[go.Table(
    header=dict(values=['Model', '10', '16', '32', '48'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[[i for i in range(0,8)], # 1st column
                       [t['10'][i] for i in range(0,8)],
                       [t['16'][i] for i in range(0,8)],
                       [t['32'][i] for i in range(0,8)],
                       [t['48'][i] for i in range(0,8)],
                       ], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=900, height=400)
fig.show()
lr = [0.00016, 0.00016, 0.00016, 0.00016, 0.0001, 0.0001, 0.0001, 0.0001]
x = [i for i in range(0, 8)]

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=lr,
                            mode='lines+markers',
                            name='B'+str(j),
                            line=dict(color=cols[temp])))


fig.show()

