#import package

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

!pip install seaborn==0.10.0.rc0

import seaborn as sns



import missingno as msno 





import cufflinks as cf

cf.set_config_file(theme='ggplot',

#                    sharing='public',

                   offline=True,

                   dimensions=(500,300),offline_show_link=False)



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots





from category_encoders.ordinal import OrdinalEncoder

from category_encoders.target_encoder import TargetEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

train.head()
target, train_id = train['target'], train['id']

test_id = test['id']

train.drop(['id'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)
#plot the missing values

msno.matrix(train);
# Plot NULL rate of training data

null_rate = [train[i].isna().sum() / len(train) for i in train.columns]

data = {'train_column':train.columns,'null_rate':null_rate}

train_null_rate = pd.DataFrame(data).drop(23)



fig = px.bar(train_null_rate, x = 'train_column',y='null_rate',

             text='null_rate',color='null_rate',

             title = 'Feature Null Rate (Train Data)')

fig.update_traces(textposition='outside',texttemplate='%{text:.2p}',textfont_size=20)

fig.update_layout(yaxis_tickformat = '%')

fig.add_shape(

        go.layout.Shape(

            type="line",

            yref= 'y', y0= 0.03, y1= 0.03,

            xref= 'x', x0= -1, x1= 23.5,

            line=dict(

                color="gray",

                width=1.5,

                dash="dash")

        ))

 

fig.update(layout_coloraxis_showscale=False) # hide colorscale

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) # this size is for kaggle kernal

fig.show()
# Plot NULL rate of Test data

null_rate = [test[i].isna().sum() / len(train) for i in test.columns]

data = {'test_column':test.columns,'null_rate':null_rate}

test_null_rate = pd.DataFrame(data)



fig = px.bar(test_null_rate, x = 'test_column',y='null_rate',

             text='null_rate',color='null_rate',

      title = 'Feature Null Rate (Test Data)')

fig.update_traces(textposition='outside',texttemplate='%{text:.2p}',textfont_size=20)

fig.update_layout(yaxis_tickformat = '%')

fig.add_shape(

        go.layout.Shape(

            type="line",

            yref= 'y', y0= 0.02, y1= 0.02,

            xref= 'x', x0= -1, x1= 23.5,

            line=dict(

                color="gray",

                width=1.5,

                dash="dash")

        ))

fig.update(layout_coloraxis_showscale=False) # hide colorscale   

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) # this size is for kaggle kernal

fig.show()
#Plot Target Distribution

target.iplot(kind='histogram',histnorm='probability',title='Total Target Distribution',bargap=0.5)
# Plot binary feature distribution for Training Data

sub_title = list(train[[col for col in train.columns if col.startswith('bin_')]].columns)

fig = make_subplots(rows=1, cols=5,subplot_titles=sub_title)

for i in range(5): 

    a = train[f'bin_{i}'].value_counts()

    fig.add_trace(       

        go.Bar(x=a.index, y=a.values,width=0.2, text=a.values,textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'bin_{i}',showlegend=False,

               textfont=dict(size=5),textangle=0,constraintext='inside',

               marker=dict(color=['#732FB0','#2FB09D'],line_width=0.5, line_color='gray')

              ),row=1, col=i+1)

    



fig.update_layout(title_text="Binary Feature Distribution (Train Data)")



# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),

                         yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')



# fig.show(config={"showLink":True})

fig.show()
# Plot binary feature distribution for Training Data

sub_title = list(train[[col for col in test.columns if col.startswith('bin_')]].columns)

fig = make_subplots(rows=1, cols=5,subplot_titles=sub_title)

for i in range(5): 

    a = test[f'bin_{i}'].value_counts()

    fig.add_trace(       

        go.Bar(x=a.index, y=a.values,width=0.2, text=a.values,textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'bin_{i}',showlegend=False,

               textfont=dict(size=5),textangle=0,constraintext='inside',

               marker=dict(color=['#732FB0','#2FB09D'],line_width=0.5, line_color='gray')

              ),row=1, col=i+1)

    



fig.update_layout(title_text="Binary Feature Distribution (Test Data)")



# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')



# fig.show(config={"showLink":True})

fig.show()
# Plot Binary Feature Target Distribution for Training Data

sub_title = list(train[[col for col in test.columns if col.startswith('bin_')]].columns)



fig = make_subplots(rows=1, cols=5,subplot_titles=sub_title)

for i in range(5): 



    a = train.groupby([f'bin_{i}','target']).size().to_frame().reset_index()

    a_0 = a[a['target']==0]

    a_1 = a[a['target']==1]

    

    if i == 0:

        

        fig.add_trace(       

            go.Bar( x=a_0[f'bin_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

                   texttemplate='%{text:.2s}',

                   name = 'target_0',

                   legendgroup = 'target_0',

                   textfont=dict(size=5),textangle=0,constraintext='inside',

                    marker=dict(

                        color=['#732FB0','#732FB0'],

                        line_width=0.5, line_color='gray')

                  ),row=1, col=i+1)





        fig.add_trace(       

            go.Bar(x=a_1[f'bin_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

                   texttemplate='%{text:.2s}',

                   name = 'target_1',

                    legendgroup = 'target_1',

                   textfont=dict(size=5),textangle=0,constraintext='inside',

                    marker=dict(

                        color=['#2FB09D','#2FB09D'],

                        line_width=0.8, line_color='gray')

                  ),row=1, col=i+1)  

        

    else:

        

        fig.add_trace(       

            go.Bar( x=a_0[f'bin_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

                   texttemplate='%{text:.2s}',

    #                name=f'bin_{i}'+' target_0',

                   name = 'target_0',

                   legendgroup = 'target_0',

                   showlegend=False,

                   textfont=dict(size=5),textangle=0,constraintext='inside',

                    marker=dict(

                        color=['#732FB0','#732FB0'],

                        line_width=0.5, line_color='gray')

                  ),row=1, col=i+1)

        

        fig.add_trace(       

            go.Bar(x=a_1[f'bin_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

                   texttemplate='%{text:.2s}',

    #                name=f'bin_{i}'+' target_1',

                   name = 'target_1',

                    legendgroup = 'target_1',

                   showlegend=False,

                   textfont=dict(size=5),textangle=0,constraintext='inside',

                    marker=dict(

                        color=['#2FB09D','#2FB09D'],

                        line_width=0.8, line_color='gray')

                  ),row=1, col=i+1)





        fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),

                         yaxis4=dict(visible=False),yaxis5=dict(visible=False))



        

fig.update_layout(title_text="Binary Feature Target Distribution (Train Data)",

                  margin=dict(l=25,t=50,b=0),

                  legend_orientation='h',width=500,height=300) # this size is for kaggle kernal





# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')



fig.show()

# fig.show(config={"showLink":True})
#Describe nominal features

train[[col for col in train.columns if col.startswith('nom_')]].describe(include=['O'])
# Plot nominal feature distribution for Training Data

sub_title = list(train[[col for col in train.columns if col.startswith('nom_')]].columns)

fig = make_subplots(rows=1, cols=5,subplot_titles=sub_title)

for i in range(5): 

    a = train[f'nom_{i}'].value_counts()

    fig.add_trace(       

        go.Bar(x=a.index, y=a.values,width=0.2, text=a.values,textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'bin_{i}',showlegend=False,

               textfont=dict(size=15),

               marker=dict(

#                    color=['#732FB0','#2FB09D'],

                   line_width=0.5, line_color='gray')

              ),row=1, col=i+1)

    



fig.update_layout(title_text="Nominal Feature (0-4) Distribution (Train Data)")



# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')

#fig.show(config={"showLink":True})

fig.show()
# Plot nominal feature distribution for Test Data

sub_title = list(test[[col for col in test.columns if col.startswith('nom_')]].columns)

fig = make_subplots(rows=1, cols=5,subplot_titles=sub_title)

for i in range(5): 

    a = test[f'nom_{i}'].value_counts()

    fig.add_trace(       

        go.Bar(x=a.index, y=a.values,width=0.2, text=a.values,textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'bin_{i}',showlegend=False,

               textfont=dict(size=15),

               marker=dict(

#                    color=['#732FB0','#2FB09D'],

                   line_width=0.5, line_color='gray')

              ),row=1, col=i+1)

    



fig.update_layout(title_text="Nominal Feature (0-4) Distribution (Test Data)")



# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')

#fig.show(config={"showLink":True})

fig.show()
# Plot Nominal Feature Target Distribution for Training Data

sub_title = list(train[[col for col in test.columns if col.startswith('nom_')]].columns)

fig = make_subplots(rows=1, cols=5,subplot_titles=sub_title)

for i in range(5): 



    a = train.groupby([f'nom_{i}','target']).size().to_frame().reset_index().sort_values(0,ascending=False)

    a_0 = a[a['target']==0]

    a_1 = a[a['target']==1]

    

    if i == 0:

        fig.add_trace(       

            go.Bar( x=a_0[f'nom_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

                   texttemplate='%{text:.2s}',

                   name = 'target_0',legendgroup = 'target_0',

                   showlegend=False,

                   textfont=dict(size=15),

                    marker=dict(

#                         color=['#732FB0'] * a_1.shape[0],

                        line_width=0.5, line_color='gray')

                  ),row=1, col=i+1)



        fig.add_trace(       

            go.Bar(x=a_1[f'nom_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

                   texttemplate='%{text:.2s}',

                   name = 'target_1',legendgroup = 'target_1',

                   showlegend=False,

                   textfont=dict(size=15),

                    marker=dict(

#                         color=['#2FB09D'] * a_1.shape[0],

                        line_width=0.8, line_color='gray')

                  ),row=1, col=i+1)

    else:

        fig.add_trace(       

            go.Bar( x=a_0[f'nom_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

                   texttemplate='%{text:.2s}',

                   name = 'target_0',legendgroup = 'target_0',

                   showlegend=False,

                   textfont=dict(size=15),

                    marker=dict(

#                         color=['#732FB0'] * a_1.shape[0],

                        line_width=0.5, line_color='gray')

                  ),row=1, col=i+1)



        fig.add_trace(       

            go.Bar(x=a_1[f'nom_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

                   texttemplate='%{text:.2s}',

                   name = 'target_1',legendgroup = 'target_1',

                   showlegend=False,

                   textfont=dict(size=15),

                    marker=dict(

#                         color=['#2FB09D'] * a_1.shape[0],

                        line_width=0.8, line_color='gray')

                  ),row=1, col=i+1)

    

fig.update_layout(title_text="Nominal Feature (0-4) Target Distribution (Train Data)")



# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='v',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')

#fig.show(config={"showLink":True})

fig.show()
# Nominal Feature (0-4) Target Rate Distribution

for i in range(5):

    data = train[[f'nom_{i}', 'target']].groupby(f'nom_{i}')['target'].value_counts().unstack()

    data['rate'] = data[1]  / (data[0] + data[1] )

    data.sort_values(by=['rate'], inplace=True)

    display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39').format({'rate' : "{:.2%}"}))
train[[col for col in train.columns if col.startswith('ord_')]].describe(include='all')
# Plot Ordinal Feature  (0-3) Target Distribution for Training Data

sub_title = list(train[[col for col in test.columns if col.startswith('ord_')]].columns)[:4]



fig = make_subplots(rows=1, cols=4,subplot_titles=sub_title)

for i in range(4): 



    a = train.groupby([f'ord_{i}','target']).size().to_frame().reset_index().sort_values(0,ascending=False)

    a_0 = a[a['target']==0]

    a_1 = a[a['target']==1]

    

    fig.add_trace(       

        go.Bar( x=a_0[f'ord_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'ord_{i}'+' target_0',

               showlegend=False,

               textfont=dict(size=15),

                marker=dict(

#                     color=['#732FB0','#732FB0'],

                    line_width=0.5, line_color='gray')

              ),row=1, col=i+1)

    

    fig.add_trace(       

        go.Bar(x=a_1[f'ord_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'ord_{i}'+' target_1',

               showlegend=False,

               textfont=dict(size=15),

                marker=dict(

#                     color=['#2FB09D','#2FB09D'],

                    line_width=0.8, line_color='gray')

              ),row=1, col=i+1)  

    

fig.update_layout(title_text="Nominal Ordinal (0-3) Target Distribution (Train Data)",

                  barmode='group')



# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')

#fig.show(config={"showLink":True})

fig.show()
# Plot Ordinal Feature (4) Target Distribution for Training Data

sub_title = list(train[[col for col in test.columns if col.startswith('ord_')]].columns)[-2:-1]



fig = make_subplots(rows=1, cols=1,subplot_titles=sub_title)

for i in [4]: 



    a = train.groupby([f'ord_{i}','target']).size().to_frame().reset_index().sort_values(0,ascending=False)

    a_0 = a[a['target']==0]

    a_1 = a[a['target']==1]

    

    fig.add_trace(       

        go.Bar( x=a_0[f'ord_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'ord_{i}'+' target_0',

               showlegend=False,

               textfont=dict(size=5),textangle=0,constraintext='inside',

                marker=dict(

#                     color=['#732FB0','#732FB0'],

                    line_width=0.5, line_color='gray')

              ),row=1, col=1)

    

    fig.add_trace(       

        go.Bar(x=a_1[f'ord_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'ord_{i}'+' target_1',

               showlegend=False,

               textfont=dict(size=5),textangle=0,constraintext='inside',

                marker=dict(

#                     color=['#2FB09D','#2FB09D'],

                    line_width=0.8, line_color='gray')

              ),row=1, col=1)  

    

fig.update_layout(title_text="Nominal Ordinal (4) Target Distribution (Train Data)",

                  barmode='group')







# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',width=500,height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')

#fig.show(config={"showLink":True})

fig.show()



# Plot Ordinal Feature (5) Target Distribution for Training Data

sub_title = list(train[[col for col in test.columns if col.startswith('ord_')]].columns)[-1:]



fig = make_subplots(rows=1, cols=1,subplot_titles=sub_title)

for i in [5]: 



    a = train.groupby([f'ord_{i}','target']).size().to_frame().reset_index().sort_values(0,ascending=False)

    a_0 = a[a['target']==0]

    a_1 = a[a['target']==1]

    

    fig.add_trace(       

        go.Bar( x=a_0[f'ord_{i}'], y=a_0[0],width=0.2, text=a_0[0],textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'ord_{i}'+' target_0',

               showlegend=False,

               textfont=dict(size=5),textangle=0,constraintext='inside',

                marker=dict(

#                     color=['#732FB0','#732FB0'],

                    line_width=0.5, line_color='gray')

              ),row=1, col=1)

    

    fig.add_trace(       

        go.Bar(x=a_1[f'ord_{i}'], y=a_1[0],width=0.2, text=a_1[0],textposition='outside',

               texttemplate='%{text:.2s}',

               name=f'ord_{i}'+' target_1',

               showlegend=False,

               textfont=dict(size=5),textangle=0,constraintext='inside',

                marker=dict(

#                     color=['#2FB09D','#2FB09D'],

                    line_width=0.8, line_color='gray')

              ),row=1, col=1)  

    

fig.update_layout(title_text="Nominal Ordinal (5) Target Distribution (Train Data)",

                  barmode='group')

# Hide the yaxis

fig.update_layout(yaxis=dict(visible=False),yaxis2=dict(visible=False),yaxis3=dict(visible=False),yaxis4=dict(visible=False),yaxis5=dict(visible=False))



# this size is for kaggle kernal

fig.update_layout(margin=dict(l=25,t=50,b=0),legend_orientation='h',height=350) 



# Set the subtitle size

for i in fig['layout']['annotations']:

    i['font'] = dict(size=10,color='black')

#fig.show(config={"showLink":True})

fig.show()
# Ordinal Feature (0-4) Target Rate Distribution

for i in range(5):

    data = train[[f'ord_{i}', 'target']].groupby(f'ord_{i}')['target'].value_counts().unstack()

    data['rate'] = data[1]  / (data[0] + data[1] )

    data.sort_values(by=['rate'], inplace=True)

    display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39').format({'rate' : "{:.2%}"}))
# day & month Target Rate Distribution

data = train[['day', 'target']].groupby('day')['target'].value_counts().unstack()

data['rate'] = data[1]  / (data[0] + data[1] )

data.sort_values(by=['rate'], inplace=True)

display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39').format({'rate' : "{:.2%}"}))



data = train[['month', 'target']].groupby('month')['target'].value_counts().unstack()

data['rate'] = data[1]  / (data[0] + data[1] )

data.sort_values(by=['rate'], inplace=True)

display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39').format({'rate' : "{:.2%}"}))
train[[col for col in train.columns if col.startswith('bin_')]].describe(include='all')
#  lable Encoding for binary features



for i in range(5):

    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'bin_{i}'].dropna().unique()) + list(test[f'bin_{i}'].dropna().unique())))))}

    ord_order_dict['NULL']=len(train[f'bin_{i}'].dropna().unique()) # mapping null value

    print(ord_order_dict)

    bin_encoding = [{'col': f'bin_{i}', 'mapping': ord_order_dict}]

    label = OrdinalEncoder(mapping=bin_encoding)    

    train['lable_' + f'bin_{i}'] =  label.fit_transform(train[f'bin_{i}'].fillna('NULL'))

    test['lable_' + f'bin_{i}'] =  label.fit_transform(test[f'bin_{i}'].fillna('NULL'))
# Target Encoding for binary features



for i in range(5):

    label = TargetEncoder()

    train['target_' + f'bin_{i}'] = label.fit_transform(train[f'bin_{i}'].fillna('NULL'), target)

    test['target_' + f'bin_{i}'] = label.transform(test[f'bin_{i}'].fillna('NULL'))
train[[col for col in train.columns if col.startswith('nom_')]].describe(include='all')
# Lable Encoding for Nominal features



for i in range(10):

    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'nom_{i}'].dropna().unique()) + list(test[f'nom_{i}'].dropna().unique())))))}

    ord_order_dict['NULL']=len(train[f'nom_{i}'].dropna().unique()) # mapping null value

#     print(ord_order_dict)

    bin_encoding = [{'col': f'nom_{i}', 'mapping': ord_order_dict}]

    label = OrdinalEncoder(mapping=bin_encoding)    

    train['lable_' + f'nom_{i}'] =  label.fit_transform(train[f'nom_{i}'].fillna('NULL'))

    test['lable_' + f'nom_{i}'] =  label.fit_transform(test[f'nom_{i}'].fillna('NULL'))
# Target Encoding for Nominal features



for i in range(10):

    label = TargetEncoder()

    train['target_' + f'nom_{i}'] = label.fit_transform(train[f'nom_{i}'].fillna('NULL'), target)

    test['target_' + f'nom_{i}'] = label.transform(test[f'nom_{i}'].fillna('NULL'))
train[[col for col in train.columns if col.startswith('ord_')]].describe(include='all')
# features 'ord_0', 'ord_1', 'ord_2'  follow the order below

ord_order = [

    [1.0, 2.0, 3.0],

    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],

    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']]



for i in range(0, 3):

    ord_order_dict = {i : j for j, i in enumerate(ord_order[i])}

    ord_order_dict['NULL']=len(train[f'ord_{i}'].dropna().unique()) # mapping null value

    print(ord_order_dict)

    bin_encoding = [{'col': f'ord_{i}', 'mapping': ord_order_dict}]

    label = OrdinalEncoder(mapping=bin_encoding)    

    train['lable_' + f'ord_{i}'] =  label.fit_transform(train[f'ord_{i}'].fillna('NULL'))

    test['lable_' + f'ord_{i}'] =  label.fit_transform(test[f'ord_{i}'].fillna('NULL'))
# features 'ord_3', 'ord_4', 'ord_5'  follow the alphabet order



for i in range(3, 6):

    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'ord_{i}'].dropna().unique()) + list(test[f'ord_{i}'].dropna().unique())))))}

    ord_order_dict['NULL']=len(train[f'ord_{i}'].dropna().unique()) # mapping null value

#     print(ord_order_dict)

    bin_encoding = [{'col': f'ord_{i}', 'mapping': ord_order_dict}]

    label = OrdinalEncoder(mapping=bin_encoding)    

    train['lable_' + f'ord_{i}'] =  label.fit_transform(train[f'ord_{i}'].fillna('NULL'))

    test['lable_' + f'ord_{i}'] =  label.fit_transform(test[f'ord_{i}'].fillna('NULL'))
# Target Encoding for Ordinal features



for i in range(6):

    label = TargetEncoder()

    train['target_' + f'ord_{i}'] = label.fit_transform(train[f'ord_{i}'].fillna('NULL'), target)

    test['target_' + f'ord_{i}'] = label.transform(test[f'ord_{i}'].fillna('NULL'))
#  lable Encoding for Day & Month features



ord_order_dict = { j+1: i for j, i in enumerate(sorted(list(set(list(train['day'].dropna().unique()) + list(test['day'].dropna().unique())))))}

ord_order_dict['NULL']=len(train['day'].unique()) # mapping null value

bin_encoding = [{'col': 'day', 'mapping': ord_order_dict}]

label = OrdinalEncoder(mapping=bin_encoding)    

       

train['lable_' + 'day'] =  label.fit_transform(train['day'].fillna('NULL'))

test['lable_' + 'day'] =  label.fit_transform(test['day'].fillna('NULL'))





ord_order_dict = { j+1: i for j, i in enumerate(sorted(list(set(list(train['month'].dropna().unique()) + list(test['month'].dropna().unique())))))}

ord_order_dict['NULL']=len(train['month'].unique()) # mapping null value

bin_encoding = [{'col': 'month', 'mapping': ord_order_dict}]

label = OrdinalEncoder(mapping=bin_encoding)    

       

train['lable_' + 'month'] =  label.fit_transform(train['month'].fillna('NULL'))

test['lable_' + 'month'] =  label.fit_transform(test['month'].fillna('NULL'))

# Target Encoding for Day & Month features

label = TargetEncoder()

train['target_' + 'day'] = label.fit_transform(train['day'].fillna('NULL'), target)

test['target_' + 'day'] = label.transform(test['day'].fillna('NULL'))



label = TargetEncoder()

train['target_' + 'month'] = label.fit_transform(train[f'ord_{i}'].fillna('NULL'), target)

test['target_' + 'month'] = label.transform(test[f'ord_{i}'].fillna('NULL'))
# target features correlations

df_target_feature = train[[col for col in train.columns if col.startswith('target_')]]



sns.set(style="white")



# Compute the correlation matrix

corr = df_target_feature.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

ax.set_title('Target Features Correlation');