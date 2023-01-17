import numpy as np

import pandas as pd 

import os 



import matplotlib 

matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)

import matplotlib.pyplot as plt

plt.style.use(['fivethirtyeight', 'dark_background'])

import seaborn as sns

from PIL import Image



import itertools

import warnings 

warnings.filterwarnings('ignore')



import io

import plotly.offline as py 

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff



from sklearn import preprocessing

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support, accuracy_score)



from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.preprocessing import LabelEncoder 

from sklearn.preprocessing import StandardScaler



from keras.layers import Input, Dense, Lambda

from keras.models import Model

from keras.objectives import binary_crossentropy

from keras.callbacks import LearningRateScheduler

from keras.utils.vis_utils import model_to_dot

from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K

from keras.callbacks import Callback
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



df.head()
df.info()
# Replace empty space with null 

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)



# Only keep non nulls 

df = df[df['TotalCharges'].notnull()]



# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# coerce necessary, unable to parse string otherwise



# Convert to float 

df['TotalCharges'] = df['TotalCharges'].astype(float)



df = df.reset_index()[df.columns]
#replace 'No internet service' to No for the following columns



replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']



for i in replace_cols : 

    df[i]  = df[i].replace({'No internet service' : 'No'})
df['SeniorCitizen'] = df['SeniorCitizen'].replace({1:'Yes',

                                                   0:'No'})
# Convert tenure to categorical 



def tenure_lab(df):

    

    if df['tenure'] <= 10:

        return "tenure-0-10"

    elif (df['tenure'] > 10) & (df['tenure'] <= 20):

        return 'tenure-11-20'

    elif (df['tenure'] > 21) & (df['tenure'] <= 30):

        return 'tenure-21-30'

    elif (df['tenure'] > 31) & (df['tenure'] <= 40):

        return 'tenure-31-40'

    elif (df['tenure'] > 41) & (df['tenure'] <= 50):

        return 'tenure-41-50'

    elif (df['tenure'] > 51) & (df['tenure'] <= 60):

        return 'tenure-51-60'

    elif df['tenure'] > 60:

        return 'tenure-60+'

    

df['tenure_group'] = df.apply(lambda df:tenure_lab(df),

                              axis = 1)
churn = df[df['Churn'] == 'Yes']

unchurn = df[df['Churn'] == 'No']
print('\nMissing values: ', df.isnull().sum().values.sum())

print('\nUnique values: ', df.nunique())
# Separate cats and nums 

ID = ['customerID']

target = ['Churn']



cat_col = df.nunique()[df.nunique() < 6].keys().tolist()

cat_col = [x for x in cat_col if x not in target]



num_col = [x for x in df.columns if x not in cat_col + target + ID]
cat_col
num_col
df.head()
churn.head()
labels = df['Churn'].value_counts().keys().tolist()



values = df['Churn'].value_counts().values.tolist()
trace = go.Pie(labels = labels,

               values = values,

               marker = dict(colors = ['dodgerblue', 'maroon'],

                              line = dict(color = 'white',

                                          width = .8)

                             ),

              hoverinfo = 'label+value+text',

              hole = .4

              )



layout = go.Layout(dict(title = 'Churn percentages')

                  )



data = [trace]

fig = go.Figure(data = data, 

                layout = layout)



py.iplot(fig)
sns.kdeplot(df['tenure'].loc[df['Churn'] == 'No'],

            label = 'unchurned', shade = True);



sns.kdeplot(df['tenure'].loc[df['Churn'] == 'Yes'],

            label = 'churn', shade = True);
sns.kdeplot(df['TotalCharges'].loc[df['Churn'] == 'No'],

            label = 'unchurned', shade = True);



sns.kdeplot(df['TotalCharges'].loc[df['Churn'] == 'Yes'],

            label = 'churn', shade = True);
def pie_chart(column):

    

    trace1 = go.Pie(values = churn[column].value_counts().values.tolist(),

                    labels = churn[column].value_counts().keys().tolist(),

                    hoverinfo = 'label+percent+name',

                    domain = dict(x = [0, .5]),

                    name = 'churned',

                    marker = dict(colors = ['dodgerblue', 'maroon'],

                                  line = dict(color = 'white',

                                              width = .8)

                                 ),

                    hole = .6

                   )

    

    trace2 = go.Pie(values = unchurn[column].value_counts().values.tolist(),

                    labels = unchurn[column].value_counts().keys().tolist(),

                    hoverinfo = 'label+percent+name',

                    marker = dict(colors = ['dodgerblue', 'maroon'],

                                  line = dict(color = 'white',

                                              width = .8)

                                 ),

                    domain = dict(x = [.5, 1]),

                    hole = .6,

                    name = 'unchurned'

                   )

    

    layout = go.Layout(dict(title = column + 'churn distribution',

                            annotations = [dict(text = 'churned',

                                                font = dict(size = 10),

                                                showarrow = False,

                                                x = .20, y = .5),

                                          dict(text = 'unchurned',

                                               font = dict(size = 10),

                                               showarrow = False,

                                               x = .90, y = .5)

                                          ]

                           )

                      )

    

    data = [trace1, trace2]

    fig = go.Figure(data = data, 

                    layout = layout)

    py.iplot(fig)                                    
def histogram(column):

    

    trace1 = go.Histogram(x = churn[column],

                          histnorm = 'percent',

                          name = 'churned',

                          marker = dict(line = dict(color = 'white',

                                                    width = .5

                                                   )

                                       ),

                          opacity = 0.8

                         )

    

    trace2 = go.Histogram(x = churn[column],

                          histnorm = 'percent',

                          name = 'unchurned',

                          marker = dict(line = dict(color = 'black',

                                                    width = .5

                                                   )

                                       ),

                          opacity = 0.8

                         )

    

    data = [trace1, trace2]

    layout = go.Layout(dict(title = column + 'churn distribution',

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = column,

                                         zerolinewidth = 1,

                                         ticklen = 4,

                                         gridwidth = 2

                                        ),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = 'percent',

                                         zerolinewidth = 1,

                                         ticklen = 5,

                                         gridwidth = 2

                                        ),

                           )

                      )

    

    fig = go.Figure(data = data, 

                    layout = layout)

    

    py.iplot(fig)
# Scatter plot matrix function 



def scatter_matrix(df):

    

    df = df.sort_values(by = 'Churn', ascending = True)

    classes = df['Churn'].unique().tolist()

    classes 

    

    class_code = {classes[k] : k for k in range(2)}

    class_code

    

    color_vals = [class_code[cl] for cl in df['Churn']]

    color_vals

    

    pl_colorscale = 'Portland'

    

    pl_colorscale

    

    text = [df.loc[k, 'Churn'] for k in range(len(df))]

    text

    

    trace = go.Splom(dimensions = [dict(label = 'tenure',

                                        values = df['tenure']),

                                   dict(label = 'MonthlyCharges',

                                        values = df['MonthlyCharges']),

                                   dict(label = 'TotalCharges',

                                        values = df['TotalCharges'])],

                    text = text,

                    marker = dict(color = color_vals,

                                  colorscale = pl_colorscale,

                                  size = 3,

                                  showscale = False,

                                  line = dict(width = .3,

                                              color = 'rgb(260, 260, 260)'

                                             )

                                 )

                    )

    

    axis = dict(showline = True,

                zeroline = False,

                gridcolor = '#fff',

                ticklen = 4

               )

    

    layout = go.Layout(dict(title = 'Scatter plot matrix for numerical features',

                            autosize = False,

                            height = 800,

                            width  = 800,

                            dragmode = "select",

                            hovermode = "closest",

                            plot_bgcolor  = 'rgba(240,240,240, 0.95)',

                            xaxis1 = dict(axis),

                            yaxis1 = dict(axis),

                            xaxis2 = dict(axis),

                            yaxis2 = dict(axis),

                            xaxis3 = dict(axis),

                            yaxis3 = dict(axis),

                           )

                      )

    

    data = [trace]

    fig = go.Figure(data = data,

                    layout = layout)

    py.iplot(fig)
for c in cat_col:

    pie_chart(c)
for n in num_col:

    histogram(n)
scatter_matrix(df)
tg_ch  =  churn['tenure_group'].value_counts().reset_index()

tg_ch.columns  = ['tenure_group','count']



tg_uch =  unchurn['tenure_group'].value_counts().reset_index()

tg_uch.columns = ['tenure_group','count']
tg_ch
trace1 = go.Bar(x = tg_ch['tenure_group'], 

                y = tg_ch['count'],

                name = 'churned',

                marker = dict(line = dict(width = .5, 

                                          color = 'black')

                             ),

                opacity = .5)



trace2 = go.Bar(x = tg_uch['tenure_group'],

                y = tg_uch['count'],

                name = 'unchurned',

                marker = dict(line = dict(width = .5,

                                          color = 'black')

                             ),

                opacity = .8)



layout = go.Layout(dict(title = 'Subscriber churn in tenure groups',

                       xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                    title = 'tenure group',

                                    zerolinewidth = 1, 

                                    ticklen = 5, 

                                    gridwidth = 2),

                       yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                    title = 'count',

                                    zerolinewidth = 1,

                                    ticklen = 5, 

                                    gridwidth = 2),

                       )

                  )



data = [trace1, trace2]

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
df[['MonthlyCharges', 'TotalCharges', 'tenure', 'tenure_group']]
# Tenure group

def tenure_scatter(tenure_group, color):

    tracer = go.Scatter(x = df[df['tenure_group'] == tenure_group]['MonthlyCharges'],

                        y = df[df['tenure_group'] == tenure_group]['TotalCharges'],

                        mode = 'markers', 

                        marker = dict(line = dict(color = 'black',

                                                  width = .3),

                                      size = 5, 

                                      color = color,

                                      symbol = 'diamond-dot',

                                     ),

                        name = tenure_group,

                        opacity = .5

                       )

    return tracer
# Churn group



def churn_scatter(churn, color):

    tracer = go.Scatter(x = df[df['Churn'] == churn]['MonthlyCharges'],

                        y = df[df['Churn'] == churn]['TotalCharges'],

                               mode = 'markers',

                               marker = dict(line = dict(color = 'black',

                                                         width = .3),

                                             size = 4,

                                             color = color,

                                             symbol = 'diamond-dot',),

                              name = 'Churn - ' + churn,

                              opacity = .5)

    return tracer 
trace1 = tenure_scatter('tenure-0-10', 'orange')

trace2 = tenure_scatter('tenure-11-20', 'green')

trace3 = tenure_scatter('tenure-21-30', 'blue')

trace4 = tenure_scatter('tenure-31-40', 'red')

trace5 = tenure_scatter('tenure-41-50', 'grey')

trace6 = tenure_scatter('tenure-51-60', 'pink')

trace7 = tenure_scatter('tenure-60+', 'yellow')



trace8 = churn_scatter('Yes', 'red')

trace9 = churn_scatter('No', 'blue')



data1 = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

data2 = [trace8, trace9]



def layout_title(title):

    layout = go.Layout(dict(title = title,

                            xaxis = dict(gridcolor = 'rgb(250, 250, 250)',

                                         title = 'Monthly charges',

                                         zerolinewidth = 1, 

                                         ticklen = 5, 

                                         gridwidth = 3),

                            yaxis = dict(gridcolor = 'rgb(250, 250, 250)',

                                         title = 'Total charges',

                                         zerolinewidth = 1, 

                                         ticklen = 5, 

                                         gridwidth = 2),

                            height = 500

                           )

                      )

    return layout



layout1 = layout_title('Charges by tenure')

layout2 = layout_title('Charges by churn')



fig1 = go.Figure(data = data1, layout = layout1)

fig2 = go.Figure(data = data2, layout = layout2)



py.iplot(fig1)

py.iplot(fig2)
mean_tenure_charge = df.groupby(['tenure_group','Churn'])[['MonthlyCharges',

                                                    'TotalCharges']].mean().reset_index()



mean_tenure_charge
# tracing function



def mean_charges(column, aggregate):

    tracer = go.Bar(x = mean_tenure_charge[mean_tenure_charge['Churn'] == aggregate]['tenure_group'],

                    y = mean_tenure_charge[mean_tenure_charge['Churn'] == aggregate][column],

                    name = aggregate, 

                    marker = dict(line = dict(width = 1)),

                    text = 'Churn'

                   )

    return tracer



# layout function



def layout_plot(title, xaxis_lab, yaxis_lab):

    layout = go.Layout(dict(title = title,

                            xaxis = dict(gridcolor = 'rgb(250, 250, 250)',

                                         title = xaxis_lab,

                                         zerolinewidth = 1,

                                         ticklen = 5,

                                         gridwidth = 3),

                            yaxis = dict(gridcolor = 'rgb(250, 250, 250)',

                                         title = yaxis_lab,

                                         zerolinewidth = 1, 

                                         ticklen = 5, 

                                         gridwidth = 2),

                           )

                      )

    return layout
trace1 = mean_charges('MonthlyCharges', 'Yes')

trace2 = mean_charges('MonthlyCharges', 'No')



layout1 = layout_plot('Mean monthly charges by tenure group',

                     'Tenure group', 'Monthly Charges')



data1 = [trace1, trace2]

fig1 = go.Figure(data = data1, 

                 layout = layout1)



trace3 = mean_charges('TotalCharges', 'Yes')

trace4 = mean_charges('TotalCharges', 'No')



layout2 = layout_plot('Mean total charges by tenure group',

                      'Tenure group', 'Total Charges')



data2 = [trace3, trace4]

fig2 = go.Figure(data = data2, 

                 layout = layout2)



py.iplot(fig1)

py.iplot(fig2)
df = df.drop(columns = 'tenure_group', axis = 1)
df.columns
ID = ['customerID']

target_col = ['Churn']



cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols if x not in target]



num_cols = [x for x in df.columns if x not in cat_cols + target_col + ID]
# Binary col label encoding 



bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()



le = LabelEncoder()

for i in bin_cols:

    df[i] = le.fit_transform(df[i])
# Multi val col



multi_cols = [i for i in cat_col if i not in bin_cols]



df = pd.get_dummies(data = df, 

                    columns = multi_cols)
num_cols
# Scaling numerical col



# Scale data before applying PCA, to ensure unit variance

    # Fitting algorithms are highly dependent on scaling of features 

    

stc = StandardScaler() # StandardScaler subtracts the mean from each features and then scale to unit variance 

scaled = stc.fit_transform(df[num_cols])

scaled = pd.DataFrame(scaled, columns = num_cols)
# drop original values, 

copy = df.copy()



df = df.drop(columns = num_cols, axis = 1)



# merge scaled values for num_col

df = df.merge(scaled, 

              left_index = True, right_index = True,

              how = 'left'

             )
df.head()
summary = (copy[[i for i in copy.columns if i not in ID]].describe().transpose().reset_index())



summary = summary.rename(columns = {'index' : 'feature'})

summary = np.around(summary, 3)
val_list = [summary['feature'], summary['count'],

            summary['mean'], summary['std'],

            summary['min'], 

            summary['25%'],

            summary['50%'], 

            summary['75%'], 

            summary['max']]
trace = go.Table(header = dict(values = summary.columns.tolist(),

                               line = dict(color = ['black']),

                               fill = dict(color = ['dodgerblue']),

                               ),

                 cells = dict(values = val_list,

                              line = dict(color = ['black']),

                              fill = dict(color = ['aliceblue', 'cornflowerblue'])

                             ),

                columnwidth = [200, 60, 100, 100, 60, 60, 80, 80, 80]) # 9 summaries



layout = go.Layout(dict(title = 'Variable summary'))

figure = go.Figure(data = [trace], 

                   layout = layout)



py.iplot(figure)
corr = df.corr()



# tick labels 

matrix_col = corr.columns.tolist()



# Array conversion

corr_array = np.array(corr)
trace = go.Heatmap(z = corr_array,

                   x = matrix_col,

                   y = matrix_col,

                   colorscale = 'Viridis',

                   colorbar = dict(title = 'Pearson corr coeficient',

                                   titleside = 'right'

                                  ),

                  )



layout = go.Layout(dict(title = 'Corr matrix of variables',

                        height = 600,

                        width = 700,

                        yaxis = dict(tickfont = dict(size = 8)

                                    ),

                        xaxis = dict(tickfont = dict(size = 9))

                       )

                  )



data = [trace]

fig = go.Figure(data = data, layout = layout)

py.iplot(fig) # Hover to check coefficients 
pca = PCA(n_components = 2)



X = df[[i for i in df.columns if i not in ID + target]]

Y = df[target + ID]
principal_comp = pca.fit_transform(X)



pca_data = pd.DataFrame(principal_comp, 

                        columns = ['PC1', 'PC2'])



pca_data = pca_data.merge(Y, 

                          left_index = True,

                          right_index = True, 

                          how = 'left')



pca_data['Churn'] = pca_data['Churn'].replace({1:'Churned',

                                               0:'Unchurned'})
def pca_scatter(target, color):

    tracer = go.Scatter(x = pca_data[pca_data['Churn'] == target]['PC1'], 

                        y = pca_data[pca_data['Churn'] == target]['PC2'],

                        name = target,

                        mode = 'markers',

                        marker = dict(color = color,

                                      line = dict(width = .5),

                                      symbol = 'diamond-open'),

                        text = ('Customer Id :' + 

                                pca_data[pca_data['Churn'] == target]['customerID'])

                       )

    return tracer 



layout = go.Layout(dict(title = 'Principal components',

                        xaxis = dict(gridcolor = 'rgb(250, 250, 250)',

                                     title = 'PC1',

                                     zerolinewidth = 1, 

                                     ticklen = 5,

                                     gridwidth = 2),

                       yaxis = dict(gridcolor = 'rgb(250, 250, 250)',

                                    title = 'PC2',

                                    zerolinewidth = 1,

                                    ticklen = 5, 

                                    gridwidth = 2),

                       height = 600

                       )

                  )



trace1 = pca_scatter('Churned', 'crimson')

trace2 = pca_scatter('Unchurned', 'dodgerblue')



data = [trace1, trace2]



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
print('shape of', principal_comp) # 2 features
pca_var = np.var(principal_comp, axis = 0)



pca_var_ratio = pca_var / np.sum(pca_var)



print(pca_var_ratio)
plt.matshow(pca.components_, cmap = 'viridis')



plt.yticks([0, 1, 2], 

           ['feature 1', 'feature 2'], 

           fontsize = 10)



plt.colorbar()



plt.xticks(range(len(df.columns)),

           df.columns, 

           rotation = 90, 

           ha = 'left')



plt.tight_layout()

plt.show()#
useless_features = list(df.columns[0:24])

useless_features
s = sns.heatmap(df[useless_features].corr(), 

                cmap = 'coolwarm')



s.set_yticklabels(s.get_yticklabels(), 

                  rotation = 45, fontsize = 8)



s.set_xticklabels(s.get_xticklabels(), 

                  rotation = 35, fontsize = 8)
useful_features = list(df.columns[25:28])

useful_features



e = sns.heatmap(df[useful_features].corr(),

                cmap = 'coolwarm')



s.set_yticklabels(s.get_yticklabels(), 

                  rotation = 45, fontsize = 8)



s.set_xticklabels(s.get_xticklabels(), 

                  rotation = 45, fontsize = 7)



plt.show()
data = df[['PaymentMethod_Electronic check',

           'PaymentMethod_Mailed check',

           'tenure']]



Y = data.values



Y_std = StandardScaler().fit_transform(Y)



mean_vec = np.mean(Y_std, axis = 0)



# Covariance matrix

cov_mat = np.cov(Y_std.T)



# eigen pairs

eig_vals, eig_vecs = np.linalg.eig(cov_mat)



# pairs in tuple

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]



eig_pairs.sort(key = lambda x: x[0], reverse = True)
total_var = sum(eig_vals)



individual_var = [(i / total_var) * 100 for i in sorted(eig_vals, reverse = True)]



cum_var_explained = np.cumsum(individual_var)

cum_var_explained
plt.figure(figsize = (16, 8))

plt.bar(range(len(individual_var)), individual_var,

        alpha = 0.8,

        align = 'center',

        label = 'explained variance by individual component',

        color = 'b')



plt.step(range(len(cum_var_explained)

              ), cum_var_explained,

         where = 'mid',

         label = 'cumulative explained variation')



plt.xlabel('PC')

plt.ylabel('explained variance ratio')

plt.legend(loc = 'best')

plt.show();
# Isolate binary col

binary_col = df.nunique()[df.nunique() == 2].keys() # extract cols with 2 values 



radar = df[binary_col]
# Radar configuration 

# plot churned & unchurned distribution



def radar_plot(df, aggregate, title):

    data_frame = df[df['Churn'] == aggregate] 

    data_frame_x = data_frame[binary_col].sum().reset_index()

    data_frame_x.columns = ['feature', 'Yes']

    data_frame_x['No'] = data_frame.shape[0] - data_frame_x['Yes'] # No = unchurned 

    data_frame_x = data_frame_x[data_frame_x['feature'] != 'Churn']

    

    # 1 = Yes, churned 

    trace1 = go.Scatterpolar(r = data_frame_x['Yes'].values.tolist(),

                             theta = data_frame_x['feature'].tolist(),

                             fill = 'toself',

                             name = '1 count',

                             mode = 'markers + lines',

                             marker = dict(size = 5)

                            )

    

    # 0 = No, unchurned 

    trace2 = go.Scatterpolar(r = data_frame_x['No'].values.tolist(),

                             theta = data_frame_x['feature'].tolist(),

                             fill = 'toself',

                             name = ' 0 count',

                             mode = 'markers + lines',

                             marker = dict(size = 5)

                            )

    

    layout = go.Layout(dict(polar = dict(radialaxis = dict(visible = True,

                                                           side = 'counterclockwise',

                                                           showline = True,

                                                           linewidth = 2,

                                                           tickwidth = 2,

                                                           gridcolor = 'white',

                                                           gridwidth = 2),

                                        angularaxis = dict(tickfont = dict(size = 10),

                                                           layer = 'below traces'),

                                        bgcolor = 'rgb(250, 250, 250)',

                                        ),

                           title = title,

                        height = 800))

    

    data = [trace2, trace1] # Unchurned = Blue, Churned = Red

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig)

    

radar_plot(radar, 1, 'Churned')

radar_plot(radar, 0, 'Unchurned')