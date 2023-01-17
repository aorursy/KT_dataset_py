# for general handling and manipulation of table data

import pandas as pd

import numpy as np



# for visualization of missing data entries

import missingno as msno



# for generation of interactive data visualization

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



# for random forest model predictions and result analysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



# start a random seed for reproducibility of results

np.random.seed(1)
train_data = pd.read_csv('../input/train.csv')
print('train_data({0[0]},{0[1]})'.format(train_data.shape))



print('Number of training examples: {0}'.format(train_data.shape[0]))

print('Number of features for each example: {0}'.format(train_data.shape[1]))
pd.DataFrame(data = {'Feature Label': train_data.columns})
no_ps = [train_data.columns[x][3:] for x in range(2, len(train_data.columns))]

train_data.columns = train_data.columns[:2].tolist() + no_ps
NA_columns = train_data.columns[train_data.isin(['-1']).any()]

NA_data_counts = train_data.isin(['-1']).sum()[train_data.isin(['-1']).any()]

pd.DataFrame(data = NA_data_counts, columns = ['# of missing entries'])
NA_data = train_data[NA_columns].replace(-1, np.NaN)

msno.matrix(df=NA_data, color = (0, 0.3, 0.3))
labels = ["Target = 0", "Target = 1"]

values = train_data["target"].value_counts().values



trace = go.Pie(labels = labels, values = values)

layout = go.Layout(title = 'Distribution of Target Variable')



fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
bin_columns = train_data.columns[train_data.columns.str.contains('_bin')]



print("# of binary features: {0}".format(len(bin_columns)))
bin_counts = train_data[bin_columns].apply(pd.value_counts)



trace = []

for i in range(bin_counts.shape[0]):

    trace_temp = go.Bar(

        x= np.asarray(bin_columns),

        y= bin_counts.values[i],

        name = bin_counts.index[i]

    )

    trace.append(trace_temp)



layout = go.Layout(

    barmode = 'stack',

    title = 'Distribution of Binary Features'

)



fig = go.Figure(data = trace, layout = layout)

iplot(fig)
cat_columns = train_data.columns[train_data.columns.str.contains('_cat')]

cat_data = pd.DataFrame(data = {'# of levels': train_data[cat_columns].max()})



cat_data
cat_counts = train_data[cat_columns].apply(pd.value_counts)



trace = []

for i in range(cat_counts.shape[0]):

    trace_temp = go.Bar(

        x= np.asarray(cat_columns),

        y= cat_counts.values[i],

        name = cat_counts.index[i]

    )

    trace.append(trace_temp)



layout = go.Layout(

    barmode = 'stack',

    title = 'Distribution of Categorical Features'

)



fig = go.Figure(data = trace, layout = layout)

iplot(fig)
#Isolate the columns that are not binary or categorical

misc_columns = train_data.columns.drop(cat_columns).drop(bin_columns).drop(["id", "target"])



#Split these columns by group

ind_columns = misc_columns[misc_columns.str.contains('ind')]

reg_columns = misc_columns[misc_columns.str.contains('reg')]

car_columns = misc_columns[misc_columns.str.contains('car')]

calc_columns = misc_columns[misc_columns.str.contains('calc')]



#create boxplots for 'ind' columns

trace1 = []

for i in range(len(ind_columns)):

    trace_temp = go.Box(

        y= np.random.choice(train_data[ind_columns[i]], 2000, replace=False),

        name = ind_columns[i]

    )



    trace1.append(trace_temp)



layout1 = go.Layout(

    title = 'Distribution of "ind" Features'

)



# create boxplots for 'reg' columns

trace2 = []

for i in range(len(reg_columns)):

    trace_temp = go.Box(

        y= np.random.choice(train_data[reg_columns[i]], 2000, replace=False),

        name = reg_columns[i],

        boxpoints = 'suspectedoutliers'

    )



    trace2.append(trace_temp)



layout2 = go.Layout(

    title = 'Distribution of "reg" Features'

)



# create boxplots for 'car' columns

trace3 = []

for i in range(len(car_columns)):

    trace_temp = go.Box(

        y= np.random.choice(train_data[car_columns[i]], 2000, replace=False),

        name = car_columns[i],

        boxpoints = 'suspectedoutliers',

    )



    trace3.append(trace_temp)



layout3 = go.Layout(

    title = 'Distribution of "car" Features'

)



# create boxplots for 'calc' columns

trace4 = []

for i in range(len(calc_columns)):

    trace_temp = go.Box(

        y= np.random.choice(train_data[calc_columns[i]], 2000, replace=False),

        name = calc_columns[i],

        boxpoints = 'suspectectedoutliers'

    )



    trace4.append(trace_temp)



layout4 = go.Layout(

    title = 'Distribution of "calc" Features'

)



fig1 = go.Figure(data = trace1, layout = layout1)

fig2 = go.Figure(data = trace2, layout = layout2)

fig3 = go.Figure(data = trace3, layout = layout3)

fig4 = go.Figure(data = trace4, layout = layout4)





iplot(fig1)

iplot(fig2)

iplot(fig3)

iplot(fig4)
# Separate rows where target is 0 or 1

target_data = [train_data["target"] == 0, train_data["target"] == 1]





trace = []

trace_temp = go.Box(

    y= np.random.choice(train_data['car_13'][target_data[0]],

                        2000,

                        replace=False

                       ),

    name = 'Target = 0',

    boxpoints = 'all',

    boxmean = 'sd'

)



trace.append(trace_temp)



trace_temp = go.Box(

    y= np.random.choice(train_data['car_13'][target_data[1]],

                        2000,

                        replace=False

                       ),

    name = 'Target = 1',

    boxpoints = 'all',

    boxmean = 'sd'

)



trace.append(trace_temp)



layout = go.Layout(

    title = 'car_13 Feature Distribution',

    width = 900,

    height = 1000,

)



fig = go.Figure(data = trace, layout = layout)

iplot(fig)
# Randomly select 10% of data for CV set

CV_index = np.random.choice(train_data.shape[0], int(train_data.shape[0]*.1), replace = False)

CV = train_data.iloc[CV_index, :]



# Use remaining 90% for model training data

train = train_data.drop(CV_index, axis = 0)



assert(train_data.shape[0] == train.shape[0]+CV.shape[0])



print('Number of training examples: {0}'.format(train.shape[0]))

print('Number of cross-validation examples: {0}'.format(CV.shape[0]))
labels = ["Target = 0", "Target = 1"]

train_values = train["target"].value_counts().values

CV_values = CV["target"].value_counts().values



#Distribution of training data set Target variable

trace1 = go.Pie(labels =  labels,

                values = train_values,

                domain= {"x": [0, 0.45]},

                hole = 0.3

               )



#Distribution of CV set Target variable

trace2 = go.Pie(labels = labels,

                values = CV_values,

                domain= {"x": [0.55, 1]},

                hole = 0.3

               )



layout = go.Layout(title = 'Distribution of Target Variable',

                    annotations = [{"text": "Train",

                                    "font": {"size": 20},

                                    "showarrow": False,

                                    "x": 0.19,

                                    "y": 0.5

                                   },

                                   {"text": "CV",

                                    "font": {"size": 20},

                                    "showarrow": False,

                                    "x": 0.8,

                                    "y": 0.5

                                   },

                                  ]

                   )



fig = go.Figure(data = [trace1, trace2], layout = layout)



iplot(fig)
# Set Random Forest Model parameters

rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)



#Fit the model to our training data

rf.fit(train.drop(["id", "target"], axis=1), train.target)

print('------training done-------')
# make a list of the data features

features = train.drop(["id", "target"],axis=1).columns.values



# Extract feature importances

importance_df = pd.DataFrame(

    data = {'features': features,

            'Importance': rf.feature_importances_

           }

).sort_values(by='Importance',

              ascending=True)



# Feature importance barplot

trace = go.Bar(

    x=importance_df.iloc[:, 0],

    y=importance_df.iloc[:, 1],

    marker=dict(color = importance_df.iloc[:, 0],

                colorscale = 'Viridis',

                reversescale = True

               ),

    name = 'Random Forest Feature Importance',

    orientation = 'h'

)



layout = go.Layout(title='Barplot of Feature Importances',

                   width = 900,

                   height = 2000,

                  )

fig = go.Figure(data=[trace], layout = layout)

iplot(fig)
#Predict probability of filing a claim (Target = 0 or 1)

rf_proba = rf.predict_proba(CV.drop(["id", "target"], axis=1))



#Isolate column of predicted probabilities that Target = 1

rf_proba_0 = rf_proba[:, 1].reshape(rf_proba.shape[0],1)
def gini(a, p):

    data = np.asarray(np.c_[a, p, np.arange(len(a))],

                      dtype=np.float

                     )

    data = data[

        np.lexsort((data[:,2], -1*data[:,1]))

    ]

    totalLosses = data[:,0].sum()

    giniSum = data[:,0].cumsum().sum() / totalLosses



    giniSum -= (len(a) + 1) / 2.

    return giniSum / len(a)



def gini_norm(a, p):

    return gini(a, p) / gini(a, a)



gini_coef = gini_norm(CV['target'], rf_proba_0)



print('Normalized Gini Coefficient: {0:0.3f}'.format(gini_coef))