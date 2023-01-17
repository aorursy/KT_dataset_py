import pandas as pd

import numpy as np

pd.set_option('max_columns',500)

pd.set_option('max_rows',500)
dataset = pd.read_excel("../input/covid19/dataset.xlsx",index_col = 0)
dataset.shape
dataset.columns
dataset.columns = [x.replace("-"," ").replace(" ","_") for x in dataset.columns]
dataset['SARS_Cov_2_exam_result'] = dataset['SARS_Cov_2_exam_result'].replace(['negative','positive'], [0,1])
dataset.columns.values
dataset = dataset.drop(columns = dataset.columns[2:5])
dataset.shape
dataset.index.duplicated().sum() # filtering by patient id
dataset['SARS_Cov_2_exam_result'].isnull().sum()
positive = dataset['SARS_Cov_2_exam_result'].value_counts(normalize = True)[1]

negative = dataset['SARS_Cov_2_exam_result'].value_counts(normalize = True)[0]
print("percentage of positive exams : {0:.3f}".format(positive * 100))

print("percentage of negative exams : {0:.3f}".format(negative * 100))
dataset.head()
dataset.describe()
from sklearn.preprocessing import LabelEncoder
columns_to_be_encoded = []

for column in dataset.columns:

    if dataset[column].dtype == 'O':

        columns_to_be_encoded.append(column)
columns_to_be_encoded
labeled = dataset[columns_to_be_encoded]



labeled = labeled.astype("str").apply(LabelEncoder().fit_transform).where(~labeled.isna(), labeled)
dataset = dataset.drop(columns = columns_to_be_encoded)
dataset = dataset.join(labeled)
dataset['Respiratory_Syncytial_Virus'].value_counts()
import missingno as msno #lib to visu missing data/
print('data nullity distribution')

bar = msno.bar(dataset,figsize=(15,8),inline = True,color = (0,0,0));
to_be_checked = []

for column in dataset.columns:

    if dataset[column].isnull().sum() == len(dataset):

        to_be_checked.append(column)
to_be_checked
dataset = dataset.drop(columns = to_be_checked)
dataset.shape
dataset.head()
to_be_checked = []

covid_posi = dataset.loc[dataset['SARS_Cov_2_exam_result'] == 1]

for column in covid_posi.columns:

    if covid_posi[column].isnull().sum() == len(covid_posi):

        to_be_checked.append(column)
to_be_checked
dataset = dataset.drop(columns = to_be_checked)
dataset.shape
dataset.shape
dataset.head()
to_be_checked = []

for column in dataset.columns:

    if dataset[column].isnull().sum()/len(dataset) >= 0.90:

        to_be_checked.append(column)
len(to_be_checked)
dataset = dataset.drop(columns = to_be_checked)
corr = dataset.corr().abs()

corr['SARS_Cov_2_exam_result']
# Selecionando o triângulo de cima da matriz de correlação

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))



# Procurando atributos com mais de 95% de correlação

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
to_drop
dataset = dataset.drop(columns = to_drop)
dataset.shape
age = dataset[['Patient_age_quantile','SARS_Cov_2_exam_result']]


import matplotlib.pyplot as plt

import numpy as np



plt.rc('axes',labelsize = 30)

plt.rc('xtick',labelsize = 20)

plt.rc('ytick',labelsize = 20)

plt.rc('legend', fontsize= 20)



labels = list(range(20))

posi = []

neg = []

for i in range(20):

    cases = age.loc[age['Patient_age_quantile'] == i]

    cases = cases.iloc[:,1].value_counts()

    neg.append(cases[0])

    posi.append(cases[1])



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize = (20,10))

rects1 = ax.bar(x - width/2, neg, width, label='Negative', color = 'blue')

rects2 = ax.bar(x + width/2, posi, width, label='Positive', color = 'r')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('cases',fontsize = 30)

ax.set_xlabel('Age quantile',fontsize = 30)

ax.set_title('Diagnosis by age',fontsize = 50)

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom',fontsize = 24)





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()


import plotly.graph_objects as go

from IPython.display import display

import numpy as np





total = [x + y for x,y in zip(posi,neg)]



infect_percent = [(x/y)*100 for x,y in zip(posi,total)]



title = 'Positive percentage per age'

analised = ['Television']

colors = ['rgb(67,67,67)']



mode_size = [12]

line_size = [4]



x_data = np.vstack((labels,)*4)



y_data = np.array([

    infect_percent

])



fig = go.Figure()



for i in range(0, 1):

    fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',

        name=analised[i],

        line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,

    ))



    # endpoints

    fig.add_trace(go.Scatter(

        x=[x_data[i][0], x_data[i][-1]],

        y=[y_data[i][0], y_data[i][-1]],

        mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

    ),

    showlegend=False,

    plot_bgcolor='white'

)



annotations = []



# Adding labels

# labeling the left_side of the plot

annotations.append(dict(xref='paper', x=0.05, y=y_data[0][0],

                              xanchor='right', yanchor='middle',

                              text='{:.4f}%'.format(y_data[0][0]),

                              font=dict(family='Arial',

                                        size=16),

                              showarrow=False))

# labeling the right_side of the plot

annotations.append(dict(xref='paper', x=0.95, y=y_data[0][19],

                              xanchor='left', yanchor='middle',

                              text='{:.1f}%'.format(y_data[0][19]),

                              font=dict(family='Arial',

                                        size=16),

                              showarrow=False))

# Title

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Positive percentage per quantile age',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

# Source

annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,

                              xanchor='center', yanchor='top',

                              text='Source: Kaggle einstein database',

                              font=dict(family='Arial',

                                        size=12,

                                        color='rgb(150,150,150)'),

                              showarrow=False))





fig.update_layout(annotations=annotations)



display(fig)
corr = dataset.corr()
posi_cor = corr['SARS_Cov_2_exam_result'].loc[corr['SARS_Cov_2_exam_result'] > 0.1].sort_values()
posi_cor
neg_cor = corr['SARS_Cov_2_exam_result'].loc[corr['SARS_Cov_2_exam_result'] < -0.1].sort_values()
neg_cor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
dataset.head()
columns = list(dataset.columns)

del columns[1:2]

columns
scaler = MinMaxScaler()

df = scaler.fit_transform(dataset[columns])
df = pd.DataFrame(df,columns = columns)
df = df.join(pd.DataFrame(dataset.iloc[:,1:2].values,columns = ['SARS_Cov_2_exam_result']))
df
df.shape
df.head()
columns = list(df.columns.values)

columns
y_columns = columns[len(columns)-1]

del columns[len(columns)-1]
X = df[columns]
from sklearn.impute import KNNImputer





imputer = KNNImputer(n_neighbors = 3,weights = 'distance')

X = pd.DataFrame(imputer.fit_transform(X),columns = columns)
y = df[y_columns]
X
y
from sklearn.feature_selection import RFE

from sklearn.svm import SVR

estimator = SVR(kernel = 'linear')

selector = RFE(estimator,10,step = 1,verbose = 1)
selector = selector.fit(X,y)
drops = selector.support_
selector.ranking_
X_selected = pd.DataFrame(X.values,columns = columns)
X_selected.head()
for column, ndrop in zip(X_selected.columns, drops):

    if ndrop == False:

        X_selected = X_selected.drop(columns = column)

        
X_selected.shape
X_selected.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_selected,y,random_state = 1)
X_train
import seaborn as sns

y_train.value_counts().plot(kind='bar', figsize=(10, 8), rot=0)

plt.xlabel("Exam result", labelpad=14)

plt.ylabel("Count of People", labelpad=14)

plt.title("Exam result distribution", y=1.02);
from imblearn.over_sampling import RandomOverSampler,SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline
ros = SMOTE()
X_train,y_train = ros.fit_resample(X_train,y_train)

#X_train,X_test,y_train,y_test = train_test_split(X_smoted,y_smoted)
X_train
y_train.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)

plt.xlabel("Exam result", labelpad=14)

plt.ylabel("Count of People", labelpad=14)

plt.title("Exam result distribution", y=1.02);
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = 700,max_depth=64,learning_rate=0.1,gamma=0,random_state= 0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix, roc_auc_score

# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)

# ROC AUC

auc = roc_auc_score(y_test, y_pred)

print('ROC AUC: %f' % auc)

# confusion matrix

matrix = confusion_matrix(y_test, y_pred)

print(matrix)



plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(matrix, cmap="Blues", annot=True,annot_kws={"size": 16},fmt = 'g')# font size