import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc

from itertools import cycle

from scipy import interp



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

data_df.head()
data_df.head()
data_df.describe()
temp = data_df["Class"].value_counts()

df = pd.DataFrame({'Class': temp.index,'values': temp.values})



trace = go.Bar(

    x = df['Class'],y = df['values'],

    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",

    marker=dict(color="Red"),

    text=df['values']

)



data = [trace]

layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',

          xaxis = dict(title = 'Class', showticklabels=True), 

          yaxis = dict(title = 'Number of transactions'),

          hovermode = 'closest',width=600

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='class')
data_df.head()
data_df[data_df['Class'] == 1].head(1)

data_df[data_df['Class'] == 0].head(1)
corr = data_df.corr(method='pearson').head()
def print_full(x):

    pd.set_option('display.max_rows', len(x))

    print(x)

    pd.reset_option('display.max_rows')
corr = data_df.corr().unstack().sort_values()

print_full(corr)
sns.heatmap(corr)
component_var = {}

for i in range(1,28):

    pca = PCA(n_components=i)

    res = pca.fit(data_df)

    component_var[i] = sum(pca.explained_variance_ratio_)

    

print(component_var)
plt.matshow(data_df.corr())

plt.show()
f = plt.figure(figsize=(19,15))

plt.matshow(data_df.corr(),fignum=f.number)

plt.xticks(range(data_df.shape[1]),data_df.columns,fontsize=14, rotation=45)

plt.yticks(range(data_df.shape[1]),data_df.columns,fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16)
corr = data_df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
pd.scatter_matrix(data_df, figsize(19,15))

plt.show()
plt.hist(data_df['Class'], color='blue', edgecolor='black', bins = int(185/5))

sns.distplot(data_df['Class'], hist=True, kde=False, bins=int(180/5), color='blue', hist_kws={'edgecolor':'black'})



plt.title('histo')

plt.xlabel('delay ')

plt.ylabel('flight')
print("Normal :", data_df['Class'][data_df['Class'] == 0].count())



print("Fraud :", data_df['Class'][data_df['Class'] == 1].count())
# separate classes into different dataset

classNormal = data_df.query('Class == 0')

classFraud = data_df.query('Class == 1')



#randomize the dataset

classNormal = classNormal.sample(frac=1)

classFraud = classFraud.sample(frac=1)
classNormaltrain = classNormal.iloc[0:6000]

classFraudtrain = classFraud



#combine become one

train = classNormaltrain.append(classFraudtrain, ignore_index=True).values

X = train[:,0:30].astype(float)

Y = train[:,30]
model = XGBClassifier()

kfold = StratifiedKFold(n_splits=10, random_state=7)



scoring = 'roc_auc'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
mean_tpr = 0.0

mean_fpr = np.linspace(0,1,100)



colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])

lw = 2



i = 0

for(train,test), color in zip(kfold.split(X,Y), colors) :

    probas_ = model.fit(X[train], Y[train]).predict_proba(X[test])

    

    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:,1])

    mean_tpr += interp(mean_fpr, fpr, tpr)

    mean_tpr[0] = 0.0

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i +=1

    

plt.plot([0,1],[0,1], linestyle='--', lw=lw, color='k', label='luck')



mean_tpr /= kfold.get_n_splits(X,Y)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label ='Mean ROC(area = %0.20f)' % mean_auc, lw=lw)



plt.xlim([-0.05, 1.05])

plt.ylim([-0.05,1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC example')

plt.legend(loc="lower right")

plt.show()