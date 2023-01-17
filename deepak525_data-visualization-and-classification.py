import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.model_selection import train_test_split,ShuffleSplit

import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from itertools import chain
from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.offline as py

from plotly.graph_objs import Scatter, Layout

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff
# Read the CSV File Using Pandas read_csv function

df = pd.read_csv('../input/data.csv')

df.T.head(50)
# print the concise summery of the dataset

df.info()
#since the dataset can also contain null values

#count total rows in each column which contain null values

df.isna().sum()
#'duplicated()' function in pandas return the duplicate row as True and othter as False

#for counting the duplicate elements we sum all the rows

sum(df.duplicated())
#deleting useless columns

#deleting the "id" column

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)
p = df.describe().T

p = p.round(4)

table = go.Table(

    columnwidth=[0.8]+[0.5]*8,

    header=dict(

        values=['Attribute'] + list(p.columns),

        line = dict(color='#506784'),

        fill = dict(color='lightblue'),

    ),

    cells=dict(

        values=[p.index] + [p[k].tolist() for k in p.columns[:]],

        line = dict(color='#506784'),

        fill = dict(color=['rgb(173, 216, 220)', '#f5f5fa'])

    )

)

py.iplot([table], filename='table-of-mining-data')
B, M = df['diagnosis'].value_counts()

s = [B,M]

print(df['diagnosis'].value_counts())

with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))



    plt.bar([0,1], s,align='center',

            label='Count')

    plt.ylabel('Count')

    plt.xlabel('Target')

    plt.legend(loc='best')

    plt.tight_layout()
data_dia = df['diagnosis']

data = df.drop('diagnosis',axis=1)

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([df['diagnosis'],data_n_2.iloc[:,0:15]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(14,5))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=45,fontsize=13)
mean_col = [col for col in df.columns if col.endswith('_mean')]

for i in range(len(mean_col)):

    sns.FacetGrid(df,hue="diagnosis",aspect=3,margin_titles=True).map(sns.kdeplot,mean_col[i],shade= True).add_legend()

    #ax.set_title('lalala')
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

#cmap = sns.diverging_palette( 240 , 10 , as_cmap = True )

sns.heatmap(df.corr(), cmap='Blues',annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.xticks(fontsize=11,rotation=70)

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 8,5

cols = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean','diagnosis']

sns_plot = sns.pairplot(data=df[cols],hue='diagnosis')
palette ={'B' : 'lightblue', 'M' : 'gold'}

edgecolor = 'grey'



# Plot +

fig = plt.figure(figsize=(12,12))

sns.set_style('whitegrid')

sns.color_palette("bright")

def plot_scatter(a,b,k):

    plt.subplot(k)

    sns.scatterplot(x = df[a], y = df[b], hue = "diagnosis",

                    data = df, palette = palette, edgecolor=edgecolor)

    plt.title(a + ' vs ' + b,fontsize=15)

    k+=1



    

plot_scatter('perimeter_mean','radius_worst',221)   

plot_scatter('area_mean','radius_worst',222)   

plot_scatter('texture_mean','texture_worst',223)   

plot_scatter('area_worst','radius_worst',224)  
fig = plt.figure(figsize=(12,12))

plot_scatter('smoothness_mean','texture_mean',221)

plot_scatter('radius_mean','fractal_dimension_worst',222)

plot_scatter('texture_mean','symmetry_mean',223)

plot_scatter('texture_mean','symmetry_se',224)
fig = plt.figure(figsize=(12,12))

plot_scatter('area_mean','fractal_dimension_mean',221)

plot_scatter('radius_mean','fractal_dimension_mean',222)

plot_scatter('area_mean','smoothness_se',223)

plot_scatter('smoothness_se','perimeter_mean',224)
plt.style.use('ggplot')

sns.set_style('whitegrid')

plt.figure(figsize=(16,6))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data,palette='Set1')

plt.xticks(rotation=40)
from collections import Counter



def detect_outliers(train_data,n,features):

    outlier_indices = []

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(train_data[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(train_data[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = train_data[(train_data[col] < Q1 - outlier_step) | (train_data[col] > Q3 + outlier_step )].index

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers

list_atributes = df.drop('diagnosis',axis=1).columns

Outliers_to_drop = detect_outliers(df,2,list_atributes)
df.loc[Outliers_to_drop]
# Drop outliers

df = df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
df.shape
group_map = {"M": 1, "B": 0}



df['diagnosis'] = df['diagnosis'].map(group_map)
target_pca = pd.DataFrame(df['diagnosis'])

data_pca = df.drop('diagnosis', axis=1)



#To make a PCA, normalize data is essential

X_pca = data_pca.values

X_std = StandardScaler().fit_transform(X_pca)



pca = PCA(svd_solver='full')

pca_std = pca.fit(X_std, target_pca).transform(X_std)



pca_std = pd.DataFrame(pca_std)

pca_std = pca_std.merge(target_pca, left_index = True, right_index = True, how = 'left')
sns.set_style('whitegrid')

plt.figure(figsize=(12,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_),marker='o')

plt.xlim(0,30,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.title('Cumulative Sum of Variance Vs Number Of Components')

plt.show()
var_pca = pd.DataFrame(pca.explained_variance_ratio_)

labels = []

for i in range(1,31):

    labels.append('Col_'+str(i))

trace = go.Pie(labels = labels, values = var_pca[0].values, opacity = 0.8,

               textfont=dict(size=15))

layout = dict(title =  'PCA : components and explained variance')

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
pca.explained_variance_ratio_
X = df.drop('diagnosis',axis=1).values

y = df['diagnosis'].values



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



#X = pd.DataFrame(preprocessing.scale(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn import metrics

def plot_confusion_metrix(y_test,model_test):

    cm = metrics.confusion_matrix(y_test, model_test)

    plt.figure(1)

    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

    classNames = ['Benign','Malignant']

    plt.title('Confusion Matrix')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames)

    plt.yticks(tick_marks, classNames)

    s = [['TN','FP'], ['FN', 'TP']]

    for i in range(2):

        for j in range(2):

            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

    plt.show()

    

from sklearn.metrics import roc_curve, auc

def report_performance(model):



    model_test = model.predict(X_test)



    print("\n\nConfusion Matrix:")

    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))

    print("\n\nClassification Report: ")

    print(metrics.classification_report(y_test, model_test))

    #cm = metrics.confusion_matrix(y_test, model_test)

    plot_confusion_metrix(y_test, model_test)



def roc_curves(model):

    predictions_test = model.predict(X_test)

    fpr, tpr, _ = roc_curve(predictions_test,y_test)

    roc_auc = auc(fpr, tpr)



    plt.figure()

    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.show()

    

def accuracy(model):

    pred = model.predict(X_test)

    accu = metrics.accuracy_score(y_test,pred)

    print("\nAcuuracy Of the Model: ",accu,"\n\n")
for i in ['linear','rbf']:

    clf = SVC(kernel=i)

    clf.fit(X_train,y_train)

    print("On "+ i + " kernel:" )

    report_performance(clf)

    roc_curves(clf)

    accuracy(clf)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None, obj_line=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    from sklearn.model_selection import learning_curve

    from matplotlib import pyplot as plt

    

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std  = np.std(train_scores, axis=1)

    test_scores_mean  = np.mean(test_scores, axis=1)

    test_scores_std   = np.std(test_scores, axis=1)

    plt.grid(True)



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Testing score")

    plt.grid(True)

    if obj_line:

        plt.axhline(y=obj_line, color='blue')



    plt.legend(loc="best")

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = SVC(kernel='linear')

plot_learning_curve(estimator, 'Kernel = Linear', X, y, cv=cv)

estimator = SVC(kernel='rbf')

plot_learning_curve(estimator, 'kernel = RBF', X, y, cv=cv)
fig = plt.figure(figsize=(16,5))

def plotlc(kernel=None,k=0):

    plt.subplot(k)

    cp = np.arange(1, 11)

    train_accuracy = np.empty(len(cp))

    test_accuracy = np.empty(len(cp))

    for i, c in enumerate(cp):

        clf = SVC(C=c,kernel = kernel)

        clf.fit(X_train, y_train)

        train_accuracy[i] = clf.score(X_train, y_train)

        test_accuracy[i] = clf.score(X_test, y_test)



        #plt.figure(figsize=(10,5))

    plt.title('Learning curves for SVM( '+ kernel+' ): Varying Value of C', size=15)

    plt.plot(cp, test_accuracy, marker ='o', label = 'Testing Accuracy')

    plt.plot(cp, train_accuracy, marker ='o', label = 'Training Accuracy')

    plt.legend(prop={'size':13})

    plt.xlabel('Value of C (c)', size=13)

    plt.ylabel('Accuracy', size=13)

    plt.xticks(cp);

#plt.show()



plotlc('linear',121)

plotlc('rbf',122)
clf = SVC(kernel='rbf',C=1)

clf.fit(X_train,y_train)

report_performance(clf)

roc_curves(clf)

accuracy(clf)