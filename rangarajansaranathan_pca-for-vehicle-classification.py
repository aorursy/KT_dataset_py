import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import pandas_profiling 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_color_codes()

sns.set(style="whitegrid")

%matplotlib inline

from scipy.stats import zscore

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



#setting up for customized printing

from IPython.display import Markdown, display

from IPython.display import HTML

def printmd(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))

    

#function to display dataframes side by side    

from IPython.display import display_html

def display_side_by_side(args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)
def distplot(figRows,figCols,xSize, ySize, features, colors):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    colors = np.array(colors).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

            plot = sns.distplot(vehicle[features[row][col]], color=colors[row][col], ax=axesplt, kde=True, hist_kws={"edgecolor":"k"})

            plot.set_xlabel(features[row][col],fontsize=20)
def scatterplot(rowFeature, colFeature, data):

    f, axes = plt.subplots(1, 1, figsize=(10, 8))

        

    plot=sns.scatterplot(x=rowFeature, y=colFeature, data=data, hue='class', style='class', ax=axes)

    plot.set_xlabel(rowFeature,fontsize=20)

    plot.set_ylabel(colFeature,fontsize=20)            
def boxplot(figRows,figCols,xSize, ySize, features, colors, hue=None, orient='h', rotation=30):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    colors = np.array(colors).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

            plot = sns.boxplot(features[row][col], data= vehicle, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            plot.set_ylabel('',fontsize=20)

            plot.set_xticklabels(rotation=rotation, labels=[features[row][col]], fontweight='demibold',fontsize='large')
def boxplot_all(xSize, ySize, palette, data):

    f, axes = plt.subplots(1, 1, figsize=(xSize, ySize))

    plot = sns.boxplot(x='variable',y='value', data= pd.melt(data), palette='Set1', ax=axes, orient='v')

    plot.set_xlabel('',fontsize=20)

    plot.set_xticklabels(rotation=60, labels=plot.get_xticklabels(),fontweight='demibold',fontsize='x-large')
def countplot(figRows,figCols,xSize, ySize, features, colors=None,palette=None,hue=None, orient=None, rotation=90):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    if(colors is not None):

        colors = np.array(colors).reshape(figRows, figCols)

    if(palette is not None):

        palette = np.array(palette).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

                

            if(colors is None):

                plot = sns.countplot(features[row][col], data=vehicle, palette=palette[row][col], ax=axesplt, orient=orient, hue=hue)

            elif(palette is None):

                plot = sns.countplot(features[row][col], data=vehicle, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            plot.set_title(features[row][col],fontsize=20)

            plot.set_xlabel(None)

            plot.set_xticklabels(rotation=rotation, labels=plot.get_xticklabels(),fontweight='demibold',fontsize='large')

            
vehicle = pd.read_csv('../input/vehicle/vehicle.csv')

vehicle.head() 
print('The total number of rows :', vehicle.shape[0])

print('The total number of columns :', vehicle.shape[1])
vehicle.info()
display(vehicle.isna().sum().sort_values())

print('===================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"Missing"** values in the data', color="red")
display(vehicle.describe().transpose())
pandas_profiling.ProfileReport(vehicle)
from sklearn.impute import SimpleImputer

sim = SimpleImputer(missing_values=np.nan, strategy='median')

cols = vehicle.columns[:18]

vehicle.iloc[:,:18] = sim.fit_transform(vehicle.iloc[:,:18])

vehicle.iloc[:,:18] = pd.DataFrame(vehicle.iloc[:,:18], columns=cols)
display(vehicle.isna().sum().sort_values())

print('===================')

printmd('**CONCLUSION**: **"Missing"** values are replaced with **Median** of the respective attributes', color="green")
pal = sns.color_palette(palette='Set1', n_colors=16)
distplot(4, 4, 20, 30, vehicle.columns[:16].tolist(), pal.as_hex())

distplot(1, 2, 8, 5, vehicle.columns[16:18].tolist(), pal.as_hex()[:2])
boxplot_all(20,30,palette=['Set1'], data =vehicle.iloc[:,:18])
boxplot(8, 1, 20, 30, orient='h', 

        features=['pr.axis_aspect_ratio',

                  'max.length_aspect_ratio',

                  'scaled_radius_of_gyration.1',

                  'radius_ratio',

                  'scaled_variance',

                  'scaled_variance.1',

                 'skewness_about',

                 'skewness_about.1'], 

        colors=pal.as_hex()[:8])
def point_box_bar_plot(row, col, figRow, figCol, palette='rocket', fontsize='large', fontweight='demibold'):

    sns.set(style="whitegrid")

    f, axes = plt.subplots(3, 1, figsize=(figRow, figCol))

    pplot=sns.pointplot(row,col, data=vehicle, ax=axes[0], linestyles=['--'])

    pplot.set_xlabel(None)

    pplot.set_xticklabels(labels=pplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)

    bxplot=sns.boxplot(row,col, data=vehicle, ax=axes[1],palette='Paired')

    bxplot.set_xlabel(None)

    bxplot.set_xticklabels(labels=bxplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)

    bplot=sns.barplot(row,col, data=vehicle, ax=axes[2],palette=palette)

    bplot.set_xlabel(row,fontsize=20)

    bplot.set_xticklabels(labels=bplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)
point_box_bar_plot('class','compactness', 15, 10, palette='summer')
point_box_bar_plot('class','max.length_aspect_ratio', 15, 10, palette='Accent_r')
point_box_bar_plot('class','max.length_rectangularity', 15, 10, palette='Blues_r')
point_box_bar_plot('class','hollows_ratio', 15, 10, palette='BrBG_r')
point_box_bar_plot('class','circularity', 15, 10, palette='BuPu_r')
point_box_bar_plot('class','distance_circularity', 15, 10, palette='CMRmap')
point_box_bar_plot('class','radius_ratio', 15, 10, palette='Dark2')
point_box_bar_plot('class','pr.axis_aspect_ratio', 15, 10, palette='GnBu_r')
point_box_bar_plot('class','scatter_ratio', 15, 10, palette='Greens_r')
point_box_bar_plot('class','elongatedness', 15, 10, palette='Greys_r')
point_box_bar_plot('class','pr.axis_rectangularity', 15, 10, palette='OrRd_r')
point_box_bar_plot('class','scaled_variance', 15, 10, palette='Paired_r')
point_box_bar_plot('class','scaled_variance.1', 15, 10, palette='PuBuGn_r')
point_box_bar_plot('class','scaled_radius_of_gyration', 15, 10, palette='PuRd_r')
point_box_bar_plot('class','scaled_radius_of_gyration.1', 15, 10, palette='Set1')
point_box_bar_plot('class','skewness_about', 15, 10, palette='Set2_r')
point_box_bar_plot('class','skewness_about.1', 15, 10, palette='Spectral')
point_box_bar_plot('class','skewness_about.2', 15, 10, palette='afmhot')
sns.pairplot(data=vehicle, hue='class', palette='Set1', diag_kind='kde')
f, axes = plt.subplots(1, 1, figsize=(15, 10))

cor_mat = vehicle.corr()

hplot = sns.heatmap(cor_mat, annot=True, fmt=".2f", ax=axes, linecolor='white', linewidths=0.2, cmap='YlGnBu')

plt.xticks(rotation=75)
scatterplot('elongatedness', 'distance_circularity', vehicle)
scatterplot('elongatedness', 'scatter_ratio', vehicle)
scatterplot('elongatedness', 'pr.axis_rectangularity', vehicle)
scatterplot('elongatedness', 'scaled_variance', vehicle)
scatterplot('elongatedness', 'scaled_variance.1', vehicle)
scatterplot('circularity', 'max.length_rectangularity', vehicle)
scatterplot('circularity', 'scaled_radius_of_gyration', vehicle)
scatterplot('distance_circularity', 'scatter_ratio', vehicle)
scatterplot('scatter_ratio', 'pr.axis_rectangularity', vehicle)
scatterplot('scatter_ratio', 'scaled_variance', vehicle)
scatterplot('scatter_ratio', 'scaled_variance.1', vehicle)
scatterplot('pr.axis_rectangularity', 'scaled_variance', vehicle)
scatterplot('pr.axis_rectangularity', 'scaled_variance.1', vehicle)
scatterplot('scaled_variance', 'scaled_variance.1', vehicle)
vehicle=vehicle.replace({'skewness_about': {0: vehicle['skewness_about'].median()}}) 

vehicle=vehicle.replace({'skewness_about.1': {0: vehicle['skewness_about.1'].median()}}) 
vehicle_new = vehicle.copy()
def CheckCorrelationUpper(data):

    # Create correlation matrix

    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    f, axes = plt.subplots(1, 1, figsize=(15, 10))

    sns.heatmap(upper,cmap='YlGnBu', annot=True, fmt=".2f", ax=axes, linecolor='white', linewidths=0.2)

    plt.xticks(rotation=75)

    return upper
upper = CheckCorrelationUpper(vehicle)
# Find index of feature columns with correlation greater than 0.90

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

printmd('List of columns with correlation higher than 0.90', color='brown')

display(to_drop)
vehicle.drop(['scatter_ratio', 'elongatedness', 'pr.axis_rectangularity', 'scaled_variance.1', 'circularity'], axis=1, inplace=True)
upper = CheckCorrelationUpper(vehicle)
def remove_outliers(col, data):

    outlier_col = col + "_outliers"

    data[outlier_col] = data[col]

    data[outlier_col]= zscore(data[outlier_col])



    condition = (data[outlier_col]>3) | (data[outlier_col]<-3)

    print(data[condition].shape)

    indices = data[condition].index

    data.drop(data[condition].index, axis = 0, inplace = True)

    data.drop(outlier_col, axis=1, inplace=True)

    return indices
vehicle.skew().sort_values()
max_length_aspect_ratio_index = remove_outliers('max.length_aspect_ratio', vehicle)

remove_outliers('pr.axis_aspect_ratio', vehicle)

remove_outliers('scaled_radius_of_gyration.1', vehicle)
vehicle.shape
boxplot(3, 1, 20, 10, orient='h', 

        features=['pr.axis_aspect_ratio',

                  'max.length_aspect_ratio',                  

                  'scaled_radius_of_gyration.1'], 

        colors=pal.as_hex()[:3])
vehicle.skew().sort_values()
from sklearn.preprocessing import LabelEncoder   # import label encoder



def lencode(col, data):

    labelencoder = LabelEncoder()

    data[col] = labelencoder.fit_transform(data[col]) # returns label encoded variable(s)

    return data
vehicle = lencode('class', vehicle)
vehicle['class'].value_counts()
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler()

vehicle.iloc[:,:13] = std_scale.fit_transform(vehicle.iloc[:,:13])
vehicle.head()
X = vehicle.loc[:, vehicle.columns != 'class']

y = vehicle['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.30, random_state=10)



printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns')

print(f'Testing set has {X_test.shape[0]} rows and {X_test.shape[1]} columns')

from sklearn.model_selection import GridSearchCV



def find_best_model(model, parameters):

    clf = GridSearchCV(model, parameters, scoring='accuracy')

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
svm = SVC()

parameters = {'kernel':('rbf', 'poly', 'sigmoid'), 'gamma':('scale','auto'), 'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 'class_weight': (None, 'balanced')}

clf = find_best_model(svm, parameters)
svm = clf.best_estimator_

svm.fit(X_train, y_train)  

y_train_pred = svm.predict(X_train)

y_test_pred = svm.predict(X_test)



#get Precision Score on train and test

accuracy_train = round(metrics.accuracy_score(y_train, y_train_pred),3)

accuracy_test = round(metrics.accuracy_score(y_test, y_test_pred),3)

accdf = pd.DataFrame([[accuracy_train],[accuracy_test]], index=['Training', 'Testing'], columns=['Accuracy'])  

accdf
kfold = KFold(n_splits=50, random_state=10)

results = cross_val_score(svm, X, y, cv=kfold)

display(results)

kfold_result = pd.DataFrame([[results.mean()*100.0, results.std()*100.0]], index=['KFold'], columns=['Mean Accuracy', 'Standard Deviation'])

kfold_result
confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])

confusion_matrix_test
vehicle_new.drop(max_length_aspect_ratio_index, axis = 0, inplace = True)

vehicle_new = lencode('class', vehicle_new)

std_scale = StandardScaler()

vehicle_new.iloc[:,:18] = std_scale.fit_transform(vehicle_new.iloc[:,:18])

X = vehicle_new.loc[:, vehicle_new.columns != 'class']

y = vehicle_new['class']
from sklearn.decomposition import PCA

pca_full = PCA(n_components=18)

pca_full.fit(X)
pca_full.explained_variance_ratio_
f, axes = plt.subplots(1, 1, figsize=(20, 5))

sns.barplot(list(range(1,19)), pca_full.explained_variance_ratio_,ax=axes)

plot = sns.pointplot(list(range(1,19)), pca_full.explained_variance_ratio_,ax=axes, color='darkgreen')

plot.set_xlabel('Attributes')

plot.set_ylabel('Explained Variance')
pca8 = PCA(n_components=8)

X_PCA8 = pca8.fit_transform(X)
p_plot_data = pd.DataFrame(X_PCA8)

p_plot_data['class'] = y
sns.pairplot(p_plot_data, diag_kind='kde', hue='class', palette='gist_heat_r')
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_PCA8, y, test_size =.30, random_state=10)



printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {X_pca_train.shape[0]} rows and {X_pca_train.shape[1]} columns')

print(f'Testing set has {X_pca_test.shape[0]} rows and {X_pca_test.shape[1]} columns')

svm.fit(X_pca_train, y_train)  

y_train_pred = svm.predict(X_pca_train)

y_test_pred = svm.predict(X_pca_test)



#get Precision Score on train and test

accuracy_train = round(metrics.accuracy_score(y_train, y_train_pred),3)

accuracy_test = round(metrics.accuracy_score(y_test, y_test_pred),3)

accdf_pca = pd.DataFrame([[accuracy_train],[accuracy_test]], index=['Training', 'Testing'], columns=['Accuracy'])  

accdf_pca
results = cross_val_score(svm, X_PCA8, y, cv=kfold)

print(results)

kfold_result_pca = pd.DataFrame([[results.mean()*100.0, results.std()*100.0]], index=['KFold'], columns=['Mean Accuracy', 'Standard Deviation'])

kfold_result_pca
confusion_matrix_test_pca = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])

confusion_matrix_test_pca
printmd('**SVM Standard**', color='brown')

display_side_by_side([accdf, kfold_result,confusion_matrix_test])
printmd('**SVM with PCA**', color='brown')

display_side_by_side([accdf_pca, kfold_result_pca,confusion_matrix_test_pca])