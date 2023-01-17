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

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LassoCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor



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
def distplot(figRows,figCols,xSize, ySize, data, features, colors):

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

            plot = sns.distplot(data[features[row][col]], color=colors[row][col], ax=axesplt, kde=True, hist_kws={"edgecolor":"k"})

            plot.set_xlabel(features[row][col],fontsize=20)
def boxplot(figRows,figCols,xSize, ySize, data, features, colors, hue=None, orient='h', rotation=30):

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

                

            plot = sns.boxplot(features[row][col], data= data, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            plot.set_xlabel(features[row][col],fontsize=20)
def boxplot2(figRows,figCols,xSize, ySize, data, features, palette=None, hue=None, orient='v', rotation=30):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)    

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

                

            plot = sns.boxplot(x=hue, y=features[row][col], data= data, palette=palette, ax=axesplt, orient=orient, hue=hue)

            plot.set_xlabel(hue,fontsize=20)

            plot.set_ylabel(features[row][col],fontsize=20)
def scatterplot(rowFeature, colFeature, data, hue=None, style=None, size=None, sizes=(10,200), palette='Set1',legend='brief', alpha=0.7):

    f, axes = plt.subplots(1, 1, figsize=(15, 8))

        

    plot=sns.scatterplot(x=rowFeature, y=colFeature, data=data, hue=hue, style=style, size=size, sizes=sizes, palette=palette, ax=axes, legend=legend, alpha=alpha)

    plot.set_xlabel(rowFeature,fontsize=20)

    plot.set_ylabel(colFeature,fontsize=20)            
def boxplot_all(xSize, ySize, palette, data):

    f, axes = plt.subplots(1, 1, figsize=(xSize, ySize))

    plot = sns.boxplot(x='variable',y='value', data= pd.melt(data), palette='Set1', ax=axes, orient='v')

    plot.set_xlabel('',fontsize=20)

    plot.set_xticklabels(rotation=60, labels=plot.get_xticklabels(),fontweight='demibold',fontsize='x-large')
def point_box_bar_plot(row, col, figRow, figCol, data, palette='rocket', fontsize='large', fontweight='demibold'):

    sns.set(style="whitegrid")

    f, axes = plt.subplots(3, 1, figsize=(figRow, figCol))

    pplot=sns.pointplot(row,col, data=data, ax=axes[0], linestyles=['--'])

    pplot.set_xlabel(None)

    pplot.set_xticklabels(labels=pplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)

    bxplot=sns.boxplot(row,col, data=data, ax=axes[1],palette='Paired')

    bxplot.set_xlabel(None)

    bxplot.set_xticklabels(labels=bxplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)

    bplot=sns.barplot(row,col, data=data, ax=axes[2],palette=palette)

    bplot.set_xlabel(row,fontsize=20)

    bplot.set_xticklabels(labels=bplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)
def kdeplot(rowFeature, colFeature, data, shade, palette='afmhot', cut=3, n_levels=10):

    f, axes = plt.subplots(1, 1, figsize=(10, 6))

    kplot = sns.kdeplot(data[rowFeature],data[colFeaturef], ax=axes, shade=True, cmap=palette,cbar=True, cut=cut, n_levels=n_levels)

    kplot.set_xlabel(rowFeature,fontsize=20)

    kplot.set_ylabel(colFeature,fontsize=20)            
def scatter_kde_plot(rowFeature, colFeature, data, hue=None, style=None, size=None, sizes=(10,200), palette='Set1',legend='brief', alpha=0.7, cut=3, n_levels=10):

    f, axes = plt.subplots(1, 2, figsize=(20, 8))

    plot=sns.scatterplot(x=rowFeature, y=colFeature, data=data, hue=hue, style=style, size=size, sizes=sizes, palette=palette, ax=axes[0], legend=legend, alpha=alpha)

    plot.set_xlabel(rowFeature,fontsize=20)

    plot.set_ylabel(colFeature,fontsize=20)

    

    kplot = sns.kdeplot(data[rowFeature],data[colFeature], ax=axes[1], shade=True, cmap='afmhot',cbar=True, cut=cut, n_levels=n_levels)

    kplot.set_xlabel(rowFeature,fontsize=20)

    kplot.set_ylabel(colFeature,fontsize=20)
def regplot(rowFeature, colFeature, data, order=1, color='blue'):

    f, axes = plt.subplots(1, 1, figsize=(15, 8))

    plot=sns.regplot(x=rowFeature, y=colFeature, data=data, order=order, ax=axes, color=color)

    plot.set_xlabel(rowFeature,fontsize=20)

    plot.set_ylabel(colFeature,fontsize=20)
def outlier_treatment(datacolumn):

    sorted(datacolumn)

    Q1,Q3 = np.percentile(datacolumn , [25,75])

    IQR = Q3 - Q1

    lower_range = Q1 - (1.5 * IQR)

    upper_range = Q3 + (1.5 * IQR)

    return lower_range,upper_range
def remove_outliers(col, data, method='zscore'):

    

    if(method == 'zscore'):

        outlier_col = col + "_outliers"

        data[outlier_col] = data[col]

        data[outlier_col]= zscore(data[outlier_col])

        condition = (data[outlier_col]>3) | (data[outlier_col]<-3)

        print(data[condition].shape)

        indices = data[condition].index

        data.drop(data[condition].index, axis = 0, inplace = True)

        data.drop(outlier_col, axis=1, inplace=True)

        return indices

    elif(method == 'capping'):

        q1,q3 = outlier_treatment(data[col])

        print('Count below minimum whisker for ', col,data[data[col] < q1][col].count())

        data[col].values[data[col].values < q1] = q1

        print('Count above maximum whisker for ', col,data[data[col] > q3][col].count())

        data[col].values[data[col].values > q3] = q3

        return data
def CheckCorrelationUpper(data):

    # Create correlation matrix

    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    f, axes = plt.subplots(1, 1, figsize=(15, 10))

    sns.heatmap(upper,cmap='YlGnBu', annot=True, fmt=".2f", ax=axes, linecolor='white', linewidths=0.2)

    plt.xticks(rotation=75)

    return upper
from sklearn.model_selection import GridSearchCV



def find_best_model(model, parameters, X_train, y_train):

    clf = GridSearchCV(model, parameters, scoring='accuracy')

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
concrete = pd.read_csv("../input/concrete.csv")

concrete.head() 
print('The total number of rows :', concrete.shape[0])

print('The total number of columns :', concrete.shape[1])
concrete.info()
display(concrete.isna().sum().sort_values())

print('===================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"No Missing"** values in the data', color="blue")
display(concrete.describe().transpose())
pandas_profiling.ProfileReport(concrete)
concrete[concrete.duplicated(subset=None)]
concrete.drop_duplicates(subset=None, keep='first', inplace=True)
concrete.shape
pal = sns.color_palette(palette='Set1', n_colors=12)
distplot(4, 2, 20, 30, data=concrete, features=concrete.columns[:8].tolist(), colors=pal.as_hex()[:8])

distplot(1, 1, 8, 5, data=concrete, features=['strength'], colors=pal.as_hex()[0])
boxplot_all(20,15,palette=['Set1'], data=concrete.iloc[:,:9])
boxplot(6, 1, 20, 30, orient='h', data=concrete,features=['slag','water','superplastic','fineagg','age','strength'], colors=pal.as_hex()[:6])
scatter_kde_plot('cement', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('cement', 'strength',data=concrete)
scatter_kde_plot('slag', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('slag', 'strength',data=concrete, order=2)
scatter_kde_plot('ash', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('ash', 'strength',data=concrete, order=2)
scatter_kde_plot('water', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('water', 'strength',data=concrete, order=3)
scatter_kde_plot('superplastic', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('superplastic', 'strength',data=concrete, order=2)
scatter_kde_plot('coarseagg', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('coarseagg', 'strength',data=concrete, order=3)
scatter_kde_plot('fineagg', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('fineagg', 'strength',data=concrete, order=3)
scatter_kde_plot('age', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('age', 'strength',data=concrete, order=2)
sns.pairplot(data=concrete, palette='muted', diag_kind='kde')
f, axes = plt.subplots(1, 1, figsize=(15, 10))

cor_mat = concrete.corr()

hplot = sns.heatmap(cor_mat, annot=True, fmt=".2f", ax=axes, linecolor='white', linewidths=0.2, cmap='YlGnBu')

plt.xticks(rotation=60)

plt.yticks(rotation=0)
scatter_kde_plot('ash', 'superplastic',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('ash', 'superplastic',data=concrete, order=2)
scatter_kde_plot('superplastic', 'water',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('water', 'superplastic',data=concrete, order=2)
scatter_kde_plot('water', 'fineagg',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('water', 'fineagg',data=concrete, order=2)
concrete[concrete == 0].count(axis=0)
concrete_zeros = concrete[['slag', 'ash', 'superplastic']]
printmd('====== **Median with Zeroes** ========== **Median without Zeroes**======', color='brown')

concrete_median = pd.DataFrame(concrete_zeros.median(), columns=['Median'], index=concrete_zeros.columns)

concrete_nozero_median = pd.DataFrame(concrete_zeros[concrete_zeros != 0].median(), columns=['Median'], index=concrete_zeros.columns)

display_side_by_side([concrete_median, concrete_nozero_median])
printmd('======= **Mean with Zeroes** ============= **Mean without Zeroes**=======', color='brown')

concrete_mean = pd.DataFrame(concrete_zeros.mean(), columns=['Mean'], index=concrete_zeros.columns)

concrete_nozero_mean = pd.DataFrame(concrete_zeros[concrete_zeros != 0].mean(), columns=['Mean'], index=concrete_zeros.columns)

display_side_by_side([concrete_mean, concrete_nozero_mean])
concrete.replace(0, np.nan, inplace=True)

concrete['slag'].fillna(concrete_zeros[concrete_zeros != 0]['slag'].mean(), inplace=True)

concrete['ash'].fillna(concrete_zeros[concrete_zeros != 0]['ash'].mean(), inplace=True)

concrete['superplastic'].fillna(concrete_zeros[concrete_zeros != 0]['superplastic'].mean(), inplace=True)
concrete.describe().transpose()
distplot(1, 2, 20, 7, data=concrete, features=['slag', 'superplastic'], colors=pal.as_hex()[:2])
concrete.skew().sort_values()
concrete[['age', 'superplastic']].describe().transpose()
concrete = remove_outliers('age', concrete, 'capping')

concrete = remove_outliers('superplastic', concrete, 'capping')
concrete[['age', 'superplastic']].describe().transpose()
concrete.skew().sort_values()
concrete['water_to_cement_ratio'] = concrete['water']/concrete['cement']
distplot(1, 1, 8, 5, data=concrete, features=['water_to_cement_ratio'], colors=pal.as_hex()[0])
scatter_kde_plot('water_to_cement_ratio', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('water_to_cement_ratio', 'strength',data=concrete, order=2)
concrete[['strength', 'water_to_cement_ratio']].corr()
concrete['coarseagg_to_fineagg_ratio'] = concrete['coarseagg']/concrete['fineagg']
distplot(1, 1, 8, 5, data=concrete, features=['coarseagg_to_fineagg_ratio'], colors=pal.as_hex()[0])
scatter_kde_plot('coarseagg_to_fineagg_ratio', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('coarseagg_to_fineagg_ratio', 'strength',data=concrete, order=3)
concrete['aggregate_to_cement_ratio'] = concrete['fineagg']/concrete['cement']
distplot(1, 1, 8, 5, data=concrete, features=['aggregate_to_cement_ratio'], colors=pal.as_hex()[0])
scatter_kde_plot('aggregate_to_cement_ratio', 'strength',data=concrete, alpha=1, size='strength', hue='strength', palette='tab10_r', cut=3, n_levels=5)
regplot('aggregate_to_cement_ratio', 'strength',data=concrete, order=3)
concrete[['strength', 'aggregate_to_cement_ratio']].corr()
X = concrete.loc[:, concrete.columns != 'strength']

y = concrete['strength']
std_scale = StandardScaler()

cols = X.columns

X_scaled = std_scale.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=cols)

X_scaled.head()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size =.30, random_state=10)

printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns')

print(f'Testing set has {X_test.shape[0]} rows and {X_test.shape[1]} columns')

#Finding optimal no. of clusters

from scipy.spatial.distance import cdist



def find_optimal_clusters(model):

    clusters=range(1,10)

    meanDistortions=[]



    for k in clusters:

        model=KMeans(n_clusters=k)

        model.fit(X_scaled)

        prediction=model.predict(X_scaled)

        meanDistortions.append(sum(np.min(cdist(X_scaled, model.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])





    plt.plot(clusters, meanDistortions, 'bx-')

    plt.xlabel('k')

    plt.ylabel('Average distortion')

    plt.title('Selecting k with the Elbow Method')
from sklearn.cluster import KMeans

kmeans = KMeans(n_jobs=-1, random_state=10)

find_optimal_clusters(kmeans)   
kmeans = KMeans(n_jobs=-1, n_clusters=3, random_state=10)

kmeans.fit(X_scaled, y)  

y_pred = kmeans.predict(X_scaled)

concrete['group'] = y_pred
boxplot2(4,3, 20, 25, data=concrete, features=concrete.columns[concrete.columns != 'group'].tolist(),hue='group', orient='v', palette='Set1')
from sklearn.model_selection import GridSearchCV



def find_best_model_gridsearch(model, parameters, X_train, y_train):

    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform



def find_best_model_randomsearch(model, parameters, X_train, y_train):

    clf = RandomizedSearchCV(model, parameters, n_jobs=-1, n_iter=50, random_state=10)

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
def show_accuracy(y_train, y_train_pred, y_test, y_test_pred):

    #get Precision Score on train and test

    accuracy_train = round(metrics.r2_score(y_train, y_train_pred),3)

    accuracy_test = round(metrics.r2_score(y_test, y_test_pred),3)

    

    accdf = pd.DataFrame([[accuracy_train],[accuracy_test]], index=['Training', 'Testing'], columns=['Accuracy'])  

    display(accdf)

    return accdf
def model_show_feature_importance(model, X_train, feature_importance=False):

    f, axes = plt.subplots(1, 1, figsize=(20, 10))

    

    if (not feature_importance):

        coef = pd.DataFrame(model.coef_.ravel())

    elif (feature_importance):

        coef = pd.DataFrame(model.feature_importances_)

    

    coef["feat"] = X_train.columns

    bplot = sns.barplot(coef["feat"],coef[0],palette="Set1",linewidth=2,edgecolor="k", ax=axes)    

    bplot.set_facecolor("white")

    bplot.axhline(0,color="k",linewidth=2)

    bplot.set_ylabel("coefficients/weights", fontdict=dict(fontsize=20))

    bplot.set_xlabel("features", fontdict=dict(fontsize=20))

    bplot.set_title('FEATURE IMPORTANCES')

    bplot.set_xticklabels(rotation=60, labels=bplot.get_xticklabels(),fontweight='demibold',fontsize='x-large')

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



def get_important_features(model, X_train, y_train, k_features):

    sfs1 = sfs(model, k_features=k_features, forward=True, scoring='r2', cv=5)

    sfs1 = sfs1.fit(X_train.values, y_train.values)

    #display(sfs1.get_metric_dict())

    fig = plot_sfs(sfs1.get_metric_dict())



    plt.title('Sequential Forward Selection (w. R^2)')

    plt.grid()

    plt.show()



    columnList = list(X_train.columns)

    feat_cols_idx = list(sfs1.k_feature_idx_)

    subsetColumnList = [columnList[i] for i in feat_cols_idx] 

    display(pd.DataFrame(subsetColumnList, columns=['Selected Columns'], index=feat_cols_idx))

    return subsetColumnList
def model_fit_predict_score(model, X_train, y_train, X_test, y_test):    

    model.fit(X_train, y_train)    

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    model_result = show_accuracy(y_train, y_train_pred, y_test, y_test_pred)

    return model_result
def model_fit_predict_score_only_important_features(model, X_train, y_train, X_test, y_test, k_features = 'best'):    

    subsetColumnList = get_important_features(model, X_train, y_train, k_features=k_features)    

    model.fit(X_train[subsetColumnList], y_train)

    

    y_train_pred = model.predict(X_train[subsetColumnList])

    y_test_pred = model.predict(X_test[subsetColumnList])

    model_result = show_accuracy(y_train, y_train_pred, y_test, y_test_pred)

    return model_result
def crossvalidation(model, X_train, y_train):

    kfold = KFold(n_splits=20, random_state=10)

    results = cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=-1)

    display(results)

    kfold_result = pd.DataFrame([[results.mean()*100.0, results.std()*100.0]], index=['KFold'], columns=['Mean Accuracy', 'Standard Deviation'])

    display(kfold_result)

    return results
def model_confidence_interval(stats, alpha = 0.95):

    sns.distplot(stats, color='blue')

    

    # confidence intervals    

    p = ((1.0-alpha)/2.0) * 100

    lower = max(0.0, np.percentile(stats, p))

    p = (alpha+((1.0-alpha)/2.0)) * 100

    upper = min(1.0, np.percentile(stats, p))

    printmd('**Model performance range at %.0f%% confidence interval is between %.1f%% and %.1f%%**' % (alpha*100, lower*100, upper*100), color='brown')
lr = LinearRegression(n_jobs=-1)
lr_result = model_fit_predict_score(lr, X_train, y_train, X_test, y_test)
model_show_feature_importance(lr, X_train)
lr_result_imp_feat = model_fit_predict_score_only_important_features(lr, X_train, y_train, X_test, y_test, k_features=6)
from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(2)

X_poly2 = poly2.fit_transform(X_scaled)
X_train_poly2, X_test_poly2, y_train_poly2, y_test_poly2 = train_test_split(X_poly2, y, test_size =.30, random_state=10)
lr_poly2_result = model_fit_predict_score(lr, X_train_poly2, y_train_poly2, X_test_poly2, y_test_poly2)
stats = crossvalidation(lr, X_train, y_train)

model_confidence_interval(stats, alpha=0.95)
pca_full = PCA(n_components=78)

pca_full.fit(X_poly2)
f, axes = plt.subplots(1, 1, figsize=(20, 20))

sns.barplot(list(range(1,79)), pca_full.explained_variance_ratio_,ax=axes)

plot = sns.pointplot(list(range(1,79)), pca_full.explained_variance_ratio_,ax=axes, color='darkgreen')

plot.set_xlabel('Attributes')

plot.set_ylabel('Explained Variance')
pca = PCA(n_components=50)

X_poly_PCA = pca.fit_transform(X_poly2)
X_train_poly_PCA, X_test_poly_PCA, y_train_poly_PCA, y_test_poly_PCA = train_test_split(X_poly_PCA, y, test_size =.30, random_state=10)
lr_poly2_result = model_fit_predict_score(lr, X_train_poly_PCA, y_train_poly_PCA, X_test_poly_PCA, y_test_poly_PCA)
lasso = LassoCV(n_jobs=-1,random_state=10)

lasso_result = model_fit_predict_score(lasso, X_train, y_train, X_test, y_test)
model_show_feature_importance(lasso, X_train)
lasso_poly2_result = model_fit_predict_score(lasso, X_train_poly2, y_train_poly2, X_test_poly2, y_test_poly2)
display(np.round(lasso.coef_, 1))

display(np.round(lasso.intercept_, 3))
coeff= np.round(lasso.coef_, 3)

len(coeff[coeff > 0])
stats = crossvalidation(lasso, X_train, y_train)

model_confidence_interval(stats)
dtree = DecisionTreeRegressor(random_state=10)

dtree_result1 = model_fit_predict_score(dtree, X_train, y_train, X_test, y_test)
dtree = DecisionTreeRegressor(random_state=10, min_samples_leaf=15)

dtree_result1 = model_fit_predict_score(dtree, X_train, y_train, X_test, y_test)
model_show_feature_importance(dtree, X_train, True)
dtree_result1_imp_feat = model_fit_predict_score_only_important_features(dtree, X_train, y_train, X_test, y_test, k_features=3)
stats = crossvalidation(dtree, X_train, y_train)

model_confidence_interval(stats)
rfor = RandomForestRegressor(n_jobs=-1)

rfor_result1 = model_fit_predict_score(rfor, X_train, y_train, X_test, y_test)
rfor = RandomForestRegressor(n_jobs=-1, max_depth=5, max_features=10)

rfor_result1 = model_fit_predict_score(rfor, X_train, y_train, X_test, y_test)
params = dict(n_estimators=range(50, 500, 50), max_depth = range(3, 20, 1), min_samples_leaf=range(1, 5, 1), max_features=('auto', 'sqrt', 'log2', None))

clf = find_best_model_randomsearch(rfor, params, X_train, y_train)
rfor = clf.best_estimator_

rfor.max_depth=5

rfor_result1 = model_fit_predict_score(rfor, X_train, y_train, X_test, y_test)
model_show_feature_importance(rfor, X_train, True)
stats = crossvalidation(rfor, X_train, y_train)

model_confidence_interval(stats)
gboost = GradientBoostingRegressor(random_state=10)

gboost_result1 = model_fit_predict_score(gboost, X_train, y_train, X_test, y_test)
gboost = GradientBoostingRegressor(random_state=10, n_estimators=100, min_samples_split=10, min_samples_leaf=30)

gboost_result1 = model_fit_predict_score(gboost, X_train, y_train, X_test, y_test)
model_show_feature_importance(gboost, X_train, True)
stats = crossvalidation(gboost, X_train, y_train)

model_confidence_interval(stats)
from xgboost.sklearn import XGBRegressor

xgr = XGBRegressor(n_jobs=-1, random_state=10)

xgr_result1 = model_fit_predict_score(xgr, X_train, y_train, X_test, y_test)
xgr =XGBRegressor(max_depth=2, n_estimators=200, n_jobs=-1, random_state=10, gamma=20)

xgr_result1 = model_fit_predict_score(xgr, X_train, y_train, X_test, y_test)
model_show_feature_importance(xgr, X_train, True)
stats = crossvalidation(xgr, X_train, y_train)

model_confidence_interval(stats)
stats = crossvalidation(gboost, X_train, y_train)

model_confidence_interval(stats)
stats = crossvalidation(xgr, X_train, y_train)

model_confidence_interval(stats)