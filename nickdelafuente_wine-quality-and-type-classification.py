import os
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #getting rid of warning from sklearn and seaborn

#Packages
from scipy.stats import skew, norm, probplot, boxcox, f_oneway
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np 
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import Counter
from sklearn.base import TransformerMixin 
%matplotlib inline
red_df = pd.read_csv('../input/winequality-red.csv', sep = ';')
white_df = pd.read_csv('../input/winequality-white.csv', sep = ';')
#creating a wine_color column to distinguish between type
red_df['wine_color'] = 'red' 
white_df['wine_color'] = 'white' 
red_df.head()
white_df.head()
white_df['label'] = white_df['quality'].apply(lambda x: 1 if x <= 5 else 2 if x <= 7 else 3)
red_df['label'] = red_df['quality'].apply(lambda x: 1 if x <= 5 else 2 if x <= 7 else 3)

wine = pd.concat([red_df, white_df], axis = 0) #Combing

#shuffle data for randomization of data points
wine = wine.sample(frac = 1, random_state = 77).reset_index(drop = True)
wine.isnull().sum()
class null_cleaner(TransformerMixin):

    def __init__(self):
        """
        fills missing values:
        -If the column is dtype object they are imputed with the most frequent
         value within the column
        -The other columns with data types are imputed with the mean
         of the corresponding column
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

wine = null_cleaner().fit_transform(wine)
wine.describe().transpose()
wine.info()
Counter(wine['quality'])
Counter(wine['label'])
sns.countplot(wine['wine_color'], palette = ['red', 'white'], edgecolor = 'black')
plt.show()
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize= (10,10), sharey = True)
plt.rc('font', size = 12)
title = fig.suptitle('Wine Type vs Quality', fontsize = 15, y = 1.05)

axes[0, 0].set_title('Red Wine')
axes[0, 0].set_xlabel('Quality')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].bar(list(red_df['quality'].value_counts().index), 
               list(red_df['quality'].value_counts().values),
               color = 'red', edgecolor = 'black')

axes[0, 1].set_title('White Wine')
axes[0, 1].set_xlabel('Quality')
axes[0, 1].bar(list(white_df['quality'].value_counts().index), 
               list(white_df['quality'].value_counts().values),
               color = 'white', edgecolor = 'black')

axes[1, 0].set_xlabel('Quality Label')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].bar(list(red_df['label'].value_counts().index), 
               list(red_df['label'].value_counts().values),
               color = 'red', edgecolor = 'black')

axes[1, 1].set_xlabel('Quality Label')
axes[1, 1].bar(list(white_df['label'].value_counts().index), 
               list(white_df['label'].value_counts().values),
               color = 'white', edgecolor = 'black')

plt.tight_layout()
fig, axes = plt.subplots(10, 2, figsize = (12,17), sharex = True)

ax = sns.boxplot(x="quality", y="fixed acidity", data=white_df, orient='v', 
    ax=axes[0, 0])
ax = sns.boxplot(x="quality", y="fixed acidity", data=red_df, orient='v', 
    ax=axes[0, 1])
ax = sns.boxplot(x="quality", y="volatile acidity", data=white_df, orient='v', 
    ax=axes[1, 0])
ax = sns.boxplot(x="quality", y="volatile acidity", data=red_df, orient='v', 
    ax=axes[1, 1])
ax = sns.boxplot(x="quality", y="citric acid", data=white_df, orient='v', 
    ax=axes[2, 0])
ax = sns.boxplot(x="quality", y="citric acid", data=red_df, orient='v', 
    ax=axes[2, 1])
ax = sns.boxplot(x="quality", y="residual sugar", data=white_df, orient='v', 
    ax=axes[3, 0])
ax = sns.boxplot(x="quality", y="residual sugar", data=red_df, orient='v', 
    ax=axes[3, 1])
ax = sns.boxplot(x="quality", y="chlorides", data=white_df, orient='v', 
    ax=axes[4, 0])
ax = sns.boxplot(x="quality", y="chlorides", data=red_df, orient='v', 
    ax=axes[4, 1])
ax = sns.boxplot(x="quality", y="free sulfur dioxide", data=white_df, orient='v', 
    ax=axes[5, 0])
ax = sns.boxplot(x="quality", y="free sulfur dioxide", data=red_df, orient='v', 
    ax=axes[5, 1])
ax = sns.boxplot(x="quality", y="density", data=white_df, orient='v', 
    ax=axes[6, 0])
ax = sns.boxplot(x="quality", y="density", data=red_df, orient='v', 
    ax=axes[6, 1])
ax = sns.boxplot(x="quality", y="pH", data=white_df, orient='v', 
    ax=axes[7, 0])
ax = sns.boxplot(x="quality", y="pH", data=red_df, orient='v', 
    ax=axes[7, 1])
ax = sns.boxplot(x="quality", y="sulphates", data=white_df, orient='v', 
    ax=axes[8, 0])
ax = sns.boxplot(x="quality", y="sulphates", data=red_df, orient='v', 
    ax=axes[8, 1])
ax = sns.boxplot(x="quality", y="alcohol", data=white_df, orient='v', 
    ax=axes[9, 0])
ax = sns.boxplot(x="quality", y="alcohol", data=red_df, orient='v', 
    ax=axes[9, 1])

axes[0,0].title.set_text('White')
axes[0,1].title.set_text('Red')
plt.tight_layout()
pd.concat([red_df.describe().T, white_df.describe().T], axis = 1, keys = ['Red Wine Statistical Description','White Wine Statistical Description'])
bad_q = wine[wine['label'] == 1].describe()
avg_q = wine[wine['label'] == 2].describe()
high_q = wine[wine['label'] == 3].describe()
pd.concat([bad_q, avg_q, high_q], axis = 0, keys = ['Bad Quality Wine', 'Average Quality Wine', 'High Quality Wine'])
def type_h_testing(feature):
    F, p = f_oneway(red_df[feature],
                    white_df[feature])
    if p <= 0.05: # Standard Measure 
        result = 'Reject'
    else:
        result = 'Accept'
    print('ANOVA test for {}:'.format(feature))
    print('F Statistic: {:.2f} \tp-value: {:.3f} \tNull Hypothese: {}'.format(F, p, result))
    
def quality_h_testing(feature):
    F, p = f_oneway(wine[wine['label'] == 1][feature],
                    wine[wine['label'] == 2][feature],
                    wine[wine['label'] == 3][feature])
    if p <= 0.05:
        result = 'Reject'
    else:
        result = 'Accept'
    print('ANOVA test for {}:'.format(feature))
    print('F Statistic: {:.2f} \tp-value: {:.3f} \tNull Hypothesis: {}'.format(F, p, result))
wine.head()
print('Anova Test for Types of Wine \n')
for column in wine.drop(['wine_color', 'label', 'quality'], axis = 1).columns:
    type_h_testing(column)
print('Anova Test for Types of Wine \n')
for column in wine.drop(['wine_color', 'label', 'quality'], axis = 1).columns:
    quality_h_testing(column)
wine.head()
# re-shuffle data to randomize data points
wine = wine.sample(frac = 1, random_state = 77).reset_index(drop = True)
type_encoder = LabelEncoder()
wine['wine_color'] = type_encoder.fit_transform(wine['wine_color'].values)
# 'white': 1, 'red': 0
sns.set(font_scale=1.5)
corr = wine.corr().drop('label', axis = 1)
type_sorted = corr.sort_values('wine_color', ascending = False).keys()
corr_matrix = corr.loc[type_sorted, type_sorted] #sorts columns
mask = np.zeros_like(corr_matrix, dtype = np.bool) 
mask[np.triu_indices_from(mask)] = True #mask for heatmap
plt.figure(figsize = (16, 9))
sns.heatmap(corr_matrix, cmap = sns.diverging_palette(h_neg = 220, h_pos = 10, s = 80, l = 60, as_cmap = True),
            annot = True, mask = mask)

plt.show()
g = sns.pairplot(data = wine, hue = 'wine_color', 
                 palette = 'Reds')

for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

fig = g.fig
fig.suptitle('Wine Attributes Pairwise Plots by Types', fontsize = 40, y = 1.05)

plt.show()
corr = wine.corr().drop('quality', axis = 1)
sort_ql = corr.sort_values('label', ascending = False).keys()
corr_matrix = corr.loc[sort_ql, sort_ql]
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = (16, 9))
sns.heatmap(corr_matrix, cmap = sns.diverging_palette(h_neg = 220, h_pos = 10, s = 80, l =60, as_cmap = True),
            annot = True, mask = mask)

plt.show()
sns.set(font_scale = 1)
fig, axes = plt.subplots(1, 2, figsize = (15,5))
fig.suptitle('Wine Type, Quality, and Alcohol Content', y = 1.02)

sns.boxplot(x = 'label', y = 'alcohol', data = wine, 
            hue = 'wine_color', palette = 'Reds', ax = axes[0])
h, l = axes[0].get_legend_handles_labels()
axes[0].set_xlabel("Wine Quality Label")
axes[0].set_ylabel("Wine Alcohol %")
axes[0].legend(h, ['White', 'Red'], title = 'Wine Color')


sns.boxplot(x = 'quality', y = 'alcohol', data = wine, 
            hue = 'wine_color', palette = 'Reds', ax = axes[1])
h2, l2 = axes[1].get_legend_handles_labels()
axes[1].legend(h2, ['White', 'Red'], title = 'Wine Color')
axes[1].set_xlabel("Wine Quality")
axes[1].set_ylabel("Wine Alcohol %")

plt.show()
f, axes = plt.subplots(1, 2, figsize = (15, 5))
f.suptitle('Wine Type, Quality, and Acidity', y = 1.02)

sns.violinplot(x = 'label', y = 'volatile acidity', data = wine, 
               hue = 'wine_color', palette = 'Reds', ax = axes[0],
               split = True,  inner = 'quart')
h, l = axes[0].get_legend_handles_labels()
axes[0].set_xlabel('Wine Quality Label')
axes[0].set_ylabel('Wine Fixed Acidity')
axes[0].legend(h, ['White', 'Red'], title = 'Wine Color')

sns.violinplot(x = 'quality', y = 'volatile acidity', data = wine, 
               hue = 'wine_color', palette = 'Reds', ax = axes[1], 
               split = True, inner = 'quart')
h2, l2 = axes[1].get_legend_handles_labels()
axes[1].set_xlabel('Wine Quality')
axes[1].set_ylabel('Wine Fixed Acidity')
axes[1].legend(h, ['White', 'Red'], title = 'Wine Color')

plt.show()
g = sns.FacetGrid(wine, col = 'wine_color', hue = 'label', aspect = 1.5)
g = g.map(plt.scatter, 'volatile acidity', 'alcohol' ,alpha = 0.8,
          edgecolor = 'white')
axes = g.axes.flatten()
axes[0].set_title('Red Wine')
axes[1].set_title('White Wine')
fig = g.fig
fig.suptitle('Wine Color, Quality, Alcohol, and volatile Acidity', y = 1.08)
l = g.add_legend(title = 'Wine Quality Class')


g = sns.FacetGrid(wine, col = 'wine_color', hue = 'label', aspect = 1.5)
g = g.map(plt.scatter, 'volatile acidity', 'total sulfur dioxide',
          edgecolor = 'white', alpha = 0.8)
axes = g.axes.flatten()
axes[0].set_title('Red Wine')
axes[1].set_title('White Wine')
l = g.add_legend(title = 'Wine Quality Class')

fig, axes = plt.subplots(6, 2, figsize = (12,17))

ax = sns.distplot(wine['volatile acidity'], ax=axes[0, 0])
ax = sns.distplot(wine['fixed acidity'], ax=axes[1, 0])
ax = sns.distplot(wine['citric acid'], ax=axes[2, 0])
ax = sns.distplot(wine['residual sugar'], ax=axes[3, 0])
ax = sns.distplot(wine['chlorides'], ax=axes[4, 0])
ax = sns.distplot(wine['free sulfur dioxide'], ax=axes[5, 0])
ax = sns.distplot(wine['total sulfur dioxide'], ax=axes[0, 1])
ax = sns.distplot(wine['density'], ax=axes[1, 1])
ax = sns.distplot(wine['pH'], ax=axes[2, 1])
ax = sns.distplot(wine['sulphates'], ax=axes[3, 1])
ax = sns.distplot(wine['alcohol'], ax=axes[4, 1])
ax = sns.distplot(wine['quality'], ax=axes[5, 1], kde = False)

plt.tight_layout()
numeric_features = list(wine.dtypes[(wine.dtypes != "str") & (wine.dtypes !='object')].index)
numeric_features.remove('wine_color')

#using scipy.stats.skew to measure skew value
skewed_features = wine[numeric_features].apply(lambda x: skew(x))
skew_values = pd.DataFrame({'skew' :skewed_features})
# Using a skew criteria of 0.7 or -0.7
skew_values = skew_values[np.absolute(skew_values) > 0.7].dropna()
print('There are {} skewed parameters, here are the largest:'.format(len(skew_values.index)))
print(skew_values.sort_values(by = 'skew', ascending = False))
maxlog = {}

#no constant since dealing with right skewed parmaters
for feature in skew_values.index:
    wine[feature], maxlog[feature] = boxcox(wine[feature])

#skewness check after data transform
skew_check = wine[skew_values.index].apply(lambda x: skew(x))
skew_check = pd.DataFrame({'new skew': skew_check})

display(pd.concat([skew_values, skew_check], axis = 1).sort_values(by = 'skew', ascending = False))
def QQ_plot(data, measure):
    fig = plt.figure(figsize = (12, 4))
    
    #grabbing mu and sigma after fitting the data to a normal distribution
    (mu, sigma) = norm.fit(data) 
    
    fig1 = fig.add_subplot(1 ,2, 1)
    sns.distplot(data, fit = norm)
    fig1.set_title(measure + ' Distribution (mu = {:.2f}, sigma = {:.2f})'.format(mu, sigma), loc = 'center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')
    
    #qq plot
    fig2 = fig.add_subplot(1, 2, 2)
    res = probplot(data, plot = fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f}, kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')
    
    plt.tight_layout()
    

for feature in skew_values.index:
    QQ_plot(wine[feature], str(feature))
    
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

cols = wine.columns.str.replace(' ', '_')
df = wine.copy()
df.columns = cols
cols = list(cols.drop(['wine_color', 'label', 'quality']))

def VRF(predict, data, y):

    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(data), columns = cols)
    features = '+'.join(cols)
    df['label'] = y.values
    
    #grabbing y and X based off regression
    y, X = dmatrices(predict + ' ~' + features, data = df, return_type = 'dataframe')
    
    #Vif factor calculations for each X
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['features'] = X.columns
    
    #VIF inspection
    display(vif.sort_values('VIF Factor', ascending = False))
    return vif

vif = VRF('label', df.loc[:, cols], wine['label'])
cols = wine.columns.str.replace(' ', '_')
df = wine.copy()
df.columns = cols

# Remove the higest correlations and run a multiple regression
cols = list(cols.drop(['wine_color', 'label', 'quality', 'density']))

vif = VRF('label', df.loc[:, cols], wine['label'])

del df, vif
class select_features(object):
    def __init__(self, select_cols):
        self.select_cols = select_cols
        
    def fit(self, X, y):
        pass
    
    def transform(self, X):
        return X.loc[:, self.select_cols]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        df = self.transform(X)
        return df
    
    def __getitem__(self, x):
        return self.X[x], self.Y[x]
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA

def pca_analysis(df, y_train, feat):
    scale = StandardScaler()
    df = pd.DataFrame(scale.fit_transform(df), index=df.index)
    pca_all = PCA(random_state=101, whiten=True).fit(df)

    my_color=y_train

    # Store results of PCA in a data frame
    result=pd.DataFrame(pca_all.transform(df), columns=['PCA%i' % i for i in range(df.shape[1])], index=df.index)

    # Plot initialisation
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Set2_r", s=60,
               edgecolor = 'white')

    # make simple, bare axis lines through space:
    xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA on the Wines dataset for " + (feat))
    plt.show()

    X_train , X_test, y, y_test = train_test_split(df , y_train, test_size=0.3, random_state=0)

    KNC = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 12, n_neighbors = 12, p  = 1, weights = 'distance')
    KNC = KNC.fit(X_train, y)
    print('KNeighbors Classifier Training Accuracy: {:2.2%}'.format(accuracy_score(y, KNC.predict(X_train))))
    y_pred = KNC.predict(X_test)
    print('KNeighbors Classifier Test Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))

    print('_' * 40)
    print('\nAccurance on', feat, ' Prediction By Number of PCA COmponents:\n')
    AccPca = pd.DataFrame(columns=['Components', 'Var_ratio', 'Train_Acc', 'Test_Acc'])

    for componets in np.arange(1, df.shape[1]):
        variance_ratio = sum(pca_all.explained_variance_ratio_[:componets])*100
        pca = PCA(n_components=componets, random_state=101, whiten=True)
        X_train_pca = pca.fit_transform(X_train)
        Components = X_train_pca.shape[1]
        KNC = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 12, n_neighbors = 12, p  = 1, weights = 'distance')
        KNC = KNC.fit(X_train_pca, y)
        Training_Accuracy = accuracy_score(y, KNC.predict(X_train_pca))
        X_test_pca = pca.transform(X_test)
        y_pred = KNC.predict(X_test_pca)
        Test_Accuracy = accuracy_score(y_test, y_pred)
        AccPca = AccPca.append(pd.DataFrame([(Components, variance_ratio, Training_Accuracy, Test_Accuracy)],
                                            columns=['Components', 'Var_ratio', 'Train_Acc', 'Test_Acc']))#], axis=0)

    AccPca.set_index('Components', inplace=True)
    display(AccPca.sort_values(by='Test_Acc', ascending=False))

#running wine color pca analysis
cols = wine.columns
cols = list(cols.drop(['wine_color', 'label', 'quality']))
pca_analysis(wine.loc[:, cols], wine['wine_color'], 'wine_color')

#running wine quality pca analysis
cols = wine.columns
cols = list(cols.drop(['wine_color', 'label', 'quality']))
pca_analysis(wine.loc[:, cols], wine['label'], 'Quality')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def LDA_analysis(df, y_train, feat):
    X_train , X_test, y, y_test = train_test_split(df , y_train, test_size=0.3, random_state=0)

    KNC = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 12, n_neighbors = 12, p  = 1, weights = 'distance')
    KNC = KNC.fit(X_train, y)
    print('KNC Training Accuracy: {:2.2%}'.format(accuracy_score(y, KNC.predict(X_train))))
    y_pred = KNC.predict(X_test)
    print('KNC Test Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))
    print('_' * 40)
    print('\nApply LDA:\n')
    lda = LDA(n_components=2, store_covariance=True)
    X_train_lda = lda.fit_transform(X_train, y)
    #X_train_lda = pd.DataFrame(X_train_lda)

    print('Number of features after LDA:',X_train_lda.shape[1])
    KNC = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 12, n_neighbors = 12, p  = 1, weights = 'distance')
    KNCr = KNC.fit(X_train_lda, y)
    print('LR Training Accuracy With LDA: {:2.2%}'.format(accuracy_score(y, KNC.predict(X_train_lda))))
    X_test_lda = lda.transform(X_test)
    y_pred = KNC.predict(X_test_lda)
    print('LR Test Accuracy With LDA: {:2.2%}'.format(accuracy_score(y_test, y_pred)))

    if X_train_lda.shape[1]==1:
        fig = plt.figure(figsize=(20,5))
        fig.add_subplot(121)
        plt.scatter(X_train_lda[y==0, 0], np.zeros((len(X_train_lda[y==0, 0]),1)), color='red', alpha=0.1)
        plt.scatter(X_train_lda[y==1, 0], np.zeros((len(X_train_lda[y==1, 0]),1)), color='blue', alpha=0.1)
        plt.title('LDA on Training Data Set')
        plt.xlabel('LDA')
        fig.add_subplot(122)
        plt.scatter(X_test_lda[y_test==0, 0], np.zeros((len(X_test_lda[y_test==0, 0]),1)), color='red', alpha=0.1)
        plt.scatter(X_test_lda[y_test==1, 0], np.zeros((len(X_test_lda[y_test==1, 0]),1)), color='blue', alpha=0.1)
        plt.title('LDA on Test Data Set')
        plt.xlabel('LDA')
    else:
        fig = plt.figure(figsize=(20,5))
        fig.add_subplot(121)
        plt.scatter(X_train_lda[y==0, 0], X_train_lda[y==0, 1], color='black', alpha=0.1)
        plt.scatter(X_train_lda[y==1, 0], X_train_lda[y==1, 1], color='yellow', alpha=0.1)
        plt.scatter(X_train_lda[y==2, 0], X_train_lda[y==2, 1], color='red', alpha=0.1)
        plt.title('LDA on Training Data Set')
        plt.xlabel('LDA')
        fig.add_subplot(122)
        plt.scatter(X_test_lda[y_test==0, 0], X_test_lda[y_test==0, 1], color='black', alpha=0.1)
        plt.scatter(X_test_lda[y_test==1, 0], X_test_lda[y_test==1, 1], color='yellow', alpha=0.1)
        plt.scatter(X_test_lda[y_test==2, 0], X_test_lda[y_test==2, 1], color='red', alpha=0.1)
        plt.title('LDA on Test Data Set')
        plt.xlabel('LDA')

    plt.show()
    
cols = wine.columns
cols = list(cols.drop(['wine_color', 'label', 'quality']))
LDA_analysis(wine.loc[:, cols], wine['wine_color'], 'Type')

cols = wine.columns
cols = list(cols.drop(['wine_color', 'label', 'quality']))
LDA_analysis(wine.loc[:, cols], wine['label'], 'Quality')
cols = wine.columns
cols = list(cols.drop(['wine_color','label']))
y = wine['wine_color']
X_train, X_test, y_train, y_test = train_test_split(wine.loc[:, cols], y, test_size=0.3, random_state = 77)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve 
clf = Pipeline([
    ('pca', PCA(random_state = 77)),
    ('clf', LogisticRegression(random_state = 77))])


#dictionary of parameters to tune
n_components = [10, 12]
whiten = [True]
C = [0.003, 0.009, 0.01, 0.1]
tol = [0.001, 0.0001, 0.01]

param_grid =\
    [{'clf__C': C
     ,'clf__solver': ['liblinear', 'saga'] 
     ,'clf__penalty': ['l1', 'l2']
     ,'clf__tol' : tol 
     ,'clf__class_weight': [None, 'balanced']
     ,'pca__n_components' : n_components
     ,'pca__whiten' : whiten
},
    {'clf__C': C
     ,'clf__max_iter': [3, 9, 2, 7, 4]
     ,'clf__solver': ['newton-cg', 'sag', 'lbfgs']
     ,'clf__penalty': ['l2']
     ,'clf__tol' : tol 
     ,'clf__class_weight': [None, 'balanced'] 
     ,'pca__n_components' : n_components
     ,'pca__whiten' : whiten
}]

#Grid search allows us to figure out the best model given some varying parameters
gs = GridSearchCV(estimator = clf, param_grid = param_grid, scoring = 'accuracy', cv = 5, verbose = 1, n_jobs = -1)
LR = Pipeline([
        #('sel', select_fetaures(select_cols=list(shadow))),
        ('scl', StandardScaler()),
        #('lda', LDA(store_covariance=True)),
        ('gs', gs)]) 

LR.fit(X_train,y_train)
predictions = LR.predict(X_test)
confusion = confusion_matrix(y_test, predictions)
ticklabels = ['False', 'True']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

print(classification_report(y_test, predictions))

auc_score = auc(roc_curve(y_test, predictions)[0],roc_curve(y_test, predictions)[1])
print('AUC score: {}'.format(auc_score))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

wtp_dnn_model = Sequential()
wtp_dnn_model.add(Dense(64, activation = 'relu', input_shape = (12,)))
wtp_dnn_model.add(Dropout(rate = 0.3))
wtp_dnn_model.add(Dense(32, activation = 'relu'))
wtp_dnn_model.add(Dense(16, activation = 'relu'))
wtp_dnn_model.add(Dense(1, activation = 'sigmoid'))

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

wtp_dnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = wtp_dnn_model.fit(X_train, y_train, epochs=55, batch_size=50, 
                            shuffle=True, validation_split=0.2, verbose=1, 
                            callbacks = [early_stop])

wtp_dnn_ypred = wtp_dnn_model.predict_classes(X_test)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
t = f.suptitle('Deep Neural Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs = np.arange(1, len(history.history['accuracy']) + 1)
ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(epochs)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epochs, history.history['loss'], label='Train Loss')
ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(epochs)
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

plt.tight_layout()
confusion = confusion_matrix(y_test, wtp_dnn_ypred)
ticklabels = ['False', 'True']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

print(classification_report(y_test, wtp_dnn_ypred))

auc_score = auc(roc_curve(y_test, wtp_dnn_ypred)[0], roc_curve(y_test, wtp_dnn_ypred)[1])
print('AUC score: {}'.format(auc_score))
X_train, X_test, y_train, y_test = train_test_split(wine.drop(['label', 'quality'], axis = 1), wine['label'], test_size=0.30, random_state=77)
from sklearn.tree import DecisionTreeClassifier

#For model evaluation
model = {}

clf = Pipeline([('clf', DecisionTreeClassifier(random_state = 77))])

# parameter dictionary for tuning
criterion = ['gini', 'entropy']
splitter = ['best']
max_depth = [8, 9 ,10, 11, 15]
min_samples_leaf = [2, 3, 5]
class_weight = ['balanced', None]

param_grid =\
    [{ 'clf__class_weight': class_weight
      ,'clf__criterion': criterion
      ,'clf__splitter': splitter
      ,'clf__max_depth': max_depth
      ,'clf__min_samples_leaf': min_samples_leaf
}]

gs = GridSearchCV(estimator = clf, param_grid = param_grid, scoring = 'accuracy',
                  cv = 5, verbose = 1, n_jobs = -1)

DT = Pipeline([
        ('scl', StandardScaler()),
        ('gs', gs)
 ]) 

DT.fit(X_train, y_train)
DT_predictions = DT.predict(X_test)

#For model evaulation later
model['DT'] = DT_predictions
print(classification_report(y_test, DT_predictions))
confusion = confusion_matrix(y_test, DT_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

#'ovo' multiclass for the average of all possible pairwise combinations, insensitive to class imbalance
auc_score = roc_auc_score(y_test, DT.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')
print('AUC score: {}'.format(auc_score))
from graphviz import Source
from sklearn import tree
from IPython.display import Image

dt = gs.best_estimator_.get_params()['clf']
dt.fit(X_train, y_train)
target_names = ['Bad', 'Average', 'High']

max_depth = 10
graph = Source(tree.export_graphviz(dt, out_file = None, class_names = target_names, filled = True,
                                 rounded= True, special_characters = False, feature_names = cols, 
                                 max_depth = max_depth))
png_data = graph.pipe(format = 'png')
with open('dtree_structure.png','wb') as f:
    f.write(png_data)
    
Image(png_data)
from sklearn.tree import DecisionTreeClassifier

#For model evaluation
model = {}

clf = Pipeline([('clf', DecisionTreeClassifier(random_state = 77))])

# parameter dictionary for tuning
criterion = ['gini', 'entropy']
splitter = ['best']
max_depth = [8, 9 ,10, 11, 15]
min_samples_leaf = [2, 3, 5]
class_weight = ['balanced', None]

param_grid =\
    [{ 'clf__class_weight': class_weight
      ,'clf__criterion': criterion
      ,'clf__splitter': splitter
      ,'clf__max_depth': max_depth
      ,'clf__min_samples_leaf': min_samples_leaf
}]

gs = GridSearchCV(estimator = clf, param_grid = param_grid, scoring = 'accuracy',
                  cv = 5, verbose = 1, n_jobs = -1)

DT = Pipeline([
        ('scl', StandardScaler()),
        ('gs', gs)
 ]) 

DT.fit(X_train, y_train)
DT_predictions = DT.predict(X_test)

#For model evaulation later
model['DT'] = DT_predictions
print(classification_report(y_test, RF_predictions))
confusion = confusion_matrix(y_test, RF_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

#'ovo' multiclass for the average of all possible pairwise combinations, insensitive to class imbalance
auc_score = roc_auc_score(y_test, RF.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')
print('AUC score: {}'.format(auc_score))
print('Accuracy difference in RFC vs DTC: {}'.format(accuracy_score(y_test, RF_predictions) - accuracy_score(y_test, DT_predictions)))
print('AUC difference in RFC vs DTC: {}'.format(auc_score - roc_auc_score(y_test, DT.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')))
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', KNeighborsClassifier())])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [n_comp_base - 5, n_comp_base - 4, n_comp_base - 3, n_comp_base - 2] 
whiten = [True, False]

param_grid =\
    [{'clf__n_neighbors': [10, 11, 12, 13] 
     ,'clf__weights': ['distance'] 
     ,'clf__algorithm' : ['ball_tree'] #, 'brute', 'auto',  'kd_tree', 'brute']
     ,'clf__leaf_size': [12, 11, 13]
     ,'clf__p': [1] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

KNNC = Pipeline([
        ('scl', StandardScaler()),
        ('gs', gs)
 ]) 

KNNC.fit(X_train,y_train)

KNNC_predictions = KNNC.predict(X_test)

#For model evaulation later
model['KNNC'] = KNNC_predictions
print(classification_report(y_test, KNNC_predictions))
confusion = confusion_matrix(y_test, KNNC_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

#'ovo' multiclass for the average of all possible pairwise combinations, insensitive to class imbalance
auc_score = roc_auc_score(y_test, KNNC.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')
print('AUC score: {}'.format(auc_score))
accuracy_score(y_test, KNNC_predictions)
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', GradientBoostingClassifier(random_state=101))])  

# a list of dictionaries to specify the parameters that we'd want to tune
#cv=None, dual=False,  scoring=None, refit=True,  multi_class='ovr'
n_components= [n_comp_base - 5, n_comp_base - 4, n_comp_base - 3, n_comp_base - 2] 
whiten = [True, False]
learning_rate =  [1e-02] #, 5e-03, 2e-02]
n_estimators= [400]
max_depth = [10]
n_comp = [2, 3, 4, 5]

param_grid =\
    [{'clf__learning_rate': learning_rate
     ,'clf__max_depth': max_depth
     ,'clf__n_estimators' : n_estimators 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
}]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

GBC = Pipeline([
        #('sel', select_fetaures(select_cols=SEL)),
        ('scl', StandardScaler()),
        ('gs', gs)
 ])  

GBC.fit(X_train,y_train)

GBC_predictions = GBC.predict(X_test)


#For model evaulation later
model['GBC'] = GBC_predictions
print(classification_report(y_test, GBC_predictions))
confusion = confusion_matrix(y_test, GBC_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

#'ovo' multiclass for the average of all possible pairwise combinations, insensitive to class imbalance
auc_score = roc_auc_score(y_test, GBC.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')
print('AUC score: {}'.format(auc_score))
from sklearn.ensemble import AdaBoostClassifier

clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', AdaBoostClassifier(random_state=101))])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [n_comp_base - 5, n_comp_base - 4, n_comp_base - 3, n_comp_base - 2] 
whiten = [True, False]
n_comp = [2, 3, 4, 5]

param_grid =\
    [{'clf__learning_rate': [2e-01, 15e-02]
     ,'clf__n_estimators': [500, 600, 700] 
     ,'clf__algorithm' : ['SAMME.R'] # 'SAMME'
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     #,'lda__n_components': n_comp
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

ADAB = Pipeline([
        #('sel', select_fetaures(select_cols=SEL)),
        ('scl', StandardScaler()),
        #('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

ADAB.fit(X_train,y_train)

ADAB_predictions = ADAB.predict(X_test)

#For model evaulation later
model['ADAB'] = ADAB_predictions
print(classification_report(y_test, ADAB_predictions))
confusion = confusion_matrix(y_test, ADAB_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

#'ovo' multiclass for the average of all possible pairwise combinations, insensitive to class imbalance
auc_score = roc_auc_score(y_test, ADAB.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')
print('AUC score: {}'.format(auc_score))
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', LogisticRegression(random_state=101))])  

# a list of dictionaries to specify the parameters that we'd want to tune

n_components= [n_comp_base - 5, n_comp_base - 4, n_comp_base - 3, n_comp_base - 2] 
whiten = [True, False]
C =  [1.0] #, 1e-06, 5e-07, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 10.0, 100.0, 1000.0]
tol = [1e-06] #, 5e-07, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]

param_grid =\
    [{'clf__C': C
     ,'clf__solver': ['liblinear', 'saga'] 
     ,'clf__penalty': ['l1', 'l2']
     ,'clf__tol' : tol 
     ,'clf__class_weight': ['balanced']
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
},
    {'clf__C': C
     ,'clf__max_iter': [3, 9, 2, 7, 4]
     ,'clf__solver': ['newton-cg', 'sag', 'lbfgs']
     ,'clf__penalty': ['l2']
     ,'clf__tol' : tol 
     ,'clf__class_weight': ['balanced'] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
}]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

LRC = Pipeline([
        ('scl', StandardScaler()),
        ('gs', gs)
 ])  

LRC.fit(X_train,y_train)

LRC_predictions = LR.predict(X_test)

#For model evaulation later
model['LRC'] = LRC_predictions
print(classification_report(y_test, LRC_predictions))
confusion = confusion_matrix(y_test, LRC_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')

#'ovo' multiclass for the average of all possible pairwise combinations, insensitive to class imbalance
auc_score = roc_auc_score(y_test, LRC.predict_proba(X_test), multi_class = 'ovo', average = 'weighted')
print('AUC score: {}'.format(auc_score))
from sklearn.svm import LinearSVC

clf = Pipeline([
        ('pca', PCA(random_state = 101)),
        ('clf', LinearSVC(random_state=101, multi_class='ovr', class_weight='balanced'))])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [n_comp_base - 5, n_comp_base - 4, n_comp_base - 3, n_comp_base - 2] 
whiten = [True, False]
C =  [0.06, 0.08, 0.07] #, 1.0, 10.0, 100.0, 1000.0]
tol = [1e-06]
max_iter = [10, 15, 9]

param_grid =\
    [{'clf__loss': ['hinge']
     ,'clf__tol': tol
     ,'clf__C': C
     ,'clf__penalty': ['l2']
     ,'clf__max_iter' : max_iter
     ,'clf__dual' : [True]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }
    ,{'clf__loss': ['squared_hinge']
     ,'clf__tol': tol
     ,'clf__C': C
     ,'clf__penalty': ['l2', 'l1']
     ,'clf__max_iter' : max_iter
     ,'clf__dual' : [False]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

LSVC = Pipeline([
        ('scl', StandardScaler()),
        #('lda', LDA(n_components = 2, store_covariance=True)),
        ('gs', gs)
 ])  

LSVC.fit(X_train,y_train)

LSVC_predictions = LSVC.predict(X_test)


#For model evaulation later
model['LSVC'] = LSVC_predictions
print(classification_report(y_test, LSVC_predictions))
confusion = confusion_matrix(y_test, LSVC_predictions)
ticklabels = ['1: Bad', '   2: Average', '   3: High']
ax = sns.heatmap(confusion, annot = True, cbar = False, fmt = 'g', 
                 yticklabels = ticklabels, xticklabels = ticklabels,
                 cmap = 'OrRd', linecolor = 'black', linewidth = 1)
ax.set_xlabel('True/Actual')
ax.set_ylabel('Predicted')
from sklearn.metrics import precision_score, recall_score, f1_score
#DONT FORGET XG

model_df = pd.DataFrame()

for key in model:
    model_df['score'] = ['accuracy', 'recall', 'precision', 'f1']
    
    accuracy = accuracy_score(y_test, model[key])
    recall = recall_score(y_test, model[key], average = 'weighted')
    precision = precision_score(y_test, model[key], average = 'weighted')
    f1 = f1_score(y_test, model[key], average = 'weighted')
    
    model_df[key] = [accuracy, recall, precision, f1]
    
    print('{}: \
        \n Accuracy score: {} \
        \n Recall score: {} \
        \n Precision score: {} \
        \n f1 score: {} \
        '.format(key, accuracy, recall, precision, f1))

model_df = model_df.set_index('score')
model_df.head()
sclf = StackingClassifier(classifiers=[RF, GBC],
                          use_probas=False,
                          average_probas=False,
                          use_features_in_secondary=False,
                          meta_classifier= RF)

sclf.fit(X_train,y_train)

sclf_predictions = sclf.predict(X_test)
