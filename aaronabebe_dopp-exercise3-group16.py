import pandas as pd 

import warnings

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn import neighbors

from sklearn.model_selection import GridSearchCV

import category_encoders as ec

from sklearn import preprocessing

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

import scipy

import seaborn as sns

import sklearn as sk

from sklearn import tree

from sklearn.tree import _tree

from sklearn.tree import DecisionTreeClassifier as DTC





warnings.filterwarnings('ignore')
# READ SOURCE CSV

raw = pd.read_csv("../input/unesco_poverty_dataset.csv") 

keys = raw.DEMO_IND.unique() 



# DEFINE BASE CSV

base = raw[['LOCATION', 'TIME']]



# FOR EVERY VAR JOIN ON LOCATION & TIME 

for i in range(0,len(keys)):

    loop = raw.loc[raw.DEMO_IND == keys[i]]

    base = pd.merge(base, loop[['LOCATION', 'TIME', 'Value']],  how='left', left_on=['LOCATION','TIME'], right_on = ['LOCATION','TIME']) 

    base.columns = base.columns.str.replace('Value', keys[i])



# DROP DUPLICATES

base = base.drop_duplicates()
# SEPARATE INTO 3 SUB-TABLES: 1970-2004, 2005-2014, 2015-2019

sub_0 = base[base['TIME'] < 2005]

sub_1 = base[(base['TIME'] >= 2005) & (base['TIME'] < 2015)]

sub_2 = base[base['TIME'] >= 2015]



# WRITE TARGET VARIABLES

sub_0['poverty'] = base['NY_GNP_PCAP_CN'].apply(lambda x: (x / 365) < 1)

sub_1['poverty'] = base['NY_GNP_PCAP_CD'].apply(lambda x: (x / 365) < 1.25)

sub_2['poverty'] = base['NY_GNP_PCAP_PP_CD'].apply(lambda x: (x / 365) < 1.9)



# RE-CONCAT SUB-DATAFRAMES

base = pd.concat([sub_0, sub_1, sub_2])



# SHOW HOW MANY COUNTRIES WERE POOR AT LEAST ONCE

poor = base[base['poverty'] == True]

perc_poor_countries_ever = round(poor['LOCATION'].drop_duplicates().shape[0] / base['LOCATION'].drop_duplicates().shape[0] * 100,2)



print('From 1970-2019, all countries considered,', perc_poor_countries_ever, '% have lived in extreme poverty at least once.')
def get_threshold_line(start, end, height):

    return go.layout.Shape(

        type='line',

        x0=start ,

        y0=height,

        x1=end,

        y1=height,

        line=dict(

            color='red',

            width=4,

            dash='dashdot'

        )

    )



def merge_continents(base):

    # ADD CONTINENTS FOR PLOTTING

    continents = pd.read_csv('../input/continents.csv')

    base = pd.merge(base, continents, left_on='LOCATION', right_on='LOCATION')

    return base







def combined_line_chart(base, y): 



    # COUNT DAILY NOT YEARLY

    base[y] = base['NY_GNP_PCAP_CD'].apply(lambda x: (x / 365))



    base = merge_continents(base)



    fig = px.line(

        base, 

        x="TIME", 

        y=y, 

        hover_name="LOCATION", 

        color="continent",

    )



    # ADD THRESHOLD LINE

    fig.add_shape(get_threshold_line(1970, 1996, 1))

    fig.add_shape(get_threshold_line(1996, 2005, 1.25))

    fig.add_shape(get_threshold_line(2005, 2019, 1.9))

    fig.update_layout(yaxis_type="log")

    return fig



combined_line_chart(base.copy(), 'target')
# GET DATA PER COLUMN

na_percent = []

na_total = []

minimum = []

maximum = []

for col in base.columns:

    na_percent.append(round(base[col].isna().sum() / base.shape[0] * 100, 2))

    na_total.append(base[col].isna().sum())

    minimum.append(base[col].min())

    maximum.append(base[col].max())



# GET VARIABLE DESCRIPTIONS

descriptions = raw['Indicator'].drop_duplicates().tolist()

descriptions.insert(0, 'LOCATION')

descriptions.insert(1, 'TIME')

descriptions.insert(38, 'poverty')



features = pd.DataFrame(

    {'descriptions': descriptions, 

    'na_percent': na_percent, 

    'na_total': na_total,

    'minimum': minimum,

    'maximum': maximum},

    index=base.columns) 



features
# READ TRANSFORMED CSV FILE

raw = pd.read_csv("../input/transformed.csv")  

feature_descriptions = pd.read_csv("../input/feature_descriptions.csv")



# FEATURES WITH LESS THAN 50% MISSING VALUES

features = feature_descriptions.where(feature_descriptions['na_percent']<=50.0).dropna(0)



# ONLY DEMOGRAFIC FEATURES!

#cols_to_drop = 7:13 + 18:25

cols = features['Unnamed: 0'].tolist()

cols = cols[0:7]+ cols[13:18] + [cols[25]]

dataset = raw[cols]

    

by_country = dataset.groupby(by=dataset['LOCATION'])  

dataset_full = pd.DataFrame(columns=cols)

dataset_full2 = pd.DataFrame(columns=cols)





for name, group in by_country :

    tdf = pd.DataFrame(columns=cols)

    tdf2 = pd.DataFrame(columns=cols) 



    tdf['TIME'] = group['TIME']

    tdf['poverty'] = group['poverty']



    # cols with all NaN values

    all_null = group.isna().all()  

    null_cols = all_null.where(all_null == 1).dropna(0).index.tolist()

    tdf[null_cols] = 0



    # cols for interpolation

    cols_to_int = all_null.where(all_null == 0).dropna(0).index.tolist()[2:]

    cols_to_int.remove('poverty')



    tdf[cols_to_int] = group[cols_to_int].interpolate(method='linear', axis=0)

    tdf['LOCATION'] = name 



    # fill the NaN values that were not interpolated

    tdf.fillna(tdf.mean(), inplace=True)



    # Another way to interpolate - take mean for the cols with all NaNs

    tdf2 = group.interpolate(method ='linear', limit_direction ='forward', axis = 0)

    tdf2 = tdf2.interpolate(method ='linear', limit_direction ='backward', axis = 0)

    tdf2['LOCATION'] = name

    tdf2.fillna(dataset.drop(labels=['LOCATION'], axis=1).mean(), inplace=True)

    dataset_full2 = pd.concat([dataset_full2,tdf2])

    

    dataset_full = pd.concat([dataset_full,tdf])



# NA -> mean    

dataset_full2.sort_index(inplace=True)

# NA -> 0

dataset_full.sort_index(inplace=True)



# dataset_full2.head(100)
def scatter(base, x, x_name, y, y_name):

    base.rename(columns={

        x: x_name, 

        y: y_name, 

        '200101': 'Population'}, 

    inplace=True)



    # fill missing values with 1 to get shown on the scatter plot

    base['Population'].fillna(1, inplace=True)



    base = merge_continents(base)



    fig = px.scatter(

        data_frame=base, 

        x=x_name, 

        y=y_name, 

        animation_frame='TIME',

        hover_name='LOCATION',

        size='Population',

        color='continent',

        size_max=60

    )

    return fig



scatter(base.copy(), 

             x='SP_DYN_TFRT_IN', 

             x_name='Fertility Rate', 

             y='NY_GDP_PCAP_CD', 

             y_name='GDP per capita')
def scatter_poor_rich(base, x, x_name, y, y_name):

    base.rename(columns={

        x: x_name,

        y: y_name, 

        '200101': 'Population'

    }, inplace=True)



    # fill missing values with 1 to get shown on the scatter plot

    base['Population'].fillna(1, inplace=True)

    

    base = merge_continents(base)



    fig = px.scatter(

            base, 

            x=x_name, 

            y=y_name,

            facet_col="poverty",

            animation_frame='TIME', 

            hover_name='LOCATION',

            size='Population',

            color='continent'

        )

    return fig





scatter_poor_rich(base.copy(), 

                       x='SP_DYN_IMRT_IN', 

                       x_name='Mortality rate, infant', 

                       y='NY_GDP_PCAP_CD', 

                       y_name='GDP per capita')
def world_map(base, y, y_name, yearly_feature=False):

    if yearly_feature:

        base[y_name] = base[y].apply(lambda x: (x / 365))

    else:

        base.rename(columns={y: y_name}, inplace=True)

    fig = px.choropleth(    

        base,

        locations="LOCATION",

        color=y_name,

        hover_name="LOCATION",

        animation_frame="TIME"

    )

    return fig



world_map(base.copy(), y='SP_RUR_TOTL_ZS', y_name='Rural population (% of total population)')
def question1(base):

    poverty_percent = {'TIME': [], 'PERCENT': []}

    for i in range(1970, 2017):

        poverty_percent['TIME'].append(i)

        poverty_percent['PERCENT'].append(base[(base['TIME']==i) & (base['poverty'])].shape[0] / base[base['TIME']==i].shape[0])



    pp = pd.DataFrame(poverty_percent)



    fig = px.line(

        pp, 

        x='TIME', 

        y='PERCENT', 

    )

    return fig



question1(dataset_full2)
# GROUND TRUTH AS NUMERIC

y = dataset_full['poverty']

y = y.apply(lambda x: 1 if x==True else 0)

X_2 = dataset_full2.drop(labels=['LOCATION', 'poverty'], axis=1)



# FUNCTIONS FOR ML 

def print_performance (classifier, X, y, scores= ['accuracy', 'precision', 'recall'], model=''):

    for score in scores:

        cv2 = cross_val_score(classifier, X, y, cv=10, scoring=score)

        cv2_m = cv2.mean()

        cv2_sd = cv2.std()

        print(model + ' ' + score +" : " + str(round(cv2_m, 5))+ ' +- '+ str(round(cv2_sd, 5)))



def r_classifier (X, y, alpha=1.0, fit_intercept=True, normalize=True, solver='auto', max_iter=1000, tol=0.0001) :

    reg = linear_model.RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=0.001, solver='auto', random_state=30)

    print_performance(reg, X , y, model='Ridge Classifier', scores= ['accuracy'])

    reg.fit(X,y)

    return reg   



ridge_2 =  r_classifier(X_2,y, alpha=0.1)
R2_coef = np.array(ridge_2.coef_)

X2_cols = X_2.columns

R2_relativ = np.abs(R2_coef)/np.abs(R2_coef).sum()

table = {'col':X2_cols, 'absolute':[], 'relative':[]}



# Fill into Dataframe

for i in range(0,len(X2_cols)):

    table['absolute'].append(round(R2_coef[0,i],6))

    table['relative'].append(round(R2_relativ[0,i],6))

    

q2_weights = pd.DataFrame.from_dict(table)

q2_weights
def just_demographic(df_poor):

    feature_descriptions = pd.read_csv("../input/feature_descriptions.csv")



    #FEATURES WITH LESS THAN 50% MISSING VALUES

    features = feature_descriptions.where(feature_descriptions['na_percent']<=50.0).dropna(0)

    

    #ONLY DEMOGRAFIC FEATURES!

    cols = features['Unnamed: 0'].tolist()

    cols = cols[0:7]+ cols[13:18] + [cols[25]]

    df_poor = df_poor[cols]

    return (df_poor)



def interpolation_df_poor(df_poor):

    columns = df_poor.columns

    countries = df_poor['LOCATION'].unique()

    for c in countries:

        df_c = df_poor[df_poor['LOCATION']==c]

        df_c = df_c.interpolate(method ='linear', 

                                    limit_direction ='forward',

                                    axis = 0)

        df_c = df_c.interpolate(method ='linear', 

                                    limit_direction ='backward',

                                    axis = 0)

        df_c.fillna(poor.drop(labels='LOCATION', axis=1).mean(), inplace =True)

        df_poor[df_poor['LOCATION']==c] = df_c

    return(df_poor)



def thresholds(df_poor):

    poor = df_poor

    thresholds = {}

    columns = poor.columns[2:-2]



    X = poor.iloc[:,2:-2]

    y = poor.iloc[:,-2:-1]

    for c in columns:

        X_ = np.array(X[c]).reshape(-1, 1)#X.iloc[:,c]

        clf_tree = DTC(criterion="gini",

                        max_depth=1, 

                        splitter="best")

        clf_tree.fit(X_,y)

        threshold = clf_tree.tree_.threshold[0]

        thresholds[c] = threshold    

    return(thresholds)





def e_poor_selectKBest(df_e_poor, score_f = f_classif, k='all'):

    e_poor = df_e_poor

    # Split dataset to train

    X = e_poor.iloc[:,2:-2] 

    

    # All the columns less the last one

    y = e_poor.iloc[:,-2:-1] 

    

    # Just the last column

    dfcolumns = pd.DataFrame(X.columns)



    scaler = MinMaxScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    

    # Create the feature selector

    bestFeatures = SelectKBest(score_func=score_f, k=k)

    fit = bestFeatures.fit(X,y)

    dfscores = pd.DataFrame(fit.scores_)



    # Create a data frame to see the impact of the features

    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    featureScores.columns = ['Features','Scores']

    featureScores.sort_values(by='Scores', ascending=False, inplace=True)

    scores = featureScores.Scores

    rel_scores = np.array(scores)/np.abs(np.array(scores)).sum()

    rel_scores.reshape(-1, 1)



    rel_scores = pd.DataFrame(rel_scores, dtype=float, columns=['Relative'])



    df_scores = pd.concat([featureScores,rel_scores], axis=1)

    return(df_scores)



def e_poor_feature_importance(df_e_poor):

    e_poor = df_e_poor

    # Split dataset to train

    X = e_poor.iloc[:,2:-2] # All the columns less the last one

    y = e_poor.iloc[:,-2:-1] # Just the last column

    model = ExtraTreesClassifier()

    model.fit(X,y)

    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    

    scores = pd.DataFrame(feat_importances, dtype=float)

    featureScores = scores.reset_index()

    featureScores.columns = ['Features','Scores']

    df_scores0 = featureScores.sort_values(by='Scores', ascending=False)

    df_scores0.reset_index(drop=True, inplace=True)

    scores = df_scores0.Scores

    rel_scores = np.array(scores)/np.abs(np.array(scores)).sum()

    rel_scores.reshape(-1, 1)

    rel_scores = pd.DataFrame(rel_scores, dtype=float, columns=['Relative'])

    df_scores = pd.concat([df_scores0,rel_scores], axis=1)

    print(df_scores)

    feat_importances.nlargest(20).plot(kind='barh', figsize = (13, 6), fontsize=12)

    plt.show()

    

def linearModel(df_e_poor, alpha=0.1, fit_intercept=True, normalize=True, solver='auto', max_iter=1000, tol=0.0001):

    e_poor = df_e_poor

    # Split dataset to train

    X = e_poor.iloc[:,2:-2] # All the columns less the last one

    #st.write(X.columns)

    y = e_poor.iloc[:,-2:-1] # Just the last column



    lm = linear_model.RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=0.001, solver='auto', random_state=30)

    lm.fit(X,y)

    coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)

    print(coefficients)

    return(lm)



def plot_correlation_matrix(df_e_poor, n=20):

    data = e_poor.iloc[:,2:-2] # All features

    # Split dataset to train

    columns = data.columns

    plt.clf()

    correlation = data.corr()



    correlation_map = np.corrcoef(data[columns].values.T)

    sns.set(font_scale=1, rc={'figure.figsize':(30,30)})

    heatmap = sns.heatmap(correlation_map, cbar=True, annot=False, 

                            square=True, fmt='.2f', yticklabels=columns.values, 

                            xticklabels=columns.values)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90, fontsize = 16)

    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 16)

    plt.show()
df_p = pd.read_csv("../input/transformed.csv")

poor = df_p

poor = just_demographic(poor)

poor = interpolation_df_poor(poor)

print (poor.head(6))

print (poor.shape)
p_poor = poor[poor['poverty'] == 1]

c_poor = p_poor.LOCATION.unique() # poor countries



# No Poor countries

no_poor = poor[(poor['poverty'] == 0) & (poor['TIME'] == 2015)]



c_no_poor = no_poor.LOCATION.unique() # no poor countries 2015



e_countries = set(c_no_poor).intersection(c_poor) # Identifying emerging countries



#e_countries # Emerging

poor['emerging'] = poor['LOCATION'].isin(e_countries)

e_poor = poor[poor['LOCATION'].isin(e_countries)]



#Thresholds

print('The thresholds to identiy boundries in some of the plots:')

thresholds(df_poor=poor)



print(e_poor.head(6))

print(e_poor.shape)
print('From a total of',len(poor.LOCATION.unique()),'countries,', len(e_countries), 'have emerged from extreme poverty')
def years_emerged(e_poor):

    years = {}

    years_total = {}

    e_countries = list()

    df_e_poor = e_poor

    for y in df_e_poor.TIME.unique():

        df_y = df_e_poor[df_e_poor['TIME'] == y]

        countries = list()

        #countries_total = list()

        for c in df_y.LOCATION.unique():

            df_c = df_y[df_y['LOCATION'] == c]

            if ((df_c.iloc[0,-2] == 0) & (c not in e_countries)):

                #st.write(df_c.iloc[0,-2])

                e_countries.append(c)

                countries.append(c)

            #break

        #st.write(y,countries)

        y = int(y)

        years[y] = countries

        years_total[y] = len(countries)

    # PLOT

    plt.bar(years_total.keys(), years_total.values(), color='g')

    plt.show()

    return(years)
years_emerged(e_poor)
print("Execution of the K Best features")

print("f_classif")

print(e_poor_selectKBest(e_poor,score_f=f_classif))

#e_poor_selectKBest(e_poor,score_f=f_classif)



print("chi2")

print(e_poor_selectKBest(e_poor,score_f=chi2))

#e_poor_selectKBest(e_poor,score_f=chi2)

print("mutual_info_classif")

print(e_poor_selectKBest(e_poor,score_f=mutual_info_classif))

#e_poor_selectKBest(e_poor,score_f=mutual_info_classif)

print("feature_importance")

e_poor_feature_importance(e_poor)

print("Linear Model")

lm = linearModel(df_e_poor = e_poor)

print("Correlation Matrix")

plot_correlation_matrix(e_poor)

def e_poor_selectKBest(df_e_poor, score_f = f_classif, k='all'):

    e_poor = df_e_poor

    # Split dataset to train

    X = e_poor.iloc[:,2:-2] # All the columns less the last one

    y = e_poor.iloc[:,-2:-1] # Just the last column

    dfcolumns = pd.DataFrame(X.columns)



    scaler = MinMaxScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    # Create the feature selector

    # perhaps a switch

    bestFeatures = SelectKBest(score_func=score_f, k=k)

    fit = bestFeatures.fit(X,y)

    dfscores = pd.DataFrame(fit.scores_)



    # Create a data frame to see the impact of the features

    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    featureScores.columns = ['Features','Scores']

    featureScores.sort_values(by='Scores', ascending=False, inplace=True)

    

    return(featureScores)



# SHOW FEATURE IMPORTANCES

selectKBest_df = pd.merge(e_poor_selectKBest(e_poor), e_poor_selectKBest(e_poor,score_f=chi2), on='Features')

selectKBest_df.columns = ['Features', 'f-classif', 'chi2']

selectKBest_df
def e_poor_feature_importance(df_e_poor):

    e_poor = df_e_poor

    # Split dataset to train

    X = e_poor.iloc[:,2:-2] # All the columns less the last one

    y = e_poor.iloc[:,-2:-1] # Just the last column

    model = ExtraTreesClassifier()

    model.fit(X,y)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    feat_importances.nlargest(20).plot(kind='barh', figsize = (13, 6), fontsize=12)



e_poor_feature_importance(e_poor)
# TRANSFORM DATA FOR BAR PLOT

Features = ['%Pop in Rural Areas', 'Population Growth', 'Fertility', 'Life Expectancy', 'Mortality Rate']

selectKBest_df['f-classif'] = np.abs(selectKBest_df['f-classif'])/np.abs(selectKBest_df['f-classif']).sum()

bar_emerge = selectKBest_df.sort_values(by='Features', ascending=False)['f-classif'].tolist()

bar_poverty = q2_weights[q2_weights.col != 'TIME'].sort_values(by='col', ascending=False)['relative'].to_list()



# PLOT

fig = go.Figure(data=[

    go.Bar(name='In Poverty', x=Features, y=bar_poverty),

    go.Bar(name='Emerged', x=Features, y=bar_emerge)

])

fig.update_layout(barmode='group')

fig.show()