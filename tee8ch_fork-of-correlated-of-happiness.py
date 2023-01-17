from pylab import *

import pandas as pd

import seaborn as sns

sns.set_style('white')

from scipy.cluster.hierarchy import dendrogram, fcluster

%matplotlib inline

hap = pd.read_csv("../input/world-happiness/2015.csv") #world happiness report 2015

world = pd.read_csv("../input/65-world-indexes-gathered/Kaggle.csv") #65 world indexes
hap.head()
world.head()
world.columns = concatenate([[hap.columns[0]], world.columns[1:]]) #equalize country first column
hap.shape, \

world.shape
print(set(hap['Country']).symmetric_difference(world['Country']))

#congo, dom rer. congo, hongkong, Palestine, 
old_names, new_names = (['Congo (Brazzaville)', 'Congo (Kinshasa)', 'Hong Kong\xc2\xa0', 'Palestinian Territories'],

                         ['Republic of the Congo',

                          'Democratic Republic of the Congo',

                          'Hong Kong',

                          'Palestine'])
hap['Country'].replace(old_names, new_names, inplace = True)

world['Country'].replace(old_names, new_names, inplace = True)
len(set(hap['Country']).intersection(world['Country'])), \

len(set(hap['Country']).symmetric_difference(world['Country']))
merged = hap[['Country','Happiness Score']].merge(world, how = 'inner', on = 'Country')
merged.drop(['Country', u'Gross domestic product GDP 2013', 'Infant Mortality 2013 per thousands', 'Gross national income GNI per capita - 2011  Dollars', 'Birth registration funder age 5 2005-2013', 'Pre-primary 2008-2014', u'International inbound tourists thausands 2013', u'International student mobility of total tetiary enrolvemnt 2013', ], axis = 1, inplace = True)
corr_matrix = merged.corr()
figure(figsize = (15, 15))

sns.heatmap(corr_matrix, xticklabels= False)
figure(figsize = (16, 16))

#sns.heatmap(merged.corr())

cg = sns.clustermap(merged.corr().applymap(abs), xticklabels = True, yticklabels = False)
#figure(figsize = (9, 6))

#a = dendrogram(cg.dendrogram_col.linkage, labels = merged.columns, color_threshold = 2, leaf_font_size= 10)

fc = fcluster(cg.dendrogram_col.linkage, 2, criterion = 'distance')
pd.Series(data = merged.columns[fc == 1])
hap_corr = pd.DataFrame(merged.corr()[abs(merged.corr()['Happiness Score']) > .2]['Happiness Score'])

hap_corr.drop('Happiness Score', axis = 0, inplace = True)

hap_corr.columns = ['Corr. with happiness']

hap_corr.sort_values('Corr. with happiness', axis = 0, ascending = False)
# sort by absolute value of correlation

hap_corr_best = hap_corr.apply(abs).sort_values('Corr. with happiness', axis = 0, ascending = False)
figure(figsize = (12,10))

sns.heatmap(corr_matrix.apply(abs).loc[hap_corr_best[:20].index,hap_corr_best[:20].index], annot= True)
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
figure(figsize = (8,6))

scores = zeros((35, len(hap_corr_best.index)-1))

for i in range(35):

    for k in range(1,len(hap_corr_best.index)):

        X, y = array(merged[hap_corr_best.index[:k]]), array(merged['Happiness Score'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        linReg = LinearRegression(normalize = True)

        linReg.fit(X_train,y_train)

        sc = linReg.score(X_test, y_test)

        scores[i][k-1] = sc

sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores)

xlabel("# most important features")

ylabel("$R^2$ score of lin. regression")

show()
from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

from keras import optimizers



def coeff_determination(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred )) 

    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



stopper = EarlyStopping(monitor='loss', min_delta=0, patience = 8, verbose= 0, mode='auto')

MinMax = MinMaxScaler()
%%time 



figure(figsize = (8,6))

scores2 = zeros((3, len(hap_corr_best.index)-1))

for k in range(1,len(hap_corr_best.index)):

    K.clear_session()

    X, y = array(merged[hap_corr_best.index[:k]]), array(merged['Happiness Score'])

    X = MinMax.fit_transform(X)

    for i in range(3):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        model = Sequential()

        model.add(Dense(2*k, input_dim = k, activation = 'relu'))

        model.add(Dense(1))



        model.compile(optimizer= 'rmsprop',

                      loss='mse', metrics = [coeff_determination])



        model.fit(X_train, y_train, epochs = 500, batch_size = 32, verbose = 0, callbacks= [stopper])

        sc = model.evaluate(X_test, y_test, verbose = 0)

        scores2[i][k-1]= sc[1]

sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores2)

xlabel("# most important features")

ylabel("$R^2$ score of lin. regression")

show()

#model.summary()
figure(figsize = (7,5))

sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores2, err_style = "unit_traces", condition = 'ANN')

sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores, color = 'r', condition = 'linear')

ylim([0,1])

show()