# Importamos librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

import cesium
from cesium.time_series import TimeSeries
import cesium.featurize as featurize

from tqdm import tnrange, tqdm_notebook

import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

from astropy.stats import LombScargle
import scipy.signal as signal
from gatspy.periodic import LombScargleFast

from imblearn.over_sampling import SMOTE

import warnings
warnings.simplefilter('ignore')

%matplotlib inline
plt.xkcd()
metadata = pd.read_csv('../input/training_set_metadata.csv')
nobjects = len(metadata)
metadata.head()
metadata
# Máscaras booleanas
# Los astros dentro de la Vía Láctea tienen redshift = 0

galactic = metadata['hostgal_specz'] == 0
extragal = metadata['hostgal_specz'] != 0
passbands = ['u','g','r','i','z','y']


targets = metadata['target'].unique()
targets.sort()

gal_targets = metadata[galactic]['target'].unique()
gal_targets.sort()

extragal_targets = metadata[extragal]['target'].unique()
extragal_targets.sort()

targets
fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(metadata['target'],hue=metadata['hostgal_photoz']==0,dodge=False )
ax.legend(['extragalactic','galactic'])
ax.set_title('amount of stars of each class')
ax.set_xlabel('class')
g = sns.pairplot(metadata[extragal][['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'mwebv', 'target']],             
                vars = [c for c in ['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'mwebv', 'target'] if c != 'target'], 
                hue = 'target',plot_kws={'alpha':0.6},diag_kws={'alpha':0.6},palette='bright')
g.fig.suptitle('metadata correlation of extragalactic stars')
g = sns.pairplot(x_vars=['hostgal_photoz_err'], y_vars=['hostgal_photoz'], data=metadata[extragal], 
                 hue="target",palette='bright', size=6, plot_kws={'alpha':0.6})
g.fig.suptitle('photometric redshift vs err')
plt.figure(figsize=(8,5))
for t in extragal_targets:
    sns.distplot(metadata[metadata['target']==t]['hostgal_specz'])
plt.title('redshift distribution by classes')

plt.xlim((0,4))
plt.xticks([z/2 for z in range(8)])

#plt.xlim((0.01,10))
#plt.xscale('log')

plt.yticks([])

plt.legend(extragal_targets)
plt.figure(figsize=(8,5))
for t in extragal_targets:
    sns.distplot(metadata[(metadata['target']==t) & (metadata['hostgal_specz']>1)]['hostgal_specz'])
plt.legend(extragal_targets)
plt.title('classes with redshift > 1')


plt.xlim((1,4))
plt.xticks([z/2 for z in range(2,8)])

plt.yticks([])

plt.legend(extragal_targets)
zmin_count = []
for z in range(8):
    zmin = z/2
    zmin_count.append(len(metadata[metadata['hostgal_specz']>zmin]['target'].unique()))
fig,ax = plt.subplots(figsize=(8,5))
plt.bar(range(8),zmin_count)
ax.set_yticks(range(max(zmin_count)+1))
ax.set_xticks(range(8))
ax.set_xticklabels(z/2 for z in range(8))
ax.set_title('number of classes with spectrometry bigger than z')
ax.set_ylabel('classes')
ax.set_xlabel('z')
fig,ax = plt.subplots(figsize=(10,7))

plt.scatter('gal_l', 'gal_b',c='target', data=metadata, alpha=0.6)
ax.set_ylabel('Galactic Latitude')
ax.set_xlabel('Galactic Longitude')

cb = plt.colorbar()
loc = np.arange(0,max(targets),max(targets)/float(len(targets)))
cb.set_ticks(loc)
cb.set_ticklabels(targets)

ax.set_title('sky distribution of classes')
fig,ax = plt.subplots(figsize=(10,7))

plt.scatter('ra', 'decl',c='target', data=metadata, alpha=0.6)
ax.set_ylabel('Declination')
ax.set_xlabel('Right ascention')

cb = plt.colorbar()
loc = np.arange(0,max(targets),max(targets)/float(len(targets)))
cb.set_ticks(loc)
cb.set_ticklabels(targets)

ax.set_title('sky distribution of classes')
fig,ax = plt.subplots(figsize=(8,5))
plt.scatter('ra','decl', data=metadata, c="mwebv",alpha=0.6)
plt.title('milky way dust')
ax.set_xlabel('right ascention')
ax.set_ylabel('declination')
plt.figure(figsize=(7,5))
plt.scatter(metadata['distmod'], metadata['hostgal_photoz'])
plt.xlabel('distmod')
#plt.xlim((0,46))
plt.ylabel('redshift')
plt.title('redshift and distmod correlation')
lcdata = pd.read_csv('../input/training_set.csv')
lcdata.head()

def starViewer (df,df_meta,object_id):
    star = df[df['object_id']==object_id]
    mjd = star['mjd']
    passband = star['passband']
    flux = star['flux']
    #flux = star['flux_corrected']
    flux_err = star['flux_err']
    #dot_size = 20/(2**(flux_err/abs(flux)))
    
    star_meta = df_meta[df_meta['object_id'] == object_id]
    starSpecZ = star_meta['hostgal_specz'].iloc[0]
    starPhotoZ = star_meta['hostgal_photoz'].iloc[0]
    starMWEBV = star_meta['mwebv'].iloc[0]
    starTarget = star_meta['target'].iloc[0]
    starDDF = star_meta['ddf'].iloc[0]
    starDistmod = star_meta['distmod'].iloc[0]

    global fig, image
    
    fig = plt.figure(figsize=(18,14))
    grid = plt.GridSpec(4, 5,wspace=0.5, hspace=0.3)

    #3D
    scatter3d = fig.add_subplot(grid[0:2,0:3], projection='3d')
    #scatter3d.scatter(mjd,passband,flux, c=passband, alpha=0.5)
    scatter3d.tricontour(mjd,passband,flux, zdir='y', alpha=0.7)
    scatter3d.view_init(30, -120)
    scatter3d.set_yticks(range(6))
    scatter3d.set_ylim(-.5,5.5)
    scatter3d.set_yticklabels(passbands)
    scatter3d.set_ylabel('passband')
    scatter3d.set_xlabel('day')
    scatter3d.set_zlabel('flux')

    #PB-FLUX
    scatterpb = fig.add_subplot(grid[2,2:3])
    scatterpb.scatter(passband, flux, c=passband, alpha=0.5)
    # Sacar tick labels para darle cardinalidad
    scatterpb.set_xticks(range(6))
    scatterpb.set_xticklabels(passbands)
    
    #MJD-FLUX
    scatterflux = fig.add_subplot(grid[2,3:5])
    scatterflux.scatter(mjd, flux, c=passband, alpha=0.5)

    #MJD-PB
    scattermjd = fig.add_subplot(grid[1,3:5])
    scattermjd.scatter(mjd, passband, c=passband, alpha=0.5)
    scattermjd.set_yticks(range(6))
    scattermjd.set_yticklabels(passbands)

    #TEXT
    starText = 'object id = ' + str(object_id) + '\ntarget = ' + str(starTarget) 
    
    starMetaText = 'specZ = ' + "{0:.4f}".format(starSpecZ) + '\nphotoZ = ' + "{0:.4f}".format(starPhotoZ) + '\nMWEBV = ' + "{0:.4f}".format(starMWEBV) + '\ndistmod = ' + "{0:.4f}".format(starDistmod) + '\nDDF = ' + str(bool(starDDF))

    legend = plt.subplot(grid[0,3:5])
    legend.axis('off')
    legend.text(0.5,0.25,starText,fontsize=24,ha='center')
    
    legendMeta = plt.subplot(grid[2,0:2])
    legendMeta.axis('off')
    legendMeta.text(0.5,0.25,starMetaText,fontsize=18,ha='center')
    
    plt.show()
metadata[metadata['hostgal_photoz']==0].head()

metadata[metadata['target']==92].head()
starViewer(lcdata,metadata,615)
star615 = lcdata[lcdata['object_id'] == 615]
star615.head()
t = star615['mjd']
mag = star615['flux']
dmag = star615['flux_err']

fig, ax = plt.subplots(figsize=(10,5))
ax.errorbar(t, mag, dmag, fmt='.k', ecolor='gray')
ax.set(xlabel='Time (days)', ylabel='magitude')
# Aplicamos Modelo de Lomb Scargle para determinar el período predominante

model = LombScargleFast().fit(t, mag, dmag)
periods, power = model.periodogram_auto(nyquist_factor=100)

fig, ax = plt.subplots()
ax.plot(periods, power)
ax.set(xlim=(0.1, 4), ylim=(0, 0.8),
       xlabel='period (days)',
       ylabel='Lomb-Scargle Power');
ax.set_title('Object ID: 615\nBest Period: 0.32 days')
t = star615['mjd']
mag = star615['flux']
dmag = star615['flux_err']
pb = star615['passband']

fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(t, mag,c=pb)
ax.set(xlabel='MJD (days)', ylabel='magitude')
ax.set_title('Star 615')

# set range and find period
model.optimizer.period_range=(0.01, 10)
period = model.best_period
print("period = {0}".format(period))
# Determinamos la nueva variable de tiempo T

pb = star615['passband']
T = star615['mjd']/(period/(2*np.pi))

star615['T'] = T
fig = plt.figure(figsize=(7,7))

plt.rc('grid', color='#999999', linewidth=2, linestyle=':')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

ax = fig.add_subplot(111, projection='polar')
lines,labels = plt.thetagrids()
c = ax.scatter(T, mag, c=pb, s=50, alpha=0.5)
plt.title('Object ID: 615\nperiod =' + "{0:.2f}".format(period))
def starPhaseViewer(df,df_meta,object_id):
    star = df[df['object_id']==object_id]

    t = star['mjd']
    mag = star['flux']
    dmag = star['flux_err']
    
    model = LombScargleFast().fit(t, mag, dmag)
    periods, power = model.periodogram_auto(nyquist_factor=100)
    
    # set range and find period
    model.optimizer.period_range=(0.01, 10)
    period = model.best_period
    print("period = {0}".format(period))

    T = star['mjd']/(period/(2*np.pi))
    T = T%(2*np.pi)
    star['T'] = T
    
    #mjd = star['mjd']
    mjd = star['T']
    passband = star['passband']
    flux = star['flux']
    flux_err = star['flux_err']
    #dot_size = 20/(2**(flux_err/abs(flux)))

    star_meta = df_meta[df_meta['object_id'] == object_id]
    starSpecZ = star_meta['hostgal_specz'].iloc[0]
    starPhotoZ = star_meta['hostgal_photoz'].iloc[0]
    starMWEBV = star_meta['mwebv'].iloc[0]
    starTarget = star_meta['target'].iloc[0]
    starDDF = star_meta['ddf'].iloc[0]
    starDistmod = star_meta['distmod'].iloc[0]

    global fig, image

    fig = plt.figure(figsize=(18,14))
    grid = plt.GridSpec(4, 5,wspace=0.5, hspace=0.3)

    #3D
    scatter3d = fig.add_subplot(grid[0:2,0:3], projection='3d')
    #scatter3d.scatter(mjd,passband,flux, c=passband, alpha=0.5)
    scatter3d.tricontour(mjd,passband,flux, zdir='y', alpha=0.7)
    scatter3d.view_init(30, -120)
    scatter3d.set_yticks(range(6))
    scatter3d.set_ylim(-.5,5.5)
    scatter3d.set_yticklabels(passbands)
    scatter3d.set_ylabel('passband')
    scatter3d.set_xlabel('phase')
    scatter3d.set_zlabel('flux')

    #PB-FLUX
    scatterpb = fig.add_subplot(grid[2,2:3])
    scatterpb.scatter(passband, flux, c=passband, alpha=0.5)
    # Sacar tick labels para darle cardinalidad
    scatterpb.set_xticks(range(6))
    scatterpb.set_xticklabels(passbands)

    #MJD-FLUX
    scatterflux = fig.add_subplot(grid[2,3:5])
    scatterflux.scatter(mjd, flux, c=passband, alpha=0.5)

    #MJD-PB
    scattermjd = fig.add_subplot(grid[1,3:5])
    scattermjd.scatter(mjd, passband, c=passband, alpha=0.5)
    scattermjd.set_yticks(range(6))
    scattermjd.set_yticklabels(passbands)

    #TEXT
    starText = 'object id = ' + str(object_id) + '\ntarget = ' + str(starTarget) 

    starMetaText = 'specZ = ' + "{0:.4f}".format(starSpecZ) + '\nphotoZ = ' + "{0:.4f}".format(starPhotoZ) + '\nMWEBV = ' + "{0:.4f}".format(starMWEBV) + '\ndistmod = ' + "{0:.4f}".format(starDistmod) + '\nDDF = ' + str(bool(starDDF))

    legend = plt.subplot(grid[0,3:5])
    legend.axis('off')
    legend.text(0.5,0.25,starText,fontsize=24,ha='center')

    legendMeta = plt.subplot(grid[2,0:2])
    legendMeta.axis('off')
    legendMeta.text(0.5,0.25,starMetaText,fontsize=18,ha='center')

    plt.show()
starPhaseViewer(lcdata,metadata,615)
lcdata[(lcdata['object_id']==615)&(lcdata['passband']==0)]['flux'].describe()
fig,ax = plt.subplots(figsize=(10,7))
#plt.scatter(x='mjd',y='flux',data=lcdata[(lcdata['object_id']==615)&(lcdata['passband']==0)])

data=lcdata[(lcdata['object_id']==615)&(lcdata['passband']==0)]
plt.plot(data['mjd'],data['flux'], marker='o')

# Plot the average line
mean_line = ax.plot([59800,60600],[-3,-3], label='Mean', linestyle='--')
median_line = ax.plot([59800,60600],[-10,-10], label='Median', linestyle='--')


ax.annotate('maximum', xy=(60300, 125), xytext=(60400, 130),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.annotate('minimum', xy=(59870, -125), xytext=(59950, -115),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.set_ylabel('flux')
ax.set_xlabel('mjd')
ax.legend()
features_to_use = ["amplitude",
                   "percent_beyond_1_std",
                   "maximum",
                   "max_slope",
                   "median",
                   "median_absolute_deviation",
                   "percent_close_to_median",
                   "minimum", 
                   "skew",
                   "std",
                   "weighted_average",
                   ### Lomb-Scargle
                   #"freq1_signif",
                   #"freq1_lambda",
                   #"freq1_amplitude1"
                  ]

def worker(tsobj):
    global features_to_use
    thisfeats = featurize.featurize_single_ts(tsobj,\
    features_to_use=features_to_use,
    raise_exceptions=False)
    return thisfeats
# Si existe el archivo de features cargarlo 
# Si no existe, calcular las features con Cesium Time Series

pathfile = '../input/master_features.csv'
features_file = Path(pathfile)

if features_file.is_file():
    # file exists
    feature_table = pd.read_csv('../input/master_features.csv')
    feature_table.drop(columns='feature',inplace=True)
    feature_table.drop(index=0,inplace=True)
    feature_table.reset_index(drop=True,inplace=True)
else:
    # file does not exsist
    tsdict = dict()
    for i in tnrange(nobjects, desc='Building Timeseries'):
        row = metadata.loc[i]
        thisid = int(row['object_id'])
        target = row['target']

        meta = {'z_spec':row['hostgal_specz'],
                'z':row['hostgal_photoz'],
                'z_err':row['hostgal_photoz_err'],
                'mwebv':row['mwebv']}

        thislc = lcdata[lcdata['object_id'] == thisid]

        pbind = [(thislc['passband'] == pb) for pb in pbmap]
        t = [thislc['mjd'][mask].reset_index(drop=True) for mask in pbind ]
        m = [thislc['flux'][mask].reset_index(drop=True) for mask in pbind ]
        e = [thislc['flux_err'][mask].reset_index(drop=True) for mask in pbind ]

        tsdict[thisid] = TimeSeries(t=t, m=m, e=e,\
                            #label=target, name=thisid, meta_features=meta,\
                            channel_names=pbnames )
        
    feature_table = pd.DataFrame()
    for star in tqdm_notebook(tsdict.keys(), desc='Featurizing...'):
        feature_table = feature_table.append(pd.DataFrame(worker(tsdict[star])).T)
    feature_table.index = tsdict.keys()
feature_table.head()    
# Armar estimador por Random Forest

def randomForest(X,y,n_estimators=300):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    targets = y['target'].unique()
    targets.sort()
    
    clf = RandomForestClassifier(n_estimators=300, criterion='gini',\
                       oob_score=False, n_jobs=-1,  #random_state=42,\
                      verbose=0, class_weight='balanced', max_features='auto')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues',
                xticklabels=targets, yticklabels=targets, 
                fmt='.0f',annot_kws={"color": 'black'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    ax.set_title('Confusion matrix')
    plt.show()

    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame(columns=X.columns)
    feature_importance.loc[0] = clf.feature_importances_
    return feature_importance
# Armar estimador por K-Neighbors

def kNeighbors(X,y,scaled=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    targets = y['target'].unique()
    
    if (scaled):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    scores_para_df = []
    for i in tqdm_notebook(range(1,40,1)):
        model = KNeighborsClassifier(n_neighbors=i)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        dict_row_score = {'score_medio':np.mean(cv_scores),'score_std':np.std(cv_scores),'n_neighbours':i}
        scores_para_df.append(dict_row_score)
    df_scores = pd.DataFrame(scores_para_df)

    df_scores['limite_inferior'] = df_scores['score_medio'] - df_scores['score_std']
    df_scores['limite_superior'] = df_scores['score_medio'] + df_scores['score_std']
    plt.plot(df_scores['n_neighbours'],df_scores['limite_inferior'],color='r')
    plt.plot(df_scores['n_neighbours'],df_scores['score_medio'],color='b')
    plt.plot(df_scores['n_neighbours'],df_scores['limite_superior'],color='r');

    best_k = df_scores.loc[df_scores.score_medio == df_scores.score_medio.max(),'n_neighbours'].values
    best_k = best_k[0]

    plt.show()
    print('Best K: ', best_k)
    
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues',
                xticklabels=targets, yticklabels=targets, 
                fmt='.0f',annot_kws={"color": 'black'})
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos');
    plt.show()

    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
X = pd.DataFrame(feature_table).copy()
y = pd.DataFrame(metadata['target'])
# Random Forest sobre matriz de Features completa

feature_importance = randomForest(X, y,300)
feature_importance
# Random Forest sobre matriz de Features galáctica

feature_importance = randomForest(X[galactic], y[galactic],300)
feature_importance
# Random Forest sobre matriz de Features extragaláctica

feature_importance = randomForest(X[extragal], y[extragal],300)
feature_importance
# Comparación con K-Neighbors

kNeighbors(X[extragal], y[extragal])
fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(metadata['target'] )
ax.set_title('amount of stars of each class')
ax.set_xlabel('class')

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()/len(metadata)*100), (p.get_x()+0.15, p.get_height()+1))

fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(metadata[extragal]['target'] )
ax.set_title('amount of stars of extragalactic classes')
ax.set_xlabel('class')

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()/len(metadata[extragal])*100), (p.get_x()+0.15, p.get_height()+1))

# Agregar las Meta-Features a la tabla

feature_table['hostgal_specz'] = metadata['hostgal_specz']
feature_table['hostgal_photoz'] = metadata['hostgal_photoz']
feature_table['hostgal_photoz_err'] = metadata['hostgal_photoz_err']
#feature_table['mwebv'] = metadata['mwebv']

# Agrega importancia al Redshift
feature_table['z2'] = metadata['hostgal_specz']**2
feature_table['z3'] = metadata['hostgal_specz']**3
feature_table['z1_higher'] = metadata['hostgal_specz']>1

feature_table.head()
X = pd.DataFrame(feature_table).copy()
y = pd.DataFrame(metadata['target'])
feature_importance = randomForest(X[extragal], y[extragal],300)
feature_importance
X_train, X_test, y_train, y_test = train_test_split(X[extragal], y[extragal], stratify=y[extragal],
                                               test_size = .10)
len(X_train), len(X_test), len(y_train), len(y_test)
sm = SMOTE()
X_res, y_res = sm.fit_sample(X_train, y_train)

X_res = pd.DataFrame(X_res,columns=X.columns)
y_res = pd.DataFrame(y_res,columns=['target'])
fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(y[extragal]['target'] )
ax.set_title('classes distribution before resampling')
ax.set_xlabel('class')
fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(y_res['target'] )
ax.set_title('classes distribution after resampling')
ax.set_xlabel('class')
randomForest(X_res,y_res)
clf = RandomForestClassifier(n_estimators=300, criterion='gini',\
                       oob_score=True, n_jobs=-1, random_state=42,\
                      verbose=1, class_weight='balanced', max_features='sqrt')
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues',
            xticklabels=extragal_targets, yticklabels=extragal_targets, 
            fmt='.0f',annot_kws={"color": 'black'})
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_aspect('equal')
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
feature_importance = pd.DataFrame(columns=X.columns)
feature_importance.loc[0] = clf.feature_importances_
feature_importance
lg = linear_model.LogisticRegression()

lg.fit(X_res, y_res)

y_pred = lg.predict(X_test)

confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues',
            xticklabels=extragal_targets, yticklabels=extragal_targets, 
            fmt='.0f',annot_kws={"color": 'black'})
ax.set_title('extragalactic logistic classifier')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_aspect('equal')
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
scores_para_df = []
for i in tqdm_notebook(range(1,40,1)):
    model = KNeighborsClassifier(n_neighbors=i)
    cv_scores = cross_val_score(model, X_res, y_res, cv=5)
    dict_row_score = {'score_medio':np.mean(cv_scores),'score_std':np.std(cv_scores),'n_neighbours':i}
    scores_para_df.append(dict_row_score)
df_scores = pd.DataFrame(scores_para_df)

df_scores['limite_inferior'] = df_scores['score_medio'] - df_scores['score_std']
df_scores['limite_superior'] = df_scores['score_medio'] + df_scores['score_std']
plt.plot(df_scores['n_neighbours'],df_scores['limite_inferior'],color='r')
plt.plot(df_scores['n_neighbours'],df_scores['score_medio'],color='b')
plt.plot(df_scores['n_neighbours'],df_scores['limite_superior'],color='r');

best_k = df_scores.loc[df_scores.score_medio == df_scores.score_medio.max(),'n_neighbours'].values
best_k = best_k[0]

plt.show()
print('Best K: ', best_k)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_res,y_res)

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues',
            xticklabels=extragal_targets, yticklabels=extragal_targets, 
            fmt='.0f',annot_kws={"color": 'black'})
plt.ylabel('Verdaderos')
plt.xlabel('Predichos');
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
randomForest(X,y)
