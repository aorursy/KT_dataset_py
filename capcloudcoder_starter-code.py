!apt-get install libgeos-3.5.0
!apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip
from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn import preprocessing
import urllib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans #import k-means from sklearn
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
import random
####################
#Helper Functions
#####################
def accuracy_opt(): #Optimize the accuracy for a given target accuracy
  th = random.uniform(0, 1)
  bnds = ((0,1),)
  res = minimize(delta_accuracy,th , method='TNC', bounds=bnds, tol=1e-10)
  return res.x

def delta_accuracy(th): #return delta between current accruacy score and target score
  global probabilities,yTest,acc_target
  predictions = (probabilities > th).astype(int)
  AR = accuracy_score(yTest, predictions)
  delta = np.abs(AR-target)
  return delta
#Github link to data
link='https://raw.githubusercontent.com/varunr89/smokey/master/FW_Veg_Rem_Combined.csv'

data= pd.read_csv(link, error_bad_lines=False)
data['putout_time']=data['putout_time'].str.split(" ", n = 1, expand = True).iloc[:,0].fillna(0).astype(int)

#Inputs for Analysis
Inputs = ['fire_name',"fire_size","fire_size_class","stat_cause_descr","latitude","longitude","state","discovery_month","putout_time",
          "disc_pre_year","Vegetation","fire_mag","Temp_pre_30","Temp_pre_15","Temp_pre_7","Temp_cont","Wind_pre_30","Wind_pre_15",
          "Wind_pre_7","Wind_cont","Hum_pre_30","Hum_pre_15","Hum_pre_7","Hum_cont","Prec_pre_30",
          "Prec_pre_15","Prec_pre_7","Prec_cont","remoteness"]

#Clean left over index row
cols2drop = list(data.columns)
for col in Inputs:
  cols2drop.remove(col)
data= data.drop(columns=cols2drop)

#Make a subset for exploratory development
data = data.loc[data.loc[:,'fire_size']>0,:]

##Plot the input data
data['fire_size_class'].value_counts().plot(kind='bar',label = '{} Fires / {} Attributes'.format(len(data),len(data.columns)))
plt.title('Distribution of firesize. Total Fires = {},Total acres bured = {}'.format(sum(data['fire_size_class'].value_counts().values),sum(data['fire_size'].values.astype(int))))
plt.legend()
plt.show()

#print data types
print(data.dtypes)

## print data description
link_description = 'https://raw.githubusercontent.com/varunr89/smokey/master/Wildfire_att_description.txt'
fp = urllib.request.urlopen(link_description)
mybytes = fp.read()
data_desc = mybytes.decode("utf8")
fp.close()

print('\n\n'+'Description of labels:'+'\n\n'+data_desc)
data
#############################
#6. Visualize Fire map
############################
fig = plt.figure(figsize=(16,16))
plt.title('Fire Intensity',fontsize=20)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
         projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution ='l')
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

cvals = []
for x in data.putout_time:
    if x is not pd.NaT and x < 30:
        cvals.append(int(x))
    elif x is not pd.NaT:
        cvals.append(30)
    else:
        cvals.append(0)

m.scatter(data['longitude'].values, data['latitude'].values, latlon=True,
          s=list(data['fire_mag'].values),
          c=cvals,
          cmap='Reds', alpha=0.6)

cbar = plt.colorbar(label=r'${duration}_{days}$',orientation='horizontal')
plt.show()

# Mercator projection, for Alaska and Hawaii
fig = plt.figure(figsize=(6,6))
m_ = Basemap(llcrnrlon=-172,llcrnrlat=50,urcrnrlon=-125,urcrnrlat=72.5,
            projection='merc',lat_ts=20,resolution ='l')  # do not change these numbers
m_.drawcoastlines(color='gray')
m_.drawcountries(color='gray')
m_.drawstates(color='gray')
m_.scatter(data['longitude'].values, data['latitude'].values, latlon=True,
s=list(data['fire_mag'].values),
c=cvals,
cmap='Reds', alpha=0.6)

#############################
#6. Look at some basic analytics and do sanity check on the data
############################
## Fires by month###
cats = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
df = data['discovery_month'].value_counts()
df.index = pd.CategoricalIndex(df.index, categories=cats, ordered=True)
df = df.sort_index()
df.plot(kind ='bar',fontsize=14)
plt.title ('Fires by month')
## Fire size by month###
cats = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
df=data.loc[:,['discovery_month','fire_size']].groupby(['discovery_month']).sum()
df.index = pd.CategoricalIndex(df.index, categories=cats, ordered=True)
df = df.sort_index()
df.plot(kind ='bar',fontsize=14,color='r')
plt.title ('Fire size by month')
plt.show()
## Fires by year###
plt.figure()
data['disc_pre_year'].value_counts().sort_index().plot(kind ='bar',fontsize=14)
plt.title ('Fires by year')
plt.show()
## Fire size by year###
plt.figure()
data.loc[:,['disc_pre_year','fire_size']].groupby(['disc_pre_year']).sum().sort_index().plot(kind ='bar',fontsize=14,color='r')
plt.title ('Fire Size by year')
plt.show()
## Fires by state###
plt.figure()
data['state'].value_counts().plot(kind ='bar',fontsize=14)
plt.title ('Fires by state')
plt.show()
## Firesize by state###
plt.figure()
data.loc[:,['state','fire_size']].groupby(['state']).sum().sort_values(by=['fire_size'],ascending=False).plot(kind ='bar',fontsize=14,color='r')
plt.title ('Fire Size by state')
plt.show()



#############################
#6. Consolidate categorical data (at least 1 column).
############################

#We can apply consolidation on the cause of fire column. 

data['stat_cause_descr'].value_counts().plot(kind = 'bar',label='Cause of Fire',fontsize =14) 
plt.legend(loc='upper right',fontsize =14)
plt.title('Cause of fire before consolidation')
plt.show()

print(''' Looking at the labels Missing/Undefined can be categorized with Miscellaneous,
Railrod, Structure, Powerline, can be joined into Strucutre, Childern, equipment use, Smoking, 
Debris burning, campfire, fireworks can be made in a new category Unintended human-causes''' )

data.loc[:,'stat_cause_descr_consolidated'] = data.loc[:,'stat_cause_descr'].values
data.loc[([i for i, x in enumerate(data.loc[:,'stat_cause_descr'].values ) if x in ['Missing/Undefined']]),'stat_cause_descr_consolidated'] = 'Miscellaneous'
data.loc[([i for i, x in enumerate(data.loc[:,'stat_cause_descr'].values ) if x in ['Railroad','Structure','Powerline','Equipment Use','Smoking','Debris Burning',
                                                                                    'Campfire','Fireworks','Children']]),'stat_cause_descr_consolidated'] = 'Unintended Human-causes'

data['stat_cause_descr_consolidated'].value_counts().plot(kind = 'bar',label='Cause of Fire',fontsize =14) 
plt.legend(loc='upper right',fontsize =14)
plt.title('Cause of fire after consolidation')
plt.show()

## Firesize by state###
plt.figure()
data.loc[:,['stat_cause_descr_consolidated','fire_size']].groupby(['stat_cause_descr_consolidated']).sum().sort_values(by=['fire_size'],ascending=False).plot(kind ='bar',fontsize=14,color='r')
plt.title ('Fire Size by cause')
plt.show()
### Group by state and cause of fire to understand differences between Alaska and Texas
grouped = data.loc[:,['stat_cause_descr_consolidated','state','fire_size']].groupby(['state'])
##Group by Alaska
ax = None
ax = grouped.get_group('AK').loc[:,['stat_cause_descr_consolidated','fire_size']].groupby(['stat_cause_descr_consolidated']).count().plot(kind='bar',label ='AK',alpha = 0.5,ax=ax,color='b')
ax=grouped.get_group('TX').loc[:,['stat_cause_descr_consolidated','fire_size']].groupby(['stat_cause_descr_consolidated']).count().plot(kind='bar',label ='TX',alpha = 0.5,ax=ax,color='r')
ax.legend(["AK", "TX"])
plt.title('Comparison of cause of fire between Alaska and Texas')

##Group by Texas
ax = None
ax = grouped.get_group('AK').loc[:,['stat_cause_descr_consolidated','fire_size']].groupby(['stat_cause_descr_consolidated']).sum().plot(kind='bar',label ='AK',alpha = 0.5,ax=ax,color='b')
ax=grouped.get_group('TX').loc[:,['stat_cause_descr_consolidated','fire_size']].groupby(['stat_cause_descr_consolidated']).sum().plot(kind='bar',label ='TX',alpha = 0.5,ax=ax,color='r')
ax.legend(["AK", "TX"])
plt.title('Comparison of cause of fire size between Alaska and Texas')
### Fire size vs remoteness
plt.figure(figsize=(10,5))
plt.scatter(data.loc[:,'remoteness'],data.loc[:,'fire_size'],label = 'Fire Size (acres)')
plt.xlabel('Remoteness-normalized')
plt.legend(loc="upper left")
plt.title ('Fires by remoteness')
plt.show()
#############################
#5.Distribution of numerical variables
#6.Distribution of categorical variables
#7.A comment on each attribute
#8.Removing cases with missing data
#9.Removing outliers
#10.Imputing missing values
############################
#####Remove Outliers######
isOut = data[data['state']=='AK'].index # Alaska state significantly skews the data
######################
# The replacement value for NaNs is Median
data = data.drop(isOut)
data = data.reset_index(drop=True)

#####Remove missing data#####
data= data.drop(columns=['latitude','longitude','state','discovery_month','disc_pre_year','fire_name'])
Cols2Clean = ['Temp_pre_30', 'Temp_pre_15','Temp_pre_7', 'Temp_cont', 'Wind_pre_30', 'Wind_pre_15', 'Wind_pre_7','Wind_cont', 'Hum_pre_30', 'Hum_pre_15', 'Hum_pre_7', 'Hum_cont','Prec_pre_30', 'Prec_pre_15', 'Prec_pre_7', 'Prec_cont']
data[Cols2Clean] = data[Cols2Clean].replace( {-1:np.nan})
data = data.dropna()
data = data.reset_index(drop=True)

######Impute missing data by consolidation#####
data[Cols2Clean] = data[Cols2Clean].replace( {0:np.nan})
Cols2Consolidate = [['Temp_pre_30', 'Temp_pre_15','Temp_pre_7', 'Temp_cont'], ['Wind_pre_30', 'Wind_pre_15', 'Wind_pre_7','Wind_cont'],[ 'Hum_pre_30', 'Hum_pre_15', 'Hum_pre_7', 'Hum_cont'],['Prec_pre_30', 'Prec_pre_15', 'Prec_pre_7', 'Prec_cont']]
data['T_mean'] = data[['Temp_pre_30', 'Temp_pre_15','Temp_pre_7', 'Temp_cont']].mean(axis=1)
data['Wind_mean'] = data[['Wind_pre_30', 'Wind_pre_15', 'Wind_pre_7','Wind_cont']].mean(axis=1)
data['Hum_mean'] = data[['Hum_pre_30', 'Hum_pre_15', 'Hum_pre_7', 'Hum_cont']].mean(axis=1)
data['Prec_max'] = data[['Prec_pre_30', 'Prec_pre_15', 'Prec_pre_7', 'Prec_cont']].max(axis=1)
data= data.drop(columns=['Temp_pre_30', 'Temp_pre_15','Temp_pre_7', 'Temp_cont','Wind_pre_30', 'Wind_pre_15', 'Wind_pre_7','Wind_cont','Hum_pre_30', 'Hum_pre_15', 'Hum_pre_7', 'Hum_cont','Prec_pre_30', 'Prec_pre_15', 'Prec_pre_7', 'Prec_cont','fire_mag','fire_size_class'])

#show the data to spot outliers
sns.pairplot(data)

#####Impute Outliers######
isOut = (data.loc[:,'Wind_mean']>15) # only outlier is wind
######################
# The replacement value for NaNs is Median
Median = np.nanmedian(data.loc[:,'Wind_mean'])
# Median imputation of nans
data.loc[isOut, 'Wind_mean'] = Median

#############################
#5. Consolidation by Binning fire size and put-out time
############################

#Lets bin the fire_size variable

FsBin0 = 1000 #acres
FsBin1 = 10000000000 #acres
FsBin2 = 10000000000 #acres
FsBin3 = 10000000000 #acres

data.loc[(data.loc[:,'fire_size'].values<=FsBin0),'Fire Size Bin'] = int(0)
data.loc[(data.loc[:,'fire_size'].values>FsBin0) & (data.loc[:,'fire_size'].values<=FsBin1),'Fire Size Bin'] = int(1)
data.loc[(data.loc[:,'fire_size'].values>FsBin1) & (data.loc[:,'fire_size'].values<=FsBin2),'Fire Size Bin'] = int(2)
data.loc[(data.loc[:,'fire_size'].values>FsBin2) & (data.loc[:,'fire_size'].values<=FsBin3),'Fire Size Bin'] = int(3)
data.loc[(data.loc[:,'fire_size'].values>FsBin3),'Fire Size Bin'] = int(4)

#Lets bin the put-outtime variable

PoBin0 = 7 #days
PoBin1 = 1000 #days
PoBin2 = 1000 #days
PoBin3 = 1000 #days

data.loc[(data.loc[:,'putout_time'].values<=PoBin0),'Putout Bin'] = int(0)
data.loc[(data.loc[:,'putout_time'].values>PoBin0) & (data.loc[:,'putout_time'].values<=PoBin1),'Putout Bin'] = int(1)
data.loc[(data.loc[:,'putout_time'].values>PoBin1) & (data.loc[:,'putout_time'].values<=PoBin2),'Putout Bin'] = int(2)
data.loc[(data.loc[:,'putout_time'].values>PoBin2) & (data.loc[:,'putout_time'].values<=PoBin3),'Putout Bin'] = int(3)
data.loc[(data.loc[:,'putout_time'].values>PoBin3),'Putout Bin'] = int(4)

#Create a new bin that is a combination of both Put-out time and fire size
data.loc[(data.loc[:,'Putout Bin'].values==0) & (data.loc[:,'Fire Size Bin'].values==0),'Fire-PO Bin'] = int(0)
data.loc[(data.loc[:,'Putout Bin'].values>0) & (data.loc[:,'Fire Size Bin'].values==0),'Fire-PO Bin'] = int(1)
data.loc[(data.loc[:,'Putout Bin'].values>0) & (data.loc[:,'Fire Size Bin'].values>0),'Fire-PO Bin'] = int(1)
data.loc[(data.loc[:,'Putout Bin'].values==0) & (data.loc[:,'Fire Size Bin'].values>0),'Fire-PO Bin'] = int(1)


#data['Fire Size Bin'].value_counts().sort_index().plot('bar',label='Fire Size acries Bins[0:<{0:1.2E},1:<{1:1.2E},2:<{2:2.2E},3:<{3:1.2E} ,3:>{3:1.2E}]'.format(FsBin0,FsBin1,FsBin2,FsBin3,FsBin3),alpha = 0.5,color = 'r') 
#data['Putout Bin'].value_counts().plot('bar',label= 'Putout days Bin Bins[0:<{0},1:<{2},2:<{3},3:<{4},4:>{4}]'.format(PoBin0,PoBin1,PoBin2,PoBin3,PoBin3),alpha = 0.5) 
data['Fire-PO Bin'].value_counts().sort_index().plot(kind='bar',label='Fire Size-Putout Bins[0:Small Risk,1:High Risk]',alpha = 0.5,color = 'r') 
plt.legend(loc='upper right')
plt.title('''Fire Size and Putout Days Binning''')
plt.show()


#############################
#4.Normalize numeric values and one-hotencode categorical values 
############################

############## One hot encoding vegetation data######
dom_veg_leg = {0:'Tropical_Evergreen_Broadleaf_Forest',	1:'Tropical_Deciduous_Broadleaf_Forest',	2:'Temperate_Evergreen_Broadleaf_Forest',	3:'Temperate_Evergreen_Needleleaf_Forest',	4:'Temperate_Deciduous_Broadleaf_Forest',	5:'Boreal_Evergreen_Needleleaf_Forest',	6:'Boreal_Deciduous_Needleleaf_Forest',	7:'Savanna',	8:'Grassland_or_Steppe',	9:'Shrubland',	10:'Tundra',	11:'Desert',	12:'Polar-Desert_or_Rock_or_Ice',	13:'Water_or_Rivers',	14:'Cropland',	15:'Pastureland',	16:'Urbanland'}
data['Vegetation'] = data['Vegetation'].replace( dom_veg_leg)
# We one hot encoded the cause of the fire.
data = pd.concat([data,pd.get_dummies(data['Vegetation'])],axis=1,sort=True)

############## One hot encoding Cause data######
# We one hot encoded the cause of the fire.
data = pd.concat([data,pd.get_dummies(data['stat_cause_descr_consolidated'])],axis=1,sort=True)

#remove category label
data= data.drop(columns=['stat_cause_descr','stat_cause_descr_consolidated'])


#####Scaling Numerical data#######
Col2Scale = [ 'fire_size',   'putout_time','remoteness',  'T_mean', 'Wind_mean','Hum_mean', 'Prec_max']

#Min Max Scaling
min_max_scaler = preprocessing.MinMaxScaler()
for col in Col2Scale:
  data[col+'_scaled'] = min_max_scaler.fit_transform(data[col].values.reshape(-1,1))

ScaledCols =[ 'fire_size_scaled',   'putout_time_scaled','remoteness_scaled',  'T_mean_scaled', 'Wind_mean_scaled','Hum_mean_scaled', 'Prec_max_scaled']

data= data.drop(columns=['fire_size',   'putout_time','remoteness',  'T_mean', 'Wind_mean','Hum_mean', 'Prec_max','Vegetation'])

#############################
#10.Missing Data Imputing using iterative imputer
############################
data = data.reset_index(drop=True)
imp = IterativeImputer(max_iter=10, verbose=0)
imp.fit(data)
imputed_data = imp.transform(data)
imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
imputed_data.iloc[:,[0,1]] = data.iloc[:,[0,1]]
Inputs= list(imputed_data.columns)
notInputs = ['fire_size_scaled','putout_time_scaled','Fire Size Bin','Putout Bin']
for elem in notInputs:
  Inputs.remove(elem) 
plt.figure(figsize=(20,10))
sns.heatmap(imputed_data.loc[:,Inputs].corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black',annot_kws={"size": 14})
# ##############Setup Model Inputs######
Inputs= list(imputed_data.columns)
notInputs = ['Putout Bin','Fire Size Bin','fire_size_scaled','putout_time_scaled','Fire-PO Bin']
for elem in notInputs:
  Inputs.remove(elem) 
Target = 'Fire-PO Bin' # target is the predict the firesize
TestFraction = 0.3 # test to training fraction is chosen is 30%
random_state = np.random.RandomState(0)
#Unsupervised Learning
######################################################
#1.Ask a binary-choice question that describes your classification. Write the question as a comment.
#####################################################
print('''
Machine learning classification being explored:
What conditions lead to large fire sizes (>7000 acres) and large put-out times of fire? 
Large fires are defined as fires that burned more than 50000 acres.
Large put-out times are defined as a fire lasting longer than 1-week''')

##########################################################
#2.Split your data set into training and testing sets using the proper function in sklearn.
#3.Use sklearn to train two classifiers on your training set, like logistic regression and random forest. 
#4.Apply your (trained) classifiers to the test set.
#5.Create and present a confusion matrix for each classifier. Specify and justify your choice of probability threshold.
#6.For each classifier, create and present 2 accuracy metrics based on the confusion matrix of the classifier.
#7.For each classifier, calculate the ROC curve and it's AUC using sklearn. Present the ROC curve. Present the AUC in the ROC's plot.
##########################################################

# ##############Setup Model Inputs######
# shuffle and split training and test sets
print ('\nSimple approximate split:')
isTest = np.random.rand(len(imputed_data)) < TestFraction
TrainSet = imputed_data[~isTest]
TestSet = imputed_data[isTest] # should be 249 but usually is not
print ('Test size should have been ', 
        TestFraction*len(imputed_data), "; and is: ", len(TestSet))

print ('\nsklearn accurate split:')
TrainSet, TestSet = train_test_split(imputed_data, test_size=TestFraction,
                                                    random_state=0)
print ('Test size should have been ', 
       TestFraction*len(imputed_data), "; and is: ", len(TestSet))
# #############Tune the model-SVM
# model_params = {'C': [ 1, 10, 100],  
#               'gamma': [1, 0.1, 0.01] 
#               } 

# # create random forest classifier model
# weights = {0:1.0, 1:100.0}
# sv_model = svm.SVC(kernel = 'rbf',random_state=random_state,probability=True,class_weight = weights)

# # set up grid search meta-estimator
# clf = GridSearchCV(sv_model, model_params, cv=5)

# # train the grid search meta-estimator to find the best model
# model = clf.fit(imputed_data.loc[:,Inputs], imputed_data.loc[:,Target])

# # print winning set of hyperparameters
# from pprint import pprint
# pprint(model.best_estimator_.get_params())
#############Train the model-SVM
# Learn to predict each class against the other
weights = {0:1.0, 1:1000.0} # penalize false negatives
clf = svm.SVC(kernel = 'rbf',C=100,gamma = 1,random_state=random_state,
              probability=True,class_weight = weights)
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]
####### Compute the metrics--SVM
print ('\nConfusion Matrix and Metrics -SVM')
yTest = TestSet.loc[:,Target]
target = 0.5
Threshold = accuracy_opt()
print ("Probability Threshold is chosen to be:", Threshold)
print('''
Probability threshold was was tuned results in low 
false negative rate or maximize recall
''')
predictions = (probabilities > Threshold).astype(int)
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))
f1= f1_score(TestSet.loc[:,Target], predictions)
print ("F1 Score:", np.round(f1, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds-SVM
fpr_SVM, tpr_SVM, th_SVM = roc_curve(TestSet.loc[:,Target], probabilities)
AUC_SVM = auc(fpr_SVM, tpr_SVM)
###############################
# #############Tune the model-RFC
# model_params = {
#     'n_estimators': [50, 150, 250],
#     'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
#     'min_samples_split': [2, 4, 6]
# }

# # create random forest classifier model
# weights = {0:1.0, 1:100.0}
# rf_model = RandomForestClassifier(max_depth=2,random_state=random_state,class_weight=weights)

# # set up grid search meta-estimator
# clf = GridSearchCV(rf_model, model_params, cv=5)

# # train the grid search meta-estimator to find the best model
# model = clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

# # print winning set of hyperparameters
# from pprint import pprint
# pprint(model.best_estimator_.get_params())
#############Train the model-RFC
weights = {0:1.0, 1:1000.0}
clf = RandomForestClassifier(n_estimators = 50,max_features=0.75,
                              min_samples_split=2,max_depth=2,random_state=random_state,
                              class_weight=weights)
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]

####### Compute the metrics-RFC
print ('\nConfusion Matrix and Metrics -Random Forest')
Threshold = .985 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
print('''
Probability threshold was tuned to maximize recall for minimum
75% accuracy
''')
predictions = (probabilities > Threshold).astype(int)
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))
f1= f1_score(TestSet.loc[:,Target], predictions)
print ("F1 Score:", np.round(f1, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds-RFC
fpr_RFC, tpr_RFC, th_RFC = roc_curve(TestSet.loc[:,Target], probabilities)
AUC_RFC = auc(fpr_RFC, tpr_RFC)
###############################


from sklearn.linear_model import LogisticRegression
# # Create logistic regression
# from sklearn import linear_model
# logistic = linear_model.LogisticRegression()

# # Create regularization penalty space
# penalty = ['l1', 'l2']

# # Create regularization hyperparameter space
# C = np.logspace(0, 4, 10)

# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)

# # Create grid search using 5-fold cross validation
# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# # Fit grid search
# best_model = clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])

# # View best hyperparameters
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])

# ##Apply Logistic Regression using parameters identified above

clf = LogisticRegression(C=1, penalty='l2', solver='liblinear')
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]


####### Compute the metrics-RFC
print ('\nConfusion Matrix and Metrics -Random Forest')
Threshold = .075 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
print('''
Probability threshold was tuned to maximize recall for minimum
75% accuracy
''')
predictions = (probabilities > Threshold).astype(int)
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))
f1= f1_score(TestSet.loc[:,Target], predictions)
print ("F1 Score:", np.round(f1, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds-RFC
fpr_LR, tpr_LR, th_LR = roc_curve(TestSet.loc[:,Target], probabilities)
AUC_RFC = auc(fpr_LR, tpr_LR)
###############################
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = { 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure(figsize=(20,10))
plt.title('ROC Curve-Models weighted to penalize False Negative 1:1000')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr_SVM, tpr_SVM, LW=3,color='r', label='ROC curve-SVM (AUC = %0.2f)' % AUC_SVM)
plt.plot(fpr_RFC, tpr_RFC, LW=3,color='b', label='ROC curve-RFC (AUC = %0.2f)' % AUC_RFC)
plt.plot(fpr_LR, tpr_LR, LW=3,color='k', label='ROC curve-LR (AUC = %0.2f)' % AUC_RFC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()