import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.color_palette("colorblind", n_colors=8, desat=.5)
plt.style.use('tableau-colorblind10')

#This function customizes a bar plot bar object to add the number of observations.
def set_bar_label(ax_ref, minval=1, orient='h'):
  for p in ax_ref.patches:
    if orient=='h':    
      if p.get_width() >=minval:
        ax_ref.annotate('{}'.format(p.get_width()),xy=(p.get_x()+p.get_width()*1.01,p.get_y()*1.05),fontsize=13)
    else:
      if p.get_height() >=minval:
          ax_ref.annotate('{}'.format(p.get_height()),xy=(p.get_x()+0.4*p.get_width(), p.get_y() + p.get_height()*1.01),fontsize=13)


UCSDB=pd.read_excel('../input/ucs-sat-db/UCS-Satellite-Database-8-1-2020.xls',index_col='Date of Launch',parse_dates=True,thousands=',')
UCSDB.index.name='Date of Launch'
UCSDB.info()
UCSDB.head(2)
UCSDB.tail(2)
#remove/drop rows with nan,
UCSDB = UCSDB[ UCSDB['Name of Satellite, Alternate Names'].notna() ]
#drop columns with fewer than 5 valid items
UCSDB.dropna( axis='columns', thresh=5, inplace=True)

#correct for extra spaces on category columns
UCSDB['Users']=UCSDB['Users'].str.strip()
#ensure numeric columns are of the correct type
UCSDB[['Dry Mass (kg.)', 'Launch Mass (kg.)', 'Eccentricity', 'Inclination (degrees)','Period (minutes)', 'Power (watts)']]=\
    UCSDB[['Dry Mass (kg.)', 'Launch Mass (kg.)', 'Eccentricity', 'Inclination (degrees)','Period (minutes)', 'Power (watts)']]\
                                                                                            .apply(pd.to_numeric,errors='coerce')
#sort the index.
UCSDB.sort_index(axis=0, inplace=True, ascending=True)
UCSDB.info()
## identify ZERO in the mass or power columns
isZeroPower_idx = UCSDB['Power (watts)'] == 0
isZeroDryMass_idx = UCSDB['Dry Mass (kg.)'] == 0
isZeroLaunchMass_idx = UCSDB['Launch Mass (kg.)'] == 0

print(' Number of entries in columns Power, Launch Mass and Dry Mass set to zero')
print('-------------------------------------------------------------------------')
print(' Power col: {}'.format(isZeroPower_idx.sum()))
print(UCSDB[isZeroPower_idx]['Name of Satellite, Alternate Names'].to_string())
print('-------------------------------------------------------------------------')
print(' Dry Mass col: {}'.format(isZeroDryMass_idx.sum()))
print(' Launch Mass col: {}'.format(isZeroLaunchMass_idx.sum()))
# Set the POWER (Watts) value for NSS-6 to 10000
UCSDB.loc[isZeroPower_idx,'Power (watts)'] = 10000
print(UCSDB[isZeroPower_idx]['Power (watts)'].to_string())
print(UCSDB.iloc[0,0:5].to_string())
print(UCSDB.iloc[-1,0:5].to_string())
print(UCSDB[(UCSDB['Class of Orbit']=='GEO') & (UCSDB['Users']=='Commercial')].iloc[0,0:8].to_string())
heaviest_sat = UCSDB[['Dry Mass (kg.)']].idxmax()
print(UCSDB.loc[heaviest_sat,['Name of Satellite, Alternate Names','Operator/Owner','Class of Orbit','Dry Mass (kg.)']].to_string())
# Determine number of satellites per country of registration and operator, for all countries
# The series returned contains the sum of satellites for all countries, sorted by value
#Note: We should remove Not registered (NR) entries from the column 'Country/Org of UN Registry' before making the count.

nbr_sats_per_cnt_reg=UCSDB['Country/Org of UN Registry'].value_counts()
nbr_sats_per_cnt_op=UCSDB['Country of Operator/Owner'].value_counts()

print('Array sizes: Per country of registration {} per country of operator {}'.format(nbr_sats_per_cnt_reg.shape, nbr_sats_per_cnt_op.shape ))
# Number of satellites based on Country of Registry, ranked for top 5
print('Satellite total count (top 5), per country of UN registration')
print('-------------------------------------------------------------')
print(UCSDB[UCSDB['Country/Org of UN Registry'].str.contains("NR")==0]['Country/Org of UN Registry'].value_counts()[:5].to_string())
# Number of satellites based on Country of Operator/Owner, ranked for top 5
print('Satellite total count (top 5), per country of Operator')
print('------------------------------------------------------')
print(UCSDB['Country of Operator/Owner'].value_counts()[:5].to_string())
# a bar plot for the top five countries, 
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(pad=0.0)
fig.set_size_inches(14,6)
ax1=UCSDB[UCSDB['Country/Org of UN Registry'].str.contains("NR")==0]['Country/Org of UN Registry'].value_counts()[:5].plot(kind='barh',ax=ax[0])
ax2=nbr_sats_per_cnt_op[0:5].plot(kind='barh')
plt.subplots_adjust(right=1.5,wspace=0.5)
# set a legend, title
ax1.set_xlabel('Number of satellites in database', fontsize=18)
ax1.set_title('Total number of satellites, per country of UN Registry', fontsize=20)
ax2.set_xlabel('Number of satellites in database', fontsize=18)
ax2.set_title('Total number of satellites, per country of Operator', fontsize=20)

#annotate
set_bar_label(ax1)
set_bar_label(ax2)
axTen = UCSDB['Launch Site'].value_counts()[:10].plot(kind='barh', color='purple',width=0.8, figsize=(8,8))
#axTen = sns.countplot(y=UCSDB['Launch Site'],order=pd.value_counts(UCSDB['Launch Site']).iloc[:10].index)
set_bar_label(axTen, orient='h')
_=axTen.set_title('Total number of launches, per launch site (top 10)', fontsize=20)

nbr_sats_per_launch_site=UCSDB['Launch Site'].value_counts()
print('List of top 10 launch sites, sorted by number of satellites launched')
print('--------------------------------------------------------------------')
print(nbr_sats_per_launch_site[0:10].to_string())
nbr_sats_per_launch_vehicle=UCSDB['Launch Vehicle'].value_counts()
print('List of launchers by number of satellites launched total, top 10')
print('---------------------------------------------------------------')
print(nbr_sats_per_launch_vehicle[0:10].to_string())
BySite_group = UCSDB.groupby(['Launch Site','Launch Vehicle']) 
nbr_Starlink_launched_from_cc = BySite_group.get_group(('Cape Canaveral','Falcon 9'))['Name of Satellite, Alternate Names'].str.contains('tarlink').sum()
nbr_NonStarlink_launched_from_cc = BySite_group.get_group(('Cape Canaveral','Falcon 9'))['Name of Satellite, Alternate Names'].str.contains('^((?!tarlink).)*$').sum()

print(' - number of Starlink satellites launched with Falcon 9 from Cape Canaveral : ', nbr_Starlink_launched_from_cc )
print(' - number of non-Starlink satellites launched with Falcon 9 from Cape Canaveral : ', nbr_NonStarlink_launched_from_cc )
print(' - total launched from Cape Canaveral using Falcon 9: ', nbr_Starlink_launched_from_cc+nbr_NonStarlink_launched_from_cc)
#To modify the code to see, say, only the first 10 launches, add [0:11] after the columns names in the print () function.
ByLaunchSite = UCSDB.groupby(['Launch Vehicle'])
display(ByLaunchSite.get_group('Falcon 9')[['Launch Site','Name of Satellite, Alternate Names']][0:16])
fig_class, ax_class = plt.subplots(nrows=1, ncols=1)
fig_class.set_size_inches(10,10)
ax_class.tick_params(axis='x', labelsize=14)
ax_class.tick_params(axis='y', labelsize=14)

axClassBar = sns.countplot(UCSDB['Class of Orbit'])
axClassBar.set_xlabel('Class of orbit',fontsize=18)
_=axClassBar.set_title('Total number of satellites, per class of orbit', fontsize=20)
set_bar_label(axClassBar,minval=1, orient='v')

cross_tab = pd.crosstab( UCSDB['Users'], UCSDB['Class of Orbit'],margins=True)
display(cross_tab)
gb=UCSDB.groupby(['Class of Orbit','Users'])
#compute the total per class of orbit, using direct computation methods. Note that this can also be accomplished through filter and logical indexing
nbr_GEO_comm_sats = gb.get_group(('GEO','Commercial'))['Name of Satellite, Alternate Names'].count() + \
              gb.get_group(('GEO','Commercial/Military'))['Name of Satellite, Alternate Names'].count() + \
              gb.get_group(('GEO','Commercial/Government'))['Name of Satellite, Alternate Names'].count()

nbr_MEO_comm_sats = gb.get_group(('MEO','Commercial'))['Name of Satellite, Alternate Names'].count() #+ \
             # gb.get_group(('MEO','Military/Commercial'))['Name of Satellite, Alternate Names'].count()

nbr_LEO_comm_sats = gb.get_group(('LEO','Commercial'))['Name of Satellite, Alternate Names'].count() + \
              gb.get_group(('LEO','Commercial/Civil'))['Name of Satellite, Alternate Names'].count() + \
              gb.get_group(('LEO','Government/Commercial'))['Name of Satellite, Alternate Names'].count() +\
              gb.get_group(('LEO','Military/Commercial'))['Name of Satellite, Alternate Names'].count()

print('Totals by class of orbit ')
print('-------------------------------------------------')
print('Number of GEO satellites for commercial use: ', nbr_GEO_comm_sats)
print('Number of MEO satellites for commercial use: ', nbr_MEO_comm_sats)
print('Number of LEO satellites for commercial use: ', nbr_LEO_comm_sats)


#to make this count work, it is important to recall that there are certain entries in the database of  "mixed use", that is,
# a satellite with payloads serving dual purposes such as commercial/military, or government/military. This is what i used a direct sum of each category in the previous cells.
#now i iwll try a logical indexing approach using the str.contains() method and looking for the keywords commercial and communication inside the columns Users and Purpose.

idx_isLEO_isCOM_isCOMMS = (UCSDB['Class of Orbit']=='LEO') & (UCSDB['Users'].str.contains('commercial',case=False) ) & (UCSDB['Purpose'].str.contains('communication',case=False))
idx_isMEO_isCOM_isCOMMS = (UCSDB['Class of Orbit']=='MEO') & (UCSDB['Users'].str.contains('commercial',case=False) ) & (UCSDB['Purpose'].str.contains('communication',case=False))
idx_isGEO_isCOM_isCOMMS = (UCSDB['Class of Orbit']=='GEO') & (UCSDB['Users'].str.contains('commercial',case=False) ) & (UCSDB['Purpose'].str.contains('communication',case=False))
#the number of entries is the sum of the boolean indexing series.
nbr_LEO_com_comms_sats =idx_isLEO_isCOM_isCOMMS.sum()
nbr_MEO_com_comms_sats =idx_isMEO_isCOM_isCOMMS.sum()
nbr_GEO_com_comms_sats =idx_isGEO_isCOM_isCOMMS.sum()
print('Total number of commercial communications satellites, by class of orbit ')
print('------------------------------------------------------------------------')
print('Number of LEO satellites for commercial communications use: ', nbr_LEO_com_comms_sats)
print('Number of MEO satellites for commercial communications use: ', nbr_MEO_com_comms_sats)
print('Number of GEO satellites for commercial communications use: ', nbr_GEO_com_comms_sats)

#a bar chart
fbar,barax=plt.subplots()
fbar.set_size_inches(7,10)
barax.bar( ['LEO', 'MEO', 'GEO'], [nbr_LEO_com_comms_sats, nbr_MEO_com_comms_sats, nbr_GEO_com_comms_sats], color=['r', 'g','b'])
#annotate bars,
set_bar_label(barax,minval=1, orient='v')
barax.set_ylabel('Number of satellites for commercial communications purpose', fontsize=14)
plt.show()
R_earth=6378.165  #in kilometres, mean radius of the earth.
UCSDB['Mean Orbital Speed (km.sec)'] = ((2*np.pi*( (UCSDB['Perigee (km)'] + UCSDB['Apogee (km)'] +2*R_earth)/2)) / (UCSDB['Period (minutes)']*60))*(1-0.25*UCSDB['Eccentricity']**2)
display( UCSDB[['Name of Satellite, Alternate Names','Class of Orbit','Mean Orbital Speed (km.sec)']].head(6))

#boolean criteria for filtering

criteria_LEO = UCSDB['Class of Orbit'] == 'LEO'
criteria_MEO = UCSDB['Class of Orbit'] == 'MEO'
criteria_GEO = UCSDB['Class of Orbit'] == 'GEO'

#mean of mean orbital speed for any non-null or non-zero entries for LEO satellites
print(' ------------------Mean orbital speed : descriptive statistics LEO--------------------------------- ')
Mean_Orbital_speed_LEO = UCSDB[criteria_LEO]['Mean Orbital Speed (km.sec)'].apply(pd.to_numeric,errors='coerce').dropna().mean()
print(UCSDB[criteria_LEO][ 'Mean Orbital Speed (km.sec)'].dropna().describe().to_string())
print( 'Mean orbital speed for LEO satellites with complete parameter set in UCSDB: {0:6.2f} km/sec.'.format ( Mean_Orbital_speed_LEO) )

##mean for any non-null or non-zero entries for MEO satellites
print('--------------------Mean orbital speed : descriptive statistics MEO--------------------------------------')
Mean_Orbital_speed_MEO = UCSDB[criteria_MEO]['Mean Orbital Speed (km.sec)'].apply(pd.to_numeric,errors='coerce').dropna().mean()
print(UCSDB[criteria_MEO]['Mean Orbital Speed (km.sec)'].dropna().describe().to_string())
print( 'Mean orbital speed for MEO satellites with complete parameter set in UCSDB: {0:6.2f} km/sec.'.format ( Mean_Orbital_speed_MEO) )

##mean for any non-null or non-zero entries for GEO satellites
print('--------------------Mean orbital speed : descriptive statistics GEO--------------------------------------')
Mean_Orbital_speed_GEO = UCSDB[criteria_GEO]['Mean Orbital Speed (km.sec)'].apply(pd.to_numeric,errors='coerce').dropna().mean()
print(UCSDB[criteria_GEO]['Mean Orbital Speed (km.sec)'].dropna().describe().to_string())
print( 'Mean orbital speed for GEO satellites with complete parameter set in UCSDB: {0:6.2f} km/sec.'.format ( Mean_Orbital_speed_GEO) )
# use sns boxplot with hue class of orbit directly, without splitting
sns.set()
fig_mp, ax_mp = plt.subplots(nrows=1, ncols=3)
fig_mp.tight_layout(pad=0.5)
fig_mp.set_size_inches(40,18)
# define outlier properties
flierprops = dict(marker='s', markersize=10)
b = sns.boxplot(x='Class of Orbit', y='Launch Mass (kg.)', data=UCSDB,  ax=ax_mp[0], linewidth=3, flierprops=flierprops, showmeans=True, whis=[5,95]).set(yticks=np.arange(0,20000,1000))
b2 = sns.boxplot(x='Class of Orbit', y='Dry Mass (kg.)', data=UCSDB,  ax=ax_mp[1], linewidth=3,  flierprops=flierprops, showmeans=True, whis=[5,95]).set(yticks=np.arange(0,6500,500))
b3 = sns.boxplot(x='Class of Orbit', y='Power (watts)', data=UCSDB, ax=ax_mp[2], linewidth=3, showmeans=True, flierprops=flierprops, whis=[5,95]).set(yticks=np.arange(0,22500,2500))

#Figure customization
for axx in ax_mp:
  axx.tick_params(axis='x', labelsize=20)
  axx.tick_params(axis='y', labelsize=20)
  axx.set_xlabel('Class of orbit',fontsize=24)
  l = axx.get_ylabel()
  axx.set_ylabel(l, fontsize=20)

ax_mp[0].set_title('Launch Mass (kg.) comparison, by class of orbit',fontsize=24)
ax_mp[1].set_title('Dry Mass (kg.) comparison, by class of orbit',fontsize=24)
_ = ax_mp[2].set_title('Power (watts) comparison, by class of orbit',fontsize=24)
criteria_LEO = UCSDB['Class of Orbit'] == 'LEO'
print('Median dry mass for a LEO satellite: {0:6.2f} kg.'.format(UCSDB[criteria_LEO]['Dry Mass (kg.)'].dropna().median()))
print('Median power generating capability for a LEO satellite: {0:6.2f} Watt.'.format( UCSDB[criteria_LEO]['Power (watts)'].dropna().median()))
print('----------------------------------------------------------------------')
#Extract the same data for MEO satellites
criteria_MEO = UCSDB['Class of Orbit'] == 'MEO'
print('Median dry mass for a MEO satellite: {0:6.2f} kg.'.format(UCSDB[criteria_MEO]['Dry Mass (kg.)'].dropna().median()))
print('Median power generating capability for a MEO satellite: {0:6.2f} Watt.'.format( UCSDB[criteria_MEO]['Power (watts)'].dropna().median()))
print('----------------------------------------------------------------------')
#Extract the same data for GEO satellites
criteria_GEO = UCSDB['Class of Orbit'] == 'GEO'
print('Median dry mass for a GEO satellite: {0:6.2f} kg.'.format(UCSDB[criteria_GEO]['Dry Mass (kg.)'].dropna().median()))
print('Median power generating capability for a GEO satellite: {0:6.2f} Watt.'.format( UCSDB[criteria_GEO]['Power (watts)'].dropna().median()))
UCSDB[criteria_LEO][['Name of Satellite, Alternate Names','Dry Mass (kg.)']][UCSDB[criteria_LEO]['Dry Mass (kg.)']>2000]
#create a copy of the data in the UCSDB database, containing the columns needed for the analysis : the name, class of orbit, Mass, power. 
SAT_DM_PWR = UCSDB[['Name of Satellite, Alternate Names','Class of Orbit','Dry Mass (kg.)',	'Power (watts)' ]]
SAT_LM_PWR = UCSDB[['Name of Satellite, Alternate Names','Class of Orbit','Launch Mass (kg.)','Power (watts)' ]]

print('Number of NaNs in POWER column : {}'.format(SAT_DM_PWR['Power (watts)'].isna().sum()))
print('Number of NaNs in Dry Mass column : {}'.format(SAT_DM_PWR['Dry Mass (kg.)'].isna().sum()))
print('Number of NaNs in Launch Mass column : {}'.format(SAT_LM_PWR['Launch Mass (kg.)'].isna().sum()))

#SAT_LM_PWR, SAT_DM_PWR contain NaN entries in numeric columns
print('Shape before removing NaNs:')
display( SAT_DM_PWR.shape)
display( SAT_LM_PWR.shape)

print('Shape after removing NaNs:')
SAT_DM_PWR = SAT_DM_PWR.dropna()
SAT_LM_PWR = SAT_LM_PWR.dropna()
display( SAT_DM_PWR.shape)
display( SAT_LM_PWR.shape)

#remove true duplicates on each dataframe
print('--------------------------------------------------------------------------------------------')
print('Shape after removing duplicates:')
SAT_DM_PWR = SAT_DM_PWR.drop_duplicates(subset=['Dry Mass (kg.)','Power (watts)'], keep='first')
SAT_LM_PWR = SAT_LM_PWR.drop_duplicates(subset=['Launch Mass (kg.)','Power (watts)'], keep='first')
display( SAT_DM_PWR.shape)
display( SAT_LM_PWR.shape)
print('-------------------------------------------------------------------------------------------')
print('Data samples to be used for modelling: ')
print('Dry Mass - Power pairs : {}'.format(SAT_DM_PWR.groupby('Class of Orbit').size().to_string()))
print('-------------------------------------------------------------------------------------------')
print('Launch Mass - Power pairs : {}'.format(SAT_LM_PWR.groupby('Class of Orbit').size().to_string()))

#compute correlation matrix of dataframes,
#Dry Mass correlation matrix
DM_corr = SAT_DM_PWR.corr()
DM_corr.style.background_gradient()
#launch mass correlation matrix
LM_corr = SAT_LM_PWR.corr()
LM_corr.style.background_gradient()
#create a figure for the scatter plots
fig_lm, ax_lm = plt.subplots(nrows=1, ncols=2)
fig_lm.tight_layout(pad=2.5)
fig_lm.set_size_inches(30,10)

#set Y column via variable to choose data to be used in regression
y_col='Dry Mass (kg.)'

if y_col=='Launch Mass (kg.)':
  X = SAT_LM_PWR['Power (watts)']
  y = SAT_LM_PWR[y_col]
  df=SAT_LM_PWR
else:
  X = SAT_DM_PWR['Power (watts)']
  y = SAT_DM_PWR[y_col]
  df=SAT_DM_PWR

#scatter plot for groups
_=sns.relplot(x="Power (watts)", y=y_col, data=df, hue='Class of Orbit', col='Class of Orbit')

#prepare two scatter plots of the data pairs selected in the cycle above.
_  = sns.scatterplot(x="Power (watts)", y=y_col, data=df, hue='Class of Orbit', ax=ax_lm[0], s=100)
ax_lm[1].scatter(X, y, c='blue')
ax_lm[1].set_xscale('log')
ax_lm[1].set_yscale('log')

str_title='{} versus Power (watts)'
ax_lm[0].set_xlabel('Power (watts)', fontsize=20)
ax_lm[0].set_ylabel(y_col, fontsize=20)
ax_lm[0].set_title(str_title.format(y_col), fontsize=20)

str_title2='{} versus Power (watts) (log scale)'
ax_lm[1].set_xlabel('Power (watts)', fontsize=20)
ax_lm[1].set_ylabel(y_col, fontsize=20)
_ =ax_lm[1].set_title(str_title2.format(y_col), fontsize=20)



cov_mat = np.cov(np.log10(X), np.log10(y) )
corr_coeff = np.corrcoef( np.log10(X), np.log10(y) )
print( 'Correlation coefficient between Log(Power) and Log({0:s})) : {1:4.2f}'.format(y_col,corr_coeff[0,1]))
# Import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

#print(np.shape(X))
#reshape data
X_data = np.log10(X.values.reshape(-1,1))
y_data = np.log10(y.values.reshape(-1,1))
#display(X_data.shape)

# Create training and test sets, set random state to the answer to life, the universe and everything.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.25, random_state=4)

# Create the regressor: reg
reg = LinearRegression()
# Create a prediction space
prediction_space = np.linspace(min(X_test), max(y_test)).reshape(-1,1)
# Fit the model to the data
reg.fit(X_data,y_data)

#perform 10-fold cross-validation
cv_scores = cross_val_score(reg, X_train,y_train,cv=5)

# Print the 10-fold cross-validation scores
#display(cv_scores)
print('Average 5-Fold CV R^2 Score: {}'.format(np.mean(cv_scores)))

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(X_test)
OLS_rsquared_score=reg.score(X_test,y_test)
OLS_rmse_score = np.sqrt(mean_squared_error(y_test,y_pred))
print( reg.coef_[0], reg.intercept_)
# Print R^2 
print('R squared score on training data : {0:4.2f}'.format(reg.score(X_train,y_train)))
print('R squared score on test data : {0:4.2f}'.format(OLS_rsquared_score))
print("Root Mean Squared Error on test data: {0:4.2f}".format(OLS_rmse_score))
print('Parameters SLOPE:  {0:4.2f}, INT: {1:4.2f}'.format(reg.coef_[0,0], reg.intercept_[0]))

# Plot regression line
fig_linreg, ax_linreg = plt.subplots(nrows=1, ncols=1)
fig_linreg.set_size_inches(10,10)
str_reg_title = 'Linear regression, Log({0:s}) = {1:4.2f}*Log(Power) + {2:4.2f}'.format(y_col,reg.coef_[0,0], reg.intercept_[0] )
plt.plot(X_test, y_pred, color='black', linewidth=3)
plt.scatter(X_train, y_train, color='purple')
plt.scatter(X_test, y_test, color='green')
plt.xlabel("Log( Power )",fontsize=20)
plt.ylabel("Log( {} )".format(y_col),fontsize=20)
plt.title(str_reg_title, fontsize=20)
plt.show()
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import GridSearchCV
# Hyperparameter : min_samples. 0-1 for percentage of samples to choose randomly from original data.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor

#This section is inspired directly by the excellent example in https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py 
Linreg = LinearRegression() 
ransacReg=RANSACRegressor()
# find optimal min_samples with grid search
min_samples = np.arange(0.5,1,0.05)
param_grid = dict(min_samples=min_samples)
grid = GridSearchCV(estimator=ransacReg, param_grid=param_grid, scoring='r2', cv=5)
grid_result = grid.fit(X_data, y_data)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
#build model fit best parameter
ransac = RANSACRegressor(min_samples=grid_result.best_params_['min_samples'])

# Create training and test sets, set random state to the answer to life, the universe and everything.
#X_GEO_train, X_GEO_test, y_GEO_train, y_GEO_test = train_test_split(X_GEO_data, y_GEO_data, test_size = 0.25, random_state=4)

# Create a prediction space
Linreg_prediction_space = np.linspace(min(X_data), max(y_data)).reshape(-1,1)

# Fit the model to the data
Linreg.fit(X_data, y_data)
ransac.fit(X_data, y_data)

#ransac.fit(X_data, y_data)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

y_pred_ransac = ransac.predict(X_data)
y_pred_Linreg = reg.predict(X_data)

#Score computation
Linreg_rsquared_score=reg.score(X_data,y_data)
Linreg_rmse_score = np.sqrt(mean_squared_error(y_data,y_pred_Linreg))
Ransac_rsquared_score = ransac.score(X_data, y_data)
Ransac_rmse_score = np.sqrt(mean_squared_error(y_data,y_pred_ransac))

# Compare estimated coefficients
print("Estimated coefficient SLOPE OLS {0:4.2f}, RANSAC {1:4.2f} ".format(Linreg.coef_[0,0], ransac.estimator_.coef_[0,0]))
print("Estimated coefficient Y-INT OLS {0:4.2f}, RANSAC {1:4.2f} ".format(Linreg.intercept_[0], ransac.estimator_.intercept_[0]))

#compare performance scores
print('R^2 score for OLS: {0:4.2f}, RANSAC model {1:4.2f}'.format( Linreg_rsquared_score,Ransac_rsquared_score))
print('RMSE score for OLS : {0:4.2f}, RANSAC model {1:4.2f}'.format( Linreg_rmse_score,Ransac_rmse_score))

#plot regression results
fig_robreg, ax_robreg = plt.subplots(nrows=1, ncols=1)
fig_robreg.set_size_inches(12,12)
ax_robreg.tick_params(axis='x', labelsize=18)
ax_robreg.tick_params(axis='y', labelsize=18)

str_robreg_title = 'Robust linear regression, Log({0:s}) = {1:4.2f}*Log(Power) + {2:4.2f}'.format(y_col,ransac.estimator_.coef_[0,0], ransac.estimator_.intercept_[0] )
lw = 2

plt.scatter(X_data[inlier_mask], y_data[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers',s=150)
plt.scatter(X_data[outlier_mask], y_data[outlier_mask], color='gold', marker='.',
            label='Outliers', s=150)
plt.plot(X_data, y_pred_Linreg, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(X_data, y_pred_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')

plt.legend(loc='lower right',fontsize=18)
plt.xlabel("Log( Power )",fontsize=14)
plt.ylabel("Log( Dry Mass )",fontsize=14)
plt.title(str_robreg_title, fontsize=18)
plt.show()

nbr_sats_per_op=UCSDB['Operator/Owner'].value_counts()
# a bar plot for the top ten operators, by number of satellites, 
axOp=UCSDB['Operator/Owner'].value_counts()[:10].plot(kind='barh', width =0.8, figsize=(8,8))
#axOp = sns.countplot(y=UCSDB['Operator/Owner'],order=pd.value_counts(UCSDB['Operator/Owner']).iloc[:-10].index)
#annotate bars,
set_bar_label(axOp)
# set a legend, title
axOp.set_xlabel('Number of satellites in database')
_= axOp.set_title('Total number of satellites, per operator', Fontsize=20)

#A closer look on GSO operators, with a commercial focus in the telecommunications domain, returns the following fleet sizes
#first we build alist with the operator's names,
GSO_comm_ops = np.unique(UCSDB['Operator/Owner'][(UCSDB['Users'] == 'Commercial') & (UCSDB['Purpose'] == 'Communications')& (UCSDB['Class of Orbit'] == 'GEO')])
#the length of the list is 74, meaning there are 74 operators meeting the three conditions set in our query above.
print('There are {} GEO satcomm operators in the list'.format(len(GSO_comm_ops)))
print('----------------------------------------------')

#alternatively, we can use a more complex but compact approach, as we did in the figures above, using Vaue_counts()
ax4 = UCSDB['Operator/Owner'][(UCSDB['Users'] == 'Commercial') & (UCSDB['Purpose'] == 'Communications') \
                              & (UCSDB['Class of Orbit'] == 'GEO')].value_counts()[0:10].plot(kind='barh',figsize=(8,8))
_=set_bar_label(ax4)
#IS_DF=UCSDB[UCSDB['Operator/Owner'].str.contains('ntelsat') & UCSDB['Users'].str.contains('Commercial')]
IS_count=np.sum(UCSDB['Operator/Owner'].str.contains('intelsat',case=False) & UCSDB['Users'].str.contains('Commercial'))
EUT_count=np.sum(UCSDB['Operator/Owner'].str.contains('eutelsat',case=False) & UCSDB['Users'].str.contains('Commercial'))
TEL_count=np.sum(UCSDB['Operator/Owner'].str.contains('telesat',case=False) & UCSDB['Users'].str.contains('Commercial'))
SES_count=np.sum(UCSDB['Operator/Owner'].str.contains('ses',case=False) & UCSDB['Users'].str.contains('Commercial')) + \
          np.sum(UCSDB['Operator/Owner'].str.contains('o3b',case=False) & UCSDB['Users'].str.contains('Commercial'))

print('UCSDB contains : ')
print('---------------------------')
print(' {} Eutelsat satellites'.format(EUT_count))
print(' {} Intelsat satellites'.format(IS_count))
print(' {} SES/O3B satellites'.format(SES_count))
print(' {} Telesat satellites'.format(TEL_count))

#extract all data points for GEO orbit, look at the longitude column
geo_flt_id = (UCSDB['Class of Orbit'] == 'GEO')
GEO_df = UCSDB[geo_flt_id]
GEO_byUser=GEO_df.groupby(['Users'])['Purpose'].count()
GEO_byUser.plot(kind='bar')
print(GEO_byUser)
#extract all data points for GEO orbit, look at the longitude column
#flt_id = (UCSDB['Class of Orbit'] == 'GEO') & (UCSDB['Users'].str.contains('commercial', case = False) )
flt_id = (UCSDB['Class of Orbit'] == 'GEO') & (UCSDB['Users'].str.contains('commercial', case = False) & (UCSDB['Purpose'].str.contains('communications', case = False) ) )
GEO_long = UCSDB[ flt_id ]['Longitude of GEO (degrees)'].astype(float)
#correct longitudes to fall in the range (-180,180)
GEO_long[ GEO_long > 180] = GEO_long[ GEO_long > 180] - 360
#make a histogram of the longitude data points, 
#histogram intervals definition and xticks locations.
bin_edges = np.arange(-180,185,5)-2.5 #every five degrees.
x_axis_ticks = np.arange(-180,190,10)
# create figure canvas and axes
fig, ax5 = plt.subplots(figsize=[15,8])
fig.tight_layout(pad=0.0)

#plot.
_ = sns.distplot(GEO_long,bins=bin_edges,kde=False,axlabel='Orbital location (deg E.)',ax=ax5, hist_kws ={'ec':'k'}) 
_ = plt.xticks(x_axis_ticks,rotation=55)
_ = plt.xlabel('Orbital location (deg E.)')
_ = plt.ylabel('Number of satellites in orbital location bin ')

set_bar_label(ax5, minval=10, orient='v')
#in case you want to explore a swarmplot,
#_ = sns.swarmplot(GEO_long, ax=ax6, orient = 'v', color='g')

DOD_GPS = UCSDB['Name of Satellite, Alternate Names'].str.contains('navstar gps',case=False).sum()
EU_GALILEO = UCSDB['Name of Satellite, Alternate Names'].str.contains('galileo',case=False).sum()
CH_BEIDOU = UCSDB['Name of Satellite, Alternate Names'].str.contains('beidou',case=False).sum()
print('Count of operational satellites (APR-2020) in three navigation constellations:')
print('-----------------------------------------------------------------------------')
print('US DoD GPS    : {} satellites '.format(DOD_GPS))
print('EU GALILEO    : {} satellites '.format(EU_GALILEO))
print('CHINA BEIDOU  : {} satellites '.format(CH_BEIDOU))
nbr_sats_per_op=UCSDB['Contractor'].value_counts()
# a bar plot for the top contractors, 
fig, ax6 = plt.subplots(figsize=[15,8])
ax6=UCSDB['Contractor'].value_counts()[:15].plot(kind='barh', width = 0.8, color='purple')
#annotate bars,
set_bar_label(ax6)
# set a legend, title
plt.xlabel('Number of satellites in database' , fontsize=18)
plt.title('Total number of satellites, per contractor/manufacturer', fontsize=20)
plt.show()
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import GridSearchCV
# Hyperparameter : min_samples. 0-1 for percentage of samples to choose randomly from original data.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor

GEO_set=SAT_DM_PWR['Class of Orbit'] == 'GEO'
X_data_GEO = SAT_DM_PWR[GEO_set]['Power (watts)']
y_data_GEO = SAT_DM_PWR[GEO_set]['Dry Mass (kg.)']

X_GEO_data = np.log10( X_data_GEO.values.reshape(-1,1) )
y_GEO_data = np.log10( y_data_GEO.values.reshape(-1,1) )

#This section is inspired directly by the excellent example in https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py 
LinregGEO = LinearRegression() 
ransacGEO=RANSACRegressor()

# find optimal min_samples with grid search
min_samples_GEO = np.arange(0.5,1,0.05)
param_grid_GEO = dict(min_samples=min_samples_GEO)
grid_GEO = GridSearchCV(estimator=ransacGEO, param_grid=param_grid_GEO, scoring='r2')
grid_result_GEO = grid_GEO.fit(X_GEO_data, y_GEO_data)

print('Best Score: ', grid_result_GEO.best_score_)
print('Best Params: ', grid_result_GEO.best_params_)

# Create a prediction space
Linreg_prediction_space = np.linspace(min(X_GEO_data), max(y_GEO_data)).reshape(-1,1)
ransacGEO = RANSACRegressor(min_samples=grid_result_GEO.best_params_['min_samples'])

# Fit the model to the data
LinregGEO.fit(X_GEO_data, y_GEO_data)
ransacGEO.fit(X_GEO_data, y_GEO_data)

inlier_mask = ransacGEO.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

y_pred_ransac = ransacGEO.predict(X_GEO_data)
y_pred_Linreg = LinregGEO.predict(X_GEO_data)

#Score computation
Linreg_rsquared_score=LinregGEO.score(X_GEO_data,y_GEO_data)
Linreg_rmse_score = np.sqrt(mean_squared_error(y_GEO_data,y_pred_Linreg))
Ransac_rsquared_score = ransacGEO.score(X_GEO_data, y_GEO_data)
Ransac_rmse_score = np.sqrt(mean_squared_error(y_GEO_data,y_pred_ransac))

# Compare estimated coefficients
print("Estimated coefficient SLOPE OLS {0:4.2f}, RANSAC {1:4.2f} ".format(LinregGEO.coef_[0,0], ransacGEO.estimator_.coef_[0,0]))
print("Estimated coefficient Y-INT OLS {0:4.2f}, RANSAC {1:4.2f} ".format(LinregGEO.intercept_[0], ransacGEO.estimator_.intercept_[0]))

#compare performance scores
print('R^2 score for OLS: {0:4.2f}, RANSAC model: {1:4.2f}'.format( Linreg_rsquared_score,Ransac_rsquared_score))
print('RMSE score for OLS: {0:4.2f}, RANSAC model: {1:4.2f}'.format( Linreg_rmse_score,Ransac_rmse_score))

#plot regression results
fig_robreg_GEO, ax_robreg_GEO = plt.subplots(nrows=1, ncols=1)
fig_robreg_GEO.set_size_inches(12,12)
ax_robreg_GEO.tick_params(axis='x', labelsize=18)
ax_robreg_GEO.tick_params(axis='y', labelsize=18)

str_robreg_GEO_title = 'Robust linear regression, GEO data set. Log({0:s}) = {1:4.2f}*Log(Power) + {2:4.2f}'.format(y_col,ransacGEO.estimator_.coef_[0,0], ransacGEO.estimator_.intercept_[0] )

lw = 2
plt.scatter(X_GEO_data[inlier_mask], y_GEO_data[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers',s=150)
plt.scatter(X_GEO_data[outlier_mask], y_GEO_data[outlier_mask], color='gold', marker='.',
            label='Outliers', s=150)
plt.plot(X_GEO_data, y_pred_Linreg, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(X_GEO_data, y_pred_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')

plt.legend(loc='lower right',fontsize=18)
plt.xlabel("Log( Power )",fontsize=18)
plt.ylabel("Log( Dry Mass )",fontsize=18)
plt.title(str_robreg_GEO_title, fontsize=20)
plt.show()



