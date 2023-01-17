# Load packages
import warnings
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)      # suppress warnings
import numpy as np                                 # linear algebra
import pandas as pd                                # data analysis
import matplotlib.pyplot as plt                    # visualization
import seaborn as sns                              # visualization
import scipy.stats as scipystats                   # statistics  
import statsmodels.formula.api as smf              # statistics
from statsmodels.api import add_constant           # statistics
from sklearn.feature_selection import SelectKBest  # feature selection
from sklearn.feature_selection import f_regression # feature selection

pd.set_option('display.float_format', lambda x: '%.1f' % x) # format decimals
sns.set(font_scale=1.5) # increse font size for seaborn charts
%matplotlib inline
SCHOOLS = pd.read_csv('../input/MA_Public_Schools_2017.csv', dtype={'School Code':'str',
                                                                    'District Code':'str'} )
HS = SCHOOLS.loc[SCHOOLS['12_Enrollment'] > 0].reset_index(drop=True)
print ("Rows: ",HS.shape[0],"   Variables: ", HS.shape[1])
HS['4YR_College_%'] = HS['% Private Four-Year'] + HS['% Public Four-Year']
HS['4YR_College_%']= HS['4YR_College_%'] * (HS['% Graduated']/100)
HS = HS.drop(['% Private Four-Year',
              '% Public Four-Year',
              '% Graduated',
              '% Public Two-Year',
              '% MA Community College',
              '% Attending College'],axis=1)
HS_30 = HS.loc[HS['12_Enrollment'] >= 30].dropna(subset=['4YR_College_%']).reset_index(drop=True)

plt.figure(figsize=(15,5))
plt.hist(HS_30['4YR_College_%']); # distribution of 4-year college attendance
plt.title('Distribution of Percentage of students attending 4-year College by School');
HS_30.isnull().sum()[HS_30.isnull().sum()>100].head()
HS_30 = HS_30[HS_30.columns[ ( (pd.isnull(HS_30).sum()) / (HS_30.shape[0]) < 0.5 ).values ]]
HS_30.head()
ST = HS_30['School Type'] # store School Type before dropping non-numeric fields
HS_30.set_index('School Name', inplace=True) # set the index to the School Name
HS_30 = HS_30.select_dtypes(include=[np.number]) # drop on-numeric fields
print ("Rows: ",HS_30.shape[0],"   Variables: ", HS_30.shape[1])
HS_30 = HS_30.fillna(HS_30.median()) # fill in missing values
y = HS_30['4YR_College_%'].values
X_df = HS_30.select_dtypes(include=[np.number]).drop(['4YR_College_%'],axis=1)
X = X_df.values
row_names = HS_30.index.values # store row names
col_names = X_df.columns.values # store column names

selector = SelectKBest(f_regression, k=20)
HS_30 = pd.DataFrame(selector.fit_transform(X, y))

vars =  (np.array(col_names)[selector.get_support()]) # variable names
fstats = (selector.scores_)[selector.get_support()] # F-statistics
for a,b in zip(vars,fstats):
   print ( "{0:40} {1:.0f}".format(a, b) )
HS_30.columns = (np.array(col_names)[selector.get_support()]) # restore column names
HS_30['School Name'] = row_names # restore row names
HS_30.set_index('School Name', inplace=True) # restore index
HS_30['4YR_College_%'] = y # restore dependent variable
HS_30['School Type']=ST.values # add back 'School Type'
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(HS_30.corr(), linewidths=0.1,cbar=True, annot=True, square=True, fmt='.1f')
plt.title('Correlation between Variables');
X = add_constant(HS_30[['% Economically Disadvantaged']])
Y = HS_30['4YR_College_%']
regr = smf.OLS(Y,X).fit()
regr.summary()
sns_plot = sns.lmplot(x='% Economically Disadvantaged', y='4YR_College_%',data=HS_30,size = 10)
plt.title('Relationship between Economic Disadvantage & College Attendance');
plt.figure(figsize=(10,10))
plt.scatter(regr.predict(), regr.resid)
plt.title('Residuals versus Predicted Values of 4-Year College Attendance');
HS_30 = HS_30[['School Type','% Economically Disadvantaged','4YR_College_%']]
HS_30  = pd.concat([HS_30, pd.Series(regr.resid, name = 'resid')], axis = 1)
HS_30  = HS_30.sort_values(ascending=False,by=['resid'])
HS_30.loc[HS_30['resid'] > 20]
HS_30.loc[(HS_30.index == "Boston Latin Academy") | 
            (HS_30.index == "O'Bryant School Math/Science") |
            (HS_30.index == "Boston Latin"), 'School Type'] = 'Exam School'
                   
HS_30.loc[(HS_30.index == "Another Course To College") | 
            (HS_30.index == "Boston Arts Academy") | 
            (HS_30.index == "Boston Community Leadership Academy") |
            (HS_30.index == "Fenway High School") |
            (HS_30.index == "Greater Egleston Community High School") |
            (HS_30.index == "Lyon Upper 9-12") |
            (HS_30.index == "New Mission High School") |
            (HS_30.index == "Quincy Upper School") |
            (HS_30.index == "TechBoston Academy"), 'School Type'] = 'Pilot School'

HS_30.loc[HS_30['resid'] > 20].sort_values(ascending=True,by=['School Type'])
HS_30.loc[HS_30['resid'] < -20].sort_values(ascending=True,by=['resid'])
HS_30.loc[(HS_30.index == "Boston Adult Academy") | 
          (HS_30.index == "Edison Academy") |
          (HS_30.index == "The Gateway to College") |
          (HS_30.index == "Greater Egleston Community High School") |
          (HS_30.index == "Boston Day and Evening Academy Charter School"), 
          'School Type'] = 'Specialist School'           
            
HS_TRAD =  HS_30.loc[(HS_30['School Type'] == 'Public School')].reset_index(drop=True) 
X = add_constant(HS_TRAD[['% Economically Disadvantaged']])
Y = HS_TRAD['4YR_College_%']
regr = smf.OLS(Y,X).fit()
HS_TRAD  = pd.concat([HS_TRAD, pd.Series(regr.resid, name = 'resid')], axis = 1)
regr.summary()
sns_scatter = sns.pairplot(HS_30,
                 x_vars=['% Economically Disadvantaged'],
                 y_vars=['4YR_College_%'],
                 hue='School Type',
                 markers=["x", "o",'D','x','o'],
                 size = 10)
plt.title('Relationship between Economic Disadvantage and College Attendance');
HS_30.loc[HS_30['% Economically Disadvantaged'] > 30].sort_values(
    ascending=False,by=['4YR_College_%']).head(10)
mask = ( (HS_30['School Type'] == 'Public School') | 
         (HS_30['School Type'] == 'Charter School') | 
         (HS_30['School Type'] == 'Pilot School') )
HS_TCP =  HS_30.loc[mask].copy() 

mask = (HS_TCP['School Type'] == "Charter School") | (HS_TCP['School Type'] == "Pilot School")
HS_TCP['Indep_School'] = np.where(mask, 1, 0)
HS_TCP['Indep_School_Mult_Econ'] = HS_TCP['Indep_School'] * HS_TCP['% Economically Disadvantaged']
X = add_constant(HS_TCP[['% Economically Disadvantaged','Indep_School_Mult_Econ']])
Y = HS_TCP['4YR_College_%']
regr = smf.OLS(Y,X).fit()
regr.summary()
plt.figure(figsize=(10,10))
plt.scatter(regr.predict(), regr.resid)
plt.title('Residuals versus Predicted Values of 4-Year College Attendance');
HS_TCP = HS_TCP[['School Type','% Economically Disadvantaged','4YR_College_%','Indep_School_Mult_Econ']]
HS_TCP  = pd.concat([HS_TCP, pd.Series(regr.resid, name = 'resid')], axis = 1)
HS_TCP = HS_TCP.sort_values(ascending=False,by=['resid'])
HS_TCP.loc[HS_TCP['resid'] > 30]