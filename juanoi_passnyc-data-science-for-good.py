import pandas as pd
import numpy as np
# Read in main data

## This DataFrame will be our main DataFrame, so we will add additional data from other sources below
schoolExplorer2016File = '../input/data-science-for-good/2016 School Explorer.csv'
df = pd.read_csv(schoolExplorer2016File,header=0)

# Convert percentage fields into numerical values, add new features, etc.

df['Average Proficiency'] = df.loc[:,'Average ELA Proficiency':'Average Math Proficiency'].mean(axis=1)

# Normalize percentage rates
percentageFields =  list(df.filter(like='Percent').columns.values) + \
                    list(df.filter(like='Rate').columns.values) + \
                    list(df.filter(like='%').columns.values)
        
fillRate = {}
minRate = 0.01 # Arbitrary value 
for f in percentageFields:
    # Transform to float
    fillRate[f] = df[f].dropna().apply(lambda x: float(x[:-1])/100).mean()
    df.fillna(value={f : str(fillRate[f])+'%'},inplace=True)
    df[f]=df[f].apply(lambda x: float(x[:-1])/100)
    
    # Normalize
    df.loc[df[f]<minRate, f] = minRate
    
def remComma(x):
    if isinstance(x,str):
        x = float(x[1:].replace(',',''))
        
    return x
    
fillRate['School Income Estimate'] = df['School Income Estimate'].dropna().apply(lambda x: float(x[1:].replace(',',''))).mean()
df['School Income Estimate']=df['School Income Estimate'].apply(remComma)
df.fillna(value={'School Income Estimate' : fillRate['School Income Estimate']},inplace=True);

# There is a typo on the 'Grade 3 Math - All Students Tested' label (Tested->tested)
df['Grade 3 Math - All Students Tested'] = df['Grade 3 Math - All Students tested']
df.drop(labels=['Grade 3 Math - All Students tested'], axis=1, inplace=True)

# Compute percentage of outstanding students in sample
for g in range(3,9): 
    df['Grade ' + str(g) + ' ELA 4s %'] = df['Grade ' +str(g)+' ELA 4s - All Students'].map(float) / df['Grade ' + str(g) + ' ELA - All Students Tested']
    df['Grade ' + str(g) + ' Math 4s %'] = df['Grade ' +str(g)+' Math 4s - All Students'].map(float) / df['Grade ' + str(g) + ' Math - All Students Tested']
    df['Grade ' + str(g) + ' 4s %'] = df[['Grade ' + str(g) + ' ELA 4s %', 'Grade ' + str(g) + ' Math 4s %']].mean(axis=1)
    
# Simply fill with zeros the entries for schools with no students taking exams
df[df.filter(like='4s %').columns]=df[df.filter(like='4s %').columns].fillna(-0.1)

# Convert coordinates into Web Mercator Format (we will need this when plotting maps)
def wgs84_to_web_mercator(df, lon="Longitude", lat="Latitude"):
    """ 
        Converts decimal longitude/latitude to Web Mercator format
        Code taken from the Bokeh tutorial on Geoplotting
        https://hub.mybinder.org/user/bokeh-bokeh-notebooks-xq55q0f7/notebooks/tutorial/09%20-%20Geographic%20Plots.ipynb
    """
    k = 6378137 # earth's radius in m
    df[lon+'WebMercator'] = df[lon] * (k * np.pi/180.0)
    df[lat+'WebMercator'] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df

wgs84_to_web_mercator(df)

# Sort entries by Location Code (will be useful later on)
df.sort_values(by='Location Code', inplace=True)
# Read in NYC DOE high school directory data

doeHSDir2016File = '../input/nyc-high-school-directory/2016-doe-high-school-directory.csv'
doeHSDir2015File = '../input/nyc-high-school-directory/2014-2015-doe-high-school-directory.csv'
HSDir2016 = pd.read_csv(doeHSDir2016File,header=0)
HSDir2015 = pd.read_csv(doeHSDir2015File,header=0)

# Prepare DOE data

labelCols = {'dbn':'DBN'}
numericCols = {'total_students':'Total Students'}
stringCols = {'extracurricular_activities':'Extracurricular Activities',
                   'language_classes':'Language Classes',
                   'advancedplacement_courses':'Advanced Placement Courses',
                   'school_sports':'School Sports',
                   'partner_highered':'Partner Higher Ed',
                   'partner_cultural':'Partner Cultural',
                   'partner_financial':'Partner Financial'}

interestingCols = {**labelCols, **numericCols, **stringCols}
    
HSDir2016.rename(columns=interestingCols,inplace=True)
HSDir2015.rename(columns=interestingCols,inplace=True)

HSDir2016.sort_values(by='DBN',inplace=True)
HSDir2015.sort_values(by='DBN',inplace=True)

hsDirFilter = HSDir2015['DBN'].isin(HSDir2016['DBN'].values)

# Generate a unified DataFrame for both 2015 and 2016.
HSDir = HSDir2016[list(interestingCols.values())].copy()

commonSchoolFilter2016 = HSDir['DBN'].isin(HSDir2015['DBN'].values)
commonSchoolFilter2015 = HSDir2015['DBN'].isin(HSDir['DBN'].values)

HSDir.loc[commonSchoolFilter2016 , 'Total Students'] = 0.5*(HSDir.loc[commonSchoolFilter2016 , 'Total Students'].values + HSDir2015.loc[commonSchoolFilter2015 , 'Total Students'].values)

def cleanSemicolons(s):
    if isinstance(s,str):
        s=s.replace(';',',')
    
    return s

for col in stringCols.values():
    for l, r in zip(HSDir.loc[commonSchoolFilter2016 , col].values , HSDir2015.loc[commonSchoolFilter2015 , col].values):
        if isinstance(l,str) and isinstance(r,str):
            l += ',' + r
        elif isinstance(r,str):
            l = r
            
    HSDir.loc[commonSchoolFilter2016 , col] = HSDir.loc[commonSchoolFilter2016 , col].map(cleanSemicolons)
    
HSDir = pd.concat([HSDir , HSDir2015.loc[~hsDirFilter,interestingCols.values()]], ignore_index=True)
HSDir.sort_values(by='DBN', inplace=True)

# Now we add all this data to the 'main' DataFrame
HSDirFilter = HSDir['DBN'].isin(df['Location Code'].values)
DFFilter = df['Location Code'].isin(HSDir['DBN'].values)

def countCommas(s):
    if isinstance(s,str):
        return 1 + s.count(',')
    else:
        return 0

df['Total Students'] = pd.Series(np.NaN, index=df.index)
df.loc[DFFilter , 'Total Students'] = HSDir.loc[HSDirFilter, 'Total Students'].values 

# Fill missing values with a simple estimate. We distinguish between school type because of the analysis below
commSchoolFilter = df['Community School?']=='Yes'
df.loc[~DFFilter & commSchoolFilter , 'Total Students'] = df.loc[commSchoolFilter,'Total Students'].mean()
df.loc[~DFFilter & ~commSchoolFilter , 'Total Students'] = df.loc[~commSchoolFilter,'Total Students'].mean()

for col in stringCols.values():
    df[col] = pd.Series(np.NaN, index=df.index)
    df.loc[DFFilter , col] = HSDir.loc[HSDirFilter, col].map(countCommas).values
    df[col].fillna(0, inplace=True)
# Read in NYC School Demographics and Accountability data

schoolDemographicsFile = '../input/ny-school-demographics-and-accountability-snapshot/2006-2012-school-demographics-and-accountability-snapshot.csv'
HSDemo = pd.read_csv(schoolDemographicsFile,header=0)

# Prepare NYC School Demographics and Accountability data

HSDemo.sort_values(by='DBN')

## Replace spurious strings being read in in numeric fields by NaNs
def replaceString(x):
    if x == '    ':
        x = np.NaN
    else:
        x = float(x)
    
    return x

gradeCountCols = HSDemo.filter(like='grade').columns.values
for l in gradeCountCols:
    HSDemo.loc[:,l] = HSDemo.loc[:,l].map(replaceString).values

    # Fill NaNs by zeros in student counts for all grades
    HSDemo.loc[:,l].fillna(np.NaN, inplace=True)

# Consider only average counts
HSDemo = HSDemo.groupby('DBN').mean()
HSDemo.reset_index(inplace=True)

genderCols = {'male_per':'Male %', 'female_per':'Female %'}
HSDemo.rename(columns=genderCols,inplace=True)

## Insert student counts to main DataFrame
HSDemoFilter = HSDemo['DBN'].isin(df['Location Code'].values)
DFFilter = df['Location Code'].isin(HSDemo['DBN'].values)

for col in list(gradeCountCols)+list(genderCols.values()):
    df[col] = pd.Series(np.NaN, index=df.index)
    df.loc[DFFilter,col] = HSDemo.loc[HSDemoFilter, col].values
# Read in TNYTimes data
TNYTimesFile = '../input/the-new-york-times-nyc-shsat-data/nyc-shsat-data.csv'
TNYTimes = pd.read_csv(TNYTimesFile,header=0)

# Prepare TNYTimes data

TNYTimes.drop(labels=['Unnamed: 4','Source: New York City Department of Education. ^Data suppressed for values of 5 students or fewer.'], axis=1, inplace=True)
TNYTimes.sort_values(by='DBN', inplace=True)

def replaceSmall(s):
    '''All values below 6 were replaced by an \'s\' in the original data. This functions replaces that placeholder for NaNs.'''
    if s=='s' or s=='s^':
        s = 0 # np.NaN
    return s

TNYTimes['Offers'] = TNYTimes['Offers'].map(replaceSmall)
TNYTimes['Testers'] = TNYTimes['Testers'].map(replaceSmall)

# Insert data into main DataFrame
df['SHSAT Takers'] = 0 # np.NaN
df['SHSAT Offers'] = 0 # np.NaN

hsFilter = TNYTimes['DBN'].isin(df['Location Code'].values)
dfFilter = df['Location Code'].isin(TNYTimes['DBN'].values)

df.loc[dfFilter , 'SHSAT Takers'] = TNYTimes.loc[hsFilter, 'Testers'].map(float).values
df.loc[dfFilter , 'SHSAT Offers'] = TNYTimes.loc[hsFilter, 'Offers'].map(float).values

df['SHSAT Takers %'] = df['SHSAT Takers'] / df['Total Students']
df['SHSAT Offers %'] = df['SHSAT Offers'] / df['SHSAT Takers']
df['SHSAT Offers %'].fillna(0, inplace=True)
import matplotlib.pyplot as plt
import seaborn as sb; sb.set()
meanAverageProficiency = df['Average Proficiency'].mean()
medianAverageProficiency = df['Average Proficiency'].median()
medianSchoolIncomeEstimate = df['School Income Estimate'].median()
df.fillna(value={'Average Proficiency' : meanAverageProficiency},inplace=True);

print('Mean Average Proficiency = ' + str(meanAverageProficiency))
print('Median Average Proficiency = ' + str(medianAverageProficiency))
print('Median School Income Estimate = ' + str(medianSchoolIncomeEstimate))
g=sb.jointplot(df['Economic Need Index'],df['Average Proficiency'],
           kind='kde',color='b');

Xs = [df['Economic Need Index'].min(), df['Economic Need Index'].max()]
Meds=[meanAverageProficiency,meanAverageProficiency]
g.ax_joint.plot(Xs, Meds, '--k')
g=sb.jointplot(df['Percent of Students Chronically Absent'],df['Average Proficiency'],
           kind='kde',color='b')

Xs = [df['Percent of Students Chronically Absent'].min(), df['Percent of Students Chronically Absent'].max()]
Meds=[meanAverageProficiency,meanAverageProficiency]
g.ax_joint.plot(Xs, Meds, '--k')

sb.jointplot(df['Economic Need Index'],df['Percent of Students Chronically Absent'],
           kind='kde',color='b');
from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(n_clusters=2, linkage= 'ward')
clusterLabels = clusterer.fit_predict(df[['Average Proficiency','Percent of Students Chronically Absent']])

cluster0 = clusterLabels == 0
cluster1 = clusterLabels == 1

g = sb.JointGrid(df['Percent of Students Chronically Absent'][cluster1],df['Average Proficiency'][cluster1],
                xlim=[df['Percent of Students Chronically Absent'].min(),df['Percent of Students Chronically Absent'].max()],
                ylim=[df['Average Proficiency'].min(),df['Average Proficiency'].max()])

g = g.plot_joint(sb.kdeplot, cmap='Blues_d')
g = g.plot_marginals(sb.distplot, color="b")
g.x = df['Percent of Students Chronically Absent'].values[cluster0]
g.y = df['Average Proficiency'].values[cluster0]
g = g.plot_joint(sb.kdeplot, cmap='Reds_d')
g = g.plot_marginals(sb.distplot, color="r")
g=sb.jointplot(df['School Income Estimate'],df['Average Proficiency'],
           kind='kde',color='b',n_levels=15)

Xs = [df['School Income Estimate'].min(), df['School Income Estimate'].max()]
Meds=[meanAverageProficiency,meanAverageProficiency]
g.ax_joint.plot(Xs, Meds, '--k')

Ys = [df['Average Proficiency'].min(), df['Average Proficiency'].max()]
Meds=[medianSchoolIncomeEstimate,medianSchoolIncomeEstimate]

g.ax_joint.plot(Meds, Ys, '--r');
ratingFields = df.filter(like='Rating').columns.values
ratingValues = df['Supportive Environment Rating'].dropna().unique()

dfTakersByRating = pd.DataFrame(columns=ratingFields,index=['Meeting/Exceeding Target','Not Meeting/Approaching Target'])

for rf in ratingFields:
    filt = df[rf].isin(['Exceeding Target','Meeting Target'])
    dfTakersByRating.loc['Meeting/Exceeding Target',rf] = df.loc[filt, 'SHSAT Takers %'].mean()
    
    filt = df[rf].isin(['Approaching Target','Not Meeting Target'])
    dfTakersByRating.loc['Not Meeting/Approaching Target',rf] = df.loc[filt, 'SHSAT Takers %'].mean()

dfTakersByRating = dfTakersByRating.reset_index(col_fill='Value').rename(columns={'index':'Score'})
dfTakersByRating = dfTakersByRating.melt(id_vars=['Score'], value_vars=ratingFields, var_name='Rating', value_name='Average SHSAT Takers %')
dfTakersByRating['Rating'] = dfTakersByRating['Rating'].map(lambda s: s.replace(' Rating',''))
fig, (ax1) = plt.subplots(figsize=(21, 8), ncols=1)
sb.barplot(x="Rating", y="Average SHSAT Takers %", hue='Score', data=dfTakersByRating, ax=ax1)
plt.show()
dfBySchoolType = df.groupby('Community School?').mean()
dfBySchoolType.reset_index(inplace=True)

percentCols=['Percent Asian','Percent Black / Hispanic', 'Percent White']

dfMinorityPercentage = dfBySchoolType.melt(id_vars=['Community School?'], value_vars=percentCols,var_name='Group', value_name='Percentage')

dfMinorityPercentage['Group'] = dfMinorityPercentage['Group'].map(lambda s: s.replace('Percent ','')).values
fig, (ax1,ax2) = plt.subplots(figsize=(21, 8), ncols=2)
sb.boxplot(df['Community School?'],df['Average Proficiency'], ax=ax1)
sb.boxplot(df['Community School?'],df['Percent of Students Chronically Absent'], ax=ax2)
plt.show()
fig, (ax1) = plt.subplots(figsize=(21, 8), ncols=1)
sb.barplot(x="Group", y="Percentage", hue='Community School?', data=dfMinorityPercentage, ax=ax1)
plt.show()
specialActivities=['Extracurricular Activities','Advanced Placement Courses','Language Classes','School Sports']
dfActivities = dfBySchoolType[['Community School?']+specialActivities]
dfActivities = dfActivities.melt(id_vars=['Community School?'], value_vars=specialActivities,var_name='Activity', value_name='Average Offer')
fig, (ax1) = plt.subplots(figsize=(17, 8), ncols=1)
sb.barplot(x="Activity", y="Average Offer", hue='Community School?', data=dfActivities, ax=ax1)
plt.show()
scoringWeights = {}
scoringWeights['Economic Need Index'] = 0.2
scoringWeights['Percent of Students Chronically Absent'] = 0.2
scoringWeights['School Income Estimate'] = 0.3
scoringWeights['Percent Black / Hispanic'] = 0.3
                                                                      
scoringFields = ['Economic Need Index','Percent of Students Chronically Absent','School Income Estimate','Percent Black / Hispanic']

scoringParams = {}
for field in scoringFields:
    scoringParams[field] = {'max':df[field].max() , 'min':df[field].min()}
    
scores = pd.DataFrame(index=[df.index], columns=['DBN','Name','Percent Black / Hispanic','Score'])
scores['DBN'] = df['Location Code'].values
scores['Name'] = df['School Name'].values
scores['Percent Black / Hispanic'] = df['Percent Black / Hispanic'].values
scores['Score'] = 0
for field in scoringFields:
    scores['Score'] += scoringWeights[field] * (df[field].values - scoringParams[field]['min'])/abs(scoringParams[field]['max']-scoringParams[field]['min'])
scores.sort_values(by='Score',ascending=False).head(15)
from bokeh.io import output_notebook, show
output_notebook()
from bokeh.plotting import figure
from bokeh.models import WMTSTileSource, ColumnDataSource, Circle, ColorBar, BasicTicker, HoverTool
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5

# web mercator coordinates
NYC = x_range,y_range = ((df['LongitudeWebMercator'].min(),df['LongitudeWebMercator'].max()) ,
                         (df['LatitudeWebMercator'].min(),df['LatitudeWebMercator'].max()))

p = figure( tools='hover, pan, wheel_zoom',
            x_range=x_range, y_range=y_range, 
            width=1250,
            height=850
          )
p.axis.visible = False

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [("Score", "@Score"),("ENI", "@ENI"), ("Black/Hispanic %", "@BHPercent")]

url = 'http://a.basemaps.cartocdn.com/dark_all/{Z}/{X}/{Y}.png'
attribution = "Tiles by Carto, under CC BY 3.0. Data by OSM, under ODbL"

p.add_tile(WMTSTileSource(url=url, attribution=attribution))

color_mapper = LinearColorMapper(palette=Viridis5)

source = ColumnDataSource(
    data=dict(
        lat=df['LatitudeWebMercator'].values,
        lon=df['LongitudeWebMercator'].values,
        Score=df['Percent Black / Hispanic'].values,
        ScoreScaled=np.exp(df['Percent Black / Hispanic'].values*2.5),
        ENI=df['Economic Need Index'].values,
        BHPercent=df['Percent Black / Hispanic'].values*100        
    )
)

circle = Circle(x="lon", y="lat", size="ScoreScaled", fill_color={'field': 'ENI', 'transform': color_mapper}, fill_alpha=0.5, line_color=None)
p.add_glyph(source, circle)

color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

p.add_layout(color_bar, 'right')

show(p)
