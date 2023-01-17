### helpers
import re
import numpy as np

def recode_species(species_value):
    """Takes a string and returns classified species"""
    if species_value in ['Cpb','cpb', 'C.p.b.']:
        return 'Cpb'
    elif species_value in ['Red-eared slider', 'RES','REs']:
        return 'Res'
    else:
        return species_value

def recode_gravid(gravid_value):
    if str(gravid_value).upper() == "YES":
        return True
    else:
        return False

def recode_sex(sex_value):
    """Takes a string and returns f, m or unknown"""
    if sex_value in ['Male','male?','m','M']:
        return 'm'
    elif sex_value in ['Female','F','f']:
        return 'f'
    else:
        return 'unknown'

def recode_season(date):
    if date.month <= 6:
        return 'spr'
    else:
        return 'fal'

def recode_decimal(dirty_decimal=''):
    """Takes a string and returns a decimal"""
    _ = []
    if not dirty_decimal:
        return 0
    if str(dirty_decimal):
        _ = re.findall(r"[-+]?\d*\.\d+|\d+",str(dirty_decimal))
    if _:
        return _[0]
    else:
        return 0

def ecdf(data):
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points: n
        n = len(data)
        # x-data for the ECDF: x
        x = np.sort(data)
        # y-data for the ECDF: y
        y = np.arange(1, n+1) / n
        return x, y
    
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    """simulate the hypothesis that two variables have identical probability distributions."""
    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

### turtels!
import pandas as pd
import numpy as np

def get_clean_data():
    fileName1 = '../input/Turtle Data.xls'
    fileName2 = '../input/MF Trapping Data.xls'
    df = clean_data(fileName1,True)
    df['Capture Location']   = 'Gresham'
    

    df2 = clean_data(fileName2)
    df2['Capture Location']   = 'Mason Flats'
    df2['Source'] = 'MF Trapping Data.xlsx|All Capture Data'
    df = df.append(df2,sort=False)
    
    return df

def clean_data(fileName,big_file=False):
    #columnNames = ['Name','Address']

    print ("Loading data " + fileName)
    df = pd.DataFrame()
    if (big_file):
        for year in range(2008,2014+1):
            print(year)
            new = pd.read_excel(fileName,sheet_name=str(year))
            new['Source'] = '{}|{}'.format('Turtle Data.xls',str(year))
            df = df.append(new,sort=False)
    else:
        df = pd.read_excel(fileName)


    #DATA CLEANING
    cleaned = df.copy()
    # decimals
    print ("Cleaning decimals ...")
    cleaned['Weight'] = cleaned['Weight'].apply(recode_decimal)
    cleaned['Weight'] = pd.to_numeric(cleaned['Weight'],downcast='float')
    cleaned['Carapace'] = cleaned['Carapace'].apply(recode_decimal)
    cleaned['Carapace'] = pd.to_numeric(cleaned['Carapace'],downcast='float')
    cleaned['Plastron'] = cleaned['Plastron'].apply(recode_decimal)
    cleaned['Plastron'] = pd.to_numeric(cleaned['Plastron'],downcast='float')
    cleaned['Annuli'] = cleaned['Annuli'].apply(recode_decimal)
    cleaned['Annuli'] = pd.to_numeric(cleaned['Annuli'],downcast='integer')

    # other
    print ("Cleaning other values ...")
    cleaned['Gender'] = cleaned['Gender'].apply(recode_sex)
    cleaned['Species'] = cleaned['Species'].apply(recode_species)
    cleaned['Gravid'] = cleaned['Gravid'].apply(recode_gravid)
    # add features
    cleaned['Age_To_Weight'] = cleaned['Annuli'] / cleaned['Weight']
    buckets = 5
    buckets = int(cleaned['Annuli'].max() / buckets)
    labels = ["{0} - {1}".format(i, i + buckets) for i in range(0, cleaned['Annuli'].max(), buckets)]
    cleaned['Annuli_Group'] = pd.cut(cleaned.Annuli, range(0, cleaned.Annuli.max()+buckets, buckets), right=False, labels=labels)
    # Calcuate Number of recaptures
    df = cleaned[['ID','Date']].groupby('ID').count()
    df.columns = ['recapture_count']
    df.reset_index(inplace=True)
    cleaned = pd.merge(cleaned,df,how='outer',on='ID')

    # recalculate annuli
    df = pd.pivot_table(cleaned[cleaned.Annuli > 0],values=['Date','Annuli'],index=['ID'],aggfunc={'Date': min,'Annuli': min})
    df.columns = ['lowest_annuli','first_date']
    df.reset_index(inplace=True)
    
    cleaned = pd.merge(cleaned,df,how='outer',on='ID')
    cleaned['date_year'] = cleaned.Date.map(lambda x: x.year)
    cleaned['first_date_year'] = cleaned.first_date.map(lambda x: x.year)
    cleaned['new_annuli'] = cleaned.date_year - cleaned.first_date_year + cleaned.lowest_annuli
    cleaned.new_annuli = np.nan_to_num(cleaned.new_annuli)

    # distinguish Spring, Fall and pregnant females (don't care about juvenilles/unknown)
    cleaned['gender_plus'] = cleaned['Gender']
    #cleaned.loc[cleaned.gender_plus != 'unknown','gender_plus'] = cleaned.Gender + '_' + cleaned.Date.apply(hlp.recode_season)
    cleaned.loc[cleaned.Gravid == True,'gender_plus'] = 'f_gra'

    cleaned['gender_seasons'] = cleaned['Gender']
    cleaned.loc[cleaned.gender_seasons != 'unknown','gender_seasons'] = cleaned.Gender + '_' + cleaned.Date.apply(recode_season)
    cleaned.loc[cleaned.Gravid == True,'gender_seasons'] = 'f_gra'
    return cleaned


#Prepping the notebook
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# my custom helper functions
import scipy

#filter data: Naitive turle and relevant
print ("Filtering Natives ...")
natives = get_clean_data()
natives = natives[natives['Weight']!=0]
natives = natives[natives['Carapace']!=0]
natives = natives[natives['Plastron']!=0]
natives = natives[natives['Species']=='Cpb']
print ("Done")

#Show some data - basic metrics
natives[['Carapace','Plastron','Annuli','Weight']].describe()
#Show some data - Gender, Location etc
natives[['Gender','Capture Location']].describe(include='all')
### Basic Swarmplot to show Carapace / Plastron / Weight by Gender
plt.style.use('seaborn-notebook')
plt.rcParams['figure.dpi'] = 300
colors = [sns.xkcd_rgb['pale red'],sns.xkcd_rgb['denim blue'],sns.xkcd_rgb['medium green']]

print ("Plotting swarm plot with " + str(natives.Weight.count()) + ' samples...')
sns.set()
_ = plt.figure(1)
_ = plt.suptitle('Native Turtles Swarmplots ')

_1 =plt.subplot(2,2,1)
_ = sns.swarmplot(x='Gender', y='Carapace', data=natives,size=3,palette=colors)
_ = plt.ylabel('Carapace')

_2 =plt.subplot(2,2,2,sharey=_1)
_ = sns.swarmplot(x='Gender', y='Plastron', data=natives,size=3,palette=colors)
_ = plt.ylabel('Plastron')
_ = plt.tight_layout(rect=[0, 0.03, 1, 0.95])
_ = plt.subplot(2,2,3)
_ = sns.swarmplot(x='Gender', y='Weight', data=natives,size=3,palette=colors)
_ = plt.xlabel('Gender')
_ = plt.ylabel('Weight')

#
natives_melted = natives.copy()
natives_melted['Weight / 10'] = natives_melted.Weight.div(10)
natives_melted = pd.melt(natives_melted[['Carapace','Plastron','Weight / 10','Gender']],'Gender',var_name='measurement')

_ = sns.swarmplot(x="measurement", y="value", hue="Gender", data=natives_melted,palette=colors,size=3)
_ = plt.show()
### Same information as beswars above just displayed as ECDFs
plt.style.use('seaborn-notebook')
plt.rcParams['figure.dpi'] = 300
#colors = [sns.xkcd_rgb['pale red'],sns.xkcd_rgb['denim blue']]


print ("Plotting Cumulative Distribution Function with " + str(natives.Weight.count()) + ' samples...')
# Compute ECDFs

_ = plt.figure(2)
_ = plt.suptitle('Native Turtles Cumulative Distribution ')

x_CarapaceF, y_CarapaceF = ecdf(natives[natives['Gender']=='f']['Carapace'])
x_CarapaceM, y_CarapaceM = ecdf(natives[natives['Gender']=='m']['Carapace'])

_ = plt.subplot(2,2,1)
_ = plt.plot(x_CarapaceF, y_CarapaceF, marker='.',linestyle = 'none',color=colors[0])
_ = plt.plot(x_CarapaceM, y_CarapaceM, marker='.',linestyle = 'none',color=colors[1])
_ = plt.margins(0.02)
_ = plt.legend(('Female', 'Male'), loc='lower right')
_ = plt.xlabel('Carapace (mm)')
_ = plt.ylabel('ECDF')

x_PlastronF, y_PlastronF = ecdf(natives[natives['Gender']=='f']['Plastron'])
x_PlastronM, y_PlastronM = ecdf(natives[natives['Gender']=='m']['Plastron'])

_ = plt.subplot(2,2,2)
_ = plt.plot(x_PlastronF, y_PlastronF, marker='.',linestyle = 'none',color=colors[0])
_ = plt.plot(x_PlastronM, y_PlastronM, marker='.',linestyle = 'none',color=colors[1])
_ = plt.margins(0.02)
_ = plt.legend(('Female', 'Male'), loc='lower right')
_ = plt.xlabel('Plastron (mm)')
_ = plt.ylabel('ECDF')
_ = plt.tight_layout(rect=[0, 0.03, 1, 0.95])

x_WeightF, y_WeightF = ecdf(natives[natives['Gender']=='f']['Weight'])
x_WeightM, y_WeightM = ecdf(natives[natives['Gender']=='m']['Weight'])

_ = plt.subplot(2,2,3)
_ = plt.plot(x_WeightF, y_WeightF, marker='.',linestyle = 'none',color=colors[0])
_ = plt.plot(x_WeightM, y_WeightM, marker='.',linestyle = 'none',color=colors[1])
_ = plt.margins(0.02)
_ = plt.legend(('Female', 'Male'), loc='lower right')
_ = plt.xlabel('Weight (g)')
_ = plt.ylabel('ECDF')
#
natives_melted = natives.copy()
natives_melted['Weight / 10'] = natives_melted.Weight.div(10)
natives_melted = pd.melt(natives_melted[['Carapace','Plastron','Weight / 10','Gender']],'Gender',var_name='measurement')

_ = sns.swarmplot(x="measurement", y="value", hue="Gender", data=natives_melted,palette=colors,size=3)
_ = plt.show()
someColumns = [
    'Date',
    'ID',
    'Capture Location',
    'Gender',
    'first_date',
    'new_annuli',
    'Carapace',
    'Plastron',
    'Weight',
]

display(natives[
    (natives.Plastron <47) |
    (natives.Carapace <47) |
    (natives.Plastron >220) 
][someColumns])


#2d histogram Carapace / Plastron / Weight - Females
sns.reset_orig()
#plt.style.use('seaborn-notebook')
plt.rcParams['figure.dpi'] = 300
## Lineplot Age + Weight, Carapace, Plastron

### FEMALE
females = natives[natives['Gender']=='f']
females = females[females['Annuli']!=0]
print ("Plotting Histogram Weight, Carapace and Plastron with " + str(females.Weight.count()) + ' samples...')

_ = plt.figure(3)
_1 =plt.subplot(2,2,1)

#_ = plt.scatter(females['Annuli'], females['Carapace'])
_ = plt.hist2d(females['Annuli'], females['Carapace'],cmap='Reds',bins=(10,10))
_ = plt.colorbar()
_ = plt.ylabel('Carapace')

_2 =plt.subplot(2,2,2,sharey=_1)
#_ = plt.scatter(females['Annuli'],females['Plastron'])
_ = plt.hist2d(females['Annuli'], females['Plastron'],cmap='Reds',bins=(10,10))
_ = plt.colorbar()
_ = plt.ylabel('Plastron')

_ = plt.suptitle('Native Females Histogram ')
_ = plt.subplot(2,2,3)
#_ = plt.scatter(females['Annuli'], females['Weight'],s=10)
_ = plt.hist2d(females['Annuli'], females['Weight'],cmap='Reds',bins=(10,10))
_ = plt.colorbar()
_ = plt.xlabel('Annuli')
_ = plt.ylabel('Weight')
_ = plt.tight_layout(rect=[0, 0.03, 1, 0.95])
##2d histogram Carapace / Plastron / Weight - Males
sns.reset_orig()
#plt.style.use('seaborn-notebook')
plt.rcParams['figure.dpi'] = 300
## Lineplot Age + Weight, Carapace, Plastron

### MALE
males = natives[natives['Gender']=='m']
males = males[males['Annuli']!=0]
print ("Plotting Histogram Weight, Carapace and Plastron with " + str(males.Weight.count()) + ' samples...')

_ = plt.figure(3)

_1 =plt.subplot(2,2,1)
#_ = plt.scatter(males['Annuli'], males['Carapace'])
_ = plt.hist2d(males['Annuli'], males['Carapace'],cmap='Blues',bins=(10,10))
_ = plt.colorbar()
_ = plt.ylabel('Carapace')

_2 =plt.subplot(2,2,2,sharey=_1)
#_ = plt.scatter(males['Annuli'],males['Plastron'])
_ = plt.hist2d(males['Annuli'], males['Plastron'],cmap='Blues',bins=(10,10))
_ = plt.colorbar()
_ = plt.ylabel('Plastron')

_ = plt.suptitle('Native Males Histogram ')
_ = plt.subplot(2,2,3)
#_ = plt.scatter(males['Annuli'], males['Weight'],s=10)
_ = plt.hist2d(males['Annuli'], males['Weight'],cmap='Blues',bins=(10,10))
_ = plt.colorbar()
_ = plt.xlabel('Annuli')
_ = plt.ylabel('Weight')
_ = plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#Hyposis: same distribution in all capture locations
native_portland = natives[(natives['Capture Location']=='Mason Flats')]
native_gresham = natives[(natives['Capture Location']=='Gresham') ]
#&(natives['Gender']=='f')
genders = ['f','m']
dimension = 'Weight'
_ = plt.figure(6)
i = 0
for gender in genders:
    i += 1
    _ = plt.subplot(1,2,i)
    for _ in range(100):
        # Generate permutation samples
        native_portland_gender = native_portland[native_portland['Gender']==gender]
        native_gresham_gender = native_portland[native_portland['Gender']==gender]
        perm_sample_1, perm_sample_2 = permutation_sample(native_portland_gender[dimension],native_gresham_gender[dimension])

        # Compute ECDFs
        x_1, y_1 = ecdf(perm_sample_1)
        x_2, y_2 = ecdf(perm_sample_2)

        # Plot ECDFs of permutation sample
        _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                     color='green', alpha=0.02)
        _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                     color='orange', alpha=0.02)

    # Create and plot ECDFs from original data
    x_1, y_1 = ecdf(native_portland_gender[dimension])
    x_2, y_2 = ecdf(native_portland_gender[dimension])
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='green',label='Portland')
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='orange',label='Gresham')
    _ = plt.legend()
    _ = plt.title(gender)
    # Label axes, set margin, and show plot
    plt.margins(0.02)
    _ = plt.xlabel(dimension)
    _ = plt.ylabel('ECDF')
plt.show()
native_females = natives[(natives['Gender']=='f')]
native_males = natives[(natives['Gender']=='m')]
corrleations = native_males.loc[slice(None),['Annuli','Weight','Carapace','Plastron']].corr()

_ = sns.heatmap(corrleations,annot=True)
_ = plt.xticks(rotation=45)
_ = plt.show()

#
more_colors = {
    'f_spr': sns.xkcd_rgb['soft pink'],
    'f_fal': sns.xkcd_rgb['hot pink'],
    'f_gra': sns.xkcd_rgb['gold'],
    'm_spr': sns.xkcd_rgb['baby blue'],
    'm_fal': sns.xkcd_rgb['denim blue'],
    'unknown': sns.xkcd_rgb['very light green'],
    'f': sns.xkcd_rgb['hot pink'],
    'm': sns.xkcd_rgb['denim blue'],
}
natives_melted13 = natives.copy()
natives_melted13['Weight / 10'] = natives_melted13.Weight.div(10)
natives_melted13 = pd.melt(natives_melted13[['Carapace','Plastron','Weight / 10','gender_plus']],'gender_plus',var_name='measurement')

sns.swarmplot(x="measurement", y="value", hue="gender_plus", data=natives_melted13,palette=more_colors,size=3)


someColumns = [
    'Carapace',
    'Plastron',
    'Weight',
    'gender_plus'
]
stats_df = natives[someColumns].groupby('gender_plus').mean()

gender_stats_parms = {
    'cellText': stats_df.values,
    'rowLabels': stats_df.index.values,
    'colLabels': ["Carapace (Mean)","Plastron (Mean)","Weight (Mean)"],
    'loc': 'top',}
    
gender_stats = plt.table(**gender_stats_parms)
plt.xticks([])
plt.tight_layout(True)

_ = plt.show()

display("Mean values per gender and location")
someColumns = [
    'Carapace',
    'Plastron',
    'Weight',
    'gender_plus',
    'Capture Location',
]
stats_df = natives[natives['Gender']!='unknown'][someColumns].groupby(['gender_plus','Capture Location']).mean()
display(stats_df)

#sns.set()
import statsmodels
natives_noUnknown = natives[natives['Gender']!='unknown'].copy()
is_robust = False


### Swarmplot with Hue 
natives = natives[natives['new_annuli']!=0]
plt.rcParams['figure.dpi'] = 500
print ("NEW ANNULI - Plotting scatter plot with hue  " + str(natives.Weight.count()) + ' samples...')
lmplotParams = {
    'x': 'new_annuli', 
    'y': 'Carapace', 
    'hue': 'gender_plus',
    'col': "Capture Location",
    'data': natives_noUnknown,
    'palette': more_colors,
    'scatter': True,
    'fit_reg': False,
    'legend_out': False,
    'x_jitter': .5,
    'y_jitter': .1,
    'scatter_kws': {"s": 7,"alpha": .7},
}
sns.set()
_ = plt.figure(5)
#lowess=True, too bus
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Carapace')
lmplotParams['y'] = 'Plastron'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Plastron')
lmplotParams['y'] = 'Weight'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Weight')

natives_melted2 = natives.copy()
natives_melted2['Weight / 10'] = natives_melted2.Weight.div(10)
natives_melted2 = pd.melt(natives_melted2[['Carapace','Plastron','Weight / 10','gender_seasons']],'gender_seasons',var_name='measurement')

_ = sns.swarmplot(x="measurement", y="value", hue="gender_seasons", data=natives_melted2,palette=more_colors,size=3)
_ = plt.show()
import statsmodels
natives2 = natives[natives['Gender']!='unknown']
is_robust = False


### Swarmplot with Hue and Linear Digression 
natives2 = natives2[natives2['new_annuli']!=0]
#colors = [sns.xkcd_rgb['pale red'],sns.xkcd_rgb['denim blue'],sns.xkcd_rgb['medium green']]
plt.rcParams['figure.dpi'] = 500
print ("NEW ANNULI - Plotting scatter plot with hue  " + str(natives2.Weight.count()) + ' samples...')
lmplotParams = {
    'x': 'new_annuli', 
    'y': 'Carapace', 
    'hue': 'gender_seasons',
    'col': "Capture Location",
    'data': natives2,
    'palette': more_colors,
    'scatter': True,
    'fit_reg': False,
    #'lowess': True,
    'legend_out': False,
    'x_jitter': .5,
    'y_jitter': .1,
    'scatter_kws': {"s": 7,"alpha": .7},
}
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Carapace')
lmplotParams['y'] = 'Plastron'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Plastron')
lmplotParams['y'] = 'Weight'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Weight')

#lmplotParams['lowess'] = False
lmplotParams['order'] = 2
lmplotParams['ci'] = None
lmplotParams['fit_reg'] = True
lmplotParams['truncate'] = True

lmplotParams['y'] = 'Carapace'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Carapace')
plt.show()
lmplotParams['y'] = 'Plastron'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Plastron')
plt.show()
lmplotParams['y'] = 'Weight'
_ = sns.lmplot(**lmplotParams)
_ = plt.ylabel('Weight')
plt.show()