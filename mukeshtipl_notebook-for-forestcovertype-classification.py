import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#flags for debugging and experimentations

KEEP_OUTLIERS=True

KEEP_ZERO_VALUES=True

EXIT_FOR_DEBUG=False



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
train.head()
train.shape,test.shape
#Keep backup of orignal data

train_orig,test_orig=train,test

train_orig.shape,test_orig.shape,train.shape,test.shape
train.dtypes
test.dtypes
test.head()
train.describe()
test.describe()
#print Count of Unique Values in each column

for column in list(train.columns):

    print ("{0:25} {1}".format(column, train[column].nunique()))
# Columns Identified which can be dropped for model

columns_to_drop=['Soil_Type7','Soil_Type15']

columns_to_drop
#print Count of Unique Values in each column

for column in list(test.columns):

    print ("{0:25} {1}".format(column, test[column].nunique()))
#Change data type of Continous variable columns to float

continuos_columns={'Elevation':float,'Aspect':float,'Slope':float,'Horizontal_Distance_To_Hydrology':float,'Vertical_Distance_To_Hydrology':float,

                   'Horizontal_Distance_To_Roadways':float,'Hillshade_9am':float,'Hillshade_Noon':float,'Hillshade_3pm':float,'Horizontal_Distance_To_Fire_Points':float}

train=train.astype(continuos_columns)

continuos_column_list=list(continuos_columns.keys())

train[continuos_column_list].dtypes
test=test.astype(continuos_columns)

test[continuos_column_list].dtypes
train_test_continuous_values=pd.DataFrame(train[continuos_column_list].nunique() ,columns=['train']).join(pd.DataFrame(test[continuos_column_list].nunique() ,columns=['test']) )

    

train_test_continuous_values


train_test_continuous_values.plot(kind='barh',figsize=(15,6))
train = train.drop(["Id"], axis = 1)



test_ids = test["Id"]



test = test.drop(["Id"], axis = 1)
#Check if there is any missing/null values in dataset

train.isnull().values.all()
test.isnull().values.all()
Cover_Type_Name=['Spruce_Fir','Lodgepole_Pine','Ponderosa_Pine','Cottonwood_Willow','Aspen','Douglas_fir','Krummholz']

Cover_Type_Name
#Uniqe Cover_Type values , will be used for chart , Soil Type by Cover Type

Cover_Type=train['Cover_Type'].unique()

Cover_Type=sorted(Cover_Type)

Cover_Type_Dict=dict(zip(Cover_Type,Cover_Type_Name))

Cover_Type_Dict
#Create a new Column having Name of Cover Type

train['Cover_Type_Name'] = train['Cover_Type'].map(Cover_Type_Dict)

train[['Cover_Type','Cover_Type_Name']].head()


Wilderness_Area_Columns=["Wilderness_Area" + str(i) for i in range(1,5)]

Wilderness_Area_Columns
Wilderness_Area_Names=['Rawah_Wilderness','Neota_Wilderness','Comanche_Peak_Wilderness','Cache_la_Poudre_Wilderness']

Wilderness_Area_Names
# Wilderness_Area_Columns

train['Wilderness_Area']=train[Wilderness_Area_Columns].idxmax(axis=1)

train.head()
#Uniqe Wilderness_Area values , will be used later

Wilderness_Area=train['Wilderness_Area'].unique()

Wilderness_Area=sorted(Wilderness_Area)

Wilderness_Area_Dict=dict(zip(Wilderness_Area,Wilderness_Area_Names))

Wilderness_Area_Dict
#Create a new Column having Name of Wilderness_Area

train['Wilderness_Area_Name'] = train['Wilderness_Area'].map(Wilderness_Area_Dict)

train[['Wilderness_Area','Wilderness_Area_Name']].head()
Soil_Type_Columns=["Soil_Type" +str(i) for i in range(1,41)]

print(Soil_Type_Columns)
#Dichotomous/Boolean Variables/columns

Boolean_Columns=Soil_Type_Columns + Wilderness_Area_Columns

print(np.shape(Boolean_Columns))

print(Boolean_Columns)
#Create a new Column Soil Type, will convert all 40 columns into one column "Reverse One Hot Encoding"

train['Soil_Type']=train[Soil_Type_Columns].idxmax(axis=1)
#Just check is Reverse One Hot coding done properly , check values of first 9 Soil type columns

#train[['Soil_Type'] + Soil_Type_Columns].loc[ train['Soil_Type']=="Soil_Type1",:]

#search_values=["Soil_Type" +str(i) for i in range(1,9)]

#train[['Soil_Type'] + Soil_Type_Columns].loc[ train['Soil_Type'].isin(search_values),:]
test['Soil_Type']=test[Soil_Type_Columns].idxmax(axis=1)
test.head()
Soil_Type_Names=["Cathedral_family_-_Rock_outcrop_complex_extremely_stony",

"Vanet_-_Ratake_families_complex_very_stony",

"Haploborolis_-_Rock_outcrop_complex_rubbly",

"Ratake_family_-_Rock_outcrop_complex_rubbly",

"Vanet_family_-_Rock_outcrop_complex_complex_rubbly",

"Vanet_-_Wetmore_families_-_Rock_outcrop_complex_stony",

"Gothic_family",

"Supervisor_-_Limber_families_complex",

"Troutville_family_very_stony",

"Bullwark_-_Catamount_families_-_Rock_outcrop_complex_rubbly",

"Bullwark_-_Catamount_families_-_Rock_land_complex_rubbly",

"Legault_family_-_Rock_land_complex_stony",

"Catamount_family_-_Rock_land_-_Bullwark_family_complex_rubbly",

"Pachic_Argiborolis_-_Aquolis_complex",

"unspecified_in_the_USFS_Soil_and_ELU_Survey",

"Cryaquolis_-_Cryoborolis_complex",

"Gateview_family_-_Cryaquolis_complex",

"Rogert_family_very_stony",

"Typic_Cryaquolis_-_Borohemists_complex",

"Typic_Cryaquepts_-_Typic_Cryaquolls_complex",

"Typic_Cryaquolls_-_Leighcan_family_till_substratum_complex",

"Leighcan_family_till_substratum_extremely_bouldery",

"Leighcan_family_till_substratum_-_Typic_Cryaquolls_complex",

"Leighcan_family_extremely_stony",

"Leighcan_family_warm_extremely_stony",

"Granile_-_Catamount_families_complex_very_stony",

"Leighcan_family_warm_-_Rock_outcrop_complex_extremely_stony",

"Leighcan_family_-_Rock_outcrop_complex_extremely_stony",

"Como_-_Legault_families_complex_extremely_stony",

"Como_family_-_Rock_land_-_Legault_family_complex_extremely_stony",

"Leighcan_-_Catamount_families_complex_extremely_stony",

"Catamount_family_-_Rock_outcrop_-_Leighcan_family_complex_extremely_stony",

"Leighcan_-_Catamount_families_-_Rock_outcrop_complex_extremely_stony",

"Cryorthents_-_Rock_land_complex_extremely_stony",

"Cryumbrepts_-_Rock_outcrop_-_Cryaquepts_complex",

"Bross_family_-_Rock_land_-_Cryumbrepts_complex_extremely_stony",

"Rock_outcrop_-_Cryumbrepts_-_Cryorthents_complex_extremely_stony",

"Leighcan_-_Moran_families_-_Cryaquolls_complex_extremely_stony",

"Moran_family_-_Cryorthents_-_Leighcan_family_complex_extremely_stony",

"Moran_family_-_Cryorthents_-_Rock_land_complex_extremely_stony"

]

print(Soil_Type_Names)


# Soil Type Name have hidden features like Stone Level, Is Complex, Is Rubbly.. lets extract these features

soil_feature_list=dict()



i=0

for Soil_Type in Soil_Type_Names:

    stony_level=0

    is_complex=0

    is_rubbly=0

    if Soil_Type.find("stony")!=-1:

        stony_level=1

        if Soil_Type.find("very")!=-1:

            stony_level=2

        if Soil_Type.find("extremely")!=-1:

            stony_level=3

    if Soil_Type.find("rubbly")!=-1:

         is_rubbly=1

    if Soil_Type.find("complex")!=-1:

         is_complex=1

    soil_quality=(stony_level,is_complex,is_rubbly)

    #print(Soil_Type_Columns[i],soil_quality)

    soil_feature_list[Soil_Type_Columns[i]]=soil_quality

    i=i+1

            

soil_feature_list
def get_soil_features(row):

    soil_type=row["Soil_Type"]

    soil_features=soil_feature_list[soil_type]

    # Soil Quality columns will have composition of all three soild related column

    soil_quality=1000 + (soil_features[0]*100) + (soil_features[1]*10) + soil_features[2] 



    return pd.Series({'0':soil_features[0],'1':soil_features[1],'2':soil_features[2],'3':soil_quality})


train[['Soil_Stony_Level',"Soil_Complex","Soil_Rubbly","Soil_Quality"]]=train.apply(lambda row:get_soil_features(row),axis=1)



#Convert Soil_Stony_Level to OHE columns , Lets keep both as stone level categorial can also acts as numeric column

soil_stony_OHE_cols=pd.get_dummies(train['Soil_Stony_Level'])

soil_stony_OHE_cols_names=["Soil_Stony_Level_" + str(colname) for colname in  soil_stony_OHE_cols.columns ]

soil_stony_OHE_cols.columns=soil_stony_OHE_cols_names

train=train.join(soil_stony_OHE_cols)
train.head()
test[['Soil_Stony_Level',"Soil_Complex","Soil_Rubbly","Soil_Quality"]]=test.apply(lambda row:get_soil_features(row),axis=1)

soil_stony_OHE_cols=pd.get_dummies(test['Soil_Stony_Level'])

soil_stony_OHE_cols_names=["Soil_Stony_Level_" + str(colname) for colname in  soil_stony_OHE_cols.columns ]

soil_stony_OHE_cols.columns=soil_stony_OHE_cols_names

test=test.join(soil_stony_OHE_cols)

test.head()
test.head()




#continuos_column_list.append('Soil_Stony_Level')

categorial_column_list = ['Cover_Type_Name', 'Wilderness_Area', 'Soil_Type','Soil_Stony_Level','Soil_Complex','Soil_Rubbly','Soil_Quality']
#exit for debug
#print Count of Unique Values in each continuos column

train_test_continuous_values
# Distribution of Target (Cover Type)

train.Cover_Type_Name.value_counts().plot(kind="barh",figsize=(10,5) ,color='green')

plt.xlabel("Count")

plt.ylabel("Cover Type")
#Function to plot charts of All features

def features_plots(data,continuous_vars,discrete_vars=None):

    plt.figure(figsize=(20,24.5))

    for i, cv in enumerate(continuous_vars):

        plt.subplot(9, 2, i+1)

        plt.hist(data[cv], bins=len(data[cv].unique()),color='green')

        plt.title(cv)

        plt.ylabel('Frequency')

    if discrete_vars is not None:

        for i, dv in enumerate(discrete_vars):

            plt.subplot(9, 2, i+ 1 + len(continuous_vars))

            data[dv].value_counts().plot(kind='bar', title=dv,color='blue')

            plt.ylabel('Frequency')
continuous_vars=continuos_column_list

discrete_vars = ['Cover_Type_Name', 'Wilderness_Area', 'Soil_Type','Soil_Stony_Level','Soil_Quality']

features_plots(train,continuous_vars,discrete_vars)
features_plots(test,continuous_vars)
number_of_columns=5

number_of_rows=2

column_names = list(continuos_column_list)

plt.figure(figsize=(4*number_of_columns,6*number_of_rows))

for i in range(0,len(column_names)):

    plt.subplot(number_of_rows +1,number_of_columns,i+1)

    dist=sns.distplot(train[column_names[i]],kde=True,color='green',vertical =False,

                      kde_kws={'color':'red','lw':3,'label':'KDE'},

                      hist_kws={'color':'green','lw':4,'label':'HIST','alpha':0.8}) 

    #dist.set_title("Distribution Plot")
number_of_columns=5

number_of_rows=2

column_names = list(continuos_column_list)

plt.figure(figsize=(4*number_of_columns,6*number_of_rows))

for i in range(0,len(column_names)):

    plt.subplot(number_of_rows +1,number_of_columns,i+1)

    dist=sns.distplot(test[column_names[i]],kde=True,color='green',vertical =False,

                      kde_kws={'color':'red','lw':3,'label':'KDE'},

                      hist_kws={'color':'green','lw':4,'label':'HIST','alpha':0.8}) 

    #dist.set_title("Distribution Plot")
train['Horizontal_Distance_To_Hydrology'].value_counts(normalize=True).head()
test['Horizontal_Distance_To_Hydrology'].value_counts(normalize=True).head()
train['Vertical_Distance_To_Hydrology'].value_counts(normalize=True).head()
test['Vertical_Distance_To_Hydrology'].value_counts(normalize=True).head()
train['Hillshade_3pm'].value_counts(normalize=True)
test['Hillshade_3pm'].value_counts(normalize=True)
train['Hillshade_3pm'].value_counts(normalize=True)[0],test['Hillshade_3pm'].value_counts(normalize=True)[0]
def replace_zeros_with_median(data,column):

    #Need to check , which option provide better results

    if KEEP_ZERO_VALUES==False:

        median=data.loc[data[column]!=0,column].median()

        data[column] = np.where(data[column] ==0, median,data[column])

    else:

        pass
#Replace 0 with median value in Horizontal_Distance_To_Hydrology , Vertical_Distance_To_Hydrology for both test and train

replace_zeros_with_median(train,'Horizontal_Distance_To_Hydrology')

train['Horizontal_Distance_To_Hydrology'].value_counts(normalize=True).head()
replace_zeros_with_median(test,'Horizontal_Distance_To_Hydrology')

test['Horizontal_Distance_To_Hydrology'].value_counts(normalize=True).head()
replace_zeros_with_median(train,'Vertical_Distance_To_Hydrology')

train['Vertical_Distance_To_Hydrology'].value_counts(normalize=True)
replace_zeros_with_median(test,'Vertical_Distance_To_Hydrology')

test['Vertical_Distance_To_Hydrology'].value_counts(normalize=True).head()
features_plots(train,['Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology'])
features_plots(test,['Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology'])
#Univariate Analysis on Traning Data for Soil Type

train['Soil_Type'].value_counts().plot(kind='bar', title="Distribution of Soil Types",color='blue',figsize = (18, 6))
# Distribution of Soil Type in test data

test['Soil_Type'].value_counts().plot(kind='bar', title="Distribution of Soil Types",color='blue',figsize = (18, 6))
groupby_soiltype_covertype=train.groupby(["Soil_Type","Cover_Type_Name"])["Soil_Type"].count().unstack("Cover_Type_Name").fillna(0)

groupby_soiltype_covertype[Cover_Type_Name].plot.bar(title="Distribution of Soil Types by Cover Types",stacked=True,figsize=(18,7))
groupby_soilquality_covertype=train.groupby(["Soil_Quality","Cover_Type_Name"])["Soil_Quality"].count().unstack("Cover_Type_Name").fillna(0)

groupby_soilquality_covertype[Cover_Type_Name].plot.bar(title="Distribution of Soil Quality by Cover Types",stacked=True,figsize=(18,7))

#groupby_soilquality_covertype
soil_quality_list=[val for val in (set(train['Soil_Quality'].values))]

groupby_covertype_soiltquality=train.groupby(["Cover_Type_Name","Soil_Quality"])["Cover_Type_Name"].count().unstack("Soil_Quality").fillna(0)

groupby_covertype_soiltquality[soil_quality_list].plot.bar(title="Distribution of Soil Quality by Cover Types",stacked=True,figsize=(18,7))

#groupby_covertype_soiltquality
pd.options.display.float_format = '{:,.4f}'.format

train['Soil_Type'].value_counts(normalize=True)
# Filter out where values are less than threshold_to_drop% , these columns can be dropped

threshold_to_drop=0.02

train_soil_type_columns_to_drop=train['Soil_Type'].value_counts(normalize=True).loc[train['Soil_Type'].value_counts(normalize=True).values<threshold_to_drop]

list(train_soil_type_columns_to_drop.index)
test['Soil_Type'].value_counts(normalize=True)
# Filter out where values are less than 1%, can be changed to drop more columns



test_soil_type_columns_to_drop=test['Soil_Type'].value_counts(normalize=True).loc[test['Soil_Type'].value_counts(normalize=True).values<threshold_to_drop]

list(test_soil_type_columns_to_drop.index)
columns_to_drop = columns_to_drop + list( set(list(train_soil_type_columns_to_drop.index) + list( test_soil_type_columns_to_drop.index)))

                        

columns_to_drop
train=train.drop(columns_to_drop,axis=1)

test=test.drop(columns_to_drop,axis=1)
def box_plots(data,continuous_vars):

    plt.figure(figsize=(15,24.5))

    for i, cv in enumerate(continuous_vars):

        plt.subplot(int(len(continuous_vars)/2) +1 , 2, i+1)

        plt.boxplot(data[cv])

        plt.title(cv)

        plt.ylabel('Value')


box_plots(train,continuos_column_list)
def get_outlier_percentage(data,column_name):

    column_values=list(data[column_name])

    q75, q25 = np.percentile(column_values, [75 ,25])

    iqr = q75 - q25

    outlier_percentage=(len(data) - len([x for x in column_values if q75+(1.5*iqr)>=x>= q25-(1.5*iqr)]))*100/float(len(data))

    return outlier_percentage
# % of Outliers for Continuous Columns

outlier_dict={}

for key in continuos_column_list:

    outlier_percentage=get_outlier_percentage(train,key)

    print(key,":",outlier_percentage)

    outlier_dict[key]=outlier_percentage
#Chart of Outliers

from matplotlib.ticker import StrMethodFormatter

plt.figure(figsize=(15,5))

plt.barh(*zip(*outlier_dict.items()))



ax=plt.gca()



ax.set_xticklabels(outlier_dict.values(),rotation=0)

ax.set_xlabel("Percentage of Outliers")

ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f} %'))

plt.show()
number_of_columns=5

number_of_rows=2

column_names = list(continuos_column_list)

plt.figure(figsize=(4*number_of_columns,6*number_of_rows))

for i in range(0,len(column_names)):

    plt.subplot(number_of_rows +1,number_of_columns,i+1)

    dist=sns.distplot(train[column_names[i]],kde=True,color='green',vertical =False,

                      kde_kws={'color':'red','lw':3,'label':'KDE'},

                      hist_kws={'color':'green','lw':4,'label':'HIST','alpha':0.8})
def remove_outliers(data,column):

    Q1=data[column].quantile(0.25)

    Q3=data[column].quantile(0.75)

    IQR=Q3-Q1

    data = data[~((data[column]< (Q1 - 1.5 * IQR)) |(data[column] > (Q3 + 1.5 * IQR)))]

    return data
def replace_outliers_with_capping(data,column):

    #Done lower degree (2%)  capping to avoid data loss

    upper_lim = data[column].quantile(.98)

    lower_lim = data[column].quantile(.02)

    data.loc[(data[column] > upper_lim),column] = upper_lim

    data.loc[(data[column] < lower_lim),column] = lower_lim

    return data
def replace_outliers_with_median(data,column):

    Q1=data[column].quantile(0.25)

    Q3=data[column].quantile(0.75)

    IQR=Q3-Q1

    upper=Q3 + (1.5*IQR)

    lower=Q1 - (1.5*IQR)

    outlier_mask=[True if x <lower or x >upper else False for x in data[column]]

    column_median=data[column].median()

    data.loc[outlier_mask,column]=column_median

    return data
#Before outlier modified

train.shape
for column in continuos_column_list:

    #Need to check which option provide better results

    

    if KEEP_OUTLIERS == False:

        #train=remove_outliers(train,column)

        #train=replace_outliers_with_median(train,column)

        train=replace_outliers_with_capping(train,column)

    else:

        pass
#After outlier removals

print(train.shape)

train.head()


features_plots(test,continuous_vars)
number_of_columns=5

number_of_rows=2

column_names = list(continuos_column_list)

plt.figure(figsize=(4*number_of_columns,6*number_of_rows))

for i in range(0,len(column_names)):

    plt.subplot(number_of_rows +1,number_of_columns,i+1)

    dist=sns.distplot(train[column_names[i]],kde=True,color='green',vertical =False,

                      kde_kws={'color':'red','lw':3,'label':'KDE'},

                      hist_kws={'color':'green','lw':4,'label':'HIST','alpha':0.8})
from sklearn import preprocessing

x_norm=  train[continuos_column_list] #Normalize numeric values 

pd.set_option('display.width',1700)

pd.set_option('precision',2)

#corr_mat=x_norm.corr(method="pearson")



corr_mat=train[ continuos_column_list +['Cover_Type']].corr(method="pearson")

corr_mat
plt.figure(figsize=(10,10))

sns.set(font_scale=0.9)

sns.heatmap(corr_mat,annot=True,square=True,vmax=0.8,cmap="inferno",fmt='.2f')
def pair_boxplot(data,continuous_vars,by_column):

    

    fig=plt.figure(figsize=(20,60))

    i=0

    for i, cv in enumerate(continuous_vars):

        ax=fig.add_subplot(int(len(continuous_vars))+1, 2, i+1)

        #data.boxplot(column=cv,return_type='axes',by=by_column,ax=ax)

        #print(len(continuous_vars))

        data.boxplot(column=cv,return_type='axes',by=by_column,ax=ax)

        ax.set_title("Box Plot for " + cv + " and " + by_column)

        ax.set_xlabel("Cover Type")

        ax.set_ylabel(cv)

        #plt.title(cv)

        #plt.ylabel('Value')
pair_boxplot(train,continuous_vars,'Cover_Type')
train.boxplot(column=['Elevation'],return_type='axes',by="Cover_Type",figsize=(15,7))

plt.show()
#Lets create a new feature to widen the gap in Elevation

train['ELV_SQR']=train['Elevation']**2

test['ELV_SQR']=test['Elevation']**2

continuos_column_list.append("ELV_SQR")

train.boxplot(column=['ELV_SQR'],return_type='axes',by="Cover_Type",figsize=(15,7))

plt.show()


#joint_plot=sns.jointplot("Cover_Type","Elevation",data=train,kind='kde',color='green')
#kde_plot=sns.kdeplot(train["Cover_Type"],train["Elevation"],cmap="PRGn",cbar=True)
sns.pairplot(train,height=4,vars=['Elevation','Slope','Aspect','Horizontal_Distance_To_Hydrology'],hue='Cover_Type')
sns.pairplot(train,height=4,vars=['Elevation','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'],hue='Cover_Type')
sns.pairplot(train,height=4,vars=['Elevation','Hillshade_9am','Hillshade_Noon','Hillshade_3pm'],hue='Cover_Type')
'''

cmap = sns.cubehelix_palette(as_cmap=True)



f, ax = plt.subplots(figsize=(20,10))

points = ax.scatter(train['Hillshade_Noon'],train['Elevation'] , c=train['Cover_Type'],label=Cover_Type_Name ,s=20, cmap='rainbow')

#plt.xticks(np.arange(0, 400,20))

#plt.axis('scaled')

f.colorbar(points)

#ax.legend(loc='upper left',fontsize='large')

#plt.legend()

plt.show()

'''
#plt.subplots(figsize=(15,10))

#sns.swarmplot("Cover_Type","Elevation",data=train)
sns.set()
'''

facet=sns.FacetGrid(train,col='Wilderness_Area_Name',hue='Cover_Type_Name',col_wrap=2,height=7,aspect=1.5)

facet.map(plt.scatter,'Horizontal_Distance_To_Hydrology','Elevation',alpha=0.5)

facet.add_legend()

'''
def pair_scatterplot(data,continuous_vars,y_column,target_column):

    #plt.figure(figsize=(20,10))

    cmap = sns.cubehelix_palette(as_cmap=True)

    for i, cv in enumerate(continuous_vars):

        #plt.subplot(7, 2, i+1)

        #data.boxplot(column=cv,return_type='axes',by=by_column)

        f, ax = plt.subplots(figsize=(20,10))

        ax.set_xlabel(cv)

        ax.set_ylabel(y_column)

        points = ax.scatter(train[cv],train[y_column] , c=train[target_column],label=Cover_Type_Name ,s=20, cmap='rainbow')

        f.colorbar(points)

    plt.show()
#pair_scatterplot(train,continuous_vars,'Elevation','Cover_Type')
#Drop Extra columns

extra_columns_in_train=['Cover_Type_Name' ,'Wilderness_Area','Wilderness_Area_Name' ,'Soil_Type']

train=train.drop(extra_columns_in_train,axis=1)

extra_columns_in_test=['Soil_Type']

test=test.drop(extra_columns_in_test,axis=1)
#Lets Remove all Soil Type columns , to check if it helps

soil_type_cols=[colname for colname in train.columns if colname.find("Soil_Type")!=-1 ]

#print(soil_type_cols)

#train=train.drop(soil_type_cols,axis=1)

#test=test.drop(soil_type_cols,axis=1)
train.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

scaler=MinMaxScaler()

train[continuos_column_list] = scaler.fit_transform(train[continuos_column_list])

test[continuos_column_list] = scaler.fit_transform(test[continuos_column_list])
test.head()
train.head()
X,Y=train.drop(['Cover_Type'], axis=1), train['Cover_Type']
from sklearn.decomposition import PCA

pca =PCA(n_components=15,whiten=True)

x_reduced=pca.fit_transform(X)
pca.explained_variance_
pca.explained_variance_ratio_
plt.plot(pca.explained_variance_ratio_)

plt.xlabel("Dimension")

plt.ylabel("Explained Variance Ratio")
#!pip install lightgbm

from lightgbm import LGBMClassifier, plot_importance

def get_LGBC():

    return LGBMClassifier(n_estimators=500,  

                     learning_rate= 0.1,

                     objective= 'multiclass', 

                     num_class=7,

                     random_state= 2019,

#                      class_weight=class_weight_lgbm,

                     n_jobs=-1)

lgbc= get_LGBC()

lgbc.fit(X,Y)



plot_importance(lgbc, ignore_zero=False, max_num_features=20)
x_train, x_test, y_train, y_test = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.2)
'''

from sklearn.svm import LinearSVC

svc_clf=LinearSVC(max_iter=10000)

svc_clf.fit(x_train,y_train)

accuracy=svc_clf.score(x_test,y_test)

print(accuracy)

'''
'''

import sklearn.decomposition 

# Create a Randomized PCA model that takes two components

randomized_pca = sklearn.decomposition.RandomizedPCA(n_components=2)



# Fit and transform the data to the model

reduced_data_rpca = randomized_pca.fit_transform(X)



# Create a regular PCA model 

pca = sklearn.decomposition.PCA(n_components=2)



# Fit and transform the data to the model

reduced_data_pca = pca.fit_transform(X)



# Inspect the shape

reduced_data_pca.shape



# Print out the data

print(reduced_data_rpca)

print(reduced_data_pca)

'''
'''

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(x_train, y_train)

train_accuracy=clf.score(x_train, y_train)

test_accuracy=clf.score(x_test,y_test)

print(train_accuracy,test_accuracy)

'''
'''

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state=10, learning_rate=0.05,

n_estimators=200, max_depth=5, max_features=20)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

train_accuracy=clf.score(x_train, y_train)

test_accuracy=clf.score(x_test,y_test)

print(train_accuracy,test_accuracy)



'''
'''

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier



# Choose the type of classifier. 

clf = GradientBoostingClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [40, 60, 90,120], 

              'max_features': [10, 20,30], 

              'learning_rate':[0.01,0.05,0.1,0.2],

              'max_depth': [2, 3, 5, 10], 

              'random_state': [5, 10, 15]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer,cv=5)

grid_obj = grid_obj.fit(x_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print(accuracy_score(y_test, predictions),clf)

'''
def get_rf():

    from sklearn.ensemble import RandomForestClassifier

    model=RandomForestClassifier(n_estimators=700,

                                              criterion='gini', 

                                              max_depth=16,

                                              min_samples_split=2,

                                              min_samples_leaf=3, 

                                              min_weight_fraction_leaf=0.0, 

                                              max_features=15, 

                                              max_leaf_nodes=None, 

                                              bootstrap=True, 

                                              oob_score=False,

                                              n_jobs=1, 

                                              random_state=10,

                                              verbose=0, 

                                              warm_start=False, 

                                              class_weight=None)

    return model
from xgboost import XGBClassifier, plot_importance

def get_xgb():

    

#model = RandomForestClassifier(n_estimators=100)

#model.fit(X_train, y_train)





#!pip install xgboost

#https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e

#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    





    model = XGBClassifier(silent=False, 

                          learning_rate=0.185,  

                          colsample_bytree = 0.6,

                          subsample = 0.8,

                          n_estimators=180,

                          objective='multi:softmax',

                          num_class=7,

                          reg_alpha = 0.2,

                          max_depth=6,

                          min_child_weight=1,

                          scale_pos_weight=0,

                          reg_lambda=1,

                          random_state=10,

                          gamma=1)

 

    return model



from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from itertools import cycle



def model_performance(model, X_train, y_train, y_test, y_score,n_classes=0):

    print ('Model : ',model)

    print( 'Test accuracy (Accuracy Score): %f'%metrics.accuracy_score(y_test, y_score))

    print ('Test accuracy (ROC AUC Score): %f'%metrics.roc_auc_score(y_test, y_score))

    #print ( 'Train accuracy: %f'%model.score(X_train, y_train))

    

    lw = 2

    

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])



    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    

    plt.figure(figsize=(20,10))

    

    

    plt.plot(fpr[2], tpr[2], color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    

    

    # Compute macro-average ROC curve and ROC area



    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])



    # Finally average it and compute AUC

    mean_tpr /= n_classes



    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves

    

    plt.plot(fpr["micro"], tpr["micro"],

             label='micro-average ROC curve (area = {0:0.2f})'

                   ''.format(roc_auc["micro"]),

             color='deeppink', linestyle=':', linewidth=4)



    plt.plot(fpr["macro"], tpr["macro"],

             label='macro-average ROC curve (area = {0:0.2f})'

                   ''.format(roc_auc["macro"]),

             color='navy', linestyle=':', linewidth=4)



    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green','yellow','blue','red'])

    for i, color in zip(range(n_classes), colors):

        plt.plot(fpr[i], tpr[i], color=color, lw=lw,

                 label='ROC curve of class {0} (area = {1:0.2f})'

                 ''.format(i, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Some extension of Receiver operating characteristic to multi-class')

    plt.legend(loc="lower right")

    plt.show()



train.shape,test.shape
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.1)
X_train.shape, X_val.shape, y_train.shape, y_val.shape




model =get_xgb()



#eval_set = [(X_val, y_val)]





eval_set = [(X_train, y_train), (X_val, y_val)]



#model=get_rf()

model.fit(X_train, y_train,eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True,early_stopping_rounds=10)



from sklearn.metrics import classification_report, accuracy_score

print("Train Accuracy:",model.score(X_train, y_train))

predictions = model.predict(X_val)

print("Test Accuracy:",accuracy_score(y_val, predictions))

model_performance(model, X_train,label_binarize(y_train,classes=[1, 2,3,4,5,6,7]),label_binarize(y_val,classes=[1, 2,3,4,5,6,7]) ,label_binarize(predictions,classes=[1, 2,3,4,5,6,7]) ,n_classes=7)
#from sklearn.metrics import classification_report, accuracy_score

results = model.evals_result()

#print(results)



epochs = len(results['validation_0']['merror'])

x_axis = range(0, epochs)



fig=plt.figure(figsize=(20,10))

# plot log loss

ax = fig.add_subplot(2,2,1)

plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')

plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test')

plt.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

#plt.show()



# plot classification error





ax = fig.add_subplot(2,2,2)

plt.plot(x_axis, results['validation_0']['merror'], label='Train')

plt.plot(x_axis, results['validation_1']['merror'], label='Test')

plt.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()

#from sklearn.metrics import classification_report, accuracy_score
#model.score(X_train, y_train)
#predictions = model.predict(X_val)

#accuracy_score(y_val, predictions)
test.head()
test_pred = model.predict(test)
#Save test predictions to file

output = pd.DataFrame({'id': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('submission_xgb_v3.csv', index=False)