

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import operator

import seaborn as sn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

file = pd.read_csv('../input/2000-16-traffic-flow-england-scotland-wales/accidents_2012_to_2014.csv',index_col='Accident_Index')

frame_list = []

frame_list.append(file)

accident_data = pd.concat(frame_list)

FatalAccidnet = accident_data[accident_data['Accident_Severity']==1]

#modify

Modify_Dataset=FatalAccidnet

Modify_Dataset["Hour"] = FatalAccidnet.Time.str.slice(0,2)

Modify_Dataset["Month"] = FatalAccidnet.Date.str.slice(3,5)



All_Modify_Dataset=accident_data

All_Modify_Dataset["Hour"] = All_Modify_Dataset.Time.str.slice(0,2)

All_Modify_Dataset["Month"] = All_Modify_Dataset.Date.str.slice(3,5)
#replace the nan value

All_Modify_Dataset=All_Modify_Dataset.drop(columns=['Did_Police_Officer_Attend_Scene_of_Accident','Location_Easting_OSGR', 'Location_Northing_OSGR','Junction_Detail','Longitude','Latitude','Number_of_Casualties','Police_Force','Time','Date'])

All_Modify_Dataset['Junction_Control'].fillna('null', inplace=True) #replace nan



pd.isnull(All_Modify_Dataset).sum() > 0
#replace the nan value

Modify_Dataset['Junction_Control'].fillna('null', inplace=True) #replace nan

Modify_Dataset['LSOA_of_Accident_Location'].fillna('null', inplace=True) #replace nan

Modify_Dataset['Hour'].fillna('null', inplace=True) #replace nan

Modify_Dataset['Road_Surface_Conditions'].fillna('null', inplace=True) #replace nan

Modify_Dataset['Carriageway_Hazards'].fillna('null', inplace=True) #replace nan

Modify_Dataset['Time'].fillna('null', inplace=True) #replace nan

Modify_Dataset['Junction_Detail'].fillna('null', inplace=True) #replace nan

Modify_Dataset['Special_Conditions_at_Site'].fillna('null', inplace=True) #replace nan

pd.isnull(Modify_Dataset).sum() > 0
Modify_Dataset=Modify_Dataset.drop(columns=['Did_Police_Officer_Attend_Scene_of_Accident','Location_Easting_OSGR', 'Location_Northing_OSGR','Junction_Detail','Longitude','Latitude','Number_of_Casualties','Police_Force','Time','Date','Accident_Severity'])





#Modify_Dataset=Modify_Dataset.drop(columns=['Did_Police_Officer_Attend_Scene_of_Accident','Location_Easting_OSGR', 'Location_Northing_OSGR','Junction_Detail','Longitude','Latitude','Number_of_Casualties','Police_Force','Time','Date','Accident_Severity'])
#include fatal & non-fatal

data2=All_Modify_Dataset

data2=data2.drop(columns=['Year'])

from kmodes.kmodes import KModes

cluster_outcome=[]

km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data2)

cluster_outcome.append(km.cluster_centroids_)
#only for fatal accident

data=Modify_Dataset

data=data.drop(columns=['Year'])

data.head()

from kmodes.kmodes import KModes

cluster_outcome=[]

km = KModes(n_clusters=10, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

cluster_outcome.append(km.cluster_centroids_)

print(cluster_outcome)
print(cluster_outcome)
# onehot-preprocessing for association rule mining-using rapid miner

dataset_onehot = Modify_Dataset.copy()

dataset_onehot = pd.get_dummies(dataset_onehot, columns=['Number_of_Vehicles', 'Month', 'Day_of_Week',

       'Local_Authority_(District)', 'Local_Authority_(Highway)',

       '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',

       'Junction_Control', '2nd_Road_Class', '2nd_Road_Number',

       'Pedestrian_Crossing-Human_Control',

       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',

       'Weather_Conditions', 'Road_Surface_Conditions',

       'Special_Conditions_at_Site', 'Carriageway_Hazards',

       'Urban_or_Rural_Area', 'LSOA_of_Accident_Location', 'Year', 'Hour'], 

                                prefix = ['Number_of_Vehicles', 'Month', 'Day_of_Week',

       'Local_Authority_(District)', 'Local_Authority_(Highway)',

       '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',

       'Junction_Control', '2nd_Road_Class', '2nd_Road_Number',

       'Pedestrian_Crossing-Human_Control',

       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',

       'Weather_Conditions', 'Road_Surface_Conditions',

       'Special_Conditions_at_Site', 'Carriageway_Hazards',

       'Urban_or_Rural_Area', 'LSOA_of_Accident_Location', 'Year', 'Hour'])

dataset_onehot.head()
#the correlation between two column

def cramers_corrected_stat(confusion_matrix):

    import scipy.stats as ss

    """ calculate Cramers V statistic for categorial-categorial association.

        uses correction from Bergsma and Wicher,

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))





tips = Modify_Dataset

confusion_matrix = pd.crosstab(tips["Day_of_Week"], tips["Hour"]).values

cramers_corrected_stat(confusion_matrix)
# Generate heatmap using cramers_corrected_stat

import scipy.stats as ss

from collections import Counter

import math 

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from scipy import stats

import numpy as np



def convert(data, to):

    converted = None

    if to == 'array':

        if isinstance(data, np.ndarray):

            converted = data

        elif isinstance(data, pd.Series):

            converted = data.values

        elif isinstance(data, list):

            converted = np.array(data)

        elif isinstance(data, pd.DataFrame):

            converted = data.as_matrix()

    elif to == 'list':

        if isinstance(data, list):

            converted = data

        elif isinstance(data, pd.Series):

            converted = data.values.tolist()

        elif isinstance(data, np.ndarray):

            converted = data.tolist()

    elif to == 'dataframe':

        if isinstance(data, pd.DataFrame):

            converted = data

        elif isinstance(data, np.ndarray):

            converted = pd.DataFrame(data)

    else:

        raise ValueError("Unknown data conversion: {}".format(to))

    if converted is None:

        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))

    else:

        return converted

    

def conditional_entropy(x, y):

    """

    Calculates the conditional entropy of x given y: S(x|y)

    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy

    :param x: list / NumPy ndarray / Pandas Series

        A sequence of measurements

    :param y: list / NumPy ndarray / Pandas Series

        A sequence of measurements

    :return: float

    """

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0.0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy



def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



def theils_u(x, y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x



def correlation_ratio(categories, measurements):

    fcat, _ = pd.factorize(categories)

    cat_num = np.max(fcat)+1

    y_avg_array = np.zeros(cat_num)

    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):

        cat_measures = measurements[np.argwhere(fcat == i).flatten()]

        n_array[i] = len(cat_measures)

        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))

    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:

        eta = 0.0

    else:

        eta = numerator/denominator

    return eta



def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,

                          return_results = False, **kwargs):



    dataset = convert(dataset, 'dataframe')

    columns = dataset.columns

    nominal_columns = columns

    corr = pd.DataFrame(index=columns, columns=columns)

    for i in range(0,len(columns)):

        for j in range(i,len(columns)):

            if i == j:

                corr[columns[i]][columns[j]] = 1.0

            else:

                if columns[i] in nominal_columns:

                    if columns[j] in nominal_columns:

                        if theil_u:

                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]],dataset[columns[j]])

                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]],dataset[columns[i]])

                        else:

                            cell = cramers_v(dataset[columns[i]],dataset[columns[j]])

                            corr[columns[i]][columns[j]] = cell

                            corr[columns[j]][columns[i]] = cell

                    else:

                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])

                        corr[columns[i]][columns[j]] = cell

                        corr[columns[j]][columns[i]] = cell

                else:

                    if columns[j] in nominal_columns:

                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])

                        corr[columns[i]][columns[j]] = cell

                        corr[columns[j]][columns[i]] = cell

                    else:

                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])

                        corr[columns[i]][columns[j]] = cell

                        corr[columns[j]][columns[i]] = cell

    corr.fillna(value=np.nan, inplace=True)

    if mark_columns:

        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]

        corr.columns = marked_columns

        corr.index = marked_columns

    if plot:

        plt.figure(figsize=(20,20))#kwargs.get('figsize',None))

        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'), cmap='coolwarm')

        plt.show()

    if return_results:

        return corr

    

    

#heat map for categorical data

Data_calculate=Modify_Dataset

Data_calculate=Data_calculate

Col=Data_calculate.columns.tolist

results = associations(Data_calculate,all,mark_columns=False, theil_u=False, plot=True,return_results = False)
#bar chart for month_year

sn.countplot(x='Month', hue='Year', data=Modify_Dataset);



# t2=Modify_Dataset[Modify_Dataset['Year'] == 2014]

# print(t2[t2['Month']=='01'].count())

#Modify_Dataset.info()