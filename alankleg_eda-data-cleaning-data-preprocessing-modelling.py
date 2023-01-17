import warnings

warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.

# import libraries for plotting

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium import plugins

%matplotlib inline
data = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

locationData = pd.read_csv("../input/loccsv/location.csv")



data.describe()
category_columns = [col for col in data.columns if data[col].dtype=="object" ]

numerical_columns = [col for col in data.columns if data[col].dtype!="object" ]

#Check if all columns are accouned for 

#print(len(category_columns)+len(numerical_columns)==len(data.columns))

print("There are {} columns. {} are category and {} are numerical".format(len(data.columns),len(category_columns),len(numerical_columns)))
print("The category columns are {}".format(category_columns))
print("There are {} missing values for the target".format(data['RainTomorrow'].isnull().sum()))
def balanceTarget (target):

    rainTodayAnalysis = target.value_counts()

    f,  ax = plt.subplots(nrows=1,ncols=2) 

    sns.barplot(rainTodayAnalysis.index,rainTodayAnalysis.values, ax=ax[0])

    sns.barplot(rainTodayAnalysis.index,rainTodayAnalysis.values/len(data), ax=ax[1])

    ax[0].set(xlabel='Rain Tomorrow', ylabel='Number of Occurrences')

    ax[1].set(xlabel='Rain Tomorrow', ylabel='Percentage of Occurrences')

    plt.tight_layout()



balanceTarget(data["RainTomorrow"])
cat_features = list(filter(lambda x: x!="RainTomorrow", category_columns))

cat_features_miss = data[cat_features].isnull().sum()

f,  ax = plt.subplots(nrows=1,ncols=2) 

sns.barplot(cat_features_miss.index,cat_features_miss.values, ax=ax[0])

sns.barplot(cat_features_miss.index,cat_features_miss.values/len(data), ax=ax[1])

ax[0].set(ylabel='Number of Occurrences')

ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=75)

ax[1].set( ylabel='Percentage of Occurrences')

ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=75)

ax[1].set_ylim(0,1)                      

plt.suptitle("Missing Values for Categorical Data")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
cat_features = list(filter(lambda x: x!="RainTomorrow", category_columns))

cat_features_dict = {}

for features in cat_features:

    cat_features_dict[features]=len(list(filter(lambda x: isinstance(x, str) or math.isnan(x)==False ,data[features].unique())))

cat_features_dict

uniqueCat = pd.DataFrame(cat_features_dict,index=["Number of Unique Values"])

uniqueCat
locationData = locationData.dropna()



m=folium.Map([-25.2744,133.7751],zoom_start=4,width="70%",height="70%",left="10%")

for lat,lon,area in zip(locationData['Latitude'],locationData['Longitude'],locationData['Location']):

     folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=3,

                            color='b',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color="green",

                           ).add_to(m)

m.save('Australia.html')

m


print("The date ranges from {} to {}".format(data["Date"].sort_index().unique()[0],data["Date"].sort_index().unique()[-1]))
dateAnalysis = data.Date.value_counts().value_counts()

dateDict = {}

for i in range(1,max(dateAnalysis.index)+1):

    if i in dateAnalysis.index:

        dateDict[i]=dateAnalysis[i]

    else:

        dateDict[i]=0

dateAnalysis=pd.DataFrame.from_dict(dateDict, orient='index',columns=["count"])
f,  ax = plt.subplots(1,1,figsize=(14,4)) 

sns.barplot(dateAnalysis.index,dateAnalysis["count"], ax=ax, color="blue")

ax.set(ylabel='Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(),rotation=75)                  

plt.suptitle("Count of locations per date")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
import branca.colormap as cm



countRainToday = data[["Location","RainToday"]]

countRainToday=countRainToday.groupby("Location")['RainToday'].apply(lambda x: (x=='Yes').sum()).reset_index(name='count')

countRainToday=countRainToday.set_index("Location").join(locationData.set_index("Location")).reset_index("Location")

countRainToday['colour']=countRainToday['count'].apply(lambda count:"darkblue" if count>=1000 else

                                         "blue" if count>=800 and count<1000 else

                                         "green" if count>=600 and count<800 else

                                         "orange" if count>=400 and count<600 else

                                         "tan" if count>=200 and count<400 else

                                         "red")
m=folium.Map([-25.2744,133.7751],zoom_start=4,width="70%",height="70%",left="10%")

for lat,lon,area,radius,colour in zip(countRainToday['Latitude'],countRainToday['Longitude'],countRainToday['Location'],countRainToday["count"],countRainToday["colour"]):

     folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=7,

                            color='b',

                            fill=True,

                            fill_opacity=0.9,

                            fill_color=colour,

                           ).add_to(m)

from branca.element import Template, MacroElement



template = """

{% macro html(this, kwargs) %}



<!doctype html>

<html lang="en">

<head>

  <meta charset="utf-8">

  <meta name="viewport" content="width=device-width, initial-scale=1">

  <title>jQuery UI Draggable - Default functionality</title>

  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">



  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>

  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

  

  <script>

  $( function() {

    $( "#maplegend" ).draggable({

                    start: function (event, ui) {

                        $(this).css({

                            right: "auto",

                            top: "auto",

                            bottom: "auto"

                        });

                    }

                });

});



  </script>

</head>

<body>



 

<div id='maplegend' class='maplegend' 

    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);

     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

     

<div class='legend-title'>Legend</div>

<div class='legend-scale'>

  <ul class='legend-labels'>

    <li><span style='background:darkblue;opacity:0.7;'></span>Over 1000 days</li>

    <li><span style='background:blue;opacity:0.7;'></span>800-1000 days</li>

    <li><span style='background:green;opacity:0.7;'></span>600-800 days</li>

    <li><span style='background:orange;opacity:0.7;'></span>400-600 days</li>

    <li><span style='background:tan;opacity:0.7;'></span>200-400 days</li>

    <li><span style='background:red;opacity:0.7;'></span>0-200 days</li>



  </ul>

</div>

</div>

 

</body>

</html>



<style type='text/css'>

  .maplegend .legend-title {

    text-align: left;

    margin-bottom: 5px;

    font-weight: bold;

    font-size: 90%;

    }

  .maplegend .legend-scale ul {

    margin: 0;

    margin-bottom: 5px;

    padding: 0;

    float: left;

    list-style: none;

    }

  .maplegend .legend-scale ul li {

    font-size: 80%;

    list-style: none;

    margin-left: 0;

    line-height: 18px;

    margin-bottom: 2px;

    }

  .maplegend ul.legend-labels li span {

    display: block;

    float: left;

    height: 16px;

    width: 30px;

    margin-right: 5px;

    margin-left: 0;

    border: 1px solid #999;

    }

  .maplegend .legend-source {

    font-size: 80%;

    color: #777;

    clear: both;

    }

  .maplegend a {

    color: #777;

    }

</style>

{% endmacro %}"""



# Code for the legend from https://nbviewer.jupyter.org/gist/talbertc-usgs/18f8901fc98f109f2b71156cf3ac81cd



macro = MacroElement()

macro._template = Template(template)



m.get_root().add_child(macro)

m.save('RainToday.html')



m
dateRainToday = data[["Date","RainToday"]]

dateRainToday['Date'] = pd.to_datetime(dateRainToday['Date'])

dateRainToday['Year'] = dateRainToday['Date'].dt.year

dateRainToday['Month'] = dateRainToday['Date'].dt.month

dateRainToday.drop("Date", axis=1, inplace = True)

years = dateRainToday['Year'].unique().tolist()

dateRainToday["Period"] = dateRainToday['Year'].apply(str) +"-"+ dateRainToday['Month'].apply(str)

dateRainToday =dateRainToday.groupby(["Year","Month","Period"])['RainToday'].apply(lambda x: (x=='Yes').sum()).reset_index(name='count')

dateRainToday.drop(["Month"], axis=1, inplace = True)

years = sorted(years, key=lambda x: int(x))

dateRainToday[dateRainToday["Year"]==2012]
g = sns.FacetGrid(dateRainToday, col="Year", col_wrap=4, height=4, ylim=(0, 500),margin_titles=True,sharey=True,sharex=False)

g.map(sns.barplot, "Period", "count", ci=None,order=None);

for ax in g.axes.ravel():

    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)

plt.subplots_adjust(hspace=0.4, wspace=0.4)
dirGust =  data[["WindGustDir"]]

dirGust =dirGust["WindGustDir"].value_counts().rename_axis('direction').reset_index(name='count')

dirGust["name"]="WindGustDir"

dir9pm =  data[["WindDir9am"]]

dir9pm=dir9pm["WindDir9am"].value_counts().rename_axis('direction').reset_index(name='count')

dir9pm["name"]="WindDir9am"

dir3pm =  data[["WindDir3pm"]]

dir3pm=dir3pm["WindDir3pm"].value_counts().rename_axis('direction').reset_index(name='count')

dir3pm["name"]="WindDir3pm"

direction = pd.concat([dirGust,dir9pm,dir3pm])

#Graph the number of directions

g = sns.FacetGrid(direction, col="name", col_wrap=3, height=4, ylim=(0, direction.max()["count"]*1.1),margin_titles=True,sharey=True,sharex=False)

g.map(sns.barplot, "direction", "count", ci=None,order=["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]);

for ax in g.axes.ravel():

    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)

plt.subplots_adjust(hspace=0.4, wspace=0.4)
print("The category columns are {}".format(numerical_columns))
num_features = list(numerical_columns)

num_features_miss = data[num_features].isnull().sum()

f,  ax = plt.subplots(nrows=1,ncols=2,figsize=(15,8)) 

sns.barplot(num_features_miss.index,num_features_miss.values, ax=ax[0])

sns.barplot(num_features_miss.index,num_features_miss.values/len(data), ax=ax[1])

ax[0].set(ylabel='Number of Occurrences')

ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=75)

ax[1].set( ylabel='Percentage of Occurrences')

ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=75)

ax[1].set_ylim(0,1)                      

plt.suptitle("Missing Values for Numerical Data")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])


appended_data = []

for feature in numerical_columns:

    name = pd.DataFrame()

    data["binned"]=pd.cut(data[feature], 10)

    name[feature]=data["binned"]

    data.drop(["binned"], axis=1, inplace=True)

    name=name[feature].value_counts().rename_axis('numerical').reset_index(name='count')

    name["name"]=feature

    appended_data.append(name)

num_data = pd.concat(appended_data)



g = sns.FacetGrid(num_data, col="name", col_wrap=4, height=4,margin_titles=True,sharey=False,sharex=True)

g.map(sns.barplot, "numerical", "count", ci=None,order=None);

for ax in g.axes.ravel():

    ax.set_xticklabels(labels="")

    ax.set(xlabel='', ylabel='')

plt.subplots_adjust(hspace=0.4, wspace=0.4)
data[numerical_columns].describe()
num_col_noRiskM =list(filter(lambda x: x!="RISK_MM", numerical_columns))

Q1 = data[num_col_noRiskM].quantile(0.25)

Q3 = data[num_col_noRiskM].quantile(0.75)

IQR = Q3 - Q1

appended_data = []

for col in num_col_noRiskM:

    aboveOutlier=data[data[col]>(Q3[col]+1.5*IQR[col])]["RainTomorrow"]

    aboveOutlier=aboveOutlier.value_counts().rename_axis('Rain').reset_index(name='counts')

    aboveOutlier["name"]=col

    aboveOutlier["total"]=aboveOutlier["counts"].sum() 

    aboveOutlier["percentageRain"]=aboveOutlier["counts"]/aboveOutlier["total"]

    aboveOutlier["percentageTotal"]=aboveOutlier["total"]/data.shape[0]

    appended_data.append(aboveOutlier)

ResultAbove = pd.concat(appended_data)

appended_data = []

for col in num_col_noRiskM:

    belowOutlier=data[data[col]<(Q1[col]-1.5*IQR[col])]["RainTomorrow"]

    belowOutlier=belowOutlier.value_counts().rename_axis('Rain').reset_index(name='counts')

    belowOutlier["name"]=col

    belowOutlier["total"]=belowOutlier["counts"].sum() 

    belowOutlier["percentageRain"]=belowOutlier["counts"]/belowOutlier["total"]

    belowOutlier["percentageTotal"]=belowOutlier["total"]/data.shape[0]

    appended_data.append(belowOutlier)

ResultBelow = pd.concat(appended_data)
g = sns.FacetGrid(ResultAbove, col="name", col_wrap=4, height=4,margin_titles=True,sharey=False,sharex=False)

g.map(sns.barplot, "Rain", "percentageRain", ci=None,order=["Yes","No"]);

for ax in g.axes.ravel():

    ax.set_xticklabels(labels="")

    ax.set(xlabel='', ylabel='')

for ax in g.axes.ravel():

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

g.fig.suptitle('Percentage of RainTomorrow for the Higher IQR ')

g.fig.subplots_adjust(top=0.9)

plt.subplots_adjust(hspace=0.4, wspace=0.4)
dataUpper = data[data[numerical_columns]>Q3+1.5*IQR].describe()

dataUpper[["Rainfall", "Evaporation", "WindSpeed9am", "WindSpeed3pm"]]

#Description of the upper quartile of these Features
ResultBelow

g = sns.FacetGrid(ResultBelow, col="name", col_wrap=4, height=4,margin_titles=True,sharey=False,sharex=False)

g.map(sns.barplot, "Rain", "percentageRain", ci=None,order=["Yes","No"]);

for ax in g.axes.ravel():

    ax.set_xticklabels(labels="")

    ax.set(xlabel='', ylabel='')

for ax in g.axes.ravel():

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

g.fig.suptitle('Percentage of RainTomorrow for the Lower IQR ')

g.fig.subplots_adjust(top=0.9)

plt.subplots_adjust(hspace=0.4, wspace=0.4)
Q1 = data[numerical_columns].quantile(0.25)

Q3 = data[numerical_columns].quantile(0.75)

IQR = Q3 - Q1
correlation = data.corr()

plt.figure(figsize=(16,12))

plt.suptitle('Correlation Heatmap of Rain in Australia Dataset', size=16, y=0.93);     

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white',linewidths=.5, center=0)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)      

plt.show()
corrTable = data.corr().unstack().sort_values(ascending = False)

corrTable=corrTable.rename_axis(['Feature 1',"Feature 2"]).reset_index(name='Correlation')

corrTable=corrTable[corrTable["Feature 1"]!=corrTable["Feature 2"]]

TopCorr=corrTable[(corrTable["Correlation"]>0.7)|(corrTable["Correlation"]<-0.7)]

TopCorr.drop_duplicates(subset='Correlation')
corrTable[corrTable["Feature 1"]=="RISK_MM"]
f = plt.figure(figsize=(18,15))

gs = f.add_gridspec(4,2, hspace=0.2, wspace=0.2)



ax1 = f.add_subplot(gs[0, 0])

ax2 = f.add_subplot(gs[0, 1])

ax11 = f.add_subplot(gs[1, 0])

ax22 = f.add_subplot(gs[1, 1])

ax3 = f.add_subplot(gs[2, 0])

ax4 = f.add_subplot(gs[2, 1])

ax5 =  f.add_subplot(gs[3, :])



humidity = data[data["Humidity3pm"]>(Q3["Humidity3pm"]+0.5*IQR["Humidity3pm"])]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax1,order=["Yes","No"])

ax1.set_title('Rainfall Tomorrow when Humidity 3pm > Q3 + 0.5 IQR = {} - Data size - {}'

              .format(Q3["Humidity3pm"]+0.5*IQR["Humidity3pm"],humidity["counts"].sum()), size=12, y=1.05)



humidity = data[data["Humidity9am"]>(Q3["Humidity9am"]+0.5*IQR["Humidity9am"])]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax2,order=["Yes","No"])

ax2.set_title('Rainfall Tomorrow when Humidity 9am > Q3 + 0.5 IQR = {} - Data size - {}'

              .format(Q3["Humidity9am"]+0.5*IQR["Humidity9am"],humidity["counts"].sum()), size=12, y=1.05)



humidity = data[data["Humidity3pm"]==100]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax11,order=["Yes","No"])

ax11.set_title('Rainfall Tomorrow when Humidity 3pm = {} - Data size - {}'

              .format(100,humidity["counts"].sum()), size=12, y=1.05)



humidity = data[data["Humidity9am"]==100]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax22,order=["Yes","No"])

ax22.set_title('Rainfall Tomorrow when Humidity 9am = {} - Data size - {}'

              .format(100,humidity["counts"].sum()), size=12, y=1.05)



humidity = data[data["Humidity9am"]<data["Humidity3pm"]]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax3,order=["Yes","No"])

ax3.set_title('Rainfall Tomorrow when Humidity increases during the day - Data size - {}'.format(humidity["counts"].sum()), size=12, y=1.05)



humidity = data[data["Humidity9am"]>data["Humidity3pm"]]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax4,order=["Yes","No"])

ax4.set_title('Rainfall Tomorrow when Humidity decreases during the day - Data size - {}'.format(humidity["counts"].sum()), size=12, y=1.05)



humidity = data[data["Humidity3pm"]>(Q3["Humidity3pm"]+0.5*IQR["Humidity3pm"])]

humidity = humidity[humidity["Humidity9am"]<humidity["Humidity3pm"]]

humidity=humidity["RainTomorrow"].value_counts().rename_axis('Rain').reset_index(name='counts')

humidity["total"]=humidity["counts"].sum()

humidity["percentage"]=humidity["counts"]/humidity["total"]

sns.barplot(humidity.Rain,humidity.percentage,ax=ax5,order=["Yes","No"])

ax5.set_title('Rainfall Tomorrow when Humidity 3pm > Q3 + 0.5 IQR ={} and Humidity increases during the day- Data size - {}'

              .format(Q3["Humidity3pm"]+0.5*IQR["Humidity3pm"],humidity["counts"].sum()), size=12, y=1.05);     





ax1.set(xlabel='', ylabel='Percentage of Occurrences')

ax2.set(xlabel='', ylabel='Percentage of Occurrences')

ax3.set(xlabel='', ylabel='Percentage of Occurrences')

ax4.set(xlabel='', ylabel='Percentage of Occurrences')

ax5.set(xlabel='', ylabel='Percentage of Occurrences')



gs.update( wspace=0.2, hspace=0.4)

f.tight_layout(pad=4.0)
data.drop(['RISK_MM'], axis=1, inplace=True)

numerical_columns.remove("RISK_MM")
data = data[~(data["Evaporation"] > (Q3["Evaporation"] + 1.5 * IQR["Evaporation"]))]

data = data[~(data["WindSpeed9am"] > (Q3["WindSpeed9am"] + 1.5 * IQR["WindSpeed9am"]))]

data = data[~(data["WindSpeed3pm"] > (Q3["WindSpeed3pm"] + 1.5 * IQR["WindSpeed3pm"]))]
data.describe()
appended_data = []

for feature in numerical_columns:

    name = pd.DataFrame()

    data["binned"]=pd.cut(data[feature], 10)

    name[feature]=data["binned"]

    data.drop(["binned"], axis=1, inplace=True)

    name=name[feature].value_counts().rename_axis('numerical').reset_index(name='count')

    name["name"]=feature

    appended_data.append(name)

num_data = pd.concat(appended_data)

g = sns.FacetGrid(num_data, col="name", col_wrap=4, height=4,margin_titles=True,sharey=False,sharex=True)

g.map(sns.barplot, "numerical", "count", ci=None,order=None);

for ax in g.axes.ravel():

    ax.set_xticklabels(labels="")

    ax.set(xlabel='', ylabel='')

plt.subplots_adjust(hspace=0.4, wspace=0.4)
def missingValues (data):

    numerical_columns = [col for col in data.columns if data[col].dtype!="object" ]

    num_features = list(numerical_columns)

    num_features_miss = data[num_features].isnull().sum()

    f,  ax = plt.subplots(nrows=1,ncols=2,figsize=(15,8)) 

    sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis', ax=ax[0])

    sns.barplot(data.isnull().sum().index,data.isnull().sum().values/len(data), ax=ax[1])

    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=70)

    ax[1].set( ylabel='Percentage of Occurrences')

    ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=75)

    ax[1].set_ylim(0,1)             

    plt.suptitle('Missing Values in the Data', size=16, y=0.93)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
missingValues(data)


def position (length,nLargest):

    numFeatures = int(length/nLargest)

    pos = np.arange(1,length/numFeatures+1) *0.4

    res = np.empty(shape=[0,0])

    for i in range(0,int(length/nLargest)*2,2):

        res=np.append(res,pos+i)

    return res

def corrT (arr):

    corrTable = data.corr().unstack().sort_values(ascending = False)

    corrTable=corrTable.rename_axis(['Feature 1',"Feature 2"]).reset_index(name='Correlation')

    corrTable["CorrelationAbs"]=abs(corrTable["Correlation"])

    corrTable=corrTable[corrTable["Feature 1"]!=corrTable["Feature 2"]]

    corrTable=corrTable[corrTable["Feature 1"].isin(arr)]

    corrTable=corrTable[~corrTable["Feature 2"].isin(arr)]

    corrTable=corrTable.loc[corrTable.groupby('Feature 1')['CorrelationAbs'].nlargest(3).index.get_level_values(1)]

    length = len(corrTable)

    pos = position(length,3)

    fig, ax=plt.subplots(figsize=(16,5))

    uelec, uind = np.unique(corrTable["Feature 2"], return_inverse=1)

    cmap = plt.cm.get_cmap("Set1")

    ax.bar(pos, corrTable["CorrelationAbs"], width=0.4, align="edge", ec="k", color=cmap(uind)  )

    handles=[plt.Rectangle((0,0),1,1, color=cmap(i), ec="k") for i in range(len(uelec))]

    ax.legend(handles=handles, labels=list(uelec),

               prop ={'size':10}, loc=9, ncol=8, 

                title=r'Feature 2')

    ax.set_xticks([x*2+1 for x in range(length//3)])

    ax.set_xticklabels(corrTable["Feature 1"].unique())

    ax.set_ylim(0, 1)

    plt.show()
arr = ['Evaporation',"Sunshine","Cloud9am","Cloud3pm"]

corrT(arr)
class medianBins():

    def __init__(self,data, missingFeature, correlatedFeature,binSize):

        self.data = data

        self.missingFeature = missingFeature

        self.correlatedFeature = correlatedFeature

        self.binSize = binSize

        print (self)

        

    def binD (self):

        binData = pd.DataFrame()

        binData[self.correlatedFeature]= self.data[self.correlatedFeature]

        binData[self.missingFeature]= data[self.missingFeature]

        binData["HCorr"]= pd.cut(binData[self.correlatedFeature], self.binSize)

        binData["HMissing"]= pd.cut(binData[self.missingFeature], self.binSize)

        binData=binData.dropna(subset=["HCorr","HMissing"])

        binData = binData.groupby(["HCorr","HMissing"])[self.missingFeature].count().rename_axis(["HCorr","HMissing"]).reset_index(name='Count')

        binData["cummulative"]=binData.groupby(["HCorr"])["Count"].apply(lambda x: x.cumsum())

        binData["bin_centres"] = binData["HMissing"].apply(lambda x: x.mid)

        return binData

    

    def median(self):

        binData= self.binD()

        median=binData.groupby(["HCorr"])["Count"].agg("sum")/2

        median=median.rename_axis(["HCorr"]).reset_index(name='median')

        binData=pd.merge(binData,median,on="HCorr")

        median=binData[binData["cummulative"]>binData["median"]]

        median =  median.loc[median.groupby(["HCorr"])['cummulative'].idxmin()]

        median=median.drop(columns=["Count","cummulative","median"])

        median["bin_centres"] = median["HMissing"].apply(lambda x: x.mid)

        return median

    

    def graph(self):

        binData = self.binD()

        pivot = binData.pivot(index="HCorr",columns="HMissing",values="Count")

        ax = pivot.plot(kind='bar', stacked=True, figsize=(18.5, 7))

        ax.set( ylabel='Count' , xLabel=self.correlatedFeature)

        ax.legend(title=self.missingFeature)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.title("Cumulative Count of the Combination {} and {} with median lines".format(self.missingFeature,self.correlatedFeature))

        return ax



impCloud3pm=medianBins(data,"Cloud3pm","Humidity3pm",10)

impCloud9am=medianBins(data,"Cloud9am","Humidity3pm",10)

impEvaporation=medianBins(data,"Evaporation","MaxTemp",10)

impSunshine=medianBins(data,"Sunshine","Humidity3pm",10)
impCloud3pm.graph()
impCloud9am.graph()
impEvaporation.graph()
impSunshine.graph()
def imputation(cols,binD):

    missingFeature = cols[0]

    corrFeature = cols[1]

    if pd.isnull(missingFeature)&pd.notna(corrFeature):

        temp=binD

        temp=temp[temp["HCorr"].apply(lambda x:x.left<corrFeature<=x.right)]

        temp["median"]=temp["Count"].sum()/2

        temp=temp[temp["cummulative"]>temp["median"]]

        #print(temp["bin_centres"].iloc[0]  ,"fixed",corrFeature)        

        try:    

            result=temp.iloc[0, temp.columns.get_loc('bin_centres')]

        except:return float('nan')

        return result

        #print(corrFeature,temp)

    elif pd.isnull(missingFeature)&pd.isnull(corrFeature):

        #print("nan")

        return float('nan')

    else:

        #print(missingFeature,"original")

        return missingFeature






"""

data["Evaporation"]=data[['Evaporation','MaxTemp']].apply(imputation,binD=impEvaporation.binD(),axis=1)

data["Cloud3pm"]=data[['Cloud3pm','Humidity3pm']].apply(imputation,binD=impCloud3pm.binD(),axis=1)

data["Cloud9am"]=data[['Cloud9am','Humidity3pm']].apply(imputation,binD=impCloud9am.binD(),axis=1)

data["Sunshine"]=data[['Sunshine','Humidity3pm']].apply(imputation,binD=impSunshine.binD(),axis=1)

data.to_csv('dataImputation.csv',index=False)

print("done")

"""



#saved to datacleanv1
data1 = pd.read_csv(r"../input/datacleanv1/dataImputation.csv")
missingValues(data1)
corrT(["WindGustSpeed", "Pressure9am", "Pressure3pm"])
impWindGustSpeed=medianBins(data1,"WindGustSpeed","WindSpeed3pm",10)

impPressure9am=medianBins(data1,"Pressure9am","MinTemp",10)

impPressure3pm=medianBins(data1,"Pressure3pm","Temp9am",10)


"""

data1["WindGustSpeed"]=data1[['WindGustSpeed','WindSpeed3pm']].apply(imputation,binD=impWindGustSpeed.binD(),axis=1)

data1["Pressure9am"]=data1[['Pressure9am','MinTemp']].apply(imputation,binD=impPressure9am.binD(),axis=1)

data1["Pressure3pm"]=data1[['Pressure3pm','Temp9am']].apply(imputation,binD=impPressure3pm.binD(),axis=1)

data1.to_csv('dataImputation.csv',index=False)

print("done")

"""



#saved to datacleanv2
data2 = pd.read_csv(r"../input/datacleanv2/dataImputation.csv")
missingValues(data2)
for col in numerical_columns:

    data2[col].fillna(data2[col].median(), inplace=True)


for col in category_columns:

    data2[col].fillna(data2[col].mode()[0], inplace=True)
missingValues(data2)
X = data2.drop(['RainTomorrow'], axis=1)

y = data2['RainTomorrow']

y=y.map(dict(Yes=1, No=0))
balanceTarget(data2["RainTomorrow"])
data2["RainTomorrow"].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0,stratify = y)
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
X_train.columns
X_train['Date']= pd.to_datetime(X_train['Date']) 

X_train['Year'] = X_train['Date'].dt.year

X_train['Month'] = X_train['Date'].dt.month

X_train['Day'] = X_train['Date'].dt.day



X_test['Date']= pd.to_datetime(X_test['Date']) 

X_test['Year'] = X_test['Date'].dt.year

X_test['Month'] = X_test['Date'].dt.month

X_test['Day'] = X_test['Date'].dt.day



#Dropping the date column

X_train.drop('Date', axis=1, inplace = True)

X_test.drop('Date', axis=1, inplace = True)
X_train.columns
X_train = pd.get_dummies(X_train, columns=['Location','WindGustDir','WindDir9am','WindDir3pm'])



X_test = pd.get_dummies(X_test, columns=['Location','WindGustDir','WindDir9am','WindDir3pm'])
X_train.columns
X_train.head()
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
allColumns = X_train.columns



from sklearn import preprocessing

scaler = preprocessing.RobustScaler()



X_train_res = scaler.fit_transform(X_train_res)

X_train_res  = pd.DataFrame(X_train_res, columns=[allColumns])



X_train = scaler.fit_transform(X_train)

X_train  = pd.DataFrame(X_train, columns=[allColumns])



X_test = scaler.transform(X_test)

X_test = pd.DataFrame(X_test, columns=[allColumns])
X_train_res = pd.read_csv(r"../input/traintest/X_train_res.csv", index_col=0)

X_train = pd.read_csv(r"../input/traintest/X_train.csv", index_col=0)

X_test = pd.read_csv(r"../input/traintest/X_test.csv", index_col=0)

y_train_res = pd.read_csv(r"../input/traintest/y_train_res.csv")

y_train = pd.read_csv(r"../input/traintest/y_train.csv", index_col=0)

y_test = pd.read_csv(r"../input/traintest/y_test.csv", index_col=0)



y_train_res = np.array(y_train_res["0"])





from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, fbeta_score



f2_score = make_scorer(fbeta_score, beta=2, pos_label=1)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import sklearn.metrics as metrics

from matplotlib import pyplot

from sklearn.metrics import roc_curve,roc_auc_score



def results(y_test, y_pred_test,y_pred_proba,model,ROC):



    print("Classifcation Report")

    print("\n")

    print("1- Rain Tomorrow, 0 - No Rain Tomorrow")

    print("\n")

    print(classification_report(y_test, y_pred_test))

    

    

    cm = confusion_matrix(y_test, y_pred_test)

    cm_df = pd.DataFrame(data=cm, columns=['Actual No Rain', 'Actual Rain'], 

                                     index=['Predict No Rain', 'Predict Rain'])

    plt.figure(figsize=(6,6))

    plt.suptitle('Confusion matrix', size=16, y=1.0);     

    ax=sns.heatmap(cm_df, square=True, annot=True, fmt='d', cbar=True)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)      

    plt.show()

    

    if ROC==True:

        auc = roc_auc_score(y_test, y_pred_proba)

        print("{} : ROC AUC = {}%".format(model,round(auc, 3)))

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        pyplot.plot(fpr, tpr, marker='.', label=model)

        pyplot.plot([0,1], [0,1], 'k--' )



        pyplot.xlabel('False Positive Rate')

        pyplot.ylabel('True Positive Rate')

        pyplot.legend()

        pyplot.show()

    print("\n")

    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
"""

# Ran the modelling and then saved it to a pickle



import pickle

from sklearn.linear_model import LogisticRegression

parameters = {'solver':['liblinear'], 'C':[100,10000],"penalty":["l1","l2"],

              "class_weight":[None,"balanced"]}

gsc=GridSearchCV(estimator=LogisticRegression(),

             param_grid=parameters,cv=5, scoring=f2_score, verbose=0, n_jobs=-1)

gr_log_bal = gsc.fit(X_train_res, y_train_res)



filename = 'logReg.sav'

pickle.dump(gr_log_bal, open(filename, 'wb'))

# Save to pickle

"""
import pickle

logReg = pickle.load(open("../input/results/logReg.sav", 'rb'))

print(logReg.param_grid)

print("\n")

y_pred_test = logReg.predict(X_test)

y_pred1 = logReg.predict_proba(X_test)[:, 1]

results(y_test, y_pred_test, y_pred1,"Logistic Regression Balance",True)

"""

from sklearn.ensemble import RandomForestClassifier



parameters = { 

    'n_estimators': [200, 500],

    'max_features': ['auto'],

    'max_depth' : [4,8],

    'criterion' :['gini', 'entropy'],

    "class_weight":[None,"balanced"],

    "oob_score":[True]

}

gsc=GridSearchCV(estimator=RandomForestClassifier(),

             param_grid=parameters,cv=5, scoring=f2_score, verbose=0, n_jobs=-1)



grid_result_Rand_bal = gsc.fit(X_train_res, y_train_res)



filename = 'RandForest.sav'

pickle.dump(grid_result_Rand_bal, open(filename, 'wb'))

# Save to pickle

"""
import pickle

randForest = pickle.load(open("../input/results/RandForest.sav", 'rb'))

print(randForest.param_grid)

print("\n")

y_pred_test = randForest.predict(X_test)

y_pred1 = randForest.predict_proba(X_test)[:, 1]

results(y_test, y_pred_test, y_pred1,"Logistic Regression Balance",True)
y.value_counts()/y.value_counts().sum()*100