#required librarys

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

import matplotlib as mpl

%matplotlib inline



from sklearn.preprocessing import LabelEncoder #For Label Encoding 

#path to file

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# Import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None) #allows us to see all the columns avoiding the dreded '...'



#Word Cloud 

from PIL import Image

from wordcloud import WordCloud, STOPWORDS



#Reading Data

df=pd.read_csv('/kaggle/input/us-accidents/US_Accidents_June20.csv')



df.head(3)
print('The shape of the raw data from the dataset is:',df.shape,',and that is just not gonna happen!')
#First we will subset our data down to just the state level

df_CT=df.loc[df['State']=='CT'].copy()

#County level 

df_FF=df_CT.loc[df_CT['County']=='Fairfield'].copy()

print('The new shape after defining the scope:',df_FF.shape,',that is something i can work with')

pd.set_option('display.max_columns', None)

df_FF.head(1)
print(df_FF.head())

print(df_FF.describe())

print(df_FF.columns)

print(df_FF.info())
print('if your interested in seeing some descriptive statistics on the overall data set click the following code button bellow')
print(df_FF.shape)

print(df_FF['Amenity'].value_counts())#Only 20 True Observations 

print(df_FF['Bump'].value_counts())#No True Observations 

print(df_FF['Crossing'].value_counts())#86 True Observations

print(df_FF['Give_Way'].value_counts())#No True Observations 

print(df_FF['Junction'].value_counts())#807 True Observations

print(df_FF['No_Exit'].value_counts())#Only 1 True Observations

print(df_FF['Railway'].value_counts())#Only 5 True Observations

print(df_FF['Roundabout'].value_counts())#No True Observations 

print(df_FF['Station'].value_counts())#Only 3 True Observations

print(df_FF['Stop'].value_counts())#Only 31 True Observations

print(df_FF['Traffic_Calming'].value_counts())#No True Observations 

print(df_FF['Traffic_Signal'].value_counts())#181 True Observations

print(df_FF['Turning_Loop'].value_counts())#No True Observations 
#The 10 columns that are almost entirely false and thus not useful

df_FF.drop(columns=['Amenity'],axis=1,inplace=True)

df_FF.drop(columns=['Bump'],axis=1,inplace=True)

df_FF.drop(columns=['Stop'],axis=1,inplace=True)

df_FF.drop(columns=['Give_Way'],axis=1,inplace=True)

df_FF.drop(columns=['No_Exit'],axis=1,inplace=True)

df_FF.drop(columns=['Railway'],axis=1,inplace=True)

df_FF.drop(columns=['Roundabout'],axis=1,inplace=True)

df_FF.drop(columns=['Station'],axis=1,inplace=True)

df_FF.drop(columns=['Traffic_Calming'],axis=1,inplace=True)

df_FF.drop(columns=['Turning_Loop'],axis=1,inplace=True)



#Just a couple of extra columns that do not give us enough information on their own to warrant staying.

df_FF.drop(columns=['Nautical_Twilight'],axis=1,inplace=True) #Closely associated with Sunset, so they will not add to our analysis 

df_FF.drop(columns=['Astronomical_Twilight'],axis=1,inplace=True)

df_FF.drop(columns=['Civil_Twilight'],axis=1,inplace=True)



#Now to convert the remaining Boolean objects into intiger objects. 

df_FF['Junction']=df_FF['Junction'].astype(int)

df_FF['Crossing']=df_FF['Crossing'].astype(int)

df_FF['Traffic_Signal']=df_FF['Traffic_Signal'].astype(int)

df_FF.head(1)
df_FF.drop(columns=['ID'],axis=1,inplace=True) #Superfluous 

df_FF.drop(columns=['Country'],axis=1,inplace=True) #This is a uniform column displaying just the 'US'

df_FF.drop(columns=['State'],axis=1,inplace=True) #This is a uniform column displaying just 'CT' 

df_FF.drop(columns=['County'],axis=1,inplace=True)#This is a uniform column displaying just 'Fairfield'

df_FF.drop(columns=['Timezone'],axis=1,inplace=True) #This is a uniform column displaying just 'US/Eastern'

df_FF.drop(columns=['Airport_Code'],axis=1,inplace=True)#Who cares about airport code, am I right?

df_FF.drop(columns=['Weather_Timestamp'],axis=1,inplace=True)#This relates to what time the weather data was recorded
df_FF.head(1)
font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 36,

        }

plt.figure(figsize=(30,12))

sns.heatmap(df_FF.isnull(),yticklabels=False,cbar=False,cmap='plasma')

plt.title('Missing Values Visualized', fontdict=font)

plt.show()
#Far to many missing values to keep these columns and they dont add much to our analysis so the decision is easy

df_FF.drop(columns=['End_Lat','End_Lng', 'Number'],axis=1,inplace=True) 

df_FF.head(1)
df_FF['Wind_Chill(F)'].fillna(df_FF['Wind_Chill(F)'].mean(),inplace=True)

df_FF['Wind_Speed(mph)'].fillna(df_FF['Wind_Speed(mph)'].mean(),inplace=True)

df_FF['Visibility(mi)'].fillna(df_FF['Visibility(mi)'].mean(),inplace=True)

df_FF['Precipitation(in)'].fillna(df_FF['Precipitation(in)'].mean(),inplace=True) 

df_FF['Temperature(F)'].fillna(df_FF['Temperature(F)'].mean(),inplace=True)

df_FF['Humidity(%)'].fillna(df_FF['Humidity(%)'].mean(),inplace=True)

df_FF['Pressure(in)'].fillna(df_FF['Pressure(in)'].mean(),inplace=True)
#Converting the giving time data into workable Date-time objects

df_FF['Start_Time']=pd.to_datetime(df_FF['Start_Time'],infer_datetime_format=True)



#Breaking the start end time into usable independant features 

df_FF['Year'] = df_FF['Start_Time'].dt.year

df_FF['Month'] = df_FF['Start_Time'].dt.month

df_FF['Day'] = df_FF['Start_Time'].dt.day

df_FF['Time_S'] = df_FF['Start_Time'].dt.hour

df_FF['Weekday']=df_FF['Start_Time'].dt.weekday

df_FF.drop(columns=['Start_Time','End_Time'],axis=1,inplace=True) # supurfluous now

df_FF[df_FF['Weather_Condition'].isnull()]
df_FF['Weather_Condition'].fillna(method='ffill',inplace=True) 

df_FF['Wind_Direction'].fillna(method='ffill',inplace=True) #We also have one missing value here we need to fix real quick
#Define bins for time (24 hour time) 0 to 6, 6-12, 12-18,18-24

timeBins=[-1,6,12,18,24]

tBin_names=['Early Morning','Morning','Afternoon','Evening']

df_FF['TimeofDay']=pd.cut(df_FF['Time_S'],timeBins,labels=tBin_names)

df_FF.head(5)
#Define bins for Season (Months) 0 to 2, 2-5, 5-8,8-11,11-12

seasonBins=[-1,2,5,8,11,12]

sBin_names=['Winter','Spring','Summer','Autumn','Winter']

df_FF['Season']=pd.cut(df_FF['Month'],seasonBins,labels=sBin_names,ordered=False)

df_FF.tail(5)
#Define bins for Day_type -1(Mon) to 4(Fri),4(Fri)-6(Sun)

seasonBins=[-1,4,6]

sBin_names=['Weekday','Weekend']

df_FF['Day_Type']=pd.cut(df_FF['Weekday'],seasonBins,labels=sBin_names,ordered=False)

df_FF.tail(5)
fig, ax =plt.subplots(2,3,figsize=(25,10))

T=sns.countplot(x='TimeofDay',hue='Severity',data=df_FF,ax=ax[0][0],palette='rocket_r')

T.legend(loc='upper right',bbox_to_anchor=(.6, 0.5, 0.5, 0.5))    

t=sns.countplot(x='Time_S',hue='Severity',data=df_FF,ax=ax[1][0],palette='rocket_r')

t.legend(loc='upper right',bbox_to_anchor=(.6, 0.5, 0.5, 0.5))   

S=sns.countplot(x='Season',hue='Severity',data=df_FF,ax=ax[0][1],palette='mako_r')

S.legend(loc='upper right',bbox_to_anchor=(.6, 0.5, 0.5, 0.5))

s=sns.countplot(x='Month',hue='Severity',data=df_FF,ax=ax[1][1],palette='mako_r')

s.legend(loc='upper right',bbox_to_anchor=(.6, 0.5, 0.5, 0.5))

W=sns.countplot(x='Day_Type',hue='Severity',data=df_FF,ax=ax[0][2],palette='rocket_r')

W.legend(loc='upper right',bbox_to_anchor=(.6, 0.5, 0.5, 0.5))

w=sns.countplot(x='Weekday',hue='Severity',data=df_FF,ax=ax[1][2],palette='rocket_r')

w.legend(loc='upper right',bbox_to_anchor=(.6, 0.5, 0.5, 0.5))





fig.show()
print('we have',df_FF['TMC'].isna().sum(), 'missing values in the TMC column')

print('out of a total',df_FF.shape)
df_FF.drop(columns='TMC',inplace=True)
font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 36,

        }

plt.figure(figsize=(30,12))

sns.heatmap(df_FF.isnull(),yticklabels=False,cbar=False,cmap='plasma')

plt.title('Missing Values Visualized', fontdict=font)

plt.show()
df_FF['Sunrise_Sunset'].replace('Night',0,inplace=True)

df_FF['Sunrise_Sunset'].replace('Day',1,inplace=True)



df_FF['Side'].replace('L',0,inplace=True)

df_FF['Side'].replace('R',1,inplace=True)
#Randomly subsetting the data to make it more manageable 

df_FF_half=df_FF.sample(frac=0.5, replace=True, random_state=101)

df_FF_half.head(1)
import folium

#Creating map of our locaiton of choice Fairfield County

FF_map = folium.Map(location=[41.40,-73.263],tiles = 'Stamen Terrain', zoom_start=10)

FF_map



from folium import plugins

from folium.plugins import HeatMap



#Making sure our data is in the correct type

df_FF_half['Start_Lat']=df_FF_half['Start_Lat'].astype(float)

df_FF_half['Start_Lng']=df_FF_half['Start_Lng'].astype(float)



#Subsetting data for visualization

df_FFHeat=df_FF_half[['Start_Lat','Start_Lng']]

df_FFHeat=df_FFHeat.dropna(axis=0,subset=['Start_Lat','Start_Lng'])



#Creating and Attaching heatmap to our map

FFHeat_data=[[row['Start_Lat'],row['Start_Lng']] for index, row in df_FFHeat.iterrows()]

HeatMap(FFHeat_data,blur=10,radius=15,gradient={0.4: 'green', 0.65: 'yellow', 1: 'red'}).add_to(FF_map)



#show

FF_map

#Reducing data to 1/8 its sice

df_FF_e=df_FF.sample(frac=0.125, replace=True, random_state=101)

df_FF_e.head(1)



#Same Map generation as before

FF_map2 = folium.Map(location=[41.40,-73.263],tiles = 'Stamen Terrain', zoom_start=10)



#Marker Creation

Marker_Map = folium.Map(location=[41.40,-73.263],tiles = 'Stamen Terrain', zoom_start=10)

for i in range(0,len(df_FF_e)):

    folium.Marker([df_FF_e['Start_Lat'].iloc[i],df_FF_e['Start_Lng'].iloc[i]]).add_to(Marker_Map) 

Marker_Map





#Adding labels and colors to our markers 

for i in range(0,len(df_FF_e)):

    Severity = df_FF_e['Severity'].iloc[i]

    if Severity == 1:

        color = 'green'

        popup = folium.Popup('Severity 1', parse_html=True) 

    elif Severity == 2:

        color = 'orange'

        popup = folium.Popup('Severity 2', parse_html=True) 



    elif Severity == 3:

        color = 'red'

        popup = folium.Popup('Severity 3', parse_html=True) 



    else:

        color = 'purple'

        popup = folium.Popup('Severity 4', parse_html=True) 



    

    #Adding our code to the map

    folium.Marker([df_FF_e['Start_Lat'].iloc[i],df_FF_e['Start_Lng'].iloc[i]],popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(Marker_Map)

#Display

Marker_Map
print(df_FF['Severity'].value_counts())

print('\n')

print('It looks like we should have enough text data for a Word Cloud of Severity levels 2, 3, and 4.')
df_FF_S4=df_FF[df_FF['Severity']==4] #Making subsets for us to use in our wordcloud generation 

df_FF_S3=df_FF[df_FF['Severity']==3]

df_FF_S2=df_FF[df_FF['Severity']==2]





#Custome Color Map for Severity 2

cmap_O = mpl.cm.Oranges(np.linspace(0,1,20))

cmap_O = mpl.colors.ListedColormap(cmap_O[10:,:-1])



#Custome Color Map for Severity 3

cmap_R = mpl.cm.Reds(np.linspace(0,1,20))

cmap_R = mpl.colors.ListedColormap(cmap_R[10:,:-1])



#Custome Color Map for Severity 4

cmap_H = mpl.cm.hot_r(np.linspace(0,1,20))

cmap_H = mpl.colors.ListedColormap(cmap_H[10:,:-1])
#Font parameters for our Title

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 36,

        }



#Creating variable with data for this Word Cloud

textS2 = ' '.join(df_FF_S2['Description'].tolist())



#Creating Mask for Word Cloud

d = '../input/car-silhouette/'

car_mask = np.array(Image.open(d + 'Car_mask3.jpg'))

stop_words=set(STOPWORDS)



#Word Cloud Creation 

Car_wc=WordCloud(width=400,height=200,mask=car_mask,random_state=101, max_font_size=450,

                 min_font_size=1,stopwords=stop_words,background_color="white",

                 scale=3,max_words=400,collocations=True,colormap=cmap_O)



#Generate Word Cloud

Car_wc.generate(str(textS2))







#show

fig=plt.figure(figsize=(30,10))

plt.ylim(-250,2700)

#plt.xlim(0,2650)

plt.gca().invert_yaxis()

#plt.gca().invert_xaxis()

plt.axis("off")

plt.title('Accidents: Severity 2',fontdict=font)

plt.imshow(Car_wc,interpolation='bilinear')

plt.show()

#Creating variable with data for this Word Cloud

textS3 = ' '.join(df_FF_S3['Description'].tolist())



#Creating Mask for Word Cloud

d = '../input/car-silhouette/'

car_mask = np.array(Image.open(d + 'Car_mask2.jpg'))

stop_words=set(STOPWORDS)



#Word Cloud Creation 

Car_wc=WordCloud(width=400,height=200,mask=car_mask,random_state=101, max_font_size=450,

                 min_font_size=1.5,stopwords=stop_words,background_color="white",

                 scale=3,max_words=400,collocations=True,colormap=cmap_R)



#Generate Word Cloud

Car_wc.generate(str(textS3))







#Show

fig=plt.figure(figsize=(30,10))

#plt.ylim(-300,2600)

#plt.xlim(0,7500)

#plt.gca().invert_xaxis()

#plt.gca().invert_yaxis()

plt.axis("off")

plt.title('Accidents: Severity 3',fontdict=font)

plt.imshow(Car_wc,interpolation='bilinear')

plt.show()
#Creating variable with data for this Word Cloud

textS4 = ' '.join(df_FF_S4['Description'].tolist())



#Creating Mask for Word Cloud

d = '../input/car-silhouette/'

car_mask = np.array(Image.open(d + 'Car_mask1.jpg'))

stop_words=set(STOPWORDS)



#Word Cloud Creation 

Car_wc=WordCloud(width=400,height=200,mask=car_mask,random_state=101, max_font_size=450,

                 min_font_size=1.5,stopwords=stop_words,background_color="white",

                 scale=3,max_words=400,collocations=True,colormap=cmap_H)



#Generate Word Cloud

Car_wc.generate(str(textS4))





#Show

fig=plt.figure(figsize=(30,10))

plt.ylim(500,1300)

plt.gca().invert_yaxis()

plt.axis("off")

plt.title('Accidents: Severity 4',fontdict=font)

plt.imshow(Car_wc,interpolation='bilinear')

plt.show()
df_FF.head(1)
df_FF.columns
#What we are trying to predict

target='Severity'



featuers_removed=['Description', 'Street','Zipcode','Source','Year', 'Month', 'Day', 

          'Time_S', 'Weekday']



#What we are using to predict our target

features=['Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)',

          'Side', 'City','Temperature(F)','Wind_Chill(F)', 'Humidity(%)',

          'Pressure(in)', 'Visibility(mi)','Wind_Direction', 'Wind_Speed(mph)',

          'Precipitation(in)','Weather_Condition', 'Junction', 'Crossing', 

          'Traffic_Signal','Sunrise_Sunset','TimeofDay', 'Season', 'Day_Type'] 



#One-Hot Encoding 

df_FF_Dummy=pd.get_dummies(df_FF[features],drop_first=True)

print(df_FF_Dummy.info())

df_FF_ML = df_FF_Dummy.reset_index()

df_FF_ML=df_FF_ML.drop('index',axis=1)

df_FF_ML.fillna(0)



#Train Test Split is a great function to break our data down. i made test 30% of total

y=df_FF_ML[target]

X=df_FF_ML.drop(target, axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#Running Model object

clf=RandomForestClassifier(n_estimators=250)



#Train Model with data

clf.fit(X_train,y_train)



#Run Model to predict accident Severity

predictions=clf.predict(X_test)



# Get the accuracy score

acc=accuracy_score(y_test, predictions)





# Model Accuracy, how often is the classifier correct?

print("[Randon forest algorithm] accuracy_score:",accuracy_score(y_test, predictions))

print('\n')

print('Confusion Matrix of results')

print(confusion_matrix(y_test,predictions))

print('\n')

print('Classificaiton Report of results')

print(classification_report(y_test,predictions))

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features

k=10

sns.barplot(x=feature_imp[:10], y=feature_imp.index[:k])

# Add labels to your graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()