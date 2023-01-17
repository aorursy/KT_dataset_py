import pandas as pd
address = '../input/melbourne-housing-snapshot/melb_data.csv'
df = pd.read_csv(address)
df.info()
df.head()
df.describe().T
df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(), cbar=False,cmap="Greens");
import missingno as msno
msno.heatmap(df)
df['Suburb'].sample(30)
df['Suburb'].value_counts()
df_kew = df.loc[df['Suburb'] == 'Kew']
df_kew['Bedroom2'].value_counts()
df_kew
import folium
locations = df_kew[['Lattitude', 'Longtitude']]
locationlist = locations.values.tolist()
len(locationlist)
locationlist[2]
map = folium.Map(location=[-37.80, 145.06], zoom_start=14)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=df['Price'][point]).add_to(map)
map
df['Price'].max()
df[['Suburb','Bedroom2']][df.Price == df['Price'].max()]
df[df['Price']==df['Price'].max()]
df_ranked = df.sort_values(by ='Price' , ascending=False)
df_ranked['Suburb'].head(10).value_counts()
df_ranked['Suburb'].head(20).value_counts()
df_ranked['Suburb'].head(50).value_counts()
df_ranked['Suburb'].head(100).value_counts()
df_sub_mean = df.groupby('Suburb').mean()
df_sub_mean.head(50)
df_sub_mean.loc[df_sub_mean.index == 'Kew']
numerical_feature=[feature for feature in df.columns if df[feature].dtype !='object']
print('There are {} numerical features.'.format(len(numerical_feature)))
year=[feature for feature in numerical_feature if "Yr" in feature or "Year" in feature]
discrete_feature = [feature for feature in numerical_feature if (df[feature].nunique())<25  and feature not in year+['Id']]
print('There are {} are discrete features'.format(len(discrete_feature)) )
year=[feature for feature in numerical_feature if "Yr" in feature or "Year" in feature]
continuous_feature=[feature for feature in numerical_feature if feature not in discrete_feature and feature not in year+['Id']]
print('There are {} are continuous features'.format(len(continuous_feature)) )
cat_feature=[feature for feature in df.columns if df[feature].dtypes=='object']
print('There are {} categorical features.'.format(len(cat_feature)))
type={'h':'house','u':'unit','t':'townhouse'}
df=df.replace({'Type':type})
df['Price'].describe()
def outlier_function(df, col_name):
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(3*IQR)
    lower_limit = first_quartile-(3*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count
for feature in continuous_feature:
    if outlier_function(df,feature)[2] > 1:
        print('{}: {} outliers'.format(feature,outlier_function(df,feature)[2]))
plt.figure(figsize=(10,8))
corr=df[continuous_feature].corr()
plt.title('Correlation heatmap of continuous features',fontsize=15)
sns.heatmap(corr);
df[continuous_feature].hist(bins=25, figsize=(22,8), layout=(2,8));
feat=[]
for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        feat.append(feature)
np.log(data[feat]).hist(bins=25, figsize=(20,5), layout=(1,5),color='green');
fig, axarr = plt.subplots(2, 2, figsize=(12, 10))

sns.distplot(df['Price'],ax=axarr[0][0],color='blue',hist_kws=dict(edgecolor="k", linewidth=2)).set_title('Distribution of Price')
sns.distplot(df['Distance'],ax=axarr[1][0],color='green',hist_kws=dict(edgecolor="k", linewidth=2)).set_title('Distribution of Distance')
sns.distplot(np.log(df['Price']),ax=axarr[0][1],color='red',hist_kws=dict(edgecolor="k", linewidth=2)).set_title('Distribution of Price(Transformed value on log scale)')
sns.distplot(df['Landsize'],ax=axarr[1][1],color='orange',hist_kws=dict(edgecolor="k", linewidth=2)).set_title('Distribution of Landsize')

plt.subplots_adjust(hspace=.6)

sns.set_style('darkgrid')
sns.despine()
sns.pairplot(np.log(df[feat]))
df['price_per_unit_area']=df['Price']/df['Landsize']
plt.figure(figsize=(10,8))
sns.scatterplot(y=np.log(df['price_per_unit_area']),x=df['Distance'],data=df,hue='Regionname').set_title("Distance from property to CBD",fontsize=20);
df[discrete_feature]
f = plt.figure(figsize=(20,20))
for i in range(len(discrete_feature)):
    f.add_subplot(6, 3, i+1)
    g=df.groupby([discrete_feature[i]])['Price'].median().sort_values()
    ax=sns.boxplot(x=discrete_feature[i],y='Price',data=df,order=list(g.index),palette='Blues')
    ax=sns.stripplot(x=discrete_feature[i],y='Price',data=df,color='red',jitter=0.2,size=2.5)
    plt.title(discrete_feature[i],fontsize=15)

plt.tight_layout()
plt.show()
corr=df[discrete_feature].corr()
sns.heatmap(corr)
plt.title('Correlation among discrete variable');
for feature in discrete_feature:
    if outlier_function(df,feature)[2] > 1:
        print('{}: {} outliers'.format(feature,outlier_function(df,feature)[2]))
df['Bedroom2']=df['Bedroom2'].astype('int')
df['Bathroom']=df['Bathroom'].astype('int')
df['Car']=df['Car'].astype('int',errors='ignore')
df['Car']=df['Car'].astype('int',errors='ignore')
fig, axarr = plt.subplots(2, 2, figsize=(14,10))
fig.suptitle('Property features',fontsize=15)

sns.countplot(df['Rooms'],ax=axarr[0][0],palette='cubehelix').set_title('No. of Rooms')
sns.countplot(df['Bedroom2'],ax=axarr[1][0],palette='cubehelix').set_title('No. of Bedrooms')
sns.boxplot(x=df['Bedroom2'],y=df['Price'],ax=axarr[1][1],palette='husl').set_title('Price with respect to number of bedrooms')
sns.boxplot(x=df['Rooms'],y=df['Price'],ax=axarr[0][1],palette='husl').set_title('Price with respect to no. of rooms')

plt.subplots_adjust(hspace=.4)
sns.set_style('darkgrid')
sns.despine()
fig, axarr = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Property features',fontsize=15)

sns.countplot(df['Bathroom'],ax=axarr[0][0],palette='husl').set_title('No. of Bathrooms')
sns.violinplot(x= "Bathroom",y="Price",data=df,palette="Set2",ax=axarr[0][1]).set_title('No. of Bathrooms vs Price')
sns.countplot(df['Car'],ax=axarr[1][0],palette='husl').set_title('No.of Cars')
sns.boxenplot(x="Car",y="Price",data=df,palette="Set1",ax=axarr[1][1]).set_title('No.of Cars vs Price');


plt.subplots_adjust(hspace=.4)
sns.set_style('darkgrid')
sns.despine()
df[cat_feature]
plt.figure(figsize=(12,8))
g=sns.countplot(df['CouncilArea'],data=df,palette='Set2')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('No.of properties in each Council Area',fontsize=15);
plt.figure(figsize=(10,8))
g=sns.stripplot(x=df['CouncilArea'],y=df['Price'],hue='Type',data=df,palette='Set1')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('Preferred Council Area by Price',fontsize=15);
plt.figure(figsize=(10,6))
g=sns.countplot(df['Regionname'],hue='Type',data=df,palette='Set2')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('Preferred region by property type',fontsize=15);
plt.figure(figsize=(10,8))
g=df.groupby(['Regionname'])['Price'].mean().sort_values()
g=sns.boxplot(x='Regionname', y='Price', data=df,order=list(g.index),palette='cubehelix')
g=sns.stripplot(x='Regionname', y='Price', data=df,color='orange',jitter=0.2,size=2.5)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title('Price with respect to Region',fontsize=15);
fig, axs = plt.subplots(ncols=2,figsize=(15,6))

sns.boxplot(x=df['Method'],y=df['Price'],ax=axs[0],palette='cubehelix').set_title('Price vs Method')

sns.stripplot(x=df['Type'],y=df['Price'],ax=axs[1],palette='hls').set_title('Price vs Type');
fig, axarr = plt.subplots(2, 1, figsize=(12, 16))
fig.suptitle('Preferred Locations',fontsize=15)

sns.scatterplot(y=df['Lattitude'],x=df['Longtitude'],hue=df['Regionname'],palette='husl',ax=axarr[0]).set_title('Lattitude-Longitude, Regionname')
sns.scatterplot(y=df['Lattitude'],x=df['Longtitude'],hue=df['Type'],palette='Set1',ax=axarr[1]).set_title('Lattitude-Longitude,Type')
plt.show()
df=df.drop(index=9968,axis=1)
fig, axarr = plt.subplots(2, 1, figsize=(12, 10))

fig.suptitle('Trend in Price over the years',fontsize=20)

sns.lineplot(x='YearBuilt',y='Price',data=df,hue='Type',palette='Set1',ax=axarr[0])

sns.lineplot(x='YearBuilt',y='Price',data=df,hue='Regionname',palette='Set1',ax=axarr[1]);
def d_date(date):
    return(date[-7:])
df['Date']=df['Date'].apply(d_date)
plt.figure(figsize=(10,8))
gg=sns.countplot(df['Date'],palette='hls',hue='Type',data=df)
gg.set_xticklabels(gg.get_xticklabels(), rotation=70)
gg.set_title('Month and year of highest sale (2016-2017)',fontsize=15);
plt.figure(figsize=(10,8))
g=df.groupby(['SellerG','Type','Method']).Price.sum().sort_values(ascending=False).head(20)
g.plot(kind='barh',color='orange',title='Top realtors by Type and Method')
plt.xlabel('Price');
plt.figure(figsize=(16,10))
plt.title('Correlation heatmap', fontsize=20)
corr_matrix=df.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix,annot=True ,cbar = True,cmap="YlGnBu",mask=mask);