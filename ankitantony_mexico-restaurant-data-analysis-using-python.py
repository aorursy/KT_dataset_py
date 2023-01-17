# For data manipulation and analysis
import pandas as pd
import numpy as np


# For graphs and plotting
import matplotlib.pyplot as plt; plt.rcdefaults()
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import seaborn as sns
# Reading csv files downloaded and stored in repository in Github

# geoplaces = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/geoplaces2.csv")
# rating_final = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/rating_final.csv")
# usercuisine = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/usercuisine.csv")
# userprofile = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/userprofile.csv")

geoplaces = pd.read_csv("/input/restaurantdata/geoplaces2.csv",error_bad_lines=False, encoding="latin-1")
# rating_final = pd.read_csv("../input/restaurantdata/rating_final.csv")
# usercuisine = pd.read_csv("../input/restaurantdata/usercuisine.csv")
# userprofile = pd.read_csv("../input/restaurantdata/userprofile.csv")
# Overwriting column with replaced value of country to Mexico
geoplaces["country"] = geoplaces["country"].replace("?", "Mexico")

# To replace the small mexico to capital letter Mexico
geoplaces["country"] = geoplaces["country"].replace("mexico", "Mexico")

# Overwriting column with ? to NA
geoplaces["address"] = geoplaces["address"].replace("?", "NA")
geoplaces["city"]= geoplaces["city"].replace("?", "NA")
geoplaces["state"]= geoplaces["state"].replace("?", "NA")

# Copying data into geoplaces_final for further analysis
geoplaces_final = geoplaces[['placeID', 'latitude', 'longitude', 'the_geom_meter', 'name', 'address',
       'city', 'state', 'country',   'alcohol', 'smoking_area',
       'dress_code', 'accessibility', 'price',  'Rambience', 'franchise',
       'area', 'other_services']]

geoplaces_final.head()
# Replacing '?' with 'NA'
userprofile["smoker"] = userprofile["smoker"].replace("?", "NA")
userprofile["ambience"] = userprofile["ambience"].replace("?","NA")
userprofile["transport"] = userprofile["transport"].replace("?","NA")
userprofile["marital_status"] = userprofile["marital_status"].replace("?","NA")
userprofile["activity"] = userprofile["activity"].replace("?","NA")
userprofile["budget"] = userprofile["budget"].replace("?","NA")

# Replacing '?' with 'no preference'
userprofile["dress_preference"] = userprofile["dress_preference"].replace("?", "no preference")

# Copying data into userprofile_final for further analysis
userprofile_final = userprofile.copy()

userprofile_final.head()
# Re-importing geoplaces dataset file for map plotting
geoplaces = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/geoplaces2.csv")

mapbox_access_token='pk.eyJ1IjoibmF2ZWVuOTIiLCJhIjoiY2pqbWlybTc2MTlmdjNwcGJ2NGt1dDFoOSJ9.z5Jt4XxKvu5voCJZBAenjQ'

mcd=geoplaces[geoplaces.country =='mexico']
mcd_lat = mcd.latitude
mcd_lon = mcd.longitude

data = [
    go.Scattermapbox(
        lat=mcd_lat,
        lon=mcd_lon,
        mode='markers',
        marker=dict(
            size=10,
            color='rgb(255, 0, 0)',
            opacity=0.4
        ))]
layout = go.Layout(
    title='Restaurants Locations',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=23,
            lon=-102
        ),
        pitch=2,
        zoom=4.5,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='restaurants')
# TO GET THE TOTAL CUSTOMERS PER CITY
city_breakdown = geoplaces[['city', 'placeID']]
city_breakdown.groupby('city').count().reset_index()

# TO CLEAN UP THE CITY NAMES FOR BETTER VISUALIZATION
slp_rep = ['san luis potosi', 'slp', 's.l.p.', 's.l.p','Soledad','san luis potos', 'san luis potosi ']

city_breakdown['city'] = city_breakdown.city.apply(lambda x: 'San Luis Potosi' if x in slp_rep else x)

#Ciudad Victoria
CDV_rep = ['Cd. Victoria', 'victoria', 'victoria ', 'Cd Victoria']
city_breakdown['city'] = city_breakdown.city.apply(lambda y: 'Ciudad Victoria' if y in CDV_rep else y)  

#Cuernavaca
city_breakdown.replace( {'city':'cuernavaca'}, {'city': 'Cuernavaca'}, inplace= True )

#?
city_breakdown.replace( {'city':'?'}, {'city': 'Unavailable'}, inplace= True )

city_breakdown2=city_breakdown.groupby('city').count().reset_index()
city_breakdown2.columns = ['city', 'TotalCount']

city_breakdown2.head()
plt.bar(city_breakdown2.city, city_breakdown2.TotalCount, align='center', alpha=0.5)

plt.ylabel('Total Count')
plt.title('Customer Count by City')
plt.xticks(city_breakdown2.city.index, city_breakdown2.city.values, fontsize=10, rotation=15)
plt.show()
DT = usercuisine.groupby('Rcuisine').count().reset_index()
DT.columns= ['CUISINE', 'TOTALNUMBER']
top5_usercuisine = DT.sort_values(by='TOTALNUMBER', ascending=False).head(5)

top5_usercuisine
plt.pie(top5_usercuisine.TOTALNUMBER , labels = top5_usercuisine.CUISINE, autopct='%.1f%%')
plt.title('Top 5 Customer Restaurant Type Preference')
plt.show()
# Unique count of users in each profession 'activity'
user_profession = userprofile.groupby('activity').count().reset_index()

plt.bar(user_profession.activity, user_profession.userID, align='center')
plt.ylabel('Number of Consumers')
plt.title('Different types of profession')
# Merging usercuisine and userprofile_final
budget_cuisine = pd.merge(left=usercuisine, right=userprofile_final,left_on='userID', right_on='userID')

final_budget_cuisine = budget_cuisine[['userID','Rcuisine','budget']]
cuisine_low = final_budget_cuisine.loc[final_budget_cuisine['budget'] == 'low']
cuisine_low_userCount = cuisine_low.groupby(["Rcuisine", "budget"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_low = cuisine_low_userCount.sort_values('COUNT',ascending=False).head(5)

# Using Seaborn to plot bar graph
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_low)

ax.set(xlabel='Preferred Cuisines', ylabel='Customer Count', title = 'Preferred Cuisine for LOW Budget customers')
plt.show()
cuisine_medium = final_budget_cuisine.loc[final_budget_cuisine['budget'] == 'medium']
cuisine_medium_userCount = cuisine_medium.groupby(["Rcuisine", "budget"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_medium = cuisine_medium_userCount.sort_values('COUNT',ascending=False).head(5)

# Using Seaborn to plot bar graph
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_medium)

ax.set(xlabel='Preferred Cuisines', ylabel='Customer Count', title = 'Preferred Cuisine for MEDIUM Budget customers')
plt.show()
cuisine_high = final_budget_cuisine.loc[final_budget_cuisine['budget'] == 'high']
cuisine_high_userCount = cuisine_high.groupby(["Rcuisine", "budget"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_high = cuisine_high_userCount.sort_values('COUNT',ascending=False).head(5)

# Using Seaborn to plot bar graph
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_high)

ax.set(xlabel='Preferred Cuisines', ylabel='Customer Count', title = 'Preferred Cuisine for HIGH Budget customers')
plt.show()
# Merging geoplaces_final and rating_final
master_placeID_userID = pd.merge(left = geoplaces_final, right = rating_final, left_on='placeID', right_on='placeID')
placeID_userID = master_placeID_userID[['name', 'userID']]

placeID_userID.head()
restaurant_userCount = placeID_userID.groupby('name')['userID'].count().reset_index(name = "USER COUNT")
top10_restaurant_userCount = restaurant_userCount.sort_values('USER COUNT', ascending = False).head(10)
top10_restaurant_userCount

ax = sns.barplot(x='name', y='USER COUNT', data = top10_restaurant_userCount)

ax.set(xlabel='Frequented Restaurants', ylabel='Customer Count', title='Top 10 Restaurants Most Frequented by Customers')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
UserProfile_cuisine = pd.merge(left=usercuisine, right=userprofile, on="userID", how="left")
Profession_perCuisine = UserProfile_cuisine[['Rcuisine', 'activity']]
Profession_perCuisine.replace( {'activity':'?'}, {'activity': 'N/A'}, inplace= True )
p_cuisine2 = Profession_perCuisine[Profession_perCuisine['Rcuisine'].isin(['American','Mexican', 'Cafe-Coffee_Shop', 'Cafeteria', 'Pizzeria'])]
p_cuisine2
df1 = p_cuisine2['Rcuisine'].groupby([p_cuisine2['Rcuisine'], p_cuisine2['activity']]).count().reset_index(name='count')
df2 = df1.pivot(index='Rcuisine', columns='activity', values='count')
df3 = df2.fillna(0)
df3
# RESTAURANT TYPE PER PROFESSION
fig, ax = plt.subplots(figsize=(12, 7))
heat_dist = sns.heatmap(df3, annot=True, fmt='g', linewidths=0.05, cmap='viridis')
# Create a table of different personalities with respect to other characteristics.
userprofile = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/userprofile.csv")
user_personality = userprofile.groupby('personality').count().reset_index()
user_personality
# Create a bar chart showing each personality's number of customers
plt.bar(user_personality.personality, user_personality.userID, align='center')
plt.ylabel('Count of customers')
plt.title('Different types of personality')
# Create a personality and cuisine merge with regard to user ID
usercuisine = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/usercuisine.csv")
userprofile = pd.read_csv("https://raw.githubusercontent.com/ankit-antony/PythonGroupAssignment/master/userprofile.csv")
personality_cuisine = pd.merge(left=usercuisine, right=userprofile,left_on='userID', right_on='userID')
new_personality_cuisine = personality_cuisine[['userID','Rcuisine','personality']]
new_personality_cuisine.head()
cuisine_hardworker = new_personality_cuisine.loc[new_personality_cuisine['personality'] == 'hard-worker']
cuisine_hardworker_userCount = cuisine_hardworker.groupby(["Rcuisine", "personality"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_hardworker = cuisine_hardworker_userCount.sort_values('COUNT',ascending=False).head(5)

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_hardworker)

ax.set(xlabel='Preferred Restaurant Type', ylabel='Customer Count', title = 'Preferred Restaurant Type for HARD-WORKER customers')
plt.show()
cuisine_thrifty = new_personality_cuisine.loc[new_personality_cuisine['personality'] == 'thrifty-protector']
cuisine_thrifty_userCount = cuisine_thrifty.groupby(["Rcuisine", "personality"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_thrifty = cuisine_thrifty_userCount.sort_values('COUNT',ascending=False).head(5)

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_thrifty)

ax.set(xlabel='Preferred Restaurant Type', ylabel='Customer Count', title = 'Preferred Restaurant Type for THRIFTY-PROTECTOR customers')
plt.show()
cuisine_hunter = new_personality_cuisine.loc[new_personality_cuisine['personality'] == 'hunter-ostentatious']
cuisine_hunter_userCount = cuisine_hunter.groupby(["Rcuisine", "personality"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_hunter = cuisine_hunter_userCount.sort_values('COUNT',ascending=False).head(5)

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_hunter)

ax.set(xlabel='Preferred Restaurant Type', ylabel='Customer Count', title = 'Preferred Restaurant Type for HUNTER-OSTENTATIOUS customers')
plt.show()
cuisine_conformist = new_personality_cuisine.loc[new_personality_cuisine['personality'] == 'conformist']
cuisine_conformist_userCount = cuisine_conformist.groupby(["Rcuisine", "personality"])["userID"].count().reset_index(name="COUNT")
top5_cuisine_conformist = cuisine_conformist_userCount.sort_values('COUNT',ascending=False).head(5)

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.barplot(x="Rcuisine", y="COUNT", data = top5_cuisine_conformist)

ax.set(xlabel='Preferred Restaurant Type', ylabel='Customer Count', title = 'Preferred Restaurant Type for CONFORMIST customers')
plt.show()
mexi_personality = new_personality_cuisine.loc[new_personality_cuisine['Rcuisine'] == 'Mexican']
mexi_personality_userCount = mexi_personality.groupby(["personality", "Rcuisine"])["userID"].count().reset_index(name="COUNT")
top50_mexi_personality = mexi_personality_userCount.sort_values('COUNT',ascending=False).head(50)

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.barplot(x="personality", y="COUNT", data = top50_mexi_personality)

ax.set(xlabel='Personality', ylabel='Customer Count', title = 'Personality of Mexican restaurant customers')
plt.show()
AVG_rating_restaurant = pd.merge(left=usercuisine, right=rating_final,left_on='userID', right_on='userID')
AVG_new_rating_restaurant = AVG_rating_restaurant[['placeID','Rcuisine','rating']]
AVG_new_rating_restaurant.head()
AVG_rating_copy = AVG_new_rating_restaurant.groupby(['placeID','Rcuisine'])['rating'].apply(lambda x: (x.sum())/x.count())
final_AVG_rating = AVG_rating_copy.reset_index(name="Average Rating")
final_AVG_rating
geo_avg_rating_merge = pd.merge(left=geoplaces_final, right=final_AVG_rating,left_on='placeID', right_on='placeID')
final_geo_avg_rating = geo_avg_rating_merge[['placeID','alcohol','smoking_area','dress_code','accessibility','Rambience','franchise','price','Rcuisine','Average Rating']]
final_geo_avg_rating.head()
final_geo_avg_rating.isnull().sum()
#To check if there is any empty cells, before proceeding with label encoding
#Outcome: no empty cells available
final_geo_avg_rating.dtypes
# Label encoding - To convert categorical values
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

copy_final_geo_avg_rating = final_geo_avg_rating.select_dtypes(include=['object'])
final_le = copy_final_geo_avg_rating.apply(encoder.fit_transform, axis=0)   # Encoding using .apply()

final_le[['placeID','Average Rating']]=final_geo_avg_rating[['placeID','Average Rating']]
final_le.head()
# Shifting 'placeID' and 'Average Rating' columns to the front of Data Frame

cols = list(final_le.columns)
cols = [cols[-2]] + cols[:-2] + [cols[-1]]
final_le = final_le[cols]

final_le.head()
# For modelling purpose we are label encoding palceID.
final_le['placeID']=encoder.fit_transform(final_le['placeID'])
final_le.head()
final_le.dtypes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# Splitting train and test data as 80/20
X=final_le.drop(['placeID','Average Rating'], axis=1)
y=final_le['Average Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# Model building.
regressor = LinearRegression()  
regressor.fit(X_train, y_train) # Training the algorithm
# Predicting on test data.
prediction =  regressor.predict(X_test)
# Compare the actual output values for X_test with the predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
comparison.head()
# To calculate 'explained variance regression score'
print("Explained Variance Score: " + str(explained_variance_score(y_test, prediction)))

# To calculate maximum residual error 
print("\nMaximum Residual Error: " + str(max_error(y_test, prediction)))

# To compute mean absolute error
print("\nMean Absolute Error: " + str(mean_absolute_error(y_test, prediction)))

# To compute mean square error
print("\nMean Square Error: " + str(mean_squared_error(y_test, prediction)))
from sklearn.preprocessing import LabelEncoder
# TO CLEAN UP THE CITY NAMES FOR BETTER VISUALIZATION
slp_rep = ['san luis potosi', 'slp', 's.l.p.', 's.l.p','Soledad','san luis potos', 'san luis potosi ']

geoplaces_final['city'] = geoplaces_final.city.apply(lambda x: 'San Luis Potosi' if x in slp_rep else x)
#Ciudad Victoria
CDV_rep = ['Cd. Victoria', 'victoria', 'victoria ', 'Cd Victoria']
geoplaces_final['city'] = geoplaces_final.city.apply(lambda y: 'Ciudad Victoria' if y in CDV_rep else y)  
#Cuernavaca
geoplaces_final.replace( {'city':'cuernavaca'}, {'city': 'Cuernavaca'}, inplace= True )
#?
geoplaces_final.replace( {'city':'?'}, {'city': 'Unavailable'}, inplace= True )
geoplaces_final2 = geoplaces_final[['placeID', 
       'city',    'alcohol', 'smoking_area',
       'dress_code', 'accessibility', 'price',  'Rambience', 'franchise',
       'area', 'other_services']]
# CONVERTING THE CATEGORICAL VARIABLES TO NUMBERS SO THAT IT CAN BE USED FOR PREDICTION
cat_cols = ['city','alcohol', 'smoking_area', 'dress_code', 'accessibility', 'Rambience', 'franchise','area', 'other_services']
for var in cat_cols:
 number = LabelEncoder()
 geoplaces_final2[var] = number.fit_transform(geoplaces_final2[var].astype('str'))
geoplaces_final2["price"] = number.fit_transform(geoplaces_final2["price"].astype('str'))
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x1= ['city','alcohol', 'smoking_area','dress_code', 'accessibility',  'Rambience', 'franchise',
    'area', 'other_services']
y1= ['price']
xTrain1, xTest1, yTrain1, yTest1 = train_test_split(geoplaces_final2[x1], geoplaces_final2[y1], test_size = 0.20, random_state = 0)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(xTrain1, yTrain1)
print("Accuracy on training set: {:.3f}".format(tree.score(xTrain1, yTrain1)))
print("Accuracy on test set: {:.3f}".format(tree.score(xTest1, yTest1)))
print("Feature importances:")
print(tree.feature_importances_)


import matplotlib.pyplot as plt
def plot_feature_importances_feat_imp(model):
    n_features = 9
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x1)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_feat_imp(tree)