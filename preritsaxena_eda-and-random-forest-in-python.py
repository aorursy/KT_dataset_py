# Importing libraries

import pandas as pd
import numpy as np
loans = pd.read_csv("../input/kiva_loans.csv")
mpi = pd.read_csv("../input/kiva_mpi_region_locations.csv")
theme = pd.read_csv("../input/loan_theme_ids.csv")
theme_region = pd.read_csv("../input/loan_themes_by_region.csv")
loans.head(5)
mpi.head(5)
theme.head(5)
theme_region.head(5)
# Loans

print(loans.shape)

# number of rows

print(len(loans))

# NAs

loans.isnull().sum()

# % of NAs

(loans.isnull().sum()/len(loans))*100

# Variables tags, region, funded time and partner id have the most missing values

# Removing tags and funded time as they are not important

loans2 = loans.drop(columns = ['tags', 'funded_time'])

(loans2.isnull().sum()/len(loans2))*100

# Now removing rows with missing values

loans3 = loans2.dropna()

loans3.shape
# Data pre-processing for formats and currencies

# Selecting the relavant columns

loans4 = loans3[['id', 'funded_amount', 'loan_amount', 'activity', 'sector', 'country_code', 'country', 'region', \
                 'partner_id', 'disbursed_time', \
                 'term_in_months','lender_count','borrower_genders', 'repayment_interval']]
loans4.shape

# Changing data formats

loans5= loans4

loans5.is_copy = False

loans5.disbursed_time = loans5.disbursed_time.str[:4]

loans5.disbursed_time.head(5)

loans5.disbursed_time.astype(int).head(5)
import seaborn as sb
import matplotlib as mlp
import matplotlib.pyplot as plt

loans5.describe()
loans5.describe(include = 'all')
# Basic graphs using seaborn

# Number of loans by sector

plt.figure(figsize=(17,8))

sb.countplot(x="sector", data=loans5, palette="Blues_d");
plt.title('Number of loans by sector')
plt.xlabel('sector')
plt.ylabel('number of loans')


plt.show()
# Loan amount given by activity

plt.figure(figsize=(17,8))

activity_wise = loans5.groupby(by=['activity'])[loans5.columns[2]].sum().sort_values(ascending = False).head(10)
sb.barplot(activity_wise.values, activity_wise.index, )
plt.xlabel('loan amount(in $10M)')
plt.title('Activity wise loan amount')

plt.show()
# Pre-processing MPI dataset

# Loans

print(mpi.shape)

# number of rows

print(len(mpi))

# NAs

mpi.isnull().sum()

# % of NAs

(mpi.isnull().sum()/len(mpi))*100

# It is a wrong representation of the data which has few inapproriate rows which have to be removeds

# Now removing rows with missing values

mpi2 = mpi.dropna()

mpi2.shape

mpi2.head(10)
# Some more pre-processing

# Data type checking for map

print(mpi2["MPI"].dtype)
print(mpi2["lat"].dtype)
print(mpi2["lon"].dtype)
# plotting for mpi dataset

import folium as folium
plt.figure(figsize=(17,8))

def plot_mpi_by_region(col_val):
    # generate a new map
    folium_map = folium.Map(zoom_start=12,
                        tiles="CartoDB dark_matter",
                        width='50%')

    # for each row in the data, add a cicle marker
    for index, row in mpi2.iterrows():

        # generate the popup message that is shown on click.
        popup_text = "World Region:  {}<br> MPI: {}"
        popup_text = popup_text.format(row["world_region"],
                          row["MPI"])

        # generate the popup message that is shown on click.
    #     popup_text = "world_region:  {}<br> country: {}<br> region: {}<br> MPI: {}"
    #     popup_text = popup_text.format(row["world_region"],
    #                       row["country"],
    #                       row["region"],
    #                       row["MPI"])

        # radius of circles
        radius = row["MPI"]*2

        # choose the color of the marker
        if row[col_val]>0.5:
            # color="#FFCE00" # orange
            # color="#007849" # green
            color="#EF0225" # red
        elif row[col_val]<0.5 and row[col_val]>0.25:
            # color="#0375B4" # blue
            # color="#FFCE00" # yellow            
            color="#FCEf00" # yellow
        else:
            color = "#1BED04" # green
            

        # add marker to the map
        folium.CircleMarker(location=(row["lat"],
                                      row["lon"]),
                            radius=radius,
                            color=color,
                            popup = popup_text,
                            fill=True).add_to(folium_map)

    return folium_map

# In the parameter area, enter the column you wish to base your size of marker upon
try:
    mpi_map = plot_mpi_by_region("MPI")
except TypeError:
    print('please enter a numeric field to get the marker')
    
mpi_map
# Pre-processing third database

# Themes

print(theme.shape)

# number of rows

print(len(theme))

# NAs

theme.isnull().sum()

# Dropping rows with blank themes

theme2 = theme.dropna()

theme2.shape
# Pre-processing fourth database

# Themes region

print(theme_region.shape)

# number of rows

print(len(theme_region))

# NAs

theme_region.isnull().sum()

# % of NAs

(theme_region.isnull().sum()/len(theme_region))*100

# It is a wrong representation of the data which has few inapproriate rows which have to be removeds

# Removing geocode_old, geocode, names, lat, lon and mpi_geo

theme_region2 = theme_region.drop(columns = ['geocode_old', 'geocode', 'names', 'lat', 'lon', 'mpi_geo'])

(theme_region2.isnull().sum()/len(theme_region2))*100

# Now removing rows with missing values

theme_region3 = theme_region2.dropna()

theme_region3.shape

theme_region3.head(10)
# EDA on loans theme dataset

# Major lenders

# Basic graphs using seaborn

# Number of loans by sector

import squarify as squarify

plt.figure(figsize=(17,8))

# field_partner = theme_region3.groupby(by=['Field Partner Name'])[theme_region3.columns[1]].count().\
#                 sort_values(ascending = False).head(10)
# sb.barplot(field_partner.values, field_partner.index, orient = "h", color = "brown")
# plt.xlabel('Number of loans given')
# plt.title('Major lenders')

# plt.show()


field_partner = theme_region3['Field Partner Name'].value_counts().sort_values(ascending = False).head(10)
squarify.plot(sizes=field_partner.values,label=field_partner.index, value=field_partner.values)
plt.title('Number of loans given by major field partners')

plt.show()
# sb.distplot(theme_region3['rural_pct'], kde = False, rug = True)

plt.hist(theme_region3['rural_pct'], 50, normed = 1, facecolor = 'green')
plt.xlabel("rural percentage")
plt.ylabel("percentage of data")
plt.title("Rural percentage histogram")

plt.show()

# This shows that most of the loans are offered to people living in regions with high rural population
#### Predictive Analytics

# Good funding in future largely depends on good repayment of loans. A lender whose who knows how his loan will be repayed
# is in a better position to secure funding for future

# Here we will try to predict what will be the repayment pattern of a loan
# We will do this using random forest

# Subsetting the dataset for random forest

loans6 = loans5[['loan_amount', 'disbursed_time','term_in_months', 'lender_count', 'repayment_interval']]

loans6.repayment_interval.astype('category')

loans6.head(5)
loans6.describe()
# Splitting the data into test and training

loans6['is_train'] = np.random.uniform(0, 1, len(loans6)) <= .75

train, test = loans6[loans6['is_train']==True], loans6[loans6['is_train']==False]

# number of observations in train and test
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
# List of predictor variables

# Traget valiable is repayment interval
features = loans6.columns[:4]

# View features
features
# One-hot encoding of target variable

y = pd.factorize(train['repayment_interval'])[0]

# View target
y
# Importing libraries

from sklearn.ensemble import RandomForestClassifier

# Initializing a random forest model

model_rf = RandomForestClassifier(n_jobs=2, random_state=0)
# Training the model on train set

model_rf.fit(train[features], y)
# Applying to test dataset

model_rf.predict(test[features])
# top 10 obs

model_rf.predict_proba(test[features])[0:10]
# Converting back to original forms of repayment intervals

preds = loans6.repayment_interval[model_rf.predict(test[features])]

preds.head(5)
# View a list of the features and their importance scores
list(zip(train[features], model_rf.feature_importances_))
# Merging loans5 and theme2

combined  = loans5.join(theme2, on = 'id', how = 'left', rsuffix = '2')

combined.head(5)
# writing out the combined dataset to an output csv file for further analysis

combined.to_csv('combined_dataset.csv', sep = ',')