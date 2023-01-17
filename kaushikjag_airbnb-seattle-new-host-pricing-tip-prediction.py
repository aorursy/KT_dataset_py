# For Analysis

import numpy as np

import pandas as pd



# For Visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# For Calculations

from math import floor



#For Modeling

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression , Lasso, Ridge

from sklearn.ensemble import RandomForestRegressor



# For Validation

from sklearn.metrics import mean_squared_error, accuracy_score



# For Storing Models

import pickle

%matplotlib inline



# For Warnings

import warnings

warnings.filterwarnings("ignore")
#Reading data

listings = pd.read_csv('/kaggle/input/seattle-listings/listings.csv')

#calendar = pd.read_csv('/kaggle/input/seattle/calendar.csv')

#reviews = pd.read_csv('/kaggle/input/seattle/reviews.csv')

neighscore = pd.read_csv('/kaggle/input/seattle-neighborhood-scores/Seattle_Scores.csv')
#get a small snapshot of the data

listings.head()
#Check for field information

listings.info()
#Check the statistical distribution for Numerical columns

listings.describe()
#Check the distribution of Categorical and Text columns

listings.describe(include=["O"])
#Listing columns

listings.columns
#Check the levels of some important listing related features

cat_level=['property_type','room_type','bed_type','cancellation_policy']

[listings[c].value_counts() for c in cat_level]
#Function for showing columns with missing values

def show_missing_values(df):

    missing_vals = df.isnull().sum().sort_values(ascending = False)

    

    return missing_vals.iloc[missing_vals.nonzero()[0]]
show_missing_values(listings)
#Function for handling missing values

def handle_missing_remove(df):

    temp_df=df.copy()

    #Fill 0 in place of missing records for number of bedrooms,bathrooms,beds

    temp_df.beds.fillna(0,inplace=True)

    temp_df.bedrooms.fillna(0,inplace=True)

    temp_df.bathrooms.fillna(0,inplace=True)

    

    #Remove columns license and squarefeet which have more than 95% missing values

    temp_df.drop(['license','square_feet'],axis=1,inplace=True)

     

    return temp_df
listings=handle_missing_remove(listings)

show_missing_values(listings)
def preprocess(df):

    

    temp_df = df.copy()

    temp_df = temp_df.replace(

            {

            'host_has_profile_pic': {'t': True, 'f': False},

            'host_identity_verified': {'t': True, 'f': False},

            'instant_bookable': {'t': True, 'f': False},

            }

    )

        

    ## Recode property_type

    def recode_prop(value):

        if value not in ['House', 'Apartment','Condominium','Townhouse','Loft']:

            return 'other_prop_type'

        return value



    temp_df['property_type'] = temp_df['property_type'].apply(recode_prop)



    ## Recode bed_type

    def recode_bed(value):

        if value not in ['Real Bed']:

            return 'other_bed_type'

        return value



    temp_df['bed_type'] = temp_df['bed_type'].apply(recode_bed)

        

    #Calculate the bedroom and bathroom share per person. Higher the share, more the comfort.

    temp_df = temp_df.assign(

        bedroom_share = temp_df.bedrooms/temp_df.accommodates,

        bathroom_share = temp_df.bathrooms/temp_df.accommodates,

        

    )

    

    df=temp_df

    print("Pre-processing completed...")

    return df
#Preprocess the listings data

listings=preprocess(listings)
#Create dummy columns by one-hot encoding

def create_dummies(df, columns = ['room_type', 'property_type', 'bed_type', 'cancellation_policy']):

    for column in columns:

        dummies = pd.get_dummies(df[column], prefix = column)

        df = pd.concat([df,dummies], axis = 1)

    return df
# Create the required dummy columns

listings = create_dummies(listings)
#Blank missing values in Amenities column

listings.loc[listings['amenities'] == '{}','amenities'] = ""
#Remove the symbols and split the amenities with | as separator

listings['amenities'] = listings['amenities'].map(

    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")\

                           for amn in amns.split(",")]))
listings['amenities'].head()
#Take the unique list of amenities across all listings

amenities = np.unique(np.concatenate(listings['amenities'].map(lambda amns: amns.split("|")).values))

amenities
#Map the presence or absence of amenities for each listing

amenities_matrix = np.array([listings['amenities'].map(lambda amns: amn in amns).values for amn in amenities])

amenities_matrix
#Make the amenities dataframe with boolean values

amen=pd.DataFrame(data=amenities_matrix.T, columns=amenities)

amen.head()
#Concat the listing id to amen dataframe

listings_amenities = pd.concat([amen,listings['id']], axis=1)

listings_amenities.head()
#Remove first column from listings_amenities whose name is ""

listings_amenities=listings_amenities.drop("",axis=1)

listings_amenities.head()
listings_amenities.columns
amenity_recode={

                'Air Conditioning':'Weather Control',

                'Indoor Fireplace':'Weather Control',

                'Heating':'Weather Control',

        

                'Carbon Monoxide Detector':'Safety Features',

                'Fire Extinguisher':'Safety Features',

                'First Aid Kit':'Safety Features',

                'Smoke Detector':'Safety Features',

                

                'Buzzer/Wireless Intercom':'Security Features',

                'Doorman':'Security Features',

                'Safety Card':'Security Features',

                'Lock on Bedroom Door':'Security Features',

                

                'Cat(s)':'Pet Friendly',

                'Dog(s)':'Pet Friendly',

                'Pets Allowed':'Pet Friendly',

                'Pets live on this property':'Pet Friendly',

                'Other pet(s)':'Pet Friendly',

                

                'Elevator in Building':'Access Friendly',

                'Wheelchair Accessible':'Access Friendly',

                

                'Essentials':'Essentials',

                'Hair Dryer':'Essentials',

                'Hangers':'Essentials',

                'Iron':'Essentials',

                'Shampoo':'Essentials',             

                

                'Cable TV':'TV',

                'TV':'TV',

                

                'Internet':'Internet',

                'Wireless Internet':'Internet',

                'Laptop Friendly Workspace':'Internet',

                

                'Dryer':'Laundry Facility',

                'Washer':'Laundry Facility',

                'Washer / Dryer':'Laundry Facility',

    

                #Leaving amenities as such which cannot be grouped

                #'Kitchen',

                #'Family/Kid Friendly', 

                #'Free Parking on Premises',

                #'Breakfast',

                #'24-Hour Check-in',

                #'Hot Tub',

                #'Pool',

                #'Gym',

                #'Smoking Allowed',

                #'Suitable for Events'

}
#Melt the amenities dataframe and recode from the dictionary

listings_amenities_melt = listings_amenities.melt(id_vars=['id'], var_name='amenity')



#Recoding and putting in new column called amenity_modified

listings_amenities_melt = listings_amenities_melt.assign(

    amenity_modified = listings_amenities_melt.amenity.replace(amenity_recode)

)



listings_amenities_melt.head()
#Pivot the melted dataframe before merging with original dataframe

listings_amenities_pivot = listings_amenities_melt.pivot_table(

    index='id',

    columns='amenity_modified',

    values='value', 

    aggfunc='max'

)



listings_amenities_pivot.head()
#Join the amenities dataframe back to the original listings dataframe

listings_joined=listings.join(listings_amenities_pivot,on="id",how="inner")

listings_joined.head()
# Generating viridis heatmap to check the correlation among the amenities

listings_selected_amenities=listings_joined[['24-Hour Check-in', 'Access Friendly', 'Breakfast', 'Essentials',

       'Family/Kid Friendly', 'Free Parking on Premises', 'Gym', 'Hot Tub',

       'Internet', 'Kitchen', 'Laundry Facility', 'Pet Friendly', 'Pool',

       'Safety Features', 'Security Features', 'Smoking Allowed',

       'Suitable for Events', 'TV', 'Weather Control']]

fig = plt.figure(figsize= (12,12))

sns.heatmap(listings_selected_amenities.corr(), annot=False, vmax=1, cmap='viridis', square=False)
#listings_am_ns=listings_joined.join(neigh_score,on="id",how="left")

listings_am_ns=pd.merge(neighscore, listings_joined, how='right', left_on=['WSName'], right_on=['neighbourhood_cleansed'])
listings_am_ns.dtypes
#Replace $ and , with "" in the price column and convert the price column to float

listings_am_ns.price=listings_am_ns.price.apply(lambda x: x.replace('$',''))

listings_am_ns.price=listings_am_ns.price.apply(lambda x: x.replace(',',''))

listings_am_ns.price=listings_am_ns.price.astype(float)
#Distribution of target

sns.distplot(listings_am_ns.price)
listings_am_ns.shape
list(listings_am_ns)
## Generating the heatmap for visualization - using Seaborn

listings_selected_am_ns = listings_am_ns[[

    'accommodates', 

    'beds','bathrooms',

    'bedroom_share','bathroom_share',

    'price'

]]

fig = plt.figure(figsize= (6,6))

sns.heatmap(listings_selected_am_ns.corr(), annot=False, vmax=1, cmap='viridis', square=False)
show_missing_values(listings_am_ns)
#Creating the train and test split

np.random.seed(2018)

train = np.random.choice([True, False], listings_am_ns.shape[0], replace=True, p=[0.8, 0.2])

listings_train = listings_am_ns.iloc[train,:]

listings_test = listings_am_ns.iloc[~train,:]
list(listings_train)
def model_listing(regr,train_cols,target_col):

    

    x_train = listings_train[train_cols].values

    x_test = listings_test[train_cols].values

    y_train = listings_train[target_col].values

    y_test = listings_test[target_col].values

    

    print("Shape of Train and Test data")

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    print(" ------------------------------------------ ")

    

    #Min Max Scaling



    #scaler = MinMaxScaler()

    #x_train = scaler.fit_transform(x_train)

    #x_test = scaler.transform(x_test)

    

    # Declare an instance of the Linear Regression model.

    rg = regr()



    # Fit the model on to the training data( Train the model ).

    rg.fit(x_train, y_train)

    

    # Use the model to predict values

    y_pred = rg.predict(x_train)



    # Calculate the Mean Squared Error using the mean_squared_error function.

    print("Training Data")

    print("R^2 value using score fn: %.3f" % rg.score(x_train,y_train))

    print("Mean Squared Error : %0.3f" % mean_squared_error(y_train,y_pred))

    print("Root Mean Squared Error : %0.3f" % (mean_squared_error(y_train,y_pred))**0.5)

    print(" ------------------------------------------ ")

    # Use the model to predict values

    y_pred = rg.predict(x_test)



    # Calculate the Mean Squared Error using the mean_squared_error function.

    print("Test Data")

    print("R^2 value using score fn: %.3f" % rg.score(x_test,y_test))

    print("Mean Squared Error : %0.3f" % mean_squared_error(y_test,y_pred))

    print("Root Mean Squared Error : %0.3f" % (mean_squared_error(y_test,y_pred)**0.5))

    print(" ------------------------------------------ ")

    #print(lm.intercept_, lm.coef_)

    

    lin_reg_coef = pd.DataFrame(list(zip(train_cols,(rg.coef_))),columns=['Feature','Coefficient'])

    print(lin_reg_coef.sort_values(by='Coefficient',ascending=False))

    print(" ------------------------------------------ ")

    

    # Plot of model's residuals:

    fig = plt.figure(figsize=(10,3))



    sns.regplot(y_test,y_pred)

    plt.title("Residuals for the model")
#Include all features which are relevant to a new host
train_cols = [

    'accommodates', 

    'beds','bathrooms',

    'bed_type_Real Bed',

    'property_type_Condominium','property_type_Townhouse',

    'room_type_Entire home/apt', 'room_type_Private room',

    'property_type_Apartment','property_type_House', 

    'cancellation_policy_flexible', 'cancellation_policy_moderate','instant_bookable'

]



target_col = 'price'



model_listing(LinearRegression,train_cols,target_col)
#Include all features which might be relevant for a new host + Neighbourhood scores
train_cols = [

    'accommodates', 

    'beds','bathrooms',

    'bed_type_Real Bed',

    'property_type_Condominium','property_type_Townhouse',

    'room_type_Entire home/apt', 'room_type_Private room',

    'property_type_Apartment','property_type_House', 

    'cancellation_policy_flexible', 'cancellation_policy_moderate','instant_bookable',

    'Walk','Transit','Bike'

]



target_col = 'price'



model_listing(LinearRegression,train_cols,target_col)
#Include basic features + neighbourhood scores + amenities
train_cols = [

    'accommodates', 

    'beds','bathrooms',

    'bed_type_Real Bed',

    'property_type_Condominium','property_type_Townhouse',

    'room_type_Entire home/apt', 'room_type_Private room',

    'property_type_Apartment','property_type_House', 

    'cancellation_policy_flexible', 'cancellation_policy_moderate','instant_bookable',

    '24-Hour Check-in', 'Access Friendly', 'Breakfast', 'Essentials',

       'Family/Kid Friendly', 'Free Parking on Premises', 'Gym', 'Hot Tub',

       'Internet', 'Kitchen', 'Laundry Facility', 'Pet Friendly', 'Pool',

       'Safety Features', 'Security Features', 'Smoking Allowed',

       'Suitable for Events', 'TV', 'Weather Control','Walk','Transit','Bike'

    

]



target_col = 'price'



model_listing(LinearRegression,train_cols,target_col)
# Function to calculate regularized cost given alpha, mse and the model coefficients

def reg_cost(alpha, mse, coeffs, model = None):

    if model == "lasso":

        return mse + alpha * np.sum(np.abs(coeffs))

    elif model == "ridge":

        return mse + alpha * np.linalg.norm(coeffs)

    else:

        return mse
alpha_levels = [0.01, 0.1, 1, 10, 100]



x_train = listings_train[train_cols].values

x_test = listings_test[train_cols].values

y_train = listings_train[target_col].values

y_test = listings_test[target_col].values



for alpha_level in alpha_levels:

    print("\n At alpha Level: %0.2f "% alpha_level)



    lasso_lm = Lasso(alpha= alpha_level)



    # Fit the model on to the training data( Train the model ).

    lasso_lm.fit(x_train, y_train)



    # Use the model to predict values

    #y_pred = np.expm1(lm.predict(x_test))

    y_pred = lasso_lm.predict(x_test)



    # Calculate the Mean Squared Error using the mean_squared_error function.

    print("Test Data")

    print("R^2 value using score fn: %.3f" % lasso_lm.score(x_test,y_test))

    print("Mean Squared Error : %0.3f" % mean_squared_error(y_test,y_pred))

    print("Root Mean Squared Error : %0.3f" % (mean_squared_error(y_test,y_pred))**0.5)

       

    # Get model complexity using the user defined fn

    print("Model Complexity: %0.3f" % reg_cost(mse = 0, alpha = 1, coeffs= lasso_lm.coef_, model= "lasso"))

    

    # Get Regularized Cost using the user defined fn

    print("Regularized Cost: %0.3f" % reg_cost(mse = mean_squared_error(y_test,y_pred), alpha = alpha_level, coeffs= lasso_lm.coef_, model= "lasso"))
train_cols = [

    'accommodates', 

    'beds','bathrooms',

    'bed_type_Real Bed',

    'property_type_Condominium','property_type_Townhouse',

    'room_type_Entire home/apt', 'room_type_Private room',

    'property_type_Apartment','property_type_House', 

    'cancellation_policy_flexible', 'cancellation_policy_moderate','instant_bookable',

    '24-Hour Check-in', 'Access Friendly', 'Breakfast', 'Essentials',

       'Family/Kid Friendly', 'Free Parking on Premises', 'Gym', 'Hot Tub',

       'Internet', 'Kitchen', 'Laundry Facility', 'Pet Friendly', 'Pool',

       'Safety Features', 'Security Features', 'Smoking Allowed',

       'Suitable for Events', 'TV', 'Weather Control','Walk','Transit','Bike'

    

]



target_col = 'price'



x_train = listings_train[train_cols].values

x_test = listings_test[train_cols].values

y_train = listings_train[target_col].values

y_test = listings_test[target_col].values



print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)



#Create a random forest regressor

clf = RandomForestRegressor(max_depth=10, n_estimators=100)



#Train the regressor

clf.fit(x_train, y_train)



#Plot variable importances for the top 10 predictors

importances = clf.feature_importances_

feat_names = train_cols

tree_result = pd.DataFrame({'feature': feat_names, 'importance': importances})

tree_result.sort_values(by='importance',ascending=False)[:10].plot(x='feature', y='importance', kind='bar',color='blue')
# Use the model to predict values

y_pred = clf.predict(x_train)



# Calculate the Mean Squared Error using the mean_squared_error function.

print("Training Data")

print("R^2 value using score fn: %.3f" % clf.score(x_train,y_train))

print("Mean Squared Error : %0.3f" % mean_squared_error(y_train,y_pred))

print("Root Mean Squared Error : %0.3f" % (mean_squared_error(y_train,y_pred))**0.5)





print(" ------------------------------------------ ")



# Use the model to predict values

y_pred = clf.predict(x_test)



# Calculate the Mean Squared Error using the mean_squared_error function.

print("Test Data")

print("R^2 value using score fn: %.3f" % clf.score(x_test,y_test))

print("Mean Squared Error : %0.3f" % mean_squared_error(y_test,y_pred))

print("Root Mean Squared Error : %0.3f" % (mean_squared_error(y_test,y_pred))**0.5)



print(" ----------------------------------- ")



# Plot of model's residuals:

fig = plt.figure(figsize=(10,3))



sns.regplot((y_test),(y_pred))

plt.title("Residuals for the model")