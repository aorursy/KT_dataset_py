# import tools for data manipulation
import pandas as pd
import numpy as np

#visualisation tools
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

#set font size
plt.rcParams['font.size']=25

#setting figure size
from IPython.core.pylabtools import figsize

#splitting test and train data
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',70)
energy = pd.read_csv('../input/energy_data.csv')
print(energy.shape,'\n')#11746 rows and 60 columns
print(energy.info())#check for missing values
energy.head()
#covert the data types
energy =energy.replace({'Not Available':np.nan})

#find columns that should be numeric and convert to float data type

for col in list(energy.columns):
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
            energy[col]=energy[col].astype(float)
energy.info()
# summary statistics upon success
energy.describe()
def missing_values_table_info(df):
    #number of null elements in the dataframe
    missing_vals=df.isnull().sum()
    
    #percantage of null elements
    missing_vals_percentage=100*missing_vals/len(df)
    
    #make a table of missing values
    missing_values_table=pd.concat([missing_vals,missing_vals_percentage],axis=1)
    
    #rename the columns
    missing_vals_rename_cols=missing_values_table.rename(columns={0:'missing values',1:'% of total'})
    
    #sort the dataframe in descending order by percentage of mising values
    missing_vals_rename_cols=missing_vals_rename_cols[
        missing_vals_rename_cols.iloc[:,1]!=0].sort_values('% of total',ascending=False).round(1)
    # Print some summary information
    print ("The dataset has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(missing_vals_rename_cols.shape[0]) +
            " columns that have missing values.")
    #return column with missing information
    return  missing_vals_rename_cols    
missing_values_table_info(energy)
#find columns with 50% of their data so as to discard them.
# this is because such columns may not be useful in our analysis
missing_df=missing_values_table_info(energy)
missing_columns=list(missing_df[missing_df['% of total']>50].index)
print('number of columns removed are {}'.format(len(missing_columns)))
#drop the columns
energy=energy.drop(columns=missing_columns,axis=1)
#check everything worked out
missing_values_table_info(energy)
#first focus on the target, which is the energy_score
figsize(10,10)
#rename the column
energy=energy.rename(columns={'ENERGY STAR Score':'score'})

#histogram of this score

plt.style.use('seaborn')
plt.hist(energy['score'].dropna(),color='blue')
plt.xlabel('energy score')
plt.ylabel('number of buildings');plt.title('score distribution')
#the anomalies are that an unusual number of managers reported 100% efficiency
#while some reported 0% efficiency. this may be a sign of total honesty or just plain lies
#using the energy use intensity which is unreported we can be able to show things clearly

figsize(10, 10)
plt.hist(energy['Site EUI (kBtu/ft²)'].dropna(), color = 'blue');
plt.xlabel('EUI'); 
plt.ylabel('Count'); plt.title('EUI Distribution');
# we have outliers and we should explore them a bit more
energy['Site EUI (kBtu/ft²)'].describe()
energy['Site EUI (kBtu/ft²)'].dropna().sort_values().head()
energy['Site EUI (kBtu/ft²)'].dropna().sort_values().tail()
energy[energy['Site EUI (kBtu/ft²)']==869265.000000]
# on the low end an extreme outlier is below 1st quartile - 3*interquatile range
# on the high end an extreme outlier is above 3rd quartile + 3*interquatile range

#find the quartiles
first_q =energy['Site EUI (kBtu/ft²)'].describe()['25%']
upper_q= energy['Site EUI (kBtu/ft²)'].describe()['75%']
#interquartile range
iqr=upper_q - first_q
                
# remove outliers
energy=energy[(energy['Site EUI (kBtu/ft²)']>(first_q-3* iqr)) & 
              (energy['Site EUI (kBtu/ft²)'] < (upper_q + 3 * iqr))] 
#histogram of this score
figsize(7,7)
plt.style.use('seaborn')
plt.hist(energy['Site EUI (kBtu/ft²)'].dropna(),color='blue')
plt.xlabel('energy score')
plt.ylabel('number of buildings');plt.title('score distribution')
#outliers have been removed and the skewness to the left eliminated.
#histogram now shows a uniform distrinution

# now investigate correlation between score and categorical features in the data. i.e building type
# end goal is to find out which features have a high correlation coefficient to energy score
# use a density plot. use buildings with more than 100 observatiions in the data so as not to clutter the image

# list of building types with more than 100 observations
types=energy.dropna(subset=['score'])
types = types['Largest Property Use Type'].value_counts()
types=list(types[types.values>100].index)

#plot the density plot
figsize(13,11)

for building_type in types:
    subset=energy[energy['Largest Property Use Type']==building_type]
    #kde plot
    sns.kdeplot(subset['score'].dropna(),label=building_type,alpha=0.6,shade=True)
# label the plot
plt.xlabel('Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Scores by Building Type', size = 28);
#from above feature we can see that use type has effect on score
#the feature is to be included in an ml model training thus hot encoding it is neccessary

#to investigate another feature. this time we'll use streets and correlation to score. code is similar to above one

streets=energy.dropna(subset=['score'])
streets=streets['Street Name'].value_counts()
streets=list(streets[streets.values>100].index)

for street in streets:
    subset=energy[energy['Street Name']==street]
    #kde plot
    sns.kdeplot(subset['score'].dropna(),label=street,alpha=0.6,shade=True)
# label the plot
plt.xlabel('Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Scores by Street', size = 28);

# there's some influence from street although not as much
# on to the last categorical variable now
# by borough(administrative centers like city wards suburbs in other areas)
boroughs=energy.dropna(subset=['score'])
boroughs=boroughs['Borough'].value_counts()
boroughs=list(boroughs[boroughs.values>100].index)

for borough in boroughs:
    subset=energy[energy['Borough']==borough]
    #kde plot
    sns.kdeplot(subset['score'].dropna(),label=street,alpha=0.6,shade=False)
# label the plot
plt.xlabel('Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Scores by Borough', size = 28);
#now to find the pearson correlation coefficient of features with the score
correlation=energy.corr()['score'].sort_values()
#most negative correlations
print(correlation.head(10),'\n')
#most positive correlations
print(correlation.tail(10),'\n')
# Select the numeric columns
numeric_subset = energy.select_dtypes('number')

# Create columns with square root and log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'score':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Select the categorical columns
categorical_subset = energy[['Borough', 'Street Name', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Drop buildings without a score
features = features.dropna(subset = ['score'])

# Find correlations with the score 
correlations_data = features.corr()['score'].dropna().sort_values()
#let's graph the most significant correlation (in terms of absolute value) in the dataset which is Site EUI (kBtu/ft^2). 
#We can color the graph by the building type to show how that affects the relationship. We will use a scatter plot

figsize(12, 10)

# Extract the building types
features['Largest Property Use Type'] = energy.dropna(subset = ['score'])['Largest Property Use Type']

# Limit to building types with more than 100 observations (from previous code)
features = features[features['Largest Property Use Type'].isin(types)]

# Use seaborn to plot a scatterplot of Score vs Log Source EUI
sns.lmplot('Site EUI (kBtu/ft²)', 'score', 
          hue = 'Largest Property Use Type', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("Site EUI", size = 28)
plt.ylabel('Score', size = 28)
plt.title('Score vs Site EUI', size = 36);

#There is a clear negative relationship between the Site EUI and the score
#We select the numeric features, adds in log transformations of all the numeric features, 
#selects and one-hot encodes the categorical features, and joins the sets of features together. 

# Copy the original data
features = energy.copy()

# Select the numeric columns
numeric_subset = energy.select_dtypes('number')

# Create columns with log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'score':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
        
# Select the categorical columns
categorical_subset = energy[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

features.shape #We result to have 11,319 buildings and 110 columns with score inclusive.
#not all feature are important
#several are highly correlated and therefore redundant. let's remove those
plot_data = energy[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()

plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
plt.xlabel('Site EUI'); plt.ylabel('Weather Norm EUI')
plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' % np.corrcoef(energy[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna(), rowvar=False)[0][1]);
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between Energy Star Score
    y = x['score']
    x = x.drop(columns = ['score'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    x = x.drop(columns = ['Weather Normalized Site EUI (kBtu/ft²)', 
                          'Water Use (All Water Sources) (kgal)',
                          'log_Water Use (All Water Sources) (kgal)',
                          'Largest Property Use Type - Gross Floor Area (ft²)'])
    
    # Add the score back in to the data
    x['score'] = y
               
    return x
# Remove the collinear features above a specified correlation coefficient in this case 0.6
features = remove_collinear_features(features, 0.6);
#drop all columns with na values
features = features.dropna(axis=1,how='all')
features.shape
#extract buildings with scores and those without scores
no_score=features[features['score'].isnull()]
with_score=features[features['score'].notnull()]

print(no_score.shape)
print(with_score.shape)
#separate the features and targets
features=with_score.drop('score',axis=1)
target= pd.DataFrame(with_score['score'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

#separate train and test data
X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.3)
# Function to calculate mean absolute error. 
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
#create a baseline
baseline_guess = np.median(y_train)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
#Saving Data

# Save the no scores, training, and testing data
no_score.to_csv('no_score.csv', index = False)
X_train.to_csv('training_features.csv', index = False)
X_test.to_csv('testing_features.csv', index = False)
y_train.to_csv('training_labels.csv', index = False)
y_test.to_csv('testing_labels.csv', index = False)
#Imputing missing values and scaling values
from sklearn.preprocessing import Imputer, MinMaxScaler

#ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# Reading in the data into dataframes 
train_features = pd.read_csv('training_features.csv')
test_features = pd.read_csv('testing_features.csv')
train_labels = pd.read_csv('training_labels.csv')
test_labels = pd.read_csv('testing_labels.csv')

# Display sizes of data
print('Training Feature Size: ', train_features.shape)
print('Testing Feature Size:  ', test_features.shape)
print('Training Labels Size:  ', train_labels.shape)
print('Testing Labels Size:   ', test_labels.shape)
#ml algorithms have a hard time understanding missing values
#in this step imputation is done to deal with the rest of the missing values(as we had dealt with some above)

#an imputter with a median filling strategy is applied here
imputer=Imputer(strategy='median')

imputer.fit(train_features)

#transforming training and testing features
X_train=imputer.transform(train_features)
X_test=imputer.transform(test_features)

print('Missing values in training features: ', np.sum(np.isnan(X_train)))
print('Missing values in testing features:  ', np.sum(np.isnan(X_test)))
#Making sure all values are finite
print (np.where(~np.isfinite(X_train)))
print (np.where(~np.isfinite(X_test)))
#normalize the data so that different algorithms perform optimally
#Creating a scaler object with a range of 0 - 1
scaler = MinMaxScaler(feature_range = (0, 1))

#Fit on the training data
scaler.fit(X_train)

#Transform both the training and testing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Converting y to one-dimensional array
y_train = np.array(train_labels).reshape((-1,))
y_test = np.array(test_labels).reshape((-1,))
#Training, testing and evaluating a model
def train_test_evaluate(model):
    
    #Train
    model.fit(X_train,y_train)
    
    #Test
    model_pred = model.predict(X_test)
    
    #Evaluate
    model_mae = mae(y_test, model_pred)
    
    #Return performance metric
    return model_mae
#Linear Regression
lr = LinearRegression()

lr_mae = train_test_evaluate(lr)

print ('Linear Regression Mean Absolute Error: %0.4f' %lr_mae, '\n')

#Support Vector Machines
svm = SVR(C = 1000, gamma = 0.1)
svm_mae = train_test_evaluate(svm)

print ('SVM Mean Absolute Error: %0.4f' %svm_mae, '\n')

#Random Forest
random_forest = RandomForestRegressor(random_state = 60)
random_forest_mae = train_test_evaluate(random_forest)

print ('Random Forest Mean Absolute Error: %0.4f' %random_forest_mae, '\n')

#Gradient Boosted Machines
gradient_boosted = GradientBoostingRegressor(random_state = 60)
gradient_boosted_mae = train_test_evaluate(gradient_boosted)

print ('Gradient Boosted Regression Mean Absolute Error: %0.4f' %gradient_boosted_mae, '\n')

#K-Nearest Neighbours                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
knn = KNeighborsRegressor(n_neighbors = 10)
knn_mae = train_test_evaluate(knn)

print ('K Nearest Neighbors Mean Absolute Error: %0.4f' %knn_mae)
plt.style.use('seaborn')
figsize(8, 6)

#A dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Linear Regression', 
                                           'Support Vector Machine', 
                                           'Random Forest', 
                                           'Gradient Boosted Machines', 'K-Nearest Neighbours'],
                                'mae': [lr_mae, svm_mae, random_forest_mae, gradient_boosted_mae, knn_mae]})

#Horizontal bar chart of MAE
model_comparison.sort_values('mae', ascending = True).plot(x = 'model', 
                                                           y = 'mae', 
                                                           kind = 'barh', 
                                                           color = 'green', 
                                                           edgecolor = 'black')

#Plot formatting
plt.ylabel('Models', size = 14) 
plt.yticks(size = 12)
plt.xlabel('Mean Absolute Error')
plt.xticks(size = 12)
plt.title('Model Comparison on Test MAE', size = 16)

#We can see that there is a use for ML because all the models significantly outperform the baseline.
#next time we'll continue on model optimisation
