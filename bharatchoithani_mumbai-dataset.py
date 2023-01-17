import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#Raading the mumbai dataset
mumbai = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Mumbai.csv')
mumbai.head()
#shape of this dataset
mumbai.shape
#Description
mumbai.describe()
#Types of features that are given in this dataset
mumbai.dtypes
#Lets look at the correlation of all numerical features with our target variable
mumbai.corr()['Price'].sort_values(ascending = False)[1:]

#We can generate one feature based on the nearest railway line to the location of the apartment
harbour = ['IT', 'Antop Hill', 'Tardeo', 'Central Avenue', 'Kewale', 'Greater Khanda', 'Mumbai CST',  'Masjid Bunder',  'Sandhurst Road',  'Dockyard Road',  'Reay Road', 'Cotton green', 'Sewri', 'Wadala', 'GTB Nagar', 'Chunabhatti', 'Kurla', 'Tilak Nagar', 'Chembur', 'Govandi', 'Mankhurd', 'Vashi', 'Sanpada', 'Jui Nagar', 'Nerul', 'Seawoods', 'Belapur', 'Kharghar', 'Mansarovar', 'Khandeshwar', 'Panvel', 'Kamothe', 'Koproli', 'Ghansoli', 'Karanjade', 'Airoli', 'Koper Khairane', 'Kopar Khairane']
central = 'IT  Nashik  Ambarnath  Samata  Balkum  thane  Kharegaon  Beturkar  Pant  Kasheli  Highway  Pokhran  Haware  Shirgaon  Tardeo  Shil kolshet  Sainath  kavesar  Navi  Vikhroli  Palava  Hiranandani  PARSIK  Vartak  taloja  Kalamboli  Vasant  Majiwada  matunga  Dombivli  Bhiwandi  Taloja  Byculla  Chinchpokli  Currey Road  Parel  Dadar  Matunga  Sion  Kurla  Vidyavihar  Ghatkopar  Vikroli  Kanjurmarg  Bhandup  Nahur  Mulund  Thane  Kalwa  Mumbra  Diva  Dombivali  Thakurli  Kalyan  Shahad  Ambivili  Titwala  Khadvali  Vashind  Asangaon  Atgaon  Khardi  Kasara  Kalyan  Ulhasnagar  Ambernath  Badlapur  Vangani  Shelu  Neral  Bhivpuri  Karjat  Palasdhari  Kelavli  Dolavli  Lowjee  Khopoli  Powai'
central = central.split()
western = 'Churchgate  Bhayander  Naigaon  Vasai  Nalasopara  Virar  Vaitarna  Saphale  Palghar  Umroli  Boisar  Vangaon   Goregaon  Malad  Kandivali  Borivali  Dahisa  Andheri  Jogeshwari  Bandra  Dadar  Mahalaxmi'
western = western.split()
western1 = ['Mumbai Central', 'Lower Parel', 'Prabhadevi', 'Matunga', 'Mahim', 'Vile Parle', 'Mira Road', 'Ville Parle', 'Nala Sopara', 'Magathane', 'Juhu', 'Dattapada', 'Worli', 'Santacruz', 'Bhayandar', 'Thakur', 'Khar West', 'Rajendra Nagar', 'kandivali', 'vile parle west', 'Tardeo', 'Jankalyan Nagar', 'Jawahar Nagar', 'Marol', 'Hanuman Nagar', 'IT']
for i in western1:
    western.append(i)
mumbai['Railway_Line'] = ''

for i in mumbai.index:
    for j in harbour:
        if j in mumbai.loc[i, 'Location']:
            mumbai.loc[i, 'Railway_Line'] = 'Harbour'
            break
for i in mumbai.index:
    for j in central:
        if j in mumbai.loc[i, 'Location']:
            mumbai.loc[i, 'Railway_Line'] = 'Central'
            break
for i in mumbai.index:
    for j in western:
        if j in mumbai.loc[i, 'Location']:
            mumbai.loc[i, 'Railway_Line'] = 'Western'
            break
for i in [451, 809, 810, 2626, 2681, 3614, 2615]:
    mumbai.loc[i, 'Railway_Line'] = 'Western'
for i in mumbai.index:
    for j in ['Ulwe', 'Uran', 'Ranjanpada', 'Dronagiri']:
        if j in mumbai.loc[i, 'Location']:
            mumbai.loc[i, 'Railway_Line'] = 'Nerul-Uran'
            break
#Rows with empty values of Railway_Line are for those apartments whose nearest railway lines are unknown to us  
mumbai['Railway_Line'].value_counts()
#Lets look at the bar plot of our newly created feature 'Railway_Line'
mumbai['Railway_Line'].value_counts().plot.bar()
#Now, lets look at the avearage price for each railway line
mumbai.groupby('Railway_Line')['Price'].mean()[1:]
#Lets convert this string value to numerical values for the machine learning algorithms that we are going to use
mumbai['Central'] = 0
mumbai['Harbour'] = 0
mumbai['Western'] = 0
mumbai['Nerul-Uran'] = 0

for i in mumbai.index:
    if mumbai.loc[i, 'Railway_Line'] == 'Central':
        mumbai.loc[i, 'Central'] = 1
    elif mumbai.loc[i, 'Railway_Line'] == 'Harbour':
        mumbai.loc[i, 'Harbour'] = 1
    elif mumbai.loc[i, 'Railway_Line'] == 'Western':
        mumbai.loc[i, 'Western'] = 1
    elif mumbai.loc[i, 'Railway_Line'] == 'Nerul-Uran':
        mumbai.loc[i, 'Nerul-Uran'] = 1
#We know that Kurla comes in both central and harbour lines so we are assigning it to harbour line aswell
for i in mumbai.index:
    if mumbai.loc[i, 'Location'] == 'Kurla':
        mumbai.loc[i, 'Harbour'] = 1
#Now lets look at the correlation of our newly created features with our target variables
for i in ['Central', 'Harbour', 'Western', 'Nerul-Uran']:
    print(mumbai[i].corr(mumbai['Price']))
mumbai.drop('Railway_Line', axis = 1, inplace = True)
#mumbai.drop('Location', axis = 1, inplace = True)
#Lets look at the bar plots for our ordinal features
for i in mumbai.columns[3:]:
    plt.title(i) 
    sns.barplot(mumbai[i].value_counts().index, mumbai[i].value_counts().values)
    plt.show()
#Earlier, we saw that 'Area' has the higgest correlation with 'Price'. Lets examine this feature
#Skewness
mumbai['Area'].skew()
#Lets visualize its skewness
mean = mumbai['Area'].mean()
plt.figure(figsize=(10,7))
plt.axvline(mean, color='r', linestyle='--')
sns.distplot(mumbai['Area'])
#As we can see this feature is positively skewed
#Lets try different techniques so as to make it somewhat symmetrical
plt.figure(figsize=(10,7))
plt.axvline(np.log(mumbai['Area']).mean(), color='r', linestyle='--')
plt.title('Log')
sns.distplot(np.log(mumbai['Area']))
plt.figure(figsize=(10,7))
plt.axvline(np.sqrt(mumbai['Area']).mean(), color='r', linestyle='--')
plt.title('Square Root')
sns.distplot(np.sqrt(mumbai['Area']))
#We can see that log transformation looks better
mumbai['Area'] = np.log(mumbai['Area'])
#We saw that 'No. of Bedrooms' also has a high correlation with 'Price'. Lets examine it
mumbai.groupby('No. of Bedrooms')['Price'].mean()
sns.barplot(mumbai.groupby('No. of Bedrooms')['Price'].mean().index, mumbai.groupby('No. of Bedrooms')['Price'].mean().values)
#Looks like apartments with 5 bedrooms are the the costliest and apartments with 1 bedroom are the cheapest
#Lets look at the skewness of out target variable
print('Skewness =', mumbai['Price'].skew())
sns.distplot(mumbai['Price'])
a = mumbai['Location']
mumbai.drop('Location', axis = 1, inplace = True)
#Lets make separate columns for the top locations
#We will plot a no. of locations vs error graph to determine the no. of locations we need
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
xgb_reg = xgb.XGBRegressor() 

loc_count = [i for i in range(10,101,10)]
errors = []


for j in loc_count:
    top_locations = a.value_counts()[:j].index.values
    for i in top_locations:
        mumbai[i] = a.apply(lambda x : 1 if i == x else 0)
    errors.append(-1 * (cross_val_score(xgb_reg, mumbai.drop('Price', axis = 1), mumbai['Price'], cv=3, scoring = 'neg_mean_absolute_error').mean()))
    mumbai.drop(top_locations, axis = 1, inplace = True)
#Looks like 30 is the best value
plt.plot(loc_count, errors)
plt.xticks(loc_count)
plt.xlabel('No. of Locations')
plt.ylabel('Error')
plt.show()
top_locations = a.value_counts()[:30].index.values
for i in top_locations:
    mumbai[i] = a.apply(lambda x : 1 if i == x else 0)
#Lets split the data into train and test sets
#We are gonna do a 70-30 train-test split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(mumbai.drop('Price', axis = 1), mumbai['Price'], test_size=0.3, random_state=42)
#Now, I have have taken help for the evaluation and hypertuning part from 'https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f' as it has been very cleary explained 
#We are gonna use xgboost 
import xgboost as xgb 
#Lets convert our dataset into dmatrix
dtrain = xgb.DMatrix(train_x, label = train_y)
dtest = xgb.DMatrix(test_x, label = test_y)
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    #Other parameters
    'objective':'reg:linear',
    'eval_metric' : 'mae'
}
num_boost_round = 999
#Cross validation matrix
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)
cv_results
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6,15)
    for min_child_weight in range(2,10)
]
#Tuning max_depth and min_child_weight
min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
params['max_depth'] = 12
params['min_child_weight'] = 3
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(5,11)]
    for colsample in [i/10. for i in range(5,11)]
]
#Tuning subsample and colsample
min_mae = float("Inf")
best_params = None
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9
#tuning learning rate
min_mae = float("Inf")
best_params = None
for eta in [.5, .4, .3, .2, .1, .05, .01, .005, .001, .0005]:
    print("CV with eta={}".format(eta))
    params['eta'] = eta
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
params['eta'] = 0.005
params
#Lets train our model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
mean_absolute_error(model.predict(dtest), test_y)
#Features in descending order according to their importance
plt.figure(figsize=(20,15)) 
xgb.plot_importance(model, ax=plt.gca())