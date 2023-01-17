# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cross_validation import train_test_split
#import polynomial_regression as plr
from sklearn.linear_model import LinearRegression, Ridge

matplotlib.rcParams.update({'font.size': 12})

dtype_dict = {'AnimalID':str,'Name':str,'DateTime':str,'OutcomeType':str,'OutcomeSubtype':str,'AnimalType':str,'SexuponOutcome':str,'AgeuponOutcome':str,'Breed':str,'Color':str}

# Load Testing and Training
testing = pd.read_csv('../input/test.csv', sep=',')
training = pd.read_csv('../input/train.csv', sep=',')

## Split into training and validation sets 
df2 = training.ix[1:]
train, validation = train_test_split(df2, test_size=0.50) 
#I think that Shuffling is built in to this
# See: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html


############## Functions ####################

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + input_feature*slope
    return predicted_values


def polynomial_dataframe(data_frame, feature, animal, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    feature_values = np.array(data_frame.loc[data_frame['AnimalType'] == animal,feature])
    if(feature == 'AgeuponOutcome'):
       ages_in_years = age_to_years(feature_values)
       poly_dataframe['power_1'] = ages_in_years
#    # first check if degree > 1
       if degree > 1:
#        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            tmp_power = pd.Series(ages_in_years).apply(lambda x: x**power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_dataframe[name] = tmp_power
       return poly_dataframe


def outcome_counts_for_feature(data_frame,feature):
    # Some parts of this code were borrowed from Andy's example on the Kaggle Scripts Board
    feature_values = np.array(data_frame[feature])
    outcome = np.array(data_frame['OutcomeType'])
    unique_outcomes = np.array(['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'], dtype=object)
    unique_features = np.unique(feature_values)
    outcome_contours = np.zeros([len(unique_outcomes),len(unique_features)])
    outcome_counts = np.zeros(len(unique_features))  
    for i in range(len(unique_features)):
        for j in range(len(unique_outcomes)):
            list1 = outcome[feature_values == unique_features[i]] #list of outcomes for that particular feature
            outcome_contours[j,i] = len(list1[list1 == unique_outcomes[j]]) #/len(list1) # fraction of total for each outcome based on that feature
        outcome_counts[i] = len(list1)
    return unique_features, outcome_contours
        
def get_residual_sum_of_squares_poly(model, data, outcome):
    # Get Residual Sum of Squares (RSS) to assess error of fit
    example_predictions = model.predict(data)
    residuals = outcome - example_predictions
    prod_res = residuals*residuals
    RSS = prod_res.sum()
    return(RSS)  

def print_coefficients(model):    
    w0=model.coef_.tolist()
    w=w0[0]
    deg = len(w)-1
    print('Learned polynomial for degree ' + str(deg) + ':')
    w.reverse()
    print(np.poly1d(w))

def polynomial_features(data, deg):
    data_copy=data.copy()
    for i in range(1,deg):
        data_copy['X'+str(i+1)]=data_copy['X'+str(i)]*data_copy['X1']
    return data_copy

def k_fold_cross_validation(k, l2_penalty, data, output, model):
    # Perform k-fold cross validation
    data1 = data.ix[1:]
    n = len(data1)
    for i in range(k):
        start = round((n*i)/k)
        end = round((n*(i+1))/k-1)
        validation0=data1[start:end+1]
        output_val = output[start:end+1]
        last = data1[end+1:n]
        first = data1[0:start] 
        train0=first.append(last)
        last_out = output[end+1:n]
        first_out = output[0:start]    
        output_train=append_ndarray(first_out, last_out)
    model.fit(train0, output_train)
    rss_validation = get_residual_sum_of_squares_poly(model, validation0, output_val) 
    return rss_validation
  
def append_ndarray(arr1, arr2):
    #Append two ndarrays
    target = []
    for i in arr1:
        target.append(i)
    for j in arr2:
        target.append(j)
#    print(target)
    return np.array(target)

def age_to_years(item):
    # Based on age_to_days function written by "Andy" on the Kaggle Scripts board
    if type(item) is str:
        item = [item]
    ages_in_years = np.zeros(len(item))
    for i in range(len(item)):
        if type(item[i]) is str:
            if 'day' in item[i]:
                ages_in_years[i] = int(item[i].split(' ')[0])/365
            if 'week' in item[i]:
                ages_in_years[i] = int(item[i].split(' ')[0])/(52) #approx
            if 'month' in item[i]:
                ages_in_years[i] = int(item[i].split(' ')[0])/12
            if 'year' in item[i]:
                ages_in_years[i] = int(item[i].split(' ')[0])    
        else:
            ages_in_years[i] = 0
    return ages_in_years

#==============================================================================
# For degree in range(1, 20+1)):
#  Learn a polynomial regression model to Age of Shelter Cats vs Adoption Rate with that 
#  degree on TRAIN data.
#  Compute the RSS on VALIDATION data for that degree.
#==============================================================================
#Report which degree had the lowest RSS on validation data 

Number_of_Polys = 12
RSS = np.zeros((Number_of_Polys))
val_output = np.array(validation['OutcomeType'])
  
print('Cross validation for different degrees of polynomial fit to data')
for degree in range(1, Number_of_Polys+1):
    polydata_train = polynomial_dataframe(train,'AgeuponOutcome', 'Cat',degree) 
    my_features = polydata_train.columns.values.tolist()
    polydata_train['OutcomeType'] = train['OutcomeType'] # add price to the data since it's the target
    unique_f,outcomes_f=outcome_counts_for_feature(polydata_train,'power_1')
    new_polydata_train = pd.DataFrame()   
    for f in my_features:
        unique_f,not_used=outcome_counts_for_feature(polydata_train,f)
        new_polydata_train[f]=unique_f
    polydata_valid = polynomial_dataframe(validation,'AgeuponOutcome', 'Cat',degree)
    polydata_valid['OutcomeType'] = validation['OutcomeType'] # add price to the data since it's the target
    unique_val_f,outcomes_val_f=outcome_counts_for_feature(polydata_valid,'power_1')
    new_polydata_valid = pd.DataFrame()  
    for f in my_features:
        unique_val_f,not_used=outcome_counts_for_feature(polydata_valid,f)
        new_polydata_valid[f]=unique_val_f
    model = LinearRegression()
    nr_adopted=outcomes_f[0].reshape((len(outcomes_f[0]),1))
    fitted = model.fit(new_polydata_train,nr_adopted)
    nr_adopted_val=outcomes_val_f[0].reshape((len(outcomes_val_f[0]),1))    
    RSS[degree-1] = get_residual_sum_of_squares_poly(model, new_polydata_valid,nr_adopted_val)
    print('Fit to ' + str(degree) + ' degree polynomial' + ' with Error, RSS: ' + str(RSS[degree-1]))


#################################################################################
#      Plot Results of 8th order polynomial fit
###################################################################################

unique_f,outcomes_f=outcome_counts_for_feature(polydata_train,'power_1')
p_deg = np.poly1d(np.polyfit(unique_f,outcomes_f[0], 6))
xp = np.linspace(0, 20, 100)
plt.plot(unique_f, outcomes_f[0], '.', xp, p_deg(xp), '--')
plt.title('Overfitted? -- Cats Adopted According to Age')
plt.xlabel('age [years]')
plt.ylabel('Number of Cats Adopted')
plt.yscale('log')
plt.show()
plt.close()

#==============================================================================
# Use Ridge Regression with L2-penalty to prevent overfitting the data
#==============================================================================
 
RSSlist=[0]
k = 10
for l2_penalty in np.logspace(-1, 0, num=10):
    model_ridge = Ridge(normalize=True, alpha=l2_penalty, solver='auto', fit_intercept=True, tol=0.001)
    RSSdatum=k_fold_cross_validation(k, l2_penalty, new_polydata_train, nr_adopted, model_ridge)
    RSSlist.append(RSSdatum)
    print('L2-penalty: ' + str(l2_penalty) + '     Error, RSS: ' + str(RSSdatum))


###################### Compare with L2-penalty = 1 using TEST data ####################################

model_ridge_best = Ridge(normalize=True, alpha=1, solver='auto', fit_intercept=True, tol=0.001)
c = new_polydata_train['power_1']
new_c=c.reshape(len(c),1) 
modelo_l21=model_ridge_best.fit(new_c, nr_adopted)
Age_Data = polynomial_dataframe(testing,'AgeuponOutcome', 'Cat',1)  
feature_values = np.array(Age_Data['power_1'])
unique_features_test = np.unique(feature_values)
new_f=unique_features_test.reshape(len(unique_features_test),1) 
example_predictions = modelo_l21.predict(new_f)


############## Plot the Predicted Fit
p1, = plt.plot(unique_f,outcomes_f[0], '.' )
p2, = plt.plot(new_f,example_predictions,'.-')
plt.title('Ridge Regression Fit, L2 = 1')
plt.xlabel('age [years]')
plt.ylabel('Number of Cats Adopted')
plt.yscale('log')
plt.show()
plt.close()

###################### Compare with L2-penalty = 0.10 using TEST data  ####################################

model_ridge_best = Ridge(normalize=True, alpha=1e-1, solver='auto', fit_intercept=True, tol=0.001)
c = new_polydata_train['power_1']
new_c=c.reshape(len(c),1) 
modelol201=model_ridge_best.fit(new_c, nr_adopted)
Age_Data = polynomial_dataframe(testing,'AgeuponOutcome', 'Cat',1)  
feature_values = np.array(Age_Data['power_1'])
unique_features_test = np.unique(feature_values)
new_f=unique_features_test.reshape(len(unique_features_test),1) 
example_predictions = modelol201.predict(new_f)

#example_predictions_6years = modelo.predict(6.0)
#example_predictions_18years = modelo.predict(18.0)

############## Plot the Predicted Fit
p1, = plt.plot(unique_f,outcomes_f[0], '.' )
p2, = plt.plot(new_f,example_predictions,'.-')
plt.title('Ridge Regression Fit, L2 = 0.10')
plt.xlabel('age [years]')
plt.ylabel('Number of Cats Adopted')
plt.yscale('log')
plt.show()
plt.close()

#############################################################################################
##   Create a DataFrame of Results and Plot based on TEST data

IDs = np.array(testing.loc[testing['AnimalType'] == 'Cat','ID'])
Cat_Ages = feature_values
Adoption_Numbers_L2_01 = []
Adoption_Numbers_L2_1 = []
for a in Cat_Ages:
    predicted_adoption_nrl201 = modelol201.predict(a)
    pa01=predicted_adoption_nrl201.tolist()
    Adoption_Numbers_L2_01.append(pa01[0][0])
    predicted_adoption_nrl21 = modelo_l21.predict(a)
    pa1=predicted_adoption_nrl21.tolist()
    Adoption_Numbers_L2_1.append(pa1[0][0])
#Avg_Adopted = np.array(Adoption_Numbers)/sum(Adoption_Numbers)*100
#Avg_Adopted.tolist()
Cat_Ages.tolist()
IDs.tolist()

Cat_Adoptions_Age = pd.DataFrame()
Cat_Adoptions_Age['ID [Cat]'] = IDs
Cat_Adoptions_Age['Age [ years]'] = Cat_Ages
Cat_Adoptions_Age['No. Adoptions with L2=0.1'] = Adoption_Numbers_L2_01
Cat_Adoptions_Age['No. Adoptions with L2=1'] = Adoption_Numbers_L2_1

p1, = plt.plot(Cat_Ages,Adoption_Numbers_L2_01, '+' )
p2, = plt.plot(Cat_Ages,Adoption_Numbers_L2_1, '.' )
p3, = plt.plot(np.linspace(0,25),np.zeros(len(np.linspace(0,25))), '-' )
plt.title('Adoptions According to Ridge Regression Fits')
plt.xlabel('age [years]')
plt.ylabel('Number of Cats Adopted')
plt.legend([p1,p2],['L2-penalty=0.10','L2-penalty=1'],loc=1)
plt.show()
plt.close()

Cat_Adoptions_Age.head()
#dtype_dict_new = {'ID [Cat]':str,'Age [ years]':str,'No. Adoptions with L2=0.1':str,'No. Adoptions with L2=1':str}
header = ['ID [Cat]','Age [ years]','No. Adoptions with L2=0.1','No. Adoptions with L2=1']

Cat_Adoptions_Age.to_csv('Cat_Adoptions_RegularizedRegressionFit.csv', columns = header)
