import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV, ElasticNet, enet_path, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from math import sqrt
import matplotlib.pyplot as plot
import pandas as pd
import csv
# Load the data using Pandas ---------------------------------------------------
# Output: Pandas Series Object

school_explorer = pd.read_csv(
    '../input/2016 School Explorer.csv')
d5_shsat = pd.read_csv(
    '../input/D5 SHSAT Registrations and Testers.csv')
school_explorer.loc[[427,1023,712,908],'School Name'] = ['P.S. 212 D12',
                                                         'P.S. 212 D30',
                                                         'P.S. 253 D21',
                                                         'P.S. 253 D27']
# List all the columns with numeric values
numeric_list = list(school_explorer.columns[[7,8]+list(range(16,27))+
                                            [28,30,32,34,36]+
                                            list(range(42,161))])
# List all the columns with percent values
percent_list = list(school_explorer.columns[list(range(18,27))+(list([28,30,32,34,36]))])
# List all the categorical data columns
category_list = list(school_explorer.columns[[27,29,31,33,35,37,38]])
# Drop the first 3 columns becaue they are mostly empty and do not have a 
# lot of predictive value
school_explorer = school_explorer.drop(['Adjusted Grade',
                                        'New?',
                                        'Other Location Code in LCGMS'], axis=1)
# Remove % signs and N/As from the percent data.  Format as a floating point numbers
for elm in percent_list:
	school_explorer[elm] = school_explorer[elm].astype('str')			# Force it to be a string
	school_explorer[elm] = school_explorer[elm].str.replace("%","")		# Replace % with blanks
	school_explorer[elm] = school_explorer[elm].str.replace("nan","0")	# Replace nan with zeros
	school_explorer[elm] = school_explorer[elm].astype('float')			# Force it to be a float
	school_explorer[elm].replace(0, np.NaN, inplace = True)				# Replace zeros with Numpy NaN
	school_explorer[elm] = school_explorer[elm].interpolate()			# Interpolate missing values (linear)
# Remove $ signs and currency formating from the School Income Estimate.  Formate as a floating point number
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].astype('str')							# Cast as a string
for elm in [",", "$", " "]:
		school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace(elm, "")			# Remove special char

school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace("nan", "0")				# Remove nans	
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].astype('float')						# Cast as a float
#school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].replace(0.0, np.NaN, inplace = True)	# Fill with Numpy NaN	
# Fill in missing data with the pandas interpolation function over columns with missing data.
for elm in ['Economic Need Index', 'Average ELA Proficiency', 'Average Math Proficiency', 'School Income Estimate']:
	school_explorer[elm] = school_explorer[elm].interpolate()
# Quantizing the categorical data.  Assign numberic values to categories. -----------------------------------
for elm in category_list:
	school_explorer[elm] = school_explorer[elm].astype('str')																# Cast as a string
	'''
	Replace all the missing data points with a '1', meaning 'Meeting Target'.  Is this a fair assumption to make?
	'''
	school_explorer[elm] = school_explorer[elm].str.replace("nan", "1")														
	attribute_map = {'Not Meeting Target':'0', 'Meeting Target':'1', 'Approaching Target':'2', 'Exceeding Target':'3'}		# Map to integer
	school_explorer[elm].replace(attribute_map, inplace = True)																# Value Swap
	school_explorer[elm] = school_explorer[elm].astype(int)																	# Cast as an integer
# Read the data from the source file and only present the data for 2016 (same year as the school explorer data)
d5_shsat_2016 = d5_shsat[['DBN', 'Enrollment on 10/31','Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']][d5_shsat['Year of SHST']==2016].groupby(['DBN'], as_index = False).agg(np.sum)
d5_shsat_2016.rename(columns={'DBN':'Location Code'}, inplace = True)
# Calculate the ratio of students who actually took the test/registerd compared to the number of students enrolled.  This is the target for prediciton.
for row in d5_shsat_2016:
	d5_shsat_2016['% Test Takers'] = (d5_shsat_2016['Number of students who took the SHSAT'] / d5_shsat_2016['Enrollment on 10/31'])
	d5_shsat_2016['% Test Registration'] = (d5_shsat_2016['Number of students who registered for the SHSAT'] / d5_shsat_2016['Enrollment on 10/31'])
# Create a list of the two target values for predition
target_list = [d5_shsat_2016['% Test Takers'], d5_shsat_2016['% Test Registration']]
# Merge with the attribute data based on the location code.
merged_data = pd.merge(school_explorer, d5_shsat_2016, on = 'Location Code')
print('Size of the School Explorer Data = ', school_explorer.shape)
print('Size of the D5 SHSAT Data = ', d5_shsat_2016.shape)
print('Size of the finished merged data = ', merged_data.shape)
# Column Names from datasets
names = merged_data.columns.get_values()
names.tolist()
# This standardized function applies to all the data, including the target values that were merged together.
norm_data = merged_data.iloc[:,13:].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
# Create the attribute matrix by slicing the normalized data up to the number of students who registered for the exam.
# This grabs everything in the normalized data except the last two columns (the target values) and converts to a list.
xNormalized = norm_data.iloc[:,:-2].values.tolist()
#print (xNormalized)
# The normalization function tries to divide by zero for rows with only zeros (std = 0) this corrects the issue.
xNormalized = [[0 if np.isnan(elm) else elm for elm in row] for row in xNormalized]

# Create the first target variable, % of enrolled students who took the exam
# This grabs the last column and converts to a list.
target_take = norm_data.iloc[:,-1].values.tolist()
#print type(target_take)

# Create the second target variable, % of enrolled students who registered the exam
# This grabs the second to last column and converts to a list.
target_reg = norm_data.iloc[:,-2].values.tolist()
#print type(target_reg)

# Convert to Numpy Array
X = np.array(xNormalized)
Y1 = np.array(target_take)
Y2 = np.array(target_reg)
schoolModel = LassoCV(cv = 2, max_iter = 10000, selection = 'random').fit(X, Y1)
alphas, coefs, _ = linear_model.lasso_path(X, Y1, return_models=False)
nattr, nalpha = coefs.shape
#find coefficient ordering
nzList = []
for iAlpha in range(1,nalpha):
    coefList = list(coefs[: ,iAlpha])
    nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
    for q in nzCoef:
        if not(q in nzList):
            nzList.append(q)
nameList = [names[nzList[i]] for i in range(len(nzList))]
print("Attributes Ordered by How Early They Enter the Model", nameList)
plot.plot(alphas, coefs.T)
plot.title("Lasso Coefficient Curves")
plot.xlabel('alpha')
plot.ylabel('Coefficients')
plot.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plot.legend(nameList, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)
plot.axis('tight')
plot.semilogx()
ax = plot.gca()
ax.invert_xaxis()
plot.show()
alphaStar = schoolModel.alpha_  #0.04263474216648463
indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
indexStar = max(indexLTalphaStar)
#here's the set of coefficients to deploy
coefStar = list(coefs[:,indexStar])
print("Best Coefficient Values ", coefStar)
#names = names.tolist()
for i in range(len(coefStar)):
    if coefStar[i] != 0.0 or -0.0:
        print('Attribute = ', names[i])
        print('Coefficient = ', coefStar[i])
    else:
        pass
absCoef = [abs(a) for a in coefStar]

#sort by magnitude
coefSorted = sorted(absCoef, reverse=True)

idxCoefSize = [absCoef.index(a) for a in coefSorted if not(a == 0.0)]

namesList2 = [names[idxCoefSize[i]] for i in range(len(idxCoefSize))]

print("Attributes Ordered by Coef Size at Optimum alpha", namesList2)
# Plot the MSE for each curve ------------------------------------------

plot.figure()
plot.plot(schoolModel.alphas_, schoolModel.mse_path_, ':')
plot.plot(schoolModel.alphas_, schoolModel.mse_path_.mean(axis=-1),
label='Average MSE Across Folds', linewidth=2)
plot.axvline(schoolModel.alpha_, linestyle='--',
label='CV Estimate of Best alpha')
plot.semilogx()
plot.legend()
ax = plot.gca()
ax.invert_xaxis()
plot.xlabel('alpha')
plot.ylabel('Mean Square Error')
plot.title('Lasso Model, Cross Validations = 2')
plot.axis('tight')
plot.show()
#print out the value of alpha that minimizes the Cv-error
print("alpha Value that Minimizes CV Error ",schoolModel.alpha_)
print("Minimum MSE ", min(schoolModel.mse_path_.mean(axis=-1)))
l1_ratio = [.1, .5, .7, .9, .95, .99, 1] #alpha
l1_ratio_MSE = []
for i in l1_ratio:
    schoolModel3 = ElasticNetCV(l1_ratio = i, cv = 2, selection = 'random').fit(X, Y1)
    alphas, coefs, _ = linear_model.enet_path(X, Y1)
    print("alpha Value that Minimizes CV Error ",schoolModel3.alpha_)
    print("Minimum MSE ", min(schoolModel3.mse_path_.mean(axis=-1)))
    l1_ratio_MSE.append(min(schoolModel3.mse_path_.mean(axis=-1)))