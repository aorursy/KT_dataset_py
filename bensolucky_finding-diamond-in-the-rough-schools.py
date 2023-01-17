# Loading python libraries
%matplotlib inline
import seaborn as sns
cm = sns.light_palette("grey", as_cmap=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import shap
import os
import math

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
############################################################
# Load the school explorer data and clean it
SCHOOL = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv', index_col=["Location Code"])
SCHOOL.columns = [x.upper() for x in SCHOOL.columns]
SCHOOL.index.name = 'DBN'

# Convert to a numeric value (i.e. 3174.99 instead of $3,174.99)
dollars = ["SCHOOL INCOME ESTIMATE"]
for col in dollars:
    SCHOOL[col] = SCHOOL[col].str.replace(',', '')
    SCHOOL[col] = SCHOOL[col].str[1:].astype(float)
# Convert these to numerics
percents = ["PERCENT ELL", "PERCENT ASIAN", "PERCENT BLACK", "PERCENT HISPANIC", 
            "PERCENT BLACK / HISPANIC", "PERCENT WHITE", "STUDENT ATTENDANCE RATE",
            "PERCENT OF STUDENTS CHRONICALLY ABSENT", "RIGOROUS INSTRUCTION %", 
            "COLLABORATIVE TEACHERS %", "SUPPORTIVE ENVIRONMENT %", 
            "EFFECTIVE SCHOOL LEADERSHIP %", "STRONG FAMILY-COMMUNITY TIES %", "TRUST %"]
for col in percents:
    SCHOOL[col] = SCHOOL[col].str[:-1].astype(float) / 100
# I'm just going to drop these ratings since we already have a % version of each
ratings = ["RIGOROUS INSTRUCTION RATING", "COLLABORATIVE TEACHERS RATING",
          "SUPPORTIVE ENVIRONMENT RATING", "EFFECTIVE SCHOOL LEADERSHIP RATING",
          "STRONG FAMILY-COMMUNITY TIES RATING", "TRUST RATING", "STUDENT ACHIEVEMENT RATING"]
SCHOOL = SCHOOL.drop(ratings,axis=1)

# Save school name and address for later use
SCHOOL_NAMES = SCHOOL[["SCHOOL NAME", "ADDRESS (FULL)"]]

# Save latitude and longitude to calculate distance from closest elite school
DISTANCE = SCHOOL[['LATITUDE', 'LONGITUDE']]

# I didn't attempt to use these variables
other_data_not_used = ["SCHOOL INCOME ESTIMATE", "OTHER LOCATION CODE IN LCGMS", "SCHOOL NAME", "SED CODE", "ADDRESS (FULL)", "GRADES", 'CITY', 'LATITUDE', 'LONGITUDE', 'ZIP']
SCHOOL = SCHOOL.drop(other_data_not_used, axis=1)

# Some more simple data cleaning / preprocessing
SCHOOL[["ADJUSTED GRADE", "NEW?"]] = SCHOOL[["ADJUSTED GRADE", "NEW?"]].replace("x", 1).fillna(0)
SCHOOL["GRADE LOW"] = SCHOOL["GRADE LOW"].replace("0K", 0).replace("PK", -1).astype(float)
SCHOOL["GRADE HIGH"] = SCHOOL["GRADE HIGH"].replace("0K", 0).astype(float)
SCHOOL["COMMUNITY SCHOOL?"] = SCHOOL["COMMUNITY SCHOOL?"].replace("Yes", 1).replace("No", 0).astype(float)

# There are a massive amount of common core Result variable.  I treat these in the following way:
# 1. Sum up all the 4s scored in ethnic, economic need and ELL sub-categories across all grade levels
#    separately for both ELA and MATH
# 2. Divide by the total number of students tested in all these grade levels
# 3. The result is the total fraction of the student body that both received a 4 and belonged to
#    that sub-category.  So for example, if my new variable MATH ELL% was 0.1 for a school,
#    then that 10% of that school's student body was an ELL student who also received a 4 in MATH.
# 4. Finally, I average ELA and MATH together
SCHOOL["ELA TESTED"] = SCHOOL[["GRADE 3 ELA - ALL STUDENTS TESTED", "GRADE 4 ELA - ALL STUDENTS TESTED", "GRADE 5 ELA - ALL STUDENTS TESTED", "GRADE 6 ELA - ALL STUDENTS TESTED", "GRADE 7 ELA - ALL STUDENTS TESTED", "GRADE 8 ELA - ALL STUDENTS TESTED"]].sum(1)
SCHOOL["ELA ALL 4%"] = SCHOOL[["GRADE 3 ELA 4S - ALL STUDENTS", "GRADE 4 ELA 4S - ALL STUDENTS", "GRADE 5 ELA 4S - ALL STUDENTS", "GRADE 6 ELA 4S - ALL STUDENTS", "GRADE 7 ELA 4S - ALL STUDENTS", "GRADE 8 ELA 4S - ALL STUDENTS"]].sum(1)
SCHOOL["ELA AAALN 4%"] = SCHOOL[["GRADE 3 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 4 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 5 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 6 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 7 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 8 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE"]].sum(1)
SCHOOL["ELA BLACK 4%"] = SCHOOL[["GRADE 3 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 4 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 5 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 6 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 7 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 8 ELA 4S - BLACK OR AFRICAN AMERICAN"]].sum(1)
SCHOOL["ELA LATINO 4%"] = SCHOOL[["GRADE 3 ELA 4S - HISPANIC OR LATINO", "GRADE 4 ELA 4S - HISPANIC OR LATINO", "GRADE 5 ELA 4S - HISPANIC OR LATINO", "GRADE 6 ELA 4S - HISPANIC OR LATINO", "GRADE 7 ELA 4S - HISPANIC OR LATINO", "GRADE 8 ELA 4S - HISPANIC OR LATINO"]].sum(1)
SCHOOL["ELA ASIAN 4%"] = SCHOOL[["GRADE 3 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 4 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 5 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 6 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 7 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 8 ELA 4S - ASIAN OR PACIFIC ISLANDER"]].sum(1)
SCHOOL["ELA WHITE 4%"] = SCHOOL[["GRADE 3 ELA 4S - WHITE", "GRADE 4 ELA 4S - WHITE", "GRADE 5 ELA 4S - WHITE", "GRADE 6 ELA 4S - WHITE", "GRADE 7 ELA 4S - WHITE", "GRADE 8 ELA 4S - WHITE"]].sum(1)
SCHOOL["ELA MULTIRACIAL 4%"] = SCHOOL[["GRADE 3 ELA 4S - MULTIRACIAL", "GRADE 4 ELA 4S - MULTIRACIAL", "GRADE 5 ELA 4S - MULTIRACIAL", "GRADE 6 ELA 4S - MULTIRACIAL", "GRADE 7 ELA 4S - MULTIRACIAL", "GRADE 8 ELA 4S - MULTIRACIAL"]].sum(1)
SCHOOL["ELA ECON 4%"] = SCHOOL[["GRADE 3 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 4 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 5 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 6 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 7 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 8 ELA 4S - ECONOMICALLY DISADVANTAGED"]].sum(1)
SCHOOL["ELA ELL 4%"] = SCHOOL[["GRADE 3 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 4 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 5 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 6 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 7 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 8 ELA 4S - LIMITED ENGLISH PROFICIENT"]].sum(1)

SCHOOL["MATH TESTED"] = SCHOOL[["GRADE 3 MATH - ALL STUDENTS TESTED", "GRADE 4 MATH - ALL STUDENTS TESTED", "GRADE 5 MATH - ALL STUDENTS TESTED", "GRADE 6 MATH - ALL STUDENTS TESTED", "GRADE 7 MATH - ALL STUDENTS TESTED", "GRADE 8 MATH - ALL STUDENTS TESTED"]].sum(1)
SCHOOL["MATH ALL 4%"] = SCHOOL[["GRADE 3 MATH 4S - ALL STUDENTS", "GRADE 4 MATH 4S - ALL STUDENTS", "GRADE 5 MATH 4S - ALL STUDENTS", "GRADE 6 MATH 4S - ALL STUDENTS", "GRADE 7 MATH 4S - ALL STUDENTS", "GRADE 8 MATH 4S - ALL STUDENTS"]].sum(1)
SCHOOL["MATH AAALN 4%"] = SCHOOL[["GRADE 3 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 4 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 5 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 6 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 7 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 8 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE"]].sum(1)
SCHOOL["MATH BLACK 4%"] = SCHOOL[["GRADE 3 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 4 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 5 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 6 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 7 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 8 MATH 4S - BLACK OR AFRICAN AMERICAN"]].sum(1)
SCHOOL["MATH LATINO 4%"] = SCHOOL[["GRADE 3 MATH 4S - HISPANIC OR LATINO", "GRADE 4 MATH 4S - HISPANIC OR LATINO", "GRADE 5 MATH 4S - HISPANIC OR LATINO", "GRADE 6 MATH 4S - HISPANIC OR LATINO", "GRADE 7 MATH 4S - HISPANIC OR LATINO", "GRADE 8 MATH 4S - HISPANIC OR LATINO"]].sum(1)
SCHOOL["MATH ASIAN 4%"] = SCHOOL[["GRADE 3 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 4 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 5 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 6 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 7 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 8 MATH 4S - ASIAN OR PACIFIC ISLANDER"]].sum(1)
SCHOOL["MATH WHITE 4%"] = SCHOOL[["GRADE 3 MATH 4S - WHITE", "GRADE 4 MATH 4S - WHITE", "GRADE 5 MATH 4S - WHITE", "GRADE 6 MATH 4S - WHITE", "GRADE 7 MATH 4S - WHITE", "GRADE 8 MATH 4S - WHITE"]].sum(1)
SCHOOL["MATH MULTIRACIAL 4%"] = SCHOOL[["GRADE 3 MATH 4S - MULTIRACIAL", "GRADE 4 MATH 4S - MULTIRACIAL", "GRADE 5 MATH 4S - MULTIRACIAL", "GRADE 6 MATH 4S - MULTIRACIAL", "GRADE 7 MATH 4S - MULTIRACIAL", "GRADE 8 MATH 4S - MULTIRACIAL"]].sum(1)
SCHOOL["MATH ECON 4%"] = SCHOOL[["GRADE 3 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 4 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 5 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 6 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 7 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 8 MATH 4S - ECONOMICALLY DISADVANTAGED"]].sum(1)
SCHOOL["MATH ELL 4%"] = SCHOOL[["GRADE 3 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 4 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 5 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 6 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 7 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 8 MATH 4S - LIMITED ENGLISH PROFICIENT"]].sum(1)

# I also save the total number of Grade 8 students tested as a rough proxy for stduent enrollment
number_of_ela_grades = (SCHOOL[["GRADE 3 ELA - ALL STUDENTS TESTED", "GRADE 4 ELA - ALL STUDENTS TESTED", "GRADE 5 ELA - ALL STUDENTS TESTED", "GRADE 6 ELA - ALL STUDENTS TESTED", "GRADE 7 ELA - ALL STUDENTS TESTED", "GRADE 8 ELA - ALL STUDENTS TESTED"]]>1).sum(1)
number_of_math_grades = (SCHOOL[["GRADE 3 MATH - ALL STUDENTS TESTED", "GRADE 4 MATH - ALL STUDENTS TESTED", "GRADE 5 MATH - ALL STUDENTS TESTED", "GRADE 6 MATH - ALL STUDENTS TESTED", "GRADE 7 MATH - ALL STUDENTS TESTED", "GRADE 8 MATH - ALL STUDENTS TESTED"]]>1).sum(1)
SCHOOL = SCHOOL.drop(SCHOOL.iloc[:,23:143].columns, axis=1)

# Here they are converted from total number, to fraction of student-body
elaCC = ["ELA TESTED","ELA ALL 4%","ELA AAALN 4%","ELA BLACK 4%","ELA LATINO 4%","ELA ASIAN 4%",
    "ELA WHITE 4%","ELA MULTIRACIAL 4%","ELA ECON 4%","ELA ELL 4%"]
mathCC = ["MATH TESTED","MATH ALL 4%","MATH AAALN 4%","MATH BLACK 4%","MATH LATINO 4%",
        "MATH ASIAN 4%","MATH WHITE 4%","MATH MULTIRACIAL 4%","MATH ECON 4%","MATH ELL 4%"]
for col in mathCC[1:]:
    SCHOOL[col] = SCHOOL[col] / SCHOOL["MATH TESTED"].values
for col in elaCC[1:]:
    SCHOOL[col] = SCHOOL[col] / SCHOOL["ELA TESTED"].values

COMMON_CORE4S = pd.DataFrame(data=(SCHOOL[elaCC[1:]].values + SCHOOL[mathCC[1:]].values)/ 2.0, index=SCHOOL.index, columns=["%Students scored 4", "%AAALN and scored 4", "%BLACK and scored 4", "%LATINO and scored 4", "%ASIAN and scored 4", "%WHITE and scored 4", "%MULTIRACIAL and scored 4", "%ECON NEED and scored 4", "%ELL and scored 4"])
COMMON_CORE4S = COMMON_CORE4S.round(2)
SCHOOL = SCHOOL.drop(elaCC, axis=1)
SCHOOL = SCHOOL.drop(mathCC, axis=1)

# We would like to identify schools with these traits, that are either doing well or potentially could do well
demographics = ["ECONOMIC NEED INDEX", "PERCENT ELL", "PERCENT ASIAN",
"PERCENT BLACK","PERCENT HISPANIC","PERCENT BLACK / HISPANIC","PERCENT WHITE"]
SCHOOL_DEMOGRAPHICS = SCHOOL[demographics]
SCHOOL_DEMOGRAPHICS.fillna(SCHOOL_DEMOGRAPHICS.mean(), inplace=True) 
OTHER_SCHOOL_DATA = SCHOOL.drop(demographics, axis=1)
SHSAT = pd.read_csv("../input/nyc-shsat-test-results-2017/nytdf.csv", index_col="DBN")
SHSAT.columns = [x.upper() for x in SHSAT.columns]
SHSAT = SHSAT[["OFFERSPERSTUDENT"]]
SHSAT.columns = ["OffersPerStudent"]

# Join with school explorer on the DBN.  
SHSAT = SHSAT.join(OTHER_SCHOOL_DATA, how='inner').iloc[:,:1]

# convert from sting xx% to numeric 0.xx
SHSAT["OffersPerStudent"].fillna("0%", inplace=True)
SHSAT["OffersPerStudent"].replace("0", "0%", inplace=True)
SHSAT["OffersPerStudent"] = SHSAT["OffersPerStudent"].str[:-1].astype(float) / 100
SAFETY = pd.read_csv("../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv", index_col="DBN")
SAFETY = SAFETY.loc[SAFETY.index.dropna()]
SAFETY.columns = [x.upper() for x in SAFETY.columns]
EliteSchools = ["Brooklyn Latin School, The", "Brooklyn Technical High School", "Bronx High School of Science", "High School for Mathematics, Science and Engineeri", "High School of American Studies at Lehman College", "Queens High School for the Sciences at York Colleg", "Staten Island Technical High School", "Stuyvesant High School"]
ELITES = SAFETY.loc[SAFETY['LOCATION NAME'].isin(EliteSchools), ['LOCATION NAME','LATITUDE','LONGITUDE','REGISTER']]
ELITES['REGISTER'] = ELITES['REGISTER'].str.replace(',', '')
ELITES['REGISTER'] = ELITES['REGISTER'].astype(int)
ELITES = ELITES.groupby(ELITES['LOCATION NAME']).mean()
SAFETY = SAFETY[['MAJOR N','OTH N','NOCRIM N','PROP N','VIO N']]
SAFETY=SAFETY.groupby(['DBN']).mean()
SAFETY.columns = ["MAJOR CRIMES", "OTHER CRIMES", "NONCRIMINAL CRIMES", "PROPERTY CRIMES", "VIOLENT CRIMES"]
#This code lifted from last example on this stackoverflow:
#https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d
for school in EliteSchools:
    DISTANCE[school] = 0.0
    EliteTuple = (ELITES.loc[school, "LATITUDE"], ELITES.loc[school, "LONGITUDE"])
    for row in DISTANCE.index:
        MidTuple = (DISTANCE.loc[row, "LATITUDE"], DISTANCE.loc[row, "LONGITUDE"])
        DISTANCE.loc[row, school] = distance(MidTuple, EliteTuple)
DISTANCE = DISTANCE.round(1)
del DISTANCE['LATITUDE']
del DISTANCE['LONGITUDE']
DISTANCE['DISTANCE_AVERAGE_SPECIALIZED_SCHOOL'] = DISTANCE.mean(1)
DISTANCE['DISTANCE_NEAREST_SPECIALIZED_SCHOOL'] = DISTANCE.min(1)
DISTANCE = DISTANCE[['DISTANCE_NEAREST_SPECIALIZED_SCHOOL', 'DISTANCE_AVERAGE_SPECIALIZED_SCHOOL']]
ENROLLMENT = pd.read_csv("../input/ny-school-demographics-and-accountability-snapshot/2006-2012-school-demographics-and-accountability-snapshot.csv", index_col=["DBN"])
ENROLLMENT = ENROLLMENT.loc[ENROLLMENT["schoolyear"]==20112012, "grade7"]
ENROLLMENT.replace('    ', "0", inplace=True)
ENROLLMENT = ENROLLMENT.astype('int')
ENROLLMENT.name = "7thGradeEnrollment"
OTHER_SCHOOL_DATA = OTHER_SCHOOL_DATA.join(ENROLLMENT, how='left')
SCHOOL_NAMES.head()
SHSAT.head().style.background_gradient(cmap=cm)
SCHOOL_DEMOGRAPHICS.head().style.background_gradient(cmap=cm)
COMMON_CORE4S.head().style.background_gradient(cmap=cm)
SAFETY.head()
DISTANCE.head()
OTHER_SCHOOL_DATA.head().style.background_gradient(cmap=cm)
# First identify a target for the model to work on, offers per student.
target = "OffersPerStudent"
# Next assemble input data: common core 4 percentages, school demographics and other school information
model_input_variables = OTHER_SCHOOL_DATA.join(COMMON_CORE4S["%Students scored 4"]).join(SAFETY, how='left').join(DISTANCE, how='left')
model_input_variables.fillna(-.01, inplace=True) # fill in missing data with an arbitary value.
model_input_variables = SHSAT[[target]].join(model_input_variables, how='inner').iloc[:,1:]

RF = RandomForestRegressor(min_samples_leaf=10, n_jobs=8, n_estimators=100, random_state=0)
# A simple grid for parameter tuning
RF_params = {"max_depth": [3,6,None],
              "max_features": [0.33,0.67,1.0],
              "min_samples_leaf": [4,9,16]}
RF_GRID = GridSearchCV(RF, RF_params, n_jobs=2, cv=2)
RF_GRID.fit(model_input_variables, SHSAT[target])
RF = RF.set_params(**RF_GRID.best_params_)
RF.fit(model_input_variables, SHSAT[target])
# delete variables which are not used or almost unused to keep the model on the simpler side
model_input_variables = model_input_variables.loc[:, RF.feature_importances_>0.01]
RF.fit(model_input_variables, SHSAT[target])
# Save the model's predictions as a new variable
SHSAT["PREDICTED"] = RF.predict(model_input_variables)
SHSAT["PREDICTED"] = SHSAT["PREDICTED"].round(2)
importances = pd.Series(index=model_input_variables.columns, data=RF.feature_importances_).sort_values(ascending=True)
importances.plot(kind='barh', figsize=(11,7), color="orange");
SHSAT["UNDERPERFORM"] = SHSAT["PREDICTED"] - SHSAT[target]
# sort all of our datasets by UNDERPERFORM
SHSAT = SHSAT.sort_values("UNDERPERFORM", ascending=False)
SCHOOL_NAMES = SCHOOL_NAMES.loc[SHSAT.index]
SCHOOL_DEMOGRAPHICS = SCHOOL_DEMOGRAPHICS.loc[SHSAT.index]
COMMON_CORE4S = COMMON_CORE4S.loc[SHSAT.index]
OTHER_SCHOOL_DATA = OTHER_SCHOOL_DATA.loc[SHSAT.index]
SAFETY = SAFETY.loc[SHSAT.index]
DISTANCE = DISTANCE.loc[SHSAT.index]
model_input_variables = model_input_variables.loc[SHSAT.index]
# Calculate shapley explanations to display later
explainer = shap.TreeExplainer(RF)
shap_values = explainer.shap_values(model_input_variables)[:,:-1]
# recombine all the variables so we can use them all at once if want to.
all_variables = SCHOOL_NAMES.join(SHSAT, how="inner").join(SCHOOL_DEMOGRAPHICS, how="inner").join(COMMON_CORE4S, how="inner").join(OTHER_SCHOOL_DATA, how="inner").join(SAFETY, how="left").join(DISTANCE, how="left")

# Plot expected vs actual
sns.lmplot(x="PREDICTED", y=target, data=SHSAT, fit_reg=True, markers='.', size = 6,
           palette="coolwarm", );
#           scatter_kws={'s':OTHER_SCHOOL_DATA['APPROXIMATE_ENROLLMENT_PER_GRADE']}); 
SCHOOL_NAMES.join(SHSAT, how='inner').head().style.background_gradient(cmap=cm)
row = SCHOOL_NAMES.index.get_loc("03M859")
print(SCHOOL_NAMES.iloc[row])
index = model_input_variables.columns + " (" + model_input_variables.iloc[row].astype(str) + ")"
pd.Series(index=index, data=shap_values[row]).sort_values(ascending=True).plot(kind='barh', figsize=(11,7));
row = SCHOOL_NAMES.index.get_loc("17K590")
print(SCHOOL_NAMES.iloc[row])
index = model_input_variables.columns + " (" + model_input_variables.iloc[row].astype(str) + ")"
pd.Series(index=index, data=shap_values[row]).sort_values(ascending=True).plot(kind='barh', figsize=(11,7));
SCHOOL_NAMES.join(SCHOOL_DEMOGRAPHICS).loc[["03M859", "17K590"]].style.background_gradient(cmap=cm, axis=1)
# First define a function to conveniantly calculate attractiveness.
def calculate_ATTRACTIVENESS():
    attract = ECON_NEED_WEIGHT * all_variables["ECONOMIC NEED INDEX"] 
    attract = attract + ELL_WEIGHT * all_variables["PERCENT ELL"]
    attract = attract + ASIAN_WEIGHT * all_variables["PERCENT ASIAN"]
    attract = attract + BLACK_WEIGHT * all_variables["PERCENT BLACK"]
    attract = attract + WHITE_WEIGHT * all_variables["PERCENT WHITE"]
    attract = attract + HISPANIC_WEIGHT * all_variables["PERCENT HISPANIC"]
    attract = attract + NONFEEDER_WEIGHT*(all_variables[target].mean()-all_variables[target])
    attract = attract + AAALN_4_WEIGHT * all_variables["%AAALN and scored 4"]
    attract = attract + BLACK_4_WEIGHT * all_variables["%BLACK and scored 4"]
    attract = attract + LATINO_4_WEIGHT * all_variables["%LATINO and scored 4"]
    attract = attract + ASIAN_4_WEIGHT * all_variables["%ASIAN and scored 4"]
    attract = attract + WHITE_4_WEIGHT * all_variables["%WHITE and scored 4"]
    attract = attract + MULTIRACIAL_4_WEIGHT * all_variables["%MULTIRACIAL and scored 4"] 
    attract = attract + ECON_4_WEIGHT * all_variables["%ECON NEED and scored 4"]
    attract = attract + ELL_4_WEIGHT * all_variables["%ELL and scored 4"]
    attract = attract / (ECON_NEED_WEIGHT + ELL_WEIGHT + ASIAN_WEIGHT + BLACK_WEIGHT + WHITE_WEIGHT + HISPANIC_WEIGHT + NONFEEDER_WEIGHT  + AAALN_4_WEIGHT + BLACK_4_WEIGHT + LATINO_4_WEIGHT + ASIAN_4_WEIGHT + WHITE_4_WEIGHT + MULTIRACIAL_4_WEIGHT + ECON_4_WEIGHT + ELL_4_WEIGHT)
    attract = attract.clip(lower=0.0)
    return attract
# First, we can assign ATTRACTIVENESS weightings to overall school demographics
ECON_NEED_WEIGHT = 1.0 # How much to weight school's Economic Need index
ELL_WEIGHT = 0.5       # How much to weight school's ELL student percentage
ASIAN_WEIGHT = 0.0     # How much to weight school's Asian student percentage
BLACK_WEIGHT = 1.0     # How much to weight school's Black student percentage
WHITE_WEIGHT = 0.0     # How much to weight school's White student percentage
HISPANIC_WEIGHT = 1.0  # How much to weight school's Hispanic student percentage
NONFEEDER_WEIGHT = 1.0 # How much to weight % of students who do not receive SHSAT offers

# We can put extra-empahsis on target groups who are already performing well on common core. 
AAALN_4_WEIGHT = 1.0       # How much to weight school's AAALN students with 4s percentage
BLACK_4_WEIGHT = 2.0       # How much to weight school's Black students with 4s percentage
LATINO_4_WEIGHT = 2.0      # How much to weight school's Latino students with 4s percentage
ASIAN_4_WEIGHT = 0.0       # How much to weight school's Asian students with 4s percentage
WHITE_4_WEIGHT = 0.0       # How much to weight school's White students with 4s percentage
MULTIRACIAL_4_WEIGHT = 1.0 # How much to weight school's Multiracial students with 4s percentage
ECON_4_WEIGHT = 2.0        # How much to weight school's Econ. disadvantaged students with 4s percentage
ELL_4_WEIGHT = 0.5         # How much to weight school's ESL students with 4s percentage
# Now create the ATTRACTIVENESS score
all_variables["ATTRACTIVENESS"] = calculate_ATTRACTIVENESS()
sns.lmplot(x="PREDICTED", y=target, data=all_variables, 
           markers='.', size = 6, fit_reg=False,
           hue="ATTRACTIVENESS",
           palette="coolwarm_r",
           legend=False,);
# First, we can assign ATTRACTIVENESS weightings to overall school demographics
ECON_NEED_WEIGHT = 1.0 # How much to weight school's Economic Need index
ELL_WEIGHT = 0.0       # How much to weight school's ELL student percentage
ASIAN_WEIGHT = 0.0     # How much to weight school's Asian student percentage
BLACK_WEIGHT = 0.0     # How much to weight school's Black student percentage
WHITE_WEIGHT = 0.0     # How much to weight school's White student percentage
HISPANIC_WEIGHT = 0.0  # How much to weight school's Hispanic student percentage
NONFEEDER_WEIGHT = 0.0 # How much to weight percentage of students who do not receive SHSAT offeres

# We can get extra-empahsis to target groups who are already performing well on common core. 
AAALN_4_WEIGHT = 0.0       # How much to weight school's AAALN students with 4s percentage
BLACK_4_WEIGHT = 0.0       # How much to weight school's Black students with 4s percentage
LATINO_4_WEIGHT = 0.0      # How much to weight school's Latino students with 4s percentage
ASIAN_4_WEIGHT = 0.0       # How much to weight school's Asian students with 4s percentage
WHITE_4_WEIGHT = 0.0       # How much to weight school's White students with 4s percentage
MULTIRACIAL_4_WEIGHT = 0.0 # How much to weight school's Multiracial students with 4s percentage
ECON_4_WEIGHT = 1.0        # How much to weight school's Economically disadvantaged students with 4s percentage
ELL_4_WEIGHT = 0.0         # How much to weight school's ESL students with 4s percentage

# recalculate the ATTRACTIVENESS score
all_variables["ATTRACTIVENESS"] = calculate_ATTRACTIVENESS()

# Plot the new ATTRACTIVENESS
sns.lmplot(x="PREDICTED", y=target, data=all_variables, 
           markers='.', size = 6, fit_reg=False,
           hue="ATTRACTIVENESS",
           palette="coolwarm_r",
           legend=False,); 
# First, we can assign ATTRACTIVENESS weightings to overall school demographics
ECON_NEED_WEIGHT = 0.0 # How much to weight school's Economic Need index
ELL_WEIGHT = 0.0       # How much to weight school's ELL student percentage
ASIAN_WEIGHT = 0.0     # How much to weight school's Asian student percentage
BLACK_WEIGHT = 0.0     # How much to weight school's Black student percentage
WHITE_WEIGHT = 0.0     # How much to weight school's White student percentage
HISPANIC_WEIGHT = 0.0  # How much to weight school's Hispanic student percentage
NONFEEDER_WEIGHT = 2.0 # How much to weight % of students who do not receive SHSAT offeres

# We can get extra-empahsis to target groups who are already performing well on common core. 
AAALN_4_WEIGHT = 1.0       # How much to weight school's AAALN students with 4s percentage
BLACK_4_WEIGHT = 1.0       # How much to weight school's Black students with 4s percentage
LATINO_4_WEIGHT = 1.0      # How much to weight school's Latino students with 4s percentage
ASIAN_4_WEIGHT = 0.0       # How much to weight school's Asian students with 4s percentage
WHITE_4_WEIGHT = 0.0       # How much to weight school's White students with 4s percentage
MULTIRACIAL_4_WEIGHT = 0.0 # How much to weight school's Multiracial students with 4s percentage
ECON_4_WEIGHT = 1.0        # How much to weight school's Econ. disadvantaged students with 4s percentage
ELL_4_WEIGHT = 1.0         # How much to weight school's ESL students with 4s percentage

# Now create the ATTRACTIVENESS score
all_variables["ATTRACTIVENESS"] = calculate_ATTRACTIVENESS()

# Plot the new ATTRACTIVENESS
sns.lmplot(x="PREDICTED", y=target, data=all_variables, 
           markers='.', size = 6, fit_reg=False,
           hue="ATTRACTIVENESS",
           palette="coolwarm_r",
           legend=False,);
recommended_schools = all_variables[["SCHOOL NAME", "ADDRESS (FULL)",target,"PREDICTED", "UNDERPERFORM", "ATTRACTIVENESS"]+list(SCHOOL_DEMOGRAPHICS.columns)+list(COMMON_CORE4S.columns[1:] )]
recommended_schools["ATTRACTIVENESS_UNDERPERFORM_COMBINED"] = recommended_schools["ATTRACTIVENESS"].rank() + recommended_schools["UNDERPERFORM"].rank()
recommended_schools["ATTRACTIVENESS"] = recommended_schools["ATTRACTIVENESS"].round(2)
recommended_schools = recommended_schools.sort_values(["ATTRACTIVENESS_UNDERPERFORM_COMBINED"], ascending=False)
del recommended_schools["ATTRACTIVENESS_UNDERPERFORM_COMBINED"]
recommended_schools.to_csv("BenS_PASSNYC_Recommendations.csv")
recommended_schools.head(30).style.background_gradient(cmap=cm)