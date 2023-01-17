import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# creating the dataframes to work with



counties = pd.read_csv('../input/covid19-preexisting-conditions/abridged_couties.csv')

confirmed = pd.read_csv('../input/covid19-preexisting-conditions/time_series_covid19_confirmed_US.csv')

deaths = pd.read_csv('../input/covid19-preexisting-conditions/time_series_covid19_deaths_US.csv')

states_april = pd.read_csv('../input/covid19/4.18states.csv')
#cleaning data sets

states_april = states_april[['Province_State', 'Confirmed', 'Deaths', 'Active','People_Tested']]

states_april = states_april.fillna(0)

states_april.head()
counties = counties[['CountyName', 'State', 'PopulationEstimate2018', 'DiabetesPercentage', 'HeartDiseaseMortality', 'StrokeMortality', 'Smokers_Percentage']]

counties = counties[counties["State"].isnull() == False]

counties = counties.fillna(0)

counties.head()
#Cleaning confirmed dataframe and grouping by state



confirmed = confirmed.dropna()

confirmed = confirmed.groupby(by="Province_State", as_index=False).mean()

confirmed = confirmed.drop(columns={"UID", "code3", "FIPS", "Lat", "Long_"})

confirmed.head()
#Cleaning deaths dataframe and grouping by state



deaths = deaths.dropna()

deaths = deaths.groupby(by="Province_State", as_index=False).mean()

deaths = deaths.drop(columns={"UID", "code3", "FIPS", "Lat", "Long_"})

deaths.head()
#Finding the death rates and positive rates for each state in the US



explore = counties[["CountyName", "State"]]

explore = explore.dropna()

explore = explore.groupby(by="State", as_index=False).count()

explore = explore.merge(states_april, left_on='State', right_on='Province_State')

explore = explore[["State", "Confirmed", "Deaths", "People_Tested"]]

explore["Death_Rate"] = (explore["Deaths"] / explore["Confirmed"]) * 100

explore["Positive_Rate"] = (explore["Confirmed"] / explore["People_Tested"]) * 100

explore = explore.drop(columns=["Confirmed", "Deaths", "People_Tested"])

explore.head()
#Let's visualize the death and positive rates according to states



import plotly.express as px



fig = px.scatter(explore, x="Death_Rate", y="Positive_Rate", text=explore["State"])



fig.update_traces(textposition='top center')



fig.update_layout(height=800, title_text='US States and Death/Positive Rates')



fig.show()
highest_death_rate = max(explore["Death_Rate"])

highest_death_rate_state = explore["State"][explore["State"].index == np.argmax(explore["Death_Rate"])].to_list()

print("The highest Death Rate is", highest_death_rate, "in the state of", *highest_death_rate_state)
lowest_death_rate = min(explore["Death_Rate"])

lowest_death_rate_state = explore["State"][explore["State"].index == np.argmin(explore["Death_Rate"])].to_list()

print("The lowest Death Rate is", lowest_death_rate, "in the state of", *lowest_death_rate_state)
highest_positive_rate = max(explore["Positive_Rate"])

highest_positive_rate_state = explore["State"][explore["State"].index == np.argmax(explore["Positive_Rate"])].to_list()

print("The highest Positive Test Rate is", highest_positive_rate, "in the state of", *highest_positive_rate_state)
lowest_positive_rate = min(explore["Positive_Rate"])

lowest_positive_rate_state = explore["State"][explore["State"].index == np.argmin(explore["Positive_Rate"])].to_list()

print("The lowest Positive Test Rate is", lowest_positive_rate, "in the state of", *lowest_positive_rate_state)
impact_smoking = counties[["CountyName", "State", "Smokers_Percentage"]]

impact_smoking = impact_smoking.dropna()

impact_smoking = impact_smoking[impact_smoking["Smokers_Percentage"] > 0]

impact_smoking.sort_values(by=["Smokers_Percentage"], ascending=False).head()
#We are grouping by state because the county data is not available in the other spreadsheet



impact_smoking = impact_smoking.groupby(by="State", as_index=False).mean()

impact_smoking.sort_values(by=["Smokers_Percentage"], ascending=False).head()
impact_smoking = impact_smoking.merge(states_april, left_on='State', right_on='Province_State')

impact_smoking = impact_smoking[["State", "Smokers_Percentage", "Confirmed", "Deaths", "People_Tested"]]

impact_smoking["Death_Rate"] = (impact_smoking["Deaths"] / impact_smoking["Confirmed"]) * 100

impact_smoking["Positive_Rate"] = (impact_smoking["Confirmed"] / impact_smoking["People_Tested"]) * 100

impact_smoking = impact_smoking.drop(columns=["Confirmed", "Deaths", "People_Tested"])

impact_smoking.sort_values(by=["Smokers_Percentage"], ascending=False).head()
#Finding correlation between state's smoking percentage vs their death rate



smoking_correlation_death = impact_smoking["Smokers_Percentage"].corr(impact_smoking["Death_Rate"])

print("The correlation is", smoking_correlation_death)
plt.scatter(impact_smoking["Smokers_Percentage"], impact_smoking["Death_Rate"])

plt.xlabel('Smokers Percentage')

plt.ylabel('Death Rate')

plt.show()
#Finding correlation between state's smoking percentage vs their positive rate



smoking_correlation_positve = impact_smoking["Smokers_Percentage"].corr(impact_smoking["Positive_Rate"])

print("The correlation is", smoking_correlation_positve)
plt.scatter(impact_smoking["Smokers_Percentage"], impact_smoking["Positive_Rate"])

plt.xlabel('Smokers Percentage')

plt.ylabel('Positive Rate')

plt.show()
impact_diabetes = counties[["CountyName", "State", "DiabetesPercentage"]]

impact_diabetes = impact_diabetes.dropna()

impact_diabetes = impact_diabetes[impact_diabetes["DiabetesPercentage"] > 0]

impact_diabetes.sort_values(by=["DiabetesPercentage"], ascending=False).head()
#We are grouping by state because the county data is not available in the other spreadsheet

#Observe the sharp decrease in % when doing this; Tippah County of MS is 33% but the entire state averages to be <15%



impact_diabetes = impact_diabetes.groupby(by="State", as_index=False).mean()

impact_diabetes.sort_values(by=["DiabetesPercentage"], ascending=False).head()
impact_diabetes = impact_diabetes.merge(states_april, left_on='State', right_on='Province_State')

impact_diabetes = impact_diabetes[["State", "DiabetesPercentage", "Confirmed", "Deaths", "People_Tested"]]

impact_diabetes["Death_Rate"] = (impact_diabetes["Deaths"] / impact_diabetes["Confirmed"]) * 100

impact_diabetes["Positive_Rate"] = (impact_diabetes["Confirmed"] / impact_diabetes["People_Tested"]) * 100

impact_diabetes = impact_diabetes.drop(columns=["Confirmed", "Deaths", "People_Tested"])

impact_diabetes.sort_values(by=["DiabetesPercentage"], ascending=False).head()
#Finding correlation between state's diabetes percentage vs their death rate



diabetes_correlation_death = impact_diabetes["DiabetesPercentage"].corr(impact_diabetes["Death_Rate"])

print("The correlation is:", diabetes_correlation_death)
plt.scatter(impact_diabetes["DiabetesPercentage"], impact_diabetes["Death_Rate"])

plt.xlabel('Diabetes Percentage')

plt.ylabel('Death Rate')

plt.show()
#Finding correlation between state's diabetes percentage vs their positive rate



smoking_correlation_positve = impact_diabetes["DiabetesPercentage"].corr(impact_diabetes["Positive_Rate"])

print("The correlation is:", smoking_correlation_positve)
plt.scatter(impact_diabetes["DiabetesPercentage"], impact_diabetes["Positive_Rate"])

plt.xlabel('Diabetes Percentage')

plt.ylabel('Positive Rate')

plt.show()
impact_heart = counties[["CountyName", "State", "HeartDiseaseMortality"]]

impact_heart = impact_heart.dropna()

impact_heart = impact_heart[impact_heart["HeartDiseaseMortality"] > 0]

impact_heart.sort_values(by=["HeartDiseaseMortality"], ascending=False).head()
#We are grouping by state because the county data is not available in the other spreadsheet



impact_heart = impact_heart.groupby(by="State", as_index=False).mean()

impact_heart.sort_values(by=["HeartDiseaseMortality"], ascending=False).head()
impact_heart = impact_heart.merge(states_april, left_on='State', right_on='Province_State')

impact_heart = impact_heart[["State", "HeartDiseaseMortality", "Confirmed", "Deaths", "People_Tested"]]

impact_heart["Death_Rate"] = (impact_heart["Deaths"] / impact_heart["Confirmed"]) * 100

impact_heart["Positive_Rate"] = (impact_heart["Confirmed"] / impact_heart["People_Tested"]) * 100

impact_heart = impact_heart.drop(columns=["Confirmed", "Deaths", "People_Tested"])

impact_heart.sort_values(by=["HeartDiseaseMortality"], ascending=False).head()
#Finding correlation between state's heart disease mortality vs their death rate



heart_correlation_death = impact_heart["HeartDiseaseMortality"].corr(impact_heart["Death_Rate"])

print("The correlation is:", heart_correlation_death)
plt.scatter(impact_heart["HeartDiseaseMortality"], impact_heart["Death_Rate"])

plt.xlabel('Heart Disease Mortality')

plt.ylabel('Death Rate')

plt.show()
#Finding correlation between state's heart rate mortality vs their positive rate



heart_correlation_positve = impact_heart["HeartDiseaseMortality"].corr(impact_heart["Positive_Rate"])

print("The correlation is:", heart_correlation_positve)
plt.scatter(impact_heart["HeartDiseaseMortality"], impact_heart["Positive_Rate"])

plt.xlabel('Heart Disease Mortality')

plt.ylabel('Positive Rate')

plt.show()
impact_stroke = counties[["CountyName", "State", "StrokeMortality"]]

impact_stroke = impact_stroke.dropna()

impact_stroke = impact_stroke[impact_stroke["StrokeMortality"] > 0]

impact_stroke.sort_values(by=["StrokeMortality"], ascending=False).head()
#We are grouping by state because the county data is not available in the other spreadsheet



impact_stroke = impact_stroke.groupby(by="State", as_index=False).mean()

impact_stroke.sort_values(by=["StrokeMortality"], ascending=False).head()
impact_stroke = impact_stroke.merge(states_april, left_on='State', right_on='Province_State')

impact_stroke = impact_stroke[["State", "StrokeMortality", "Confirmed", "Deaths", "People_Tested"]]

impact_stroke["Death_Rate"] = (impact_stroke["Deaths"] / impact_stroke["Confirmed"]) * 100

impact_stroke["Positive_Rate"] = (impact_stroke["Confirmed"] / impact_stroke["People_Tested"]) * 100

impact_stroke = impact_stroke.drop(columns=["Confirmed", "Deaths", "People_Tested"])

impact_stroke.sort_values(by=["StrokeMortality"], ascending=False).head()
#Finding correlation between state's stroke mortality vs their death rate



stroke_correlation_death = impact_stroke["StrokeMortality"].corr(impact_stroke["Death_Rate"])

print("The correlation is:", stroke_correlation_death)
plt.scatter(impact_stroke["StrokeMortality"], impact_stroke["Death_Rate"])

plt.xlabel('Stroke Mortality')

plt.ylabel('Death Rate')

plt.show()
#Finding correlation between state's stroke mortality vs their positive test rate



stroke_correlation_positve = impact_stroke["StrokeMortality"].corr(impact_stroke["Positive_Rate"])

print("The correlation is:", stroke_correlation_positve)
plt.scatter(impact_stroke["StrokeMortality"], impact_stroke["Positive_Rate"])

plt.xlabel('Stroke Mortality')

plt.ylabel('Positive Rate')

plt.show()
impact_stroke_smoking = impact_stroke.merge(impact_smoking, on='State')[['State', 'StrokeMortality', 'Smokers_Percentage', 'Positive_Rate_x']]

impact_stroke_smoking.head()
X_positive = impact_stroke_smoking[['StrokeMortality', 'Smokers_Percentage']]

X_positive.head()
Y_positive = np.array(impact_stroke_smoking['Positive_Rate_x'])

Y_positive
from sklearn.model_selection import train_test_split



np.random.seed(41)

X_positive_train, X_positive_test, Y_positive_train, Y_positive_test = train_test_split(X_positive, Y_positive, test_size = 0.10)
from sklearn import linear_model as lm



linear_model_positive = lm.LinearRegression(fit_intercept=True)

linear_model_positive.fit(X_positive_train, Y_positive_train)
linear_model_positive.predict(X_positive_train)
def rmse(predicted, actual):

    return np.sqrt(np.mean((actual - predicted)**2))
rmse(linear_model_positive.predict(X_positive_train), Y_positive_train)
import seaborn as sns



#We would hope to see a horizontal line of points at 0.



residuals = Y_positive_test - linear_model_positive.predict(X_positive_test)

ax = sns.regplot(Y_positive_test, residuals)

ax.set_xlabel('Death Rate (Test Data)')

ax.set_ylabel('Residuals (Actual Rate - Predicted Rate)')

ax.set_title("Residuals vs. Death Rate on Test Data");
impact_stroke_heart = impact_stroke.merge(impact_heart, on='State')[['State', 'StrokeMortality', 'HeartDiseaseMortality', 'Death_Rate_x']]

impact_stroke_heart.head()
X_death = impact_stroke_heart[['StrokeMortality', 'HeartDiseaseMortality']]

X_death.head()
Y_death = np.array(impact_stroke_heart['Death_Rate_x'])

Y_death
np.random.seed(41)

X_death_train, X_death_test, Y_death_train, Y_death_test = train_test_split(X_death, Y_death, test_size = 0.10)
X_death.head()
def normalize(data):



    output = (data - np.mean(data))/np.std(data)

    return output.replace(np.nan, 0)

    

X_normalized = normalize(X_death_train)

X_normalized.head()
linear_model_death = lm.LinearRegression(fit_intercept=True)

linear_model_death.fit(normalize(X_death_train), Y_death_train)
linear_model_death.predict(normalize(X_death_train))
rmse(linear_model_death.predict(normalize(X_death_train)), Y_death_train)
#Ideally, we would see a horizontal line of points at 0.



residuals = Y_death_test - linear_model_death.predict(normalize(X_death_test))

ax = sns.regplot(Y_death_test, residuals)

ax.set_xlabel('Death Rate (Test Data)')

ax.set_ylabel('Residuals (Actual Rate - Predicted Rate)')

ax.set_title("Residuals vs. Death Rate on Test Data");
Positive_Rate_Test_RMSE = rmse(linear_model_positive.predict(X_positive_test), Y_positive_test)

Death_Rate_Test_RMSE = rmse(linear_model_death.predict(normalize(X_death_test)), Y_death_test)



print("The Positive Rate Test RMSE is {}".format(Positive_Rate_Test_RMSE))

print("The Death Rate Test RMSE is {}".format(Death_Rate_Test_RMSE))
#Best possible value is 1.0 as r^2 measures goodness of fit



from sklearn.metrics import r2_score



Positive_Rate_Test_R2 = r2_score(Y_positive_test, linear_model_positive.predict(X_positive_test))

Death_Rate_Test_R2 = r2_score(Y_death_test, linear_model_death.predict(normalize(X_death_test)))



print("The Positive Rate Test R^2 is {}".format(Positive_Rate_Test_R2))

print("The Death Rate Test R^2 is {}".format(Death_Rate_Test_R2))