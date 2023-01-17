import pandas as pd
from IPython.display import display
pd.options.display.max_columns = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pylab as pl
import scikitplot as skplt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import plotly.graph_objs as go
import plotly.offline as py
%matplotlib inline
import os
from PIL import  Image
import io
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.figure_factory as ff
import matplotlib.ticker as mtick
import scipy
tcc = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
tcc.head()
senior = (tcc['Churn'].value_counts()*100.0 /len(tcc)).plot(kind='pie',\
        labels = ['No', 'Yes'], figsize = (7,7) , colors = ['yellow','blue'])

senior.set_title('Churn rate')
senior.legend(labels=['No','Yes']);
gb = tcc.groupby("gender")["Churn"].value_counts().to_frame().rename({"Churn": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "gender", y = "Number of Customers", data = gb, hue = "Churn", palette = sns.color_palette("hls", 8)).set_title("Gender and relative Churn Rates in our population");
senior = (tcc['SeniorCitizen'].value_counts()*100.0 /len(tcc)).plot(kind='pie',\
        labels = ['No', 'Yes'], figsize = (7,7) , colors = ['yellow','green'], fontsize = 15)

senior.set_title('Seniors')
senior.legend(labels=['Non Senior','Senior']);
gb = tcc.groupby("SeniorCitizen")["Churn"].value_counts().to_frame().rename({"Churn": "Number of Customers"}, axis = 1).reset_index()
gb.replace([0, 1], ["Young", "Senior"], inplace = True)
gb
tp = gb.groupby("SeniorCitizen")["Number of Customers"].sum().to_frame().reset_index().rename({"Number of Customers": "# Customers in Age Group"}, axis = 1)
gb = pd.merge(gb, tp, on = "SeniorCitizen")
gb["Churn Rate in Age Group"] = gb["Number of Customers"]/gb["# Customers in Age Group"]
gb = gb[gb.Churn == "Yes"]

sns.barplot(x = "SeniorCitizen", y = "Churn Rate in Age Group", data = gb).set_title("Churn Rate for Young and Senior customers");
phone = (tcc['PhoneService'].value_counts()*100.0 /len(tcc)).plot(kind='bar', stacked = True,\
                                                rot = 0, color = ['red','lightblue'])
  
phone.yaxis.set_major_formatter(mtick.PercentFormatter())
phone.set_ylabel('Customers')
phone.set_xlabel('Phone Service')
phone.set_ylabel('Customers')
phone.set_title('Phone service distribution');
internet = (tcc['InternetService'].value_counts()*100.0 /len(tcc)).plot(kind='pie',\
        labels = ['Fiber optic', 'DSL', 'No'], figsize = (7,7) , colors = ['orange','purple', 'black'], fontsize = 15)

senior.set_title('Seniors')
senior.legend(labels=['Non Senior','Senior']);
plt.hist(tcc.tenure)
plt.xlabel('tenure')
plt.title("Tenure Distribution");
contract = (tcc['Contract'].value_counts()*100.0 /len(tcc)).plot(kind='bar', stacked = True,\
                                                rot = 0, color = ['orange','blue','magenta'])
  
contract.yaxis.set_major_formatter(mtick.PercentFormatter())
contract.set_ylabel('Customers')
contract.set_xlabel('Contract')
contract.set_ylabel('Customers')
contract.set_title('Contract distribution');
tcc.columns
missing_values = []
for col in tcc.columns:
    missing_values.append(tcc[col].isna().any())

missing_values = pd.DataFrame(np.array(missing_values).reshape(1, 21))
missing_values.columns = tcc.columns
missing_values_table   = tcc.append(missing_values).tail(1)
missing_values_table   = missing_values_table.astype(bool)
missing_values_table   = missing_values_table.transpose()
missing_values_table.columns = ["Missing?"]

missing_values_table["dtype"] = tcc.dtypes
missing_values_table
try:
    tcc.TotalCharges.astype("float64")
except ValueError:
    print("We can't convert this column to floats, there must be some non-convertible values")
print(tcc.TotalCharges.value_counts().head())
print("")
print("We have 11 observations that take an empty string value. Let's drop that. The string we want to drop is:")
tcc.TotalCharges.value_counts().index[1]
tcc.drop(tcc[tcc.TotalCharges == " "].index, axis = 0, inplace = True)
tcc.reset_index(drop = True, inplace = True)
tcc.TotalCharges = tcc.TotalCharges.astype("float64")
for col in tcc.columns:
    print("{0}: {1}".format(col, tcc.loc[:, col].unique()))
fig, axis = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 10))

gb = tcc.groupby("InternetService")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "InternetService", y = "% of customers", data = gb, hue = "Churn", ax = axis[0][0]).set_title("Internet Service and Churn");

gb = tcc.groupby("PhoneService")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "PhoneService", y = "% of customers", data = gb, hue = "Churn", ax = axis[0][1]).set_title("Phone Service and Churn");

gb = tcc.groupby("MultipleLines")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "MultipleLines", y = "% of customers", data = gb, hue = "Churn", ax = axis[0][2]).set_title("Multiple Lines Phone Option and Churn");

gb = tcc.groupby("Contract")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "Contract", y = "% of customers", data = gb, hue = "Churn", ax = axis[1][1]).set_title("Contract Type and Churn");
gb = tcc.groupby("SeniorCitizen")["InternetService"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"InternetService": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "SeniorCitizen", y = "% of customers", data = gb, hue = "InternetService").set_title("Age Group and Internet Connection");
fig, axis = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 10))

gb = tcc.groupby("OnlineSecurity")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "OnlineSecurity", y = "% of customers", data = gb, hue = "Churn", ax = axis[0][0]).set_title("OnlineSecurity Internet Service and Churn")

gb = tcc.groupby("OnlineBackup")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "OnlineBackup", y = "% of customers", data = gb, hue = "Churn", ax = axis[0][1]).set_title("OnlineBackup Internet Service and Churn")

gb = tcc.groupby("DeviceProtection")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "DeviceProtection", y = "% of customers", data = gb, hue = "Churn", ax = axis[0][2]).set_title("DeviceProtection Internet Service and Churn")

gb = tcc.groupby("TechSupport")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "TechSupport", y = "% of customers", data = gb, hue = "Churn", ax = axis[1][0]).set_title("TechSupport Internet Service and Churn")

gb = tcc.groupby("StreamingTV")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "StreamingTV", y = "% of customers", data = gb, hue = "Churn", ax = axis[1][1]).set_title("StreamingTV Internet Service and Churn")

gb = tcc.groupby("StreamingMovies")["Churn"].value_counts()/len(tcc)
gb = gb.to_frame().rename({"Churn": "% of customers"}, axis = 1).reset_index()
sns.barplot(x = "StreamingMovies", y = "% of customers", data = gb, hue = "Churn", ax = axis[1][2]).set_title("StreamingMovies Internet Service and Churn");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0]).groupby("tenure")["OnlineSecurity"].sum().to_frame().reset_index()
sns.lmplot("tenure", "OnlineSecurity", data = gb, line_kws={'color': 'red'}, lowess = True);
ax = plt.gca()
ax.set_title("Number of OnlineSecurity subscribers per Tenure level");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0]).groupby("tenure")["OnlineBackup"].sum().to_frame().reset_index()
sns.lmplot("tenure", "OnlineBackup", data = gb, line_kws={'color': 'red'}, lowess = True)
ax = plt.gca()
ax.set_title("Number of OnlineBackup subscribers per Tenure level");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0]).groupby("tenure")["DeviceProtection"].sum().to_frame().reset_index()
sns.lmplot("tenure", "DeviceProtection", data = gb, line_kws={'color': 'red'}, lowess = True)
ax = plt.gca()
ax.set_title("Number of DeviceProtection subscribers per Tenure level");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0]).groupby("tenure")["TechSupport"].sum().to_frame().reset_index()
sns.lmplot("tenure", "TechSupport", data = gb, line_kws={'color': 'red'}, lowess = True)
ax = plt.gca()
ax.set_title("Number of TechSupport subscribers per Tenure level");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0]).groupby("tenure")["StreamingTV"].sum().to_frame().reset_index()
sns.lmplot("tenure", "StreamingTV", data = gb, line_kws={'color': 'red'}, lowess = True)
ax = plt.gca()
ax.set_title("Number of StreamingTV subscribers per Tenure level");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0]).groupby("tenure")["StreamingMovies"].sum().to_frame().reset_index()
sns.lmplot("tenure", "StreamingMovies", data = gb, line_kws={'color': 'red'}, lowess = True)
ax = plt.gca()
ax.set_title("Number of StreamingMovies subscribers per Tenure level");
gb = tcc[(tcc.OnlineSecurity != "No internet service")].replace(["Yes", "No"], [1, 0])
gb["AllServices"] = gb.OnlineSecurity*gb.OnlineBackup*gb.DeviceProtection*gb.TechSupport*gb.StreamingTV*gb.StreamingMovies
sns.lmplot("tenure", "AllServices", data = gb, line_kws={'color': 'red'}, lowess = True);
ax = plt.gca()
ax.set_title("Percentage of subscribers to all services per Tenure level");
tvc = gb.tenure.value_counts()
i = []
v = []
for tenure in tvc.index:
    i.append(tenure)
    v.append(len(gb[(gb.tenure == tenure) & (gb.AllServices == 1)])/len(gb[gb.tenure == tenure]))
df = pd.DataFrame(data = v, index = i, columns = ["%AllServices"]).reset_index().sort_values("index").reset_index(drop = True).rename({"index": "tenure"}, axis = 1)
sns.lmplot("tenure", "%AllServices", data = df, line_kws={'color': 'red'}, lowess = True)
ax = plt.gca()
ax.set_title("Percentage of Customers with all Additional Services Active per Tenure level");
plt.plot(df.tenure, df["%AllServices"]);
ax = plt.gca()
ax.set_title("Trend in percentage of customers subscribed to all services for each tenure level");
tcc = pd.get_dummies(tcc.iloc[:, 1 :])
tcc.head()
tcc.dtypes
for col in tcc.columns:
    print("{0}: {1}".format(col, tcc.loc[:, col].unique()))
tcc.head()
sns.boxplot(x = tcc.MonthlyCharges).set_title("Monthly Charges Box Plot");
sns.distplot(tcc.MonthlyCharges).set_title("Monthly Charges Distribution");
sns.boxplot(x = tcc.TotalCharges).set_title("Total Charges Box Plot");
sns.distplot(tcc.TotalCharges).set_title("Total Charges Distribution");
tcc.columns
sns.lmplot("TotalCharges", "Churn_Yes", data = tcc, line_kws={'color': 'red'}, lowess = True, size = 5)
plt.suptitle("Chances of Churn relative to Total Charges");
sns.lmplot("MonthlyCharges", "Churn_Yes", data = tcc, line_kws={'color': 'red'}, lowess = True, size = 5)
plt.suptitle("Chances of Churn relative to Monthly Charges");
sns.lmplot("tenure", "Churn_Yes", data = tcc, line_kws={'color': 'red'}, lowess = True, size = 5)
plt.suptitle("Chances of Churn relative to Tenure");
tcc.columns
# Pearson correlation coefficient
print("Coefficient:",scipy.stats.pearsonr(tcc["MonthlyCharges"], tcc["TotalCharges"])[0])
print("p-value:",scipy.stats.pearsonr(tcc["MonthlyCharges"], tcc["TotalCharges"])[1])
tcc.head()
# Intercept
tcc["intercept"] = 1.0

variables = tcc.copy()[['SeniorCitizen', 'tenure', 'MonthlyCharges', 
       'gender_Female', 'Partner_Yes', 'PhoneService_Yes',
        'Dependents_Yes', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'intercept']]

# Setting the model
logistical_regression = sm.Logit(tcc["Churn_Yes"], variables)

# Fitting the model
fitted_model = logistical_regression.fit()
fitted_model.summary2()
# Transforming PaymentMethod
tcc["PaymentMethod_Automatic"] = tcc["PaymentMethod_Bank transfer (automatic)"] + tcc["PaymentMethod_Credit card (automatic)"]
variables = tcc[['SeniorCitizen', 'tenure', 'MonthlyCharges', 
       'gender_Female', 'Partner_Yes',
        'Dependents_Yes', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Automatic', 'intercept']]

# Setting the model
logistical_regression = sm.Logit(tcc["Churn_Yes"], variables)

# Fitting the model
fitted_model = logistical_regression.fit()
fitted_model.summary2()
variables = tcc[['SeniorCitizen', 'tenure', 'MonthlyCharges', 
        'Dependents_Yes', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Automatic', 'intercept']]

# Setting the model
logistical_regression = sm.Logit(tcc["Churn_Yes"], variables)

# Fitting the model
fitted_model = logistical_regression.fit()
fitted_model.summary2()
tcc["tenure"].describe()
tcc["tenure_0:18"]  = 0
tcc["tenure_19:36"] = 0
tcc["tenure_37:54"] = 0
tcc["tenure_55:72"] = 0

tcc.loc[tcc.tenure <= 18, "tenure_0:18"] = 1
tcc.loc[((tcc.tenure >= 19) & (tcc.tenure <= 36)), "tenure_19:36"] = 1
tcc.loc[((tcc.tenure >= 37) & (tcc.tenure <= 54)), "tenure_37:54"] = 1
tcc.loc[tcc.tenure >= 55, "tenure_55:72"] = 1
tcc.head()
tcc.columns
variables = tcc[['SeniorCitizen', 'MonthlyCharges', 
        'Dependents_Yes', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'TechSupport_Yes', 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Automatic', 'tenure_19:36',
       'tenure_37:54', 'tenure_55:72','intercept']]

# Setting the model
logistical_regression = sm.Logit(tcc["Churn_Yes"], variables)

# Fitting the model
fitted_model = logistical_regression.fit()
fitted_model.summary2()
vif = pd.DataFrame()
vif["Variables"]  = variables.columns[0:-1]
vif["VIF Factor"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1]-1)]
vif
variables = tcc[['SeniorCitizen',
        'Dependents_Yes', 'MultipleLines_Yes',
         'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'TechSupport_Yes', "OnlineBackup_Yes", "DeviceProtection_Yes", 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Automatic', 'tenure_19:36',
       'tenure_37:54', 'tenure_55:72','intercept']]

vif = pd.DataFrame()
vif["Variables"]  = variables.columns[0:-1]
vif["VIF Factor"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1]-1)]
vif
# Setting the model
logistical_regression = sm.Logit(tcc["Churn_Yes"], variables)

# Fitting the model
fitted_model = logistical_regression.fit()
fitted_model.summary2()
variables = tcc[['SeniorCitizen',
        'Dependents_Yes', 'MultipleLines_Yes',
         'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'TechSupport_Yes', "OnlineBackup_Yes", 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Automatic', 'tenure_19:36',
       'tenure_37:54', 'tenure_55:72','intercept']]

# Setting the model
logistical_regression = sm.Logit(tcc["Churn_Yes"], variables)

# Fitting the model
fitted_model = logistical_regression.fit()
fitted_model.summary2()
margeff = fitted_model.get_margeff()
margeff.summary()
# Compute the predicted probability
pred = np.array([ 1 - fitted_model.predict(), fitted_model.predict() ])

skplt.metrics.plot_ks_statistic(tcc["Churn_Yes"], pred.T, figsize=(5, 5));
# Lift Chart
skplt.metrics.plot_lift_curve(tcc["Churn_Yes"], pred.T, figsize=(5, 5));
# Gain Chart
skplt.metrics.plot_cumulative_gain(tcc["Churn_Yes"], pred.T, figsize=(5, 5));
churn = fitted_model.predict()
def bootstrap_replicate(data, function):
    bs_sample = np.random.choice(data, len(data))
    return function(bs_sample)

bs_results = np.empty(10000)

for i in range(10000):
    bs_results[i] = bootstrap_replicate(churn, np.mean)

_ = plt.hist(bs_results, bins = 30, normed = True)
_ = plt.xlabel("Mean of Predicted Churners")
_ = plt.ylabel("Empirical Probability Density Function")
plt.show()
print("Bootstrap confidence interval:", np.percentile(bs_results, [2.5, 97.5]))
print("Bootstrap mean:", bs_results.mean())
print("Sample mean:", tcc.Churn_Yes.mean())
low_tenure = tcc[tcc["tenure_0:18"]+tcc["tenure_19:36"]==1]

variables = low_tenure[['SeniorCitizen',
        'Dependents_Yes', 'MultipleLines_Yes',
         'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
        'TechSupport_Yes', "OnlineBackup_Yes", 'StreamingTV_Yes',
       'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Automatic', 'tenure_19:36',
       'tenure_37:54', 'tenure_55:72','intercept']]

# Setting the model
logistical_regression_low = sm.Logit(low_tenure["Churn_Yes"], variables)

# Fitting the model
fitted_model_low = logistical_regression.fit()
fitted_model_low.summary2()
