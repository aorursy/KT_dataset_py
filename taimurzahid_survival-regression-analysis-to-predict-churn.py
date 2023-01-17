!pip install lifelines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from lifelines import *
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.info()
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})

sns.countplot(x="gender", 
              data = df, 
              color = 'gray'
             ).set_title('Gender Distribution Among the Customers')
sns.countplot(x = 'SeniorCitizen', hue = 'Partner', 
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Distribution of Senior Citizens grouped by Partners')
sns.countplot(x = 'Dependents',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Customers with Dependents')
sns.countplot(x = 'Dependents', hue = 'Partner', 
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Dependents and Partner Distribution among the Customers')
sns.countplot(x = 'Dependents', hue = 'SeniorCitizen', 
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Dependents and Senior Citizen Distribution')
sns.countplot(x = 'PhoneService', 
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Customers with Phone Service')
sns.countplot(x = 'MultipleLines',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Gender Distribution Among the Three Classes of Population')
sns.countplot(x = 'PhoneService', hue = 'MultipleLines',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Phone Service and Multiple Lines')
sns.countplot(x = 'InternetService', 
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service')
sns.countplot(x = 'OnlineSecurity', 
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Online Security')
sns.countplot(x = 'InternetService', hue = 'OnlineSecurity',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service and Online Security')
sns.countplot(x = 'InternetService', hue = 'OnlineBackup',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service and Online Backup')
sns.countplot(x = 'InternetService', hue = 'DeviceProtection',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service and Device Protection')
sns.countplot(x = 'InternetService', hue = 'TechSupport',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service and Tech Support')
sns.countplot(x = 'InternetService', hue = 'StreamingTV',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service and Streaming TV')
sns.countplot(x = 'InternetService', hue = 'StreamingMovies',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Internet Service and Streaming Movies')
sns.countplot(x = 'Contract',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Types of Contracts among the Customers')
sns.countplot(x = 'Contract', hue = 'PaperlessBilling',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Contract Type and Paperless Billing')
sns.countplot(x = 'Contract', hue = 'PaymentMethod',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Contract Type and Payment Method')
sns.countplot(x = 'Churn', hue = 'gender',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Gender')
sns.countplot(x = 'Churn', hue = 'SeniorCitizen',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Senior or Normal Citizens')
sns.countplot(x = 'Churn', hue = 'Partner',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Partners')
sns.countplot(x = 'Churn', hue = 'Dependents',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Dependents or No Dependents')
sns.countplot(x = 'Churn', hue = 'PhoneService',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Phone Service')
sns.countplot(x = 'Churn', hue = 'MultipleLines',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Multiple Phone Lines')
sns.countplot(x = 'Churn', hue = 'InternetService',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by type of Internet Service')
sns.countplot(x = 'Churn', hue = 'OnlineSecurity',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by the type of Online Security')
sns.countplot(x = 'Churn', hue = 'OnlineBackup',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Online Backup')
sns.countplot(x = 'Churn', hue = 'DeviceProtection',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Device Protection')
sns.countplot(x = 'Churn', hue = 'TechSupport',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Tech Support')
sns.countplot(x = 'Churn', hue = 'StreamingTV',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Streaming TV')
sns.countplot(x = 'Churn', hue = 'StreamingMovies',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Streaming Movies')
sns.countplot(x = 'Churn', hue = 'Contract',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by the Contract Type')
sns.countplot(x = 'Churn', hue = 'PaperlessBilling',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by Customers with or without Paperless Billing')
sns.countplot(x = 'Churn', hue = 'PaymentMethod',
              data = df, color = 'gray', 
              edgecolor=sns.color_palette('gray', 1)
             ).set_title('Churning Customers grouped by the type of Payment Method')
# Checking for Null Values
# In this case, there are no Null Values since we cannot see any lines in the figure below
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="Greens")
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200)
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
df[['Churn', 'gender', 'customerID']].groupby(['gender', 'Churn']).count()
df[['Churn', 'SeniorCitizen', 'customerID']].groupby(['SeniorCitizen', 'Churn']).count()
df[['gender','SeniorCitizen','Churn', 'customerID']].groupby(['gender','SeniorCitizen', 'Churn']).count()
print(df.gender.value_counts())
df['Female'] = df['gender'] == 'Female'
df["Female"] = df["Female"].astype(int)
df.drop('gender', axis = 1, inplace = True)
print(df.Female.value_counts())
print(df.Partner.value_counts())
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
print(df.Partner.value_counts())
df.SeniorCitizen.value_counts()
print(df.Dependents.value_counts())
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
print(df.Dependents.value_counts())
print(df.PhoneService.value_counts())
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
print(df.PhoneService.value_counts())
print(df.MultipleLines.value_counts())
df['MultipleLines'] = df['MultipleLines'].map({'Yes' : 1, 'No' : 0, 'No phone service' : 0})
print(df.MultipleLines.value_counts())
df.InternetService.value_counts()
print(df.OnlineSecurity.value_counts())
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})
print(df.OnlineSecurity.value_counts())
print(df.OnlineBackup.value_counts())
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})
print(df.OnlineBackup.value_counts())
print(df.DeviceProtection.value_counts())
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})
print(df.DeviceProtection.value_counts())
print(df.TechSupport.value_counts())
df['TechSupport'] = df['TechSupport'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})
print(df.TechSupport.value_counts())
print(df.StreamingTV.value_counts())
df['StreamingTV'] = df['StreamingTV'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})
print(df.StreamingTV.value_counts())
print(df.StreamingMovies.value_counts())
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})
print(df.StreamingMovies.value_counts())
df.Contract.value_counts()
print(df.PaperlessBilling.value_counts())
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
print(df.PaperlessBilling.value_counts())
df.PaymentMethod.value_counts()
print(df.Churn.value_counts())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(df.Churn.value_counts())
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan).astype(float)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace = True)
df.info()
df.head()
T = df['tenure']
E = df['Churn']
# from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)
#print(kmf.cumulative_density_)
kmf.plot_cumulative_density()
kmf.plot_survival_function() 
median_ = kmf.median_survival_time_
# The estimated median time to event. np.inf if doesn’t exist.
print('Median Value: ' + str(median_))
seniorCitizen = (df['SeniorCitizen'] == 1)

kmf.fit(T[~seniorCitizen], E[~seniorCitizen], label = 'Not Senior Citizens')
ax = kmf.plot()

kmf.fit(T[seniorCitizen], E[seniorCitizen], label = 'Senior Citizens')
ax = kmf.plot(ax=ax)
Partner = (df['Partner'] == 1)

kmf.fit(T[~Partner], E[~Partner], label = 'Without Partner')
ax = kmf.plot()

kmf.fit(T[Partner], E[Partner], label = 'With Partner')
ax = kmf.plot(ax=ax)
Dependents = (df['Dependents'] == 1)

kmf.fit(T[~Dependents], E[~Dependents], label = 'Without Dependents')
ax = kmf.plot()

kmf.fit(T[Dependents], E[Dependents], label = 'With Dependents')
ax = kmf.plot(ax=ax)
PhoneService = (df['PhoneService'] == 1)

kmf.fit(T[~PhoneService], E[~PhoneService], label = 'Without Phone Service')
ax = kmf.plot()

kmf.fit(T[PhoneService], E[PhoneService], label = 'With Phone Service')
ax = kmf.plot(ax=ax)
MultipleLines = (df['MultipleLines'] == 1)

kmf.fit(T[~MultipleLines], E[~MultipleLines], label = 'Without MultipleLines')
ax = kmf.plot()

kmf.fit(T[MultipleLines], E[MultipleLines], label = 'With MultipleLines')
ax = kmf.plot(ax=ax)
InternetServiceDSL = (df['InternetService'] == 'DSL')
InternetServiceFiberOptic = (df['InternetService'] == 'Fiber optic')
NoInternetService = (df['InternetService'] == 'No')


kmf.fit(T[InternetServiceDSL], E[InternetServiceDSL], label = 'DSL')
ax = kmf.plot()

kmf.fit(T[InternetServiceFiberOptic], E[InternetServiceFiberOptic], label = 'Fiber Optics')
ax = kmf.plot(ax=ax)

kmf.fit(T[NoInternetService], E[NoInternetService], label = 'No Internet Services')
ax = kmf.plot(ax=ax)
OnlineSecurity = (df['OnlineSecurity'] == 1)

kmf.fit(T[~OnlineSecurity], E[~OnlineSecurity], label = 'Without Online Security')
ax = kmf.plot()

kmf.fit(T[OnlineSecurity], E[OnlineSecurity], label = 'With Online Security')
ax = kmf.plot(ax=ax)
OnlineBackup = (df['OnlineBackup'] == 1)

kmf.fit(T[~OnlineBackup], E[~OnlineBackup], label = 'Without Online Backup')
ax = kmf.plot()

kmf.fit(T[OnlineBackup], E[OnlineBackup], label = 'With Online Backup')
ax = kmf.plot(ax=ax)
DeviceProtection = (df['DeviceProtection'] == 1)

kmf.fit(T[~DeviceProtection], E[~DeviceProtection], label = 'Without Device Protection')
ax = kmf.plot()

kmf.fit(T[DeviceProtection], E[DeviceProtection], label = 'With Device Protection')
ax = kmf.plot(ax=ax)
TechSupport = (df['TechSupport'] == 1)

kmf.fit(T[~TechSupport], E[~TechSupport], label = 'Without Tech Support')
ax = kmf.plot()

kmf.fit(T[TechSupport], E[TechSupport], label = 'With Tech Support')
ax = kmf.plot(ax=ax)
StreamingTV = (df['StreamingTV'] == 1)

kmf.fit(T[~StreamingTV], E[~StreamingTV], label = 'Without Streaming TV')
ax = kmf.plot()

kmf.fit(T[StreamingTV], E[StreamingTV], label = 'With Streaming TV')
ax = kmf.plot(ax=ax)
StreamingMovies = (df['StreamingMovies'] == 1)

kmf.fit(T[~StreamingMovies], E[~StreamingMovies], label = 'Without Streaming Movies')
ax = kmf.plot()

kmf.fit(T[StreamingMovies], E[StreamingMovies], label = 'With Streaming Movies')
ax = kmf.plot(ax=ax)
PaperlessBilling = (df['PaperlessBilling'] == 1)

kmf.fit(T[~PaperlessBilling], E[~PaperlessBilling], label = 'Without Paperless Billing')
ax = kmf.plot()

kmf.fit(T[PaperlessBilling], E[PaperlessBilling], label = 'With Paperless Billing')
ax = kmf.plot(ax=ax)
ElectronicCheck = (df['PaymentMethod'] == 'Electronic check')
MailedCheck = (df['PaymentMethod'] == 'Mailed check')
BankTransfer = (df['PaymentMethod'] == 'Bank transfer (automatic)')
CreditCard = (df['PaymentMethod'] == 'Credit card (automatic)')

kmf.fit(T[ElectronicCheck], E[ElectronicCheck], label = 'Electronic Check')
ax = kmf.plot()

kmf.fit(T[MailedCheck], E[MailedCheck], label = 'Mailed Check')
ax = kmf.plot(ax=ax)

kmf.fit(T[BankTransfer], E[BankTransfer], label = 'Bank Transfer')
ax = kmf.plot(ax=ax)

kmf.fit(T[CreditCard], E[CreditCard], label = 'Credit Card')
ax = kmf.plot(ax=ax)
Female = (df['Female'] == 1)

kmf.fit(T[~Female], E[~Female], label = 'Male Customers')
ax = kmf.plot()

kmf.fit(T[Female], E[Female], label = 'Female Customers')
ax = kmf.plot(ax=ax)
from lifelines.utils import median_survival_times

# The estimated median time to event. np.inf if doesn’t exist.
median_ci = median_survival_times(kmf.confidence_interval_)
median_ci
cols_of_interest = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                    'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaperlessBilling', 'PaymentMethod', 
                    'MonthlyCharges', 'TotalCharges', 'Female', 'Churn']
data = df[cols_of_interest]
data.head()
data = pd.get_dummies(data)
# Dropping these columns to avoid 'Matrix Singularity Error when training the Model'
data.drop('Contract_Two year', axis = 1, inplace = True)
data.drop('PaymentMethod_Mailed check', axis = 1, inplace = True)
data.drop('InternetService_Fiber optic', axis = 1, inplace = True)
data.head()
cph = CoxPHFitter()
cph.fit(data, 'tenure', event_col = 'Churn', show_progress = True)
cph.print_summary()
cph.plot() 
# Produces a visual representation of the coefficients (i.e. log hazard ratios), 
# including their standard errors and magnitudes.
cph.params_ # The estimated coefficients.
cph.plot_covariate_groups('Contract_Month-to-month', [0, 1], cmap='coolwarm')

# "we can plot what the survival curves look like as we vary a single covariate while holding everything else equal. 
# This is useful to understand the impact of a covariate, given the model. To do this, we use the plot_covariate_groups() 
# method and give it the covariate of interest, and the values to display."
cph.predict_expectation(data)
cph.predict_log_partial_hazard(data)
# The event_observed variable provided
cph.event_observed
cph.baseline_cumulative_hazard_
sns.lineplot(data=cph.baseline_cumulative_hazard_)
cph.plot_covariate_groups('InternetService_No', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups('OnlineSecurity', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups('InternetService_DSL', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups('Contract_One year', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups('PhoneService', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups('SeniorCitizen', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups('Dependents', [0, 1], cmap='coolwarm')
cph.plot_covariate_groups(['SeniorCitizen', 'Partner'], 
                            [
                                [0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1],
                            ],
                            cmap='coolwarm')
#plt.savefig('SeniorCitizen with Partner.png')
plt.title("SeniorCitizen with Partner");
cph.plot_covariate_groups(['SeniorCitizen', 'Dependents'], 
                            [
                                [0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1],
                            ],
                            cmap='coolwarm')
#plt.savefig('Media/SeniorCitizen with Dependents.png')
plt.title("SeniorCitizen with Dependents");
cph.plot_covariate_groups(
    ['PhoneService', 'InternetService_No'],
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    cmap='coolwarm')
#plt.savefig('Media/Internet Services and Phone Services.png')
plt.title("Internet Services and Phone Services");
data['ID'] = df['customerID']
data.head()
cph.predict_survival_function(data.drop('ID', axis = 1))
cph.predict_median(data.drop('ID', axis = 1))
cph.predict_partial_hazard(data.drop('ID', axis = 1))
cph.predict_median(data)
input_ = data.loc[data['ID'] == '3668-QPYBK']
input_.head()
results = cph.predict_survival_function(input_.drop('ID', axis = 1))
sns.lineplot(data = results, 
             legend = False).set_title('Survival of the Customer - 3668-QPYBK')
cph.predict_partial_hazard(input_.drop('ID', axis = 1))
from lifelines.utils import k_fold_cross_validation

cph = CoxPHFitter()

print(np.mean(k_fold_cross_validation(cph, data.drop('ID', axis = 1), duration_col='tenure', event_col='Churn')))