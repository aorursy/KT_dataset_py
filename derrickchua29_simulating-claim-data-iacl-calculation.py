import pandas as pd
import numpy as np
import datetime
pd.options.display.max_columns = 100
"""Define parameters of data"""
# Number of entries
PolicyCount = 15000
# 9-year period
YearEndCap = 2017
YearStartCap = 2007
# Dates
DateEndCap = datetime.date(YearEndCap, 12, 31)    # year, month, day
DateStartCap = datetime.date(YearStartCap, 12, 31)  # year, month, day
"""Create Main DataFrame filled with NaN's"""
# Establish initial data-frame
columns_1 = ['Insured_ID', 'Insured_Date', 'Claims_Number', 'Claims_Amount', 'Transaction_Date',
           'Insured_Year', 'Insured_Quarter',
           'Transaction_Year', 'Transaction_Quarter']
ClaimsData = pd.DataFrame(columns=columns_1)
# Insured_ID's
ClaimsData['Insured_ID'] = list(range(1, PolicyCount+1))
print(ClaimsData['Insured_ID'])
# Insured_Date's
# Random distribution
import random
for row in range(0, PolicyCount):
    n_days = (DateEndCap-DateStartCap).days
    random_days = random.randint(0, n_days-1)
    Random_Insured_Date = DateStartCap + datetime.timedelta(days=1) + datetime.timedelta(days=random_days)
    ClaimsData.loc[row, 'Insured_Date'] = Random_Insured_Date
print(ClaimsData['Insured_Date'])
# Claims_Number's
# Poisson random distribution
# Poisson parameters
Lambda = 10
Size = 1
for row in range(0, PolicyCount):
    ClaimCount = np.random.poisson(1, 1)
    ClaimsData.loc[row, 'Claims_Number'] = ClaimCount

# Remove the square brackets (i.e.a list within a list) by passing into a list & back into df again
ClaimsData['Claims_Number'] = pd.DataFrame(ClaimsData['Claims_Number'].values.tolist())
print(ClaimsData['Claims_Number'])
"""Special Case if need to simulate claims amount minimum & maximum limit. E.g. Reinsurance cases XOL"""
import random
def trunc_amt(mu, sigma, bottom, top):
    a = random.lognormal(mu,sigma)
    while (bottom <= a <= top) == False:
        a = random.lognormal(mu,sigma)
    return a
# Claims_Amount's
# Gaussian random distribution
# Gaussian parameters
MeanClaimAmt = 10
StdDevClaimAmt = 4
for row in range(0, PolicyCount):
    if ClaimsData.loc[row, 'Claims_Number'] == 0:
        # Impute 0 so that ClaimAmount is 0
        ClaimsData.loc[row, 'Claims_Amount'] = 0
    else:
        ClaimNumber = ClaimsData.loc[row, 'Claims_Number']
        num = np.random.lognormal(MeanClaimAmt, StdDevClaimAmt, ClaimNumber).sum()
        ClaimsData.loc[row, 'Claims_Amount'] = num

# Remove the square brackets (i.e.a list within a list) by passing into a list & back into df again
ClaimsData['Claims_Amount'] = pd.DataFrame(ClaimsData['Claims_Amount'].values.tolist())
print(ClaimsData['Claims_Amount'])
# Transaction_Date's
# Random distribution
import random
for row in range(0, PolicyCount):
    DateStart = ClaimsData.loc[row, 'Insured_Date']
    if ClaimsData.loc[row, 'Claims_Number'] == 0:
        # Impute InsuredDate so that Lag(i.e.DevelopmentPeriod) will be 0
        ClaimsData.loc[row, 'Transaction_Date'] = DateStart
    elif (DateEndCap-DateStart).days <=0:
        ClaimsData.loc[row, 'Transaction_Date'] = DateStart
    else:
        n_days = (DateEndCap-DateStart).days
        random_days = random.randint(1, n_days) # Min 1 day to avoid conflict of zero days and no claims
        Random_Transaction_Date = DateStart + datetime.timedelta(days=random_days)
        ClaimsData.loc[row, 'Transaction_Date'] = Random_Transaction_Date
print(ClaimsData['Transaction_Date'])
display(ClaimsData.head(10))
# Extract & Impute Date Components
# Jan-Mar=1, Apr-Jun=2, July-Sep=3, Oct-Dec=4
# Insured Year
ClaimsData['Insured_Year'] = ClaimsData['Insured_Date'].apply(lambda x: x.year)
ClaimsData['Transaction_Year'] = ClaimsData['Transaction_Date'].apply(lambda x: x.year)
# Insured Month
ClaimsData['Insured_Quarter'] = ClaimsData['Insured_Date'].apply(lambda x: x.month)
ClaimsData['Transaction_Quarter'] = ClaimsData['Transaction_Date'].apply(lambda x: x.month)
print(ClaimsData[['Insured_Date', 'Insured_Year', 'Transaction_Date', 'Transaction_Year']])
# Year ONLY lag
ClaimsData['Year_Only_Lag'] = ClaimsData['Transaction_Year'] - ClaimsData['Insured_Year']
print(ClaimsData)
# Compile Past Claims Data
# Incremental Claims Amount
py_data = ClaimsData['Claims_Amount'].groupby([ClaimsData['Insured_Year'], ClaimsData['Year_Only_Lag']]).sum().reset_index()
# Convert into data-frame
py_data = pd.DataFrame(py_data)
# Cumulative Claims Amount
py_data["cumsum"] = py_data["Claims_Amount"].groupby(py_data["Insured_Year"]).cumsum()
print(py_data)
# Establish Inflation Index
# Create data-frame of Cumulative inflation rates
columns_2 = ['Year', 'CumPastInflation']
Inflation_df = pd.DataFrame(columns=columns_2)
# Past Inflation Years
Inflation_df['Year'] = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
# Past Inflation Index
Inflation_df['CumPastInflation'] = [1.32, 1.27, 1.28, 1.22, 1.16, 1.12, 1.09, 1.07, 1.05, 1.04, 1.00]
display(Inflation_df)
# Uplift (Past Inflation) for Incremental Claims
py_data['Inflated_Claims_Amount'] = py_data['Claims_Amount']

for row in range(0, len(py_data['Insured_Year'])):
    InsuredYear = py_data.loc[row,'Insured_Year']
    LagYear = py_data.loc[row,'Year_Only_Lag']
    TransactionYear = InsuredYear + LagYear
    for year in range(0, len(Inflation_df['Year'])):
        CurrentYearInflation = Inflation_df.loc[year,'Year']
        if  CurrentYearInflation == InsuredYear:
            CurrentYearPerc = Inflation_df.loc[Inflation_df['Year'] == TransactionYear,'CumPastInflation']
            ToYearPerc = Inflation_df.loc[Inflation_df['Year']==YearEndCap,'CumPastInflation'].values[0]
            Uplift = ToYearPerc / CurrentYearPerc
            py_data['Inflated_Claims_Amount'][row] = py_data['Inflated_Claims_Amount'][row]*Uplift
        else:
             py_data['Inflated_Claims_Amount'][row] = py_data['Inflated_Claims_Amount'][row]

print(py_data)
# Get Uplift (Past Inflation) Cumulative Claims
py_data['Inflated_cumsum'] = py_data['Inflated_Claims_Amount'].groupby(py_data['Insured_Year']).cumsum()
# Inflated
py_data['Inflated_LossDF'] = 1

for row in range(0, len(py_data['Insured_Year'])):
    InsuredYear = py_data.loc[row, 'Insured_Year']
    LagYr = py_data.loc[row, 'Year_Only_Lag']
    CurrentYear = py_data.loc[row, 'Insured_Year'] + py_data.loc[row, 'Year_Only_Lag']
    CurrCumAmt = py_data.loc[row, 'Inflated_cumsum']

    if CurrentYear > YearEndCap or len(py_data.loc[(py_data['Insured_Year'] == InsuredYear) & (
            py_data['Year_Only_Lag'] == (LagYr + 1)), 'Inflated_cumsum']) == 0:
        NextCumAmt = 0
    else:
        NextCumAmt = py_data.loc[(py_data['Insured_Year'] == InsuredYear) & (
                    py_data['Year_Only_Lag'] == (LagYr + 1)), 'Inflated_cumsum'].values[0]

    LDF = NextCumAmt / CurrCumAmt
    py_data.loc[row, 'Inflated_LossDF'] = LDF

print(py_data['Inflated_LossDF'])
# Non Inflated
py_data['LossDF'] = 1

for row in range(0, len(py_data['Insured_Year'])):
    InsuredYear = py_data.loc[row, 'Insured_Year']
    LagYr = py_data.loc[row, 'Year_Only_Lag']
    CurrentYear = py_data.loc[row, 'Insured_Year'] + py_data.loc[row, 'Year_Only_Lag']
    CurrCumAmt = py_data.loc[row, 'cumsum']

    if CurrentYear > YearEndCap or len(py_data.loc[(py_data['Insured_Year'] == InsuredYear) & (
            py_data['Year_Only_Lag'] == (LagYr + 1)), 'cumsum']) == 0:
        NextCumAmt = 0
    else:
        # .values[0] code to output only values and not entire row
        NextCumAmt = py_data.loc[
            (py_data['Insured_Year'] == InsuredYear) & (py_data['Year_Only_Lag'] == (LagYr + 1)), 'cumsum'].values[0]

    LDF = NextCumAmt / CurrCumAmt
    py_data.loc[row, 'LossDF'] = LDF

print(py_data['LossDF'])
"""Claims Data - Single Plot"""
def SinglePlotPartialClaims(DataFrameName, InsuredYearColumn, LagYearColumn, ValueColumn):
    import matplotlib.pyplot as plt
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    Filtered_df = pd.DataFrame(DataFrameName[[InsuredYearColumn, LagYearColumn, ValueColumn]])
    Filtered_df.columns = Filtered_NewColumnNames
    """Unique Insured Years List"""
    InsuredYr_List = list(DataFrameName[InsuredYearColumn].unique())
    """Unique Lag Years List"""
    LagYr_List = list(DataFrameName[LagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Color_List = ALL_Colors[:len(InsuredYr_List)]
    """LineStyle List"""
    ALL_LineStyle = ['-', '--', '-.', ':','-','-','-','-','-','-','-','-','-']
    LineStyle_List = ALL_LineStyle[:len(InsuredYr_List)]
    """MarkerStyle List"""# First 4x empty 
    ALL_Markers = ['','','','','^','.','o','*', '+', '1', '2', '3', '4']
    Marker_List = ALL_Markers[:len(InsuredYr_List)]
    """Loop Plot"""
    for row_A in range(0,len(InsuredYr_List)):
        plt.figure(2, figsize=(10,5))
        Year_i = InsuredYr_List[row_A]
        SubFiltered_df = Filtered_df.loc[Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(SubFiltered_df['Year_Only_Lag'], SubFiltered_df['ClaimAmt'], 
                 label=str(Year_i), linestyle='-', color=Color_List[row_A])
    """Plot Attributes"""    
    plt.xlabel('Developement Year')
    plt.ylabel('Claims Value')
    plt.title('Single Plot Partial Claims Data')
    plt.legend()
    plt.show()
"""Claims Data - Sub Plot"""
def SubPlotPartialClaims(DataFrameName, InsuredYearColumn, LagYearColumn, ValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    Filtered_df = pd.DataFrame(DataFrameName[[InsuredYearColumn, LagYearColumn, ValueColumn]])
    Filtered_df.columns = Filtered_NewColumnNames
    """Unique Insured Years List"""
    InsuredYr_List = list(DataFrameName[InsuredYearColumn].unique())
    """Unique Lag Years List"""
    LagYr_List = list(DataFrameName[LagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Color_List = ALL_Colors[:len(InsuredYr_List)]
    """LineStyle List"""
    ALL_LineStyle = ['-', '--', '-.', ':','-','-','-','-','-','-','-','-','-']
    LineStyle_List = ALL_LineStyle[:len(InsuredYr_List)]
    """MarkerStyle List"""# First 4x empty 
    ALL_Markers = ['','','','','^','.','o','*', '+', '1', '2', '3', '4']
    Marker_List = ALL_Markers[:len(InsuredYr_List)]
    """Plot Attributes"""
    fig = plt.figure(2, figsize=(10,14))
    plt.xticks([]) # remove initial blank plot default ticks
    plt.yticks([]) # remove initial blank plot default ticks
    plt.title('Sub Plot Partial Claims Data')
    rcParams['axes.titlepad'] = 70 # position title
    plt.box(on=None) # Remove boundary line
    """Loop Plot"""
    i=0
    for row_A in range(0,len(InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1+i)
        Year_i = InsuredYr_List[row_A]
        SubFiltered_df = Filtered_df.loc[Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(SubFiltered_df['Year_Only_Lag'], SubFiltered_df['ClaimAmt'], 
                 label=str(Year_i), marker='o', linestyle='-', color=Color_List[row_A])
        plt.xticks(np.arange(0, (YearEndCap-YearStartCap), step=1))
        plt.legend()
        i += 1
        """Plot Attributes"""
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value')
    
    fig.tight_layout() # set size
    plt.show()
"""Loss Development Ratios"""
def SinglePlotLDF(DataFrameName, Columns):
    import matplotlib.pyplot as  plt
    """Create New df"""
    Filtered_df = pd.DataFrame(DataFrameName[Columns])    
    """Lag Years"""
    LagYears_List = list(range(0, len(DataFrameName)))
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Color_List = ALL_Colors[:len(Columns)]
    """Loop Plot"""
    plt.figure(2, figsize=(10,5))
    for row_A in range(0,len(Columns)):
        Column_i = Columns[row_A]
        plt.plot(LagYears_List, Filtered_df[Column_i], label=str(Column_i), linestyle='-', color=Color_List[row_A])
        plt.legend()         
    """Plot Attributes"""    
    plt.xlabel('Developement Year')
    plt.ylabel('Ratio')
    plt.title('Loss Development Factors')
    plt.show()
# Incremental Claims Amount
# Inflated
py_triangle_inflated = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Inflated_Claims_Amount"])
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
sns.distplot(py_data['Inflated_Claims_Amount'], kde=False, fit=stats.lognorm) # norm, pareto, loggamma, gompertz
plt.show()
display(py_triangle_inflated)
# Incremental Claims Amount
# Non-Inflated
py_triangle = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Claims_Amount"])
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
sns.distplot(py_data['Claims_Amount'], kde=False, fit=stats.lognorm) # norm, pareto, loggamma, gompertz
plt.show()
display(py_triangle)
# Cumulative Claims Amount
# Inflated
py_triangle_cum_inflated = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Inflated_cumsum"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_cumsum')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_cumsum')
display(py_triangle_cum_inflated)
# Cumulative Claims Amount
# Non-Inflated
py_triangle_cum = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["cumsum"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='cumsum')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='cumsum')
display(py_triangle_cum)
# Individual Loss Development factors
# Inflated
py_InflatedLossDF_triangle = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Inflated_LossDF"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_LossDF')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_LossDF')
display(py_InflatedLossDF_triangle)
# Individual Loss Development factors
# Non-Inflated
py_LossDF_triangle = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["LossDF"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='LossDF')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='LossDF')
display(py_LossDF_triangle)
# Create a Temp Df of Predicted Years & LagYears rates
columns_3 = ['InsuredYear', 'PredictedYear_Only_Lag',
             'Previous_cumsum', 'Predicted_cumsum', 'Predicted_Incremental',
             'Previous_Inflated_cumsum', 'Predicted_Inflated_cumsum', 'Predicted_Inflated_Incremental']
Temp_df = pd.DataFrame(columns=columns_3)
# +1 due to 31 Dec 2017 (also not a Bday) & +1 due to range exlusion of last value cap
InsuredYr = list(range(YearStartCap + 1, YearEndCap + 1, 1))  # [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
Temp_df['InsuredYear'] = InsuredYr
Lags = list(range(0, YearEndCap - YearStartCap, 1))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Temp_df['PredictedYear_Only_Lag'] = Lags

# Establish Predicted data-frame
Predicted_df = pd.DataFrame(columns=columns_3)
# Coordinates of predicted Insured Years & Lag Years
x = 1 # Do nothing
i = 0 # For loop impute indexing
for row in range(0, len(Temp_df['InsuredYear'])):
    BaseYr = Temp_df.loc[row, 'InsuredYear']
    for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
        LagYr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
        P_yr = BaseYr + Temp_df.loc[lag, 'PredictedYear_Only_Lag']
        if P_yr > YearEndCap:

            Predicted_df.loc[i, 'InsuredYear'] = BaseYr
            Predicted_df.loc[i, 'PredictedYear_Only_Lag'] = LagYr
            i += 1
        else:
            x = x

print(Predicted_df[['InsuredYear', 'PredictedYear_Only_Lag']])
# Impute latest cumulative amounts available
# Inflated
for row in range(0, len(Predicted_df)):
    Base = Predicted_df.loc[row, 'InsuredYear']
    Lag = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    PredYr = Base + Lag

    if Base == YearEndCap:
        PrevInflatedCumSum = py_data.loc[(py_data['Insured_Year'] == Base), 'Inflated_cumsum'].values[0]

    else:
        if PredYr > YearEndCap or len(py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'Inflated_cumsum']) == 0:
            MaxLag = py_data.loc[(py_data['Insured_Year'] == Base), 'Year_Only_Lag'].max()
            PrevInflatedCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == MaxLag), 'Inflated_cumsum'].values[0]

        else:
            PrevInflatedCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'Inflated_cumsum'].values[0]

    Predicted_df.loc[row, 'Previous_Inflated_cumsum'] = PrevInflatedCumSum

print(Predicted_df['Previous_Inflated_cumsum'])
# Impute latest cumulative amounts available
# Non-Inflated
for row in range(0, len(Predicted_df)):
    Base = Predicted_df.loc[row, 'InsuredYear']
    Lag = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    PredYr = Base + Lag

    if Base == YearEndCap:
        PrevCumSum = py_data.loc[(py_data['Insured_Year'] == Base), 'cumsum'].values[0]

    else:
        if PredYr > YearEndCap or len(
                py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'cumsum']) == 0:
            MaxLag = py_data.loc[(py_data['Insured_Year'] == Base), 'Year_Only_Lag'].max()
            PrevCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == MaxLag), 'cumsum'].values[0]

        else:
            PrevCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'cumsum'].values[0]

    Predicted_df.loc[row, 'Previous_cumsum'] = PrevCumSum
    
print(Predicted_df['Previous_cumsum'])
# Establish averaged-year-to-year LDF
columns_4 = ['Year_Only_Lag',
             'SimpleMeanLossDF', 'VolWtdLossDF',
             'CumToUlt_SimpleMeanLossDF', 'CumToUlt_VolWtdLossDF',
             'SimpleMeanLossDF_5year', 'VolWtdLossDF_5year',
             'SimpleMeanLossDF_3year', 'VolWtdLossDF_3year',
             'SelectLossDF'
             'Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF',
             'Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF',
             'Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year',
             'Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year',
             'Inflated_SelectLossDF']
LossDF_df = pd.DataFrame(columns=columns_4)
Lags = list(range(0, YearEndCap-YearStartCap, 1))
LossDF_df['Year_Only_Lag'] = Lags
display(LossDF_df)
# Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    # due to 0 input so exlude last value
    SimpleMeanLossDF = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:-1].mean()
    LossDF_df.loc[i, 'Inflated_SimpleMeanLossDF'] = SimpleMeanLossDF
    # Volume Weighted
    Deno = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'].sum()
    Neum = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:-1].sum()
    VolWtdLossDF = Deno / Neum
    LossDF_df.loc[i, 'Inflated_VolWtdLossDF'] = VolWtdLossDF
    i += 1

# [::-1] to flip or invert the row order
LossDF_df['Inflated_CumToUlt_SimpleMeanLossDF']=LossDF_df['Inflated_SimpleMeanLossDF'][::-1].cumprod()
LossDF_df['Inflated_CumToUlt_VolWtdLossDF']=LossDF_df['Inflated_VolWtdLossDF'][::-1].cumprod()

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF'])
display(LossDF_df[['Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF', 'Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF']])
# Non-Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    # due to 0 input so exlude last value
    SimpleMeanLossDF = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'LossDF'][:-1].mean()
    LossDF_df.loc[i, 'SimpleMeanLossDF'] = SimpleMeanLossDF
    # Volume Weighted
    Deno = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'cumsum'].sum()
    Neum = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'cumsum'][:-1].sum()
    VolWtdLossDF = Deno / Neum
    LossDF_df.loc[i, 'VolWtdLossDF'] = VolWtdLossDF
    i += 1

# [::-1] to flip or invert the row order
LossDF_df['CumToUlt_SimpleMeanLossDF']=LossDF_df['SimpleMeanLossDF'][::-1].cumprod()
LossDF_df['CumToUlt_VolWtdLossDF']=LossDF_df['VolWtdLossDF'][::-1].cumprod()

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['SimpleMeanLossDF', 'VolWtdLossDF'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['CumToUlt_SimpleMeanLossDF', 'CumToUlt_VolWtdLossDF'])
display(LossDF_df[['SimpleMeanLossDF', 'VolWtdLossDF', 'CumToUlt_SimpleMeanLossDF', 'CumToUlt_VolWtdLossDF']])
# Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    Year_A = 5   # 5 Year
    SimpleMeanLossDF_Ayear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_A].mean()
    LossDF_df.loc[i, 'Inflated_SimpleMeanLossDF_5year'] = SimpleMeanLossDF_Ayear
    Year_B = 3   # 3 Year
    SimpleMeanLossDF_Byear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_B].mean()
    LossDF_df.loc[i, 'Inflated_SimpleMeanLossDF_3year'] = SimpleMeanLossDF_Byear
    # Volume Weighted
    Deno_A = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_A].sum()
    Neum_A = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_A].sum()
    VolWtdLossDF_A = Deno_A / Neum_A
    LossDF_df.loc[i, 'Inflated_VolWtdLossDF_5year'] = VolWtdLossDF_A
    Deno_B = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_B].sum()
    Neum_B = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_B].sum()
    VolWtdLossDF_B = Deno_B / Neum_B
    LossDF_df.loc[i, 'Inflated_VolWtdLossDF_3year'] = VolWtdLossDF_B
    i += 1

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year'])    
display(LossDF_df[['Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year', 'Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year']])
# Non Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    Year_A = 5   # 5 Year
    SimpleMeanLossDF_Ayear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_A].mean()
    LossDF_df.loc[i, 'SimpleMeanLossDF_5year'] = SimpleMeanLossDF_Ayear
    Year_B = 3   # 3 Year
    SimpleMeanLossDF_Byear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_B].mean()
    LossDF_df.loc[i, 'SimpleMeanLossDF_3year'] = SimpleMeanLossDF_Byear
    # Volume Weighted
    Deno_A = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_A].sum()
    Neum_A = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_A].sum()
    VolWtdLossDF_A = Deno_A / Neum_A
    LossDF_df.loc[i, 'VolWtdLossDF_5year'] = VolWtdLossDF_A
    Deno_B = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_B].sum()
    Neum_B = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_B].sum()
    VolWtdLossDF_B = Deno_B / Neum_B
    LossDF_df.loc[i, 'VolWtdLossDF_3year'] = VolWtdLossDF_B
    i += 1

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['SimpleMeanLossDF_5year', 'VolWtdLossDF_5year'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['SimpleMeanLossDF_3year', 'VolWtdLossDF_3year'])
display(LossDF_df[['SimpleMeanLossDF_5year', 'VolWtdLossDF_5year', 'SimpleMeanLossDF_3year', 'VolWtdLossDF_3year']])
LossDF_df['Inflated_SelectLossDF'] = LossDF_df['Inflated_VolWtdLossDF']
LossDF_df['SelectLossDF'] = LossDF_df['VolWtdLossDF']
# Predict Cumulative Claim Amounts
# Inflated
# Set Equal for easy reference
Predicted_df['Predicted_Inflated_cumsum'] = Predicted_df['Previous_Inflated_cumsum']
lagyearlimit = (YearEndCap - YearStartCap) - 1
x = 1  # Do nothing
for row in range(0, len(Predicted_df)):
    PredLagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    BaseInsuredYr = Predicted_df.loc[row, 'InsuredYear']
    MaxLagYr = py_data.loc[(py_data['Insured_Year'] == BaseInsuredYr), 'Year_Only_Lag'].max()
    for r in range(0, len(LossDF_df)):
        if (LossDF_df.loc[r, 'Year_Only_Lag'] == lagyearlimit):
            x = x  # To avoid NaN
        elif (LossDF_df.loc[r, 'Year_Only_Lag'] == MaxLagYr):
            # LDF multiplication
            LDF = LossDF_df.loc[(LossDF_df['Year_Only_Lag'] >= MaxLagYr) & (LossDF_df['Year_Only_Lag'] <= (PredLagYr - 1)), 'Inflated_SelectLossDF'].prod()
            Predicted_df.loc[row, 'Predicted_Inflated_cumsum'] = Predicted_df.loc[row, 'Predicted_Inflated_cumsum'] * LDF
        else:
            x = x  # Do nothing
            
print(Predicted_df['Predicted_Inflated_cumsum'])
# Predict Cumulative Claim Amounts
# Non-Inflated
# Set Equal for easy reference
Predicted_df['Predicted_cumsum'] = Predicted_df['Previous_cumsum']
lagyearlimit = (YearEndCap - YearStartCap) - 1
x = 1  # Do nothing
for row in range(0, len(Predicted_df)):
    PredLagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    BaseInsuredYr = Predicted_df.loc[row, 'InsuredYear']
    MaxLagYr = py_data.loc[(py_data['Insured_Year'] == BaseInsuredYr), 'Year_Only_Lag'].max()
    for r in range(0, len(LossDF_df)):
        if (LossDF_df.loc[r, 'Year_Only_Lag'] == lagyearlimit):
            x = x  # To avoid NaN
        elif (LossDF_df.loc[r, 'Year_Only_Lag'] == MaxLagYr):
            # LDF multiplication
            LDF = LossDF_df.loc[(LossDF_df['Year_Only_Lag'] >= MaxLagYr) & (LossDF_df['Year_Only_Lag'] <= (PredLagYr - 1)), 'SelectLossDF'].prod()
            Predicted_df.loc[row, 'Predicted_cumsum'] = Predicted_df.loc[row, 'Predicted_cumsum'] * LDF
        else:
            x = x  # Do nothing

print(Predicted_df['Predicted_cumsum'])
# Data-type adjustments
# Years
Predicted_df[['InsuredYear','PredictedYear_Only_Lag']]=Predicted_df[['InsuredYear','PredictedYear_Only_Lag']].astype(int)
# Amounts
Predicted_df[['Predicted_cumsum','Previous_cumsum']]=Predicted_df[['Predicted_cumsum','Previous_cumsum']].astype(float)
Predicted_df[['Predicted_Inflated_cumsum','Previous_Inflated_cumsum']]=Predicted_df[['Predicted_Inflated_cumsum','Previous_Inflated_cumsum']].astype(float)
# Predict Incremental Amount
# Inflated
for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrCum = Predicted_df.loc[row, 'Predicted_Inflated_cumsum']
    # For which we can't look up in Predicted_df
    if len(Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_Inflated_cumsum']) == 0:
        PrevCum = py_data.loc[(py_data['Insured_Year'] == InsurYr) & (py_data['Year_Only_Lag'] == LagYr - 1), 'Inflated_cumsum'].values[0]
    # For which we can look up in Predicted_df
    else:
        PrevCum = Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_Inflated_cumsum'].values[0]

    Predicted_df.loc[row, 'Predicted_Inflated_Incremental'] = (CurrCum - PrevCum)

Predicted_df[['Predicted_Inflated_Incremental']] = Predicted_df[['Predicted_Inflated_Incremental']].astype(float)
PredictedInflatedIncrementalTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"],columns=["PredictedYear_Only_Lag"],values=["Predicted_Inflated_Incremental"])

# print(PredictedInflatedIncrementalTriangle)
display(PredictedInflatedIncrementalTriangle)
# Predict Incremental Amount
# Non-Inflated
for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrCum = Predicted_df.loc[row, 'Predicted_cumsum']

    if len(Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_cumsum']) == 0:
        PrevCum = py_data.loc[(py_data['Insured_Year'] == InsurYr) & (py_data['Year_Only_Lag'] == LagYr - 1), 'cumsum'].values[0]
    else:
        PrevCum = Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_cumsum'].values[0]

    Predicted_df.loc[row, 'Predicted_Incremental'] = CurrCum - PrevCum

Predicted_df[['Predicted_Incremental']] = Predicted_df[['Predicted_Incremental']].astype(float)
PredictedIncrementalTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"],values=["Predicted_Incremental"])

# print(PredictedIncrementalTriangle)
display(PredictedIncrementalTriangle)
# Project (Future Inflation) Predicted Incremental Amount
# Inflated
FutureInflation = Inflation_df.loc[(Inflation_df['Year'] == (YearEndCap + 1)), 'CumPastInflation'].values[0]

Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'] = Predicted_df['Predicted_Inflated_Incremental']
for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrIncremAmt = Predicted_df.loc[row, 'Predicted_Inflated_Incremental']
    Predicted_df.loc[row, 'FutureUplifted_Predicted_Inflated_Incremental'] = CurrIncremAmt * (FutureInflation ** LagYr)
    
print(Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'])
# Project (Future Inflation) Predicted Incremental Amount
# Non-Inflated
# Set equal for easy reference
Predicted_df['FutureUplifted_Predicted_Incremental'] = Predicted_df['Predicted_Incremental']
FutureInflation = Inflation_df.loc[(Inflation_df['Year'] == (YearEndCap + 1)), 'CumPastInflation'].values[0]

for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrIncremAmt = Predicted_df.loc[row, 'Predicted_Incremental']

    Predicted_df.loc[row, 'FutureUplifted_Predicted_Incremental'] = CurrIncremAmt * (FutureInflation ** LagYr)
    
print(Predicted_df['FutureUplifted_Predicted_Incremental'])
# Incremental
# Non-Inflated
PredictedTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["FutureUplifted_Predicted_Incremental"])
# Inflated
PredictedInflatedTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["FutureUplifted_Predicted_Inflated_Incremental"])
SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Incremental')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Incremental')
display(PredictedTriangle)
SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Inflated_Incremental')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Inflated_Incremental')
display(PredictedInflatedTriangle)
# Cumulative
# Non-Inflated
PredictedCumTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["Predicted_cumsum"])
# Inflated
PredictedInflatedCumTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["Predicted_Inflated_cumsum"])
SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_cumsum')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_cumsum')
display(PredictedCumTriangle)
SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_Inflated_cumsum')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_Inflated_cumsum')
display(PredictedInflatedCumTriangle)
def SinglePlotFullClaims(PastDataFrameName, PastInsuredYearColumn, PastLagYearColumn, PastValueColumn, 
                   FutureDataFrameName, FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    # Past
    Past_Filtered_df = pd.DataFrame(PastDataFrameName[[PastInsuredYearColumn, PastLagYearColumn, PastValueColumn]])
    Past_Filtered_df.columns = Filtered_NewColumnNames
    # Future
    Future_Filtered_df = pd.DataFrame(FutureDataFrameName[[FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn]])
    Future_Filtered_df.columns = Filtered_NewColumnNames    
    """Unique Insured Years List"""
    # Past
    Past_InsuredYr_List = list(PastDataFrameName[PastInsuredYearColumn].unique())
    # Future
    Future_InsuredYr_List = list(FutureDataFrameName[FutureInsuredYearColumn].unique())
    """Unique Lag Years List"""
    # Past
    Past_LagYr_List = list(PastDataFrameName[PastLagYearColumn].unique())
    # Future
    Future_LagYr_List = list(FutureDataFrameName[FutureLagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Past_Color_List = ALL_Colors[:len(Past_InsuredYr_List)]
    Future_Color_List = ALL_Colors[:len(Future_InsuredYr_List)]
    """Plotting"""
    fig = plt.figure(2, figsize=(8,12))
    plt.title('Single Plot Full Claims Data')
    """Full Loop Plot"""
    Full_Filtered_df = pd.concat([Past_Filtered_df, Future_Filtered_df])
    for row_A in range(0,len(Past_InsuredYr_List)):
        Year_i = Past_InsuredYr_List[row_A]
        Full_SubFiltered_df = Full_Filtered_df.loc[Full_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Full_SubFiltered_df['Year_Only_Lag'], Full_SubFiltered_df['ClaimAmt'], 
                 label=('Predicted %d' % Year_i), linestyle='--', color=Past_Color_List[row_A])
        plt.legend()
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value')    
    """Past Loop Plot"""
    for row_A in range(0,len(Past_InsuredYr_List)):
        Year_i = Past_InsuredYr_List[row_A]
        Past_SubFiltered_df = Past_Filtered_df.loc[Past_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Past_SubFiltered_df['Year_Only_Lag'], Past_SubFiltered_df['ClaimAmt'], 
                 label=('Historical %d' % Year_i), linestyle='-', color=Past_Color_List[row_A], marker='o')
        plt.legend()
    #"""Future Loop Plot"""
    #for row_B in range(0,len(Future_InsuredYr_List)):
    #    Year_i = Future_InsuredYr_List[row_B]
    #    Future_SubFiltered_df = Future_Filtered_df.loc[Future_Filtered_df['Insured_Year'].isin([Year_i])]
    #    plt.plot(Future_SubFiltered_df['Year_Only_Lag'], Future_SubFiltered_df['ClaimAmt'], 
    #             label=str(Year_i), linestyle='--', color=Future_Color_List[row_B])    
    
    """Plot Attributes"""    
    plt.show()
def SubPlotFullClaims(PastDataFrameName, PastInsuredYearColumn, PastLagYearColumn, PastValueColumn, 
                   FutureDataFrameName, FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    # Past
    Past_Filtered_df = pd.DataFrame(PastDataFrameName[[PastInsuredYearColumn, PastLagYearColumn, PastValueColumn]])
    Past_Filtered_df.columns = Filtered_NewColumnNames
    # Future
    Future_Filtered_df = pd.DataFrame(FutureDataFrameName[[FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn]])
    Future_Filtered_df.columns = Filtered_NewColumnNames    
    """Unique Insured Years List"""
    # Past
    Past_InsuredYr_List = list(PastDataFrameName[PastInsuredYearColumn].unique())
    # Future
    Future_InsuredYr_List = list(FutureDataFrameName[FutureInsuredYearColumn].unique())
    """Unique Lag Years List"""
    # Past
    Past_LagYr_List = list(PastDataFrameName[PastLagYearColumn].unique())
    # Future
    Future_LagYr_List = list(FutureDataFrameName[FutureLagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Past_Color_List = ALL_Colors[:len(Past_InsuredYr_List)]
    Future_Color_List = ALL_Colors[:len(Future_InsuredYr_List)]
    """Plotting"""
    fig = plt.figure(2, figsize=(12,16))
    plt.xticks([]) # remove initial blank plot default ticks
    plt.yticks([]) # remove initial blank plot default ticks
    plt.title('Sub Plot Full Claims Data')
    rcParams['axes.titlepad'] = 50 # position title
    plt.box(on=None) # Remove boundary line
    """Full Loop Plot"""
    Full_Filtered_df = pd.concat([Past_Filtered_df, Future_Filtered_df])
    i=0
    for row_A in range(0,len(Past_InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1+i)
        Year_i = Past_InsuredYr_List[row_A]
        Full_SubFiltered_df = Full_Filtered_df.loc[Full_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Full_SubFiltered_df['Year_Only_Lag'], Full_SubFiltered_df['ClaimAmt'], 
                 label=('Predicted %d' % Year_i), linestyle='--', color=Past_Color_List[row_A])
        plt.legend()
        i += 1
        plt.xticks(np.arange(0, (YearEndCap-YearStartCap), step=1))
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value') 
    """Past Loop Plot"""
    i=0
    for row_A in range(0,len(Past_InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1+i)
        Year_i = Past_InsuredYr_List[row_A]
        Past_SubFiltered_df = Past_Filtered_df.loc[Past_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Past_SubFiltered_df['Year_Only_Lag'], Past_SubFiltered_df['ClaimAmt'], 
                 label=('Historical %d' % Year_i), linestyle='-', color=Past_Color_List[row_A], marker='o')
        plt.legend()
        i += 1
    #"""Future Loop Plot"""
    #for row_B in range(0,len(Future_InsuredYr_List)):
    #    Year_i = Future_InsuredYr_List[row_B]
    #    Future_SubFiltered_df = Future_Filtered_df.loc[Future_Filtered_df['Insured_Year'].isin([Year_i])]
    #    plt.plot(Future_SubFiltered_df['Year_Only_Lag'], Future_SubFiltered_df['ClaimAmt'], 
    #             label=str(Year_i), linestyle='--', color=Future_Color_List[row_B])    
    """Plot Attributes"""    
    fig.tight_layout()
    plt.show()
SinglePlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_cumsum")
SubPlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_cumsum")
SinglePlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="Inflated_cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_Inflated_cumsum")
SubPlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="Inflated_cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_Inflated_cumsum")
InflatedReserves = Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'].sum()
print(InflatedReserves)
NonInflatedReserves = Predicted_df['FutureUplifted_Predicted_Incremental'].sum()
print(NonInflatedReserves)
PercDiff = 100*(InflatedReserves/NonInflatedReserves-1)
print('Percentage Difference {}'.format(PercDiff))