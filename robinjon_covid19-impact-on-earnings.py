import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install yahooquery
from yahooquery import Ticker
!pip install lightgbm
import matplotlib.pyplot as plt

ticker_details = pd.read_csv('../input/ticker-details-2/ticker_details (2).csv')#[0]


#nasdaq_list = pd.read_csv('../input/stock-market-dataset/symbols_valid_meta.csv')#[0]
#nasdaq_list.head()
symbols = ticker_details["ticker"].to_list()
# from yahooquery import Ticker
# all_financial_data = pd.DataFrame()
# all_financial_data_y = pd.DataFrame()
# count = 0


# # Make asynchronous requests
# for each in symbols:
#     count = count + 1
#     print(each, count)
#     faang = Ticker(each,asynchronous=True)
#     all_q = faang.all_financial_data('q')

#     all_a = faang.all_financial_data('a')


#     if (not(type(all_q) == str)):
#         #all_q["symbol"] = each
#         all_financial_data = all_financial_data.append(all_q)

#     if (not(type(all_a) == str)):
#         #all_a["symbol"] = each
#         all_financial_data_y = all_financial_data_y.append(all_a)
    
# all_financial_data.to_csv("all_financial_data.csv")
# all_financial_data_y.to_csv("all_financial_data_y.csv")
# all_financial_data.reset_index(inplace = True)#.head()
# all_financial_data_y.reset_index(inplace = True)
#Downloaded data from Yahoo. I am not making this public because of potential copyright issues. Suggest you use the above lines to create your own dataframe
all_financial_data = pd.read_csv("../input/all-financial-data/./all_financial_data.csv")
all_financial_data_y = pd.read_csv("../input/all-financial-data/./all_financial_data_y.csv")
all_financial_data.columns.to_list()

all_financial_data_filt = all_financial_data[all_financial_data["periodType"] == "3M"]
all_financial_data_filt.sort_values(by = ["symbol", "asOfDate"], inplace = True)
#Shortlist 3 quarters - FY19Q2 and FY20Q2

all_financial_data_2019_q2 = all_financial_data_filt[all_financial_data_filt["asOfDate"].isin(["2019-06-30", "2019-07-31", "2019-08-31"])]#["symbol"].unique()##.value_counts()
all_financial_data_2020_q2 = all_financial_data_filt[all_financial_data_filt["asOfDate"].isin(["2020-06-30", "2020-07-31", "2020-08-31"])]#["symbol"].unique()##.value_counts()
all_financial_data_2019_q3 = all_financial_data_filt[all_financial_data_filt["asOfDate"].isin(["2019-09-30", "2019-10-31", "2019-11-30"])]#["symbol"].unique()##.value_counts()
all_financial_data_2019_q2.set_index("symbol", inplace = True)
all_financial_data_2020_q2.set_index("symbol", inplace = True)
all_financial_data_2019_q3.set_index("symbol", inplace = True)
#Retain only common sticker symbols

all_financial_data_2019_q2_t = all_financial_data_2019_q2[~all_financial_data_2019_q2.NetIncome.isna()].merge(right = all_financial_data_2020_q2[~all_financial_data_2020_q2.NetIncome.isna()]["SalariesAndWages"], left_on = "symbol", right_on = "symbol", how = 'inner').drop(columns = ["SalariesAndWages_x", "SalariesAndWages_y"])
all_financial_data_2020_q2_t = all_financial_data_2020_q2[~all_financial_data_2020_q2.NetIncome.isna()].merge(right = all_financial_data_2019_q2[~all_financial_data_2019_q2.NetIncome.isna()]["SalariesAndWages"], left_on = "symbol", right_on = "symbol", how = 'inner').drop(columns = ["SalariesAndWages_x", "SalariesAndWages_y"])

all_financial_data_2019_q2_t.drop_duplicates(inplace = True)
all_financial_data_2020_q2_t.drop_duplicates(inplace = True)

all_financial_data_filt["asOfDate"].value_counts()
# Which companies have more than 10% decline in NetIncome?

test_yoy_q = pd.to_numeric(all_financial_data_2020_q2_t.NetIncome) - pd.to_numeric(all_financial_data_2019_q2_t.NetIncome)

output = test_yoy_q/abs(pd.to_numeric(all_financial_data_2019_q2_t.NetIncome))
output_pc = (output<-.1)

percentage = sum(test_yoy_q<0)/len(test_yoy_q)
print('Percentage of companies with negative earnings growth from 2019Q2 to 2020Q2 = {:2.2%}'.format(percentage))
all_financial_data_2019_q2_t = all_financial_data_2019_q2[~all_financial_data_2019_q2.NetIncome.isna()].merge(right = all_financial_data_2019_q3[~all_financial_data_2019_q3.NetIncome.isna()]["SalariesAndWages"], left_on = "symbol", right_on = "symbol", how = 'inner').drop(columns = ["SalariesAndWages_x", "SalariesAndWages_y"])
all_financial_data_2019_q3_t = all_financial_data_2019_q3[~all_financial_data_2019_q3.NetIncome.isna()].merge(right = all_financial_data_2019_q2[~all_financial_data_2019_q2.NetIncome.isna()]["SalariesAndWages"], left_on = "symbol", right_on = "symbol", how = 'inner').drop(columns = ["SalariesAndWages_x", "SalariesAndWages_y"])

all_financial_data_2019_q2_t.drop_duplicates(inplace = True)
all_financial_data_2019_q3_t.drop_duplicates(inplace = True)

test_qoq = pd.to_numeric(all_financial_data_2019_q3_t.NetIncome) - pd.to_numeric(all_financial_data_2019_q2_t.NetIncome)

percentage = sum(test_qoq<0)/len(test_qoq)
print('Percentage of companies with negative earnings growth from 2019Q2 to 2019Q3 = {:2.2%}'.format(percentage))
#Filter only yearly values. Harmonize rows so that only common rows are retained

all_financial_data_filt_y = all_financial_data_y[all_financial_data_y["periodType"] == "12M"]
all_financial_data_filt_y.drop_duplicates(inplace = True)
all_financial_data_filt_y.sort_values(by = ["symbol", "asOfDate"], inplace = True)#head()#.columns#['symbol'].value_counts()

all_financial_data_2019_y = all_financial_data_filt_y[pd.to_datetime(all_financial_data_filt_y["asOfDate"]).dt.year == 2019]
all_financial_data_2019_y.set_index("symbol", inplace = True)

all_financial_data_2019_y_t_t = all_financial_data_2019_y[~all_financial_data_2019_y.NetIncome.isna()].merge(right = all_financial_data_2020_q2_t[~all_financial_data_2020_q2_t.NetIncome.isna()]["BasicAverageShares"], left_on = "symbol", right_on = "symbol", how = 'inner').drop(columns = ["BasicAverageShares_x", "BasicAverageShares_y"])


#Remove zero revenue lines
all_financial_data_2019_y_t_t = all_financial_data_2019_y_t_t[all_financial_data_2019_y_t_t["TotalRevenue"] != 0]
#Augment with industry sector etc.
ticker_details = ticker_details[ticker_details["country"] == "United States"]
ticker_details.drop(columns = ["country"], inplace = True)
#Harmonize rows so that only common rows are retained
all_financial_data_2019_y_t = all_financial_data_2019_y_t_t.merge(right = ticker_details, how = 'inner', left_on = "symbol", right_on = "ticker")#.drop(columns = "ticker")
output_pc = output_pc.loc[all_financial_data_2019_y_t.ticker]
#Feature selection with 2 objectives:
# 1. Include fields with high level of data availability (less NaNs etc.)
# 2. Avoid multi-collinearity - e.g. Gross Profit = Revenue - COGS, so cant pcik three, just pick just two etc. 


all_financial_data_2019_y_t["MarketCapToNetIncome"] = all_financial_data_2019_y_t["MarketCap"]/all_financial_data_2019_y_t["NetIncome"]
all_financial_data_2019_y_t["OpexToRevenue"] = all_financial_data_2019_y_t["OperatingExpense"]/all_financial_data_2019_y_t["TotalRevenue"]
all_financial_data_2019_y_t["EBITToRevenue"] = all_financial_data_2019_y_t["EBIT"]/all_financial_data_2019_y_t["TotalRevenue"]
all_financial_data_2019_y_t["NetIncomeToRevenue"] = all_financial_data_2019_y_t["NetIncome"]/all_financial_data_2019_y_t["TotalRevenue"]
all_financial_data_2019_y_t["AccountsPayableToRevenue"] = all_financial_data_2019_y_t["AccountsPayable"]/all_financial_data_2019_y_t["TotalRevenue"]
all_financial_data_2019_y_t["AccountsReceivableToRevenue"] = all_financial_data_2019_y_t["AccountsReceivable"]/all_financial_data_2019_y_t["TotalRevenue"]
all_financial_data_2019_y_t["CapitalExpenditureToRevenue"] = all_financial_data_2019_y_t["CapitalExpenditure"]/all_financial_data_2019_y_t["TotalRevenue"]
all_financial_data_2019_y_t["FreeCashFlowToRevenue"] = all_financial_data_2019_y_t["FreeCashFlow"]/all_financial_data_2019_y_t["TotalRevenue"]





all_financial_data_2019_y_for_model = all_financial_data_2019_y_t[['TotalRevenue',
                                                                   # 'GrossProfit',
                                                                #"OperatingExpense",
                                                                "OpexToRevenue",
                                                               #'SellingGeneralAndAdministration',
                                                                "EBITToRevenue",
                                                                #'EBIT', 
                                                                "NetIncomeToRevenue",
                                                                #'NetIncome', 
                                                                "BasicEPS",#'industry', 
                                                                   'sector', 'fullTimeEmployees', "ticker", 
                                                                #"AccountsPayable", "AccountsReceivable", "CapitalExpenditure", #"CashAndCashEquivalents", 
                                                                #"FreeCashFlow", 
                                                                "AccountsPayableToRevenue", "AccountsReceivableToRevenue", "CapitalExpenditureToRevenue", "FreeCashFlowToRevenue",
                                                                "MarketCap", 
                                                                "MarketCapToNetIncome"]]


#all_financial_data_2019_y_for_model['industry'] = all_financial_data_2019_y_for_model['industry'].astype('category')
all_financial_data_2019_y_for_model['sector'] = all_financial_data_2019_y_for_model['sector'].astype('category')
all_financial_data_2019_y_for_model.shape
import lightgbm
from sklearn.model_selection import train_test_split
cats = [#"industry", 
    "sector"]
all_financial_data_2019_y_for_model.index = all_financial_data_2019_y_for_model["ticker"]

for each in all_financial_data_2019_y_for_model.columns:
    if each not in cats+["ticker"]:
        all_financial_data_2019_y_for_model[each] = pd.to_numeric(all_financial_data_2019_y_for_model[each])

x, x_test, y, y_test = train_test_split(all_financial_data_2019_y_for_model.drop(columns = ["ticker"]), output_pc, test_size=0.2, random_state=42)


train_data = lightgbm.Dataset(x, label=y, categorical_feature=cats)
test_data = lightgbm.Dataset(x_test, label=y_test, categorical_feature=cats)


#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.4,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 1
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=[train_data, test_data],
                       num_boost_round=5000,
                       early_stopping_rounds=100)
lightgbm.plot_importance(model)
lightgbm.plot_tree(model, figsize=(100, 150))
#Prep for shap
non_cats = set(all_financial_data_2019_y_for_model.columns) - set(cats) - {"ticker"}
x = x.append(x_test)
x_shap = x.copy()
for each in cats:
    x_shap[each] = x_shap[each].cat.codes
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], x_shap)
shap.summary_plot(shap_values[0], x, plot_type="bar")
shap.summary_plot(shap_values[1], x_shap)
print((x_shap["CapitalExpenditureToRevenue"]>-.2).value_counts())
shap.dependence_plot("CapitalExpenditureToRevenue",shap_values[1][x_shap["CapitalExpenditureToRevenue"]>-.2], x_shap[x_shap["CapitalExpenditureToRevenue"]>-.2])
#MarketCapToNetIncome - Faith of market in valuations
mask = ((x_shap["MarketCapToNetIncome"]<100) & (x_shap["MarketCapToNetIncome"]>0))
print(mask.value_counts())
shap.dependence_plot("MarketCapToNetIncome",shap_values[1][mask], x_shap[mask])
#MarketCapToNetIncome - Faith of market in valuations
mask = ((x_shap["sector"]<50) & (x_shap["sector"]>=0))
print(mask.value_counts())
shap.dependence_plot("sector",shap_values[1][mask], x_shap[mask])
pd.DataFrame(x["sector"].cat.categories.to_list(), columns =["Sector"])
shap.initjs()
sample_t = "CRM"
print(output_pc[sample_t])
shap.force_plot(explainer.expected_value[1], shap_values[1][x_shap.index == sample_t,:], x_shap.loc[sample_t,:])
print((x_shap["OpexToRevenue"]<2).value_counts())

shap.dependence_plot("OpexToRevenue",shap_values[1][x_shap["OpexToRevenue"]<2], x_shap[x_shap["OpexToRevenue"]<2])
#AccountsPayableToRevenue, possinle industry trends, highr accounta payable indicative of cashflow problems?
print(((x_shap["AccountsPayableToRevenue"]<.2) & (x_shap["AccountsPayableToRevenue"]>0)).value_counts())
shap.dependence_plot("AccountsPayableToRevenue",shap_values[1][(x_shap["AccountsPayableToRevenue"]<.2) & (x_shap["AccountsPayableToRevenue"]>0)], x_shap[(x_shap["AccountsPayableToRevenue"]<.2) & (x_shap["AccountsPayableToRevenue"]>0)])
#FreeCashFlowToRevenue
print(((x_shap["FreeCashFlowToRevenue"]<.1) & (x_shap["FreeCashFlowToRevenue"]>-.0)).value_counts())
shap.dependence_plot("FreeCashFlowToRevenue",shap_values[1][(x_shap["FreeCashFlowToRevenue"]<.1) & (x_shap["FreeCashFlowToRevenue"]>-.0)], x_shap[(x_shap["FreeCashFlowToRevenue"]<.1) & (x_shap["FreeCashFlowToRevenue"]>-.0)])
#FreeCashFlowToRevenue
print(((x_shap["NetIncomeToRevenue"]<.1) & (x_shap["NetIncomeToRevenue"]>-.0)).value_counts())
shap.dependence_plot("NetIncomeToRevenue",shap_values[1][(x_shap["NetIncomeToRevenue"]<.1) & (x_shap["NetIncomeToRevenue"]>-.0)], x_shap[(x_shap["NetIncomeToRevenue"]<.1) & (x_shap["NetIncomeToRevenue"]>-.0)])
#MarketCapToNetIncome - Faith of market in valuations
mask = ((x_shap["BasicEPS"]<5) & (x_shap["MarketCapToNetIncome"]>0))
print(mask.value_counts())
shap.dependence_plot("BasicEPS",shap_values[1][mask], x_shap[mask])
#MarketCapToNetIncome - Faith of market in valuations
mask = ((x_shap["TotalRevenue"]<100000000000))
print(mask.value_counts())
shap.dependence_plot("TotalRevenue",shap_values[1][mask], x_shap[mask])